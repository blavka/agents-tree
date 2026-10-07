import json
import os
import stat

import pytest

import agents_tree.cli
from agents_tree import herdr

CLAUDE_PANE = {"pane_id": "w1:p1", "agent": "claude", "cwd": "/w/app",
               "agent_session": {"agent": "claude", "kind": "id", "value": "sess-1"}}


@pytest.fixture
def fake_herdr(tmp_path, monkeypatch):
    """A herdr stand-in: logs its argv, answers `pane get` from panes.json, passes config check."""
    log = tmp_path / "calls.log"
    panes = tmp_path / "panes.json"
    panes.write_text("{}")
    script = tmp_path / "herdr"
    script.write_text(f"""#!/usr/bin/env python3
import json, sys
open({str(log)!r}, "a").write(json.dumps(sys.argv[1:]) + "\\n")
if sys.argv[1:3] == ["pane", "get"]:
    pane = json.load(open({str(panes)!r})).get(sys.argv[3])
    print(json.dumps({{"result": {{"pane": pane, "type": "pane_info"}}}} if pane else {{}}))
if sys.argv[1:3] == ["config", "check"]:
    sys.exit(1 if "BROKEN" in open({str(tmp_path / "config.toml")!r}).read() else 0)
""")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("HERDR_BIN_PATH", str(script))
    monkeypatch.setenv("HERDR_CONFIG_PATH", str(tmp_path / "config.toml"))
    monkeypatch.setenv("HERDR_PLUGIN_ID", "agents-tree")
    monkeypatch.setenv("HERDR_PLUGIN_STATE_DIR", str(tmp_path / "state"))

    class Fake:
        config = tmp_path / "config.toml"

        def set_panes(self, mapping):
            panes.write_text(json.dumps(mapping))

        def focus(self, pane_id):
            context = json.dumps({"focused_pane_id": pane_id})
            monkeypatch.setenv("HERDR_PLUGIN_CONTEXT_JSON", context)

        def calls(self):
            if not log.exists():
                return []
            return [json.loads(line) for line in log.read_text().splitlines()]

    return Fake()


GROK_PANE = {"pane_id": "w1:p2", "agent": "grok", "cwd": "/w/app",
             "agent_session": {"agent": "grok", "kind": "id", "value": "grok-sess"}}


@pytest.mark.parametrize(("pane", "target"), [
    (CLAUDE_PANE, ("claude", "sess-1")),
    (GROK_PANE, ("grok", "grok-sess")),
    ({"agent": "claude", "cwd": "/w/app", "foreground_cwd": "/w/app/sub"},
     ("claude", "/w/app/sub")),
    ({"agent": "claude", "cwd": "/w/app"}, ("claude", "/w/app")),
    ({"agent": "grok", "cwd": "/w/app"}, ("grok", "/w/app")),
    ({"agent": "codex", "agent_session": {"agent": "codex", "kind": "id", "value": "x"}},
     ("claude", "")),
    ({"cwd": "/w/app"}, ("claude", "")),
    ({}, ("claude", "")),
])
def test_target_for(pane, target):
    assert herdr.target_for(pane) == target


def test_open_passes_the_focused_session_to_the_pane(fake_herdr):
    fake_herdr.set_panes({"w1:p1": CLAUDE_PANE})
    fake_herdr.focus("w1:p1")

    herdr.run(["open", "focused"])

    assert fake_herdr.calls()[-1] == [
        "plugin", "pane", "open", "--plugin", "agents-tree", "--entrypoint", "tree",
        "--placement", "overlay", "--env", "AGENTS_TREE_PROVIDER=claude",
        "--env", "AGENTS_TREE_TARGET=sess-1"]


def test_open_all_ignores_the_focused_session(fake_herdr):
    fake_herdr.set_panes({"w1:p1": CLAUDE_PANE})
    fake_herdr.focus("w1:p1")

    herdr.run(["open", "all", "split"])

    call = fake_herdr.calls()[-1]
    assert call[call.index("--placement") + 1] == "split"
    assert call[-1] == "AGENTS_TREE_TARGET="


def test_open_on_our_own_running_pane_closes_it(fake_herdr):
    fake_herdr.focus("w1:p9")
    mark = herdr._mark_path("w1:p9")
    mark.parent.mkdir(parents=True)
    mark.write_text(str(os.getpid()))

    herdr.run(["open", "focused"])

    assert fake_herdr.calls()[-1] == ["plugin", "pane", "close", "w1:p9"]


def test_a_mark_left_by_a_dead_pane_does_not_count(fake_herdr):
    fake_herdr.focus("w1:p9")
    mark = herdr._mark_path("w1:p9")
    mark.parent.mkdir(parents=True)
    mark.write_text("999999999")

    herdr.run(["open", "focused"])

    assert fake_herdr.calls()[-1][:3] == ["plugin", "pane", "open"]


def test_a_pane_titled_like_us_is_not_mistaken_for_ours(fake_herdr):
    fake_herdr.set_panes({"w1:p3": {"pane_id": "w1:p3", "label": "agents-tree"}})
    fake_herdr.focus("w1:p3")

    herdr.run(["open", "focused"])

    assert fake_herdr.calls()[-1][:3] == ["plugin", "pane", "open"]


def test_open_without_context_opens_all_sessions(fake_herdr, monkeypatch):
    monkeypatch.delenv("HERDR_PLUGIN_CONTEXT_JSON", raising=False)

    herdr.run(["open", "focused"])

    assert fake_herdr.calls()[-1][-1] == "AGENTS_TREE_TARGET="


def test_bad_arguments_print_usage(capsys):
    assert herdr.run(["open", "focused", "sideways"]) == 2
    assert "usage:" in capsys.readouterr().err


def test_setup_keys_appends_a_block_and_remove_keys_takes_it_out(fake_herdr):
    original = '[[keys.command]]\nkey = "prefix+f"\ntype = "plugin_action"\ncommand = "x.y"\n'
    fake_herdr.config.write_text(original)

    assert herdr.run(["keys", "setup"]) == 0
    text = fake_herdr.config.read_text()
    assert text.startswith(original.rstrip("\n"))
    assert 'key = "prefix+a"' in text and 'command = "agents-tree.open"' in text
    assert 'key = "prefix+shift+a"' in text and 'command = "agents-tree.open-all"' in text
    assert (fake_herdr.config.parent / "config.toml.agents-tree-backup").read_text() == original
    assert ["server", "reload-config"] in fake_herdr.calls()

    assert herdr.run(["keys", "remove"]) == 0
    assert fake_herdr.config.read_text() == original


def test_setup_keys_twice_keeps_one_block(fake_herdr):
    herdr.run(["keys", "setup"])
    herdr.run(["keys", "setup"])

    assert fake_herdr.config.read_text().count(herdr.BEGIN) == 1


def test_setup_keys_leaves_bound_keys_alone(fake_herdr):
    fake_herdr.config.write_text('[[keys.command]]\nkey = "prefix+a"\ncommand = "other.thing"\n')

    herdr.run(["keys", "setup"])

    block = fake_herdr.config.read_text().split(herdr.BEGIN)[1]
    assert 'key = "prefix+a"' not in block
    assert 'key = "prefix+shift+a"' in block


def test_single_quoted_or_uppercase_bindings_count_as_taken(fake_herdr):
    fake_herdr.config.write_text("[[keys.command]]\nkey = 'Prefix+A'\ncommand = 'x.y'\n")

    herdr.run(["keys", "setup"])

    block = fake_herdr.config.read_text().split(herdr.BEGIN)[1]
    assert 'key = "prefix+a"' not in block


def test_setup_keys_refuses_a_config_that_already_fails_check(fake_herdr):
    fake_herdr.config.write_text("BROKEN\n")

    assert herdr.run(["keys", "setup"]) == 1
    assert fake_herdr.config.read_text() == "BROKEN\n"


def test_unterminated_block_is_refused_and_the_config_kept(fake_herdr):
    broken = f'{herdr.BEGIN}\n[[keys.command]]\nkey = "prefix+a"\n\n[ui]\nimportant = true\n'
    fake_herdr.config.write_text(broken)

    assert herdr.run(["keys", "remove"]) == 1
    assert herdr.run(["keys", "setup"]) == 1
    assert fake_herdr.config.read_text() == broken


def test_remove_keys_backs_up_first(fake_herdr):
    herdr.run(["keys", "setup"])
    with_keys = fake_herdr.config.read_text()

    herdr.run(["keys", "remove"])

    assert (fake_herdr.config.parent / "config.toml.agents-tree-backup").read_text() == with_keys


def test_pane_marks_itself_while_it_runs_and_passes_its_target(fake_herdr, monkeypatch):
    seen = []

    def fake_main(argv):
        seen.append((argv, herdr.is_our_pane("w1:p7")))
        return 0

    monkeypatch.setattr(agents_tree.cli, "main", fake_main)
    monkeypatch.setenv("HERDR_PANE_ID", "w1:p7")
    monkeypatch.setenv("AGENTS_TREE_TARGET", "-weird-dir")

    herdr.run(["pane"])

    assert seen == [(["-w", "--provider", "claude", "--", "-weird-dir"], True)]
    assert not herdr.is_our_pane("w1:p7")


def test_pane_shows_a_crash_and_waits(monkeypatch, capsys):
    def boom(argv):
        raise RuntimeError("kaput")

    waited = []
    monkeypatch.setattr(agents_tree.cli, "main", boom)
    monkeypatch.setattr("builtins.input", lambda prompt="": waited.append(prompt))

    assert herdr.run(["pane"]) == 1
    assert "kaput" in capsys.readouterr().err
    assert waited


def test_marks_fall_back_to_a_private_directory_not_tmp(monkeypatch, tmp_path):
    monkeypatch.delenv("HERDR_PLUGIN_STATE_DIR", raising=False)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))

    d = herdr._private_marks_dir()

    assert d == tmp_path / "run" / "agents-tree-panes"
    assert stat.S_IMODE(d.stat().st_mode) == 0o700


def test_a_symlinked_marks_directory_is_refused(monkeypatch, tmp_path):
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "agents-tree-panes").symlink_to(tmp_path / "elsewhere")
    monkeypatch.setenv("HERDR_PLUGIN_STATE_DIR", str(tmp_path / "state"))

    with pytest.raises(PermissionError):
        herdr._private_marks_dir()


def test_a_planted_symlink_mark_is_not_followed(fake_herdr, monkeypatch, tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me")
    mark = herdr._mark_path("w1:p7")
    mark.parent.mkdir(parents=True, mode=0o700)
    mark.symlink_to(victim)
    monkeypatch.setattr(agents_tree.cli, "main", lambda argv: 0)
    monkeypatch.setenv("HERDR_PANE_ID", "w1:p7")

    herdr.run(["pane"])

    assert victim.read_text() == "keep me"
