import json
import stat

import pytest

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

    class Fake:
        def set_panes(self, mapping):
            panes.write_text(json.dumps(mapping))

        def focus(self, pane_id):
            monkeypatch.setenv("HERDR_PLUGIN_CONTEXT_JSON", json.dumps({"focused_pane_id": pane_id}))

        def calls(self):
            return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

        config = tmp_path / "config.toml"

    return Fake()


@pytest.mark.parametrize(("pane", "target"), [
    (CLAUDE_PANE, "sess-1"),
    ({"agent": "claude", "cwd": "/w/app", "foreground_cwd": "/w/app/sub"}, "/w/app/sub"),
    ({"agent": "claude", "cwd": "/w/app"}, "/w/app"),
    ({"agent": "codex", "agent_session": {"agent": "codex", "kind": "id", "value": "x"}}, ""),
    ({"cwd": "/w/app"}, ""),
    ({}, ""),
])
def test_target_for(pane, target):
    assert herdr.target_for(pane) == target


def test_open_passes_the_focused_session_to_the_pane(fake_herdr):
    fake_herdr.set_panes({"w1:p1": CLAUDE_PANE})
    fake_herdr.focus("w1:p1")

    herdr.run(["open", "focused"])

    assert fake_herdr.calls()[-1] == [
        "plugin", "pane", "open", "--plugin", "agents-tree", "--entrypoint", "tree",
        "--placement", "overlay", "--env", "AGENTS_TREE_TARGET=sess-1"]


def test_open_all_ignores_the_focused_session(fake_herdr):
    fake_herdr.set_panes({"w1:p1": CLAUDE_PANE})
    fake_herdr.focus("w1:p1")

    herdr.run(["open", "all", "split"])

    call = fake_herdr.calls()[-1]
    assert call[call.index("--placement") + 1] == "split"
    assert call[-1] == "AGENTS_TREE_TARGET="


def test_open_on_our_own_pane_closes_it(fake_herdr):
    fake_herdr.set_panes({"w1:p9": {"pane_id": "w1:p9", "label": "agents-tree"}})
    fake_herdr.focus("w1:p9")

    herdr.run(["open", "focused"])

    assert fake_herdr.calls()[-1] == ["plugin", "pane", "close", "w1:p9"]


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


def test_single_quoted_or_uppercase_bindings_count_as_taken(fake_herdr):
    fake_herdr.config.write_text("[[keys.command]]\nkey = 'Prefix+A'\ncommand = 'x.y'\n")

    herdr.run(["keys", "setup"])

    block = fake_herdr.config.read_text().split(herdr.BEGIN)[1]
    assert 'key = "prefix+a"' not in block


def test_pane_shows_a_crash_and_waits(monkeypatch, capsys):
    import agents_tree.cli

    def boom(argv):
        raise RuntimeError("kaput")

    waited = []
    monkeypatch.setattr(agents_tree.cli, "main", boom)
    monkeypatch.setattr("builtins.input", lambda prompt="": waited.append(prompt))

    assert herdr.run(["pane"]) == 1
    assert "kaput" in capsys.readouterr().err
    assert waited


def test_pane_passes_the_target_after_a_double_dash(monkeypatch):
    import agents_tree.cli

    seen = []
    monkeypatch.setattr(agents_tree.cli, "main", lambda argv: seen.append(argv) or 0)
    monkeypatch.setenv("AGENTS_TREE_TARGET", "-weird-dir")

    herdr.run(["pane"])

    assert seen == [["-w", "--", "-weird-dir"]]
