import json
import re
from pathlib import Path

import pytest
from conftest import assistant

import agents_tree
from agents_tree import cli
from agents_tree.cli import main, parse_args
from agents_tree.model import Agent, Session

ROOT = Path(__file__).resolve().parent.parent


def test_prints_tree_and_exits_zero(fake, capsys):
    fake.session("abc123", "/w/a", [assistant(0)])

    assert main(["abc123", "--color", "never"]) == 0

    out = capsys.readouterr().out.splitlines()
    assert "CONTEXT" in out[0] and out[1].startswith("– 1 inactive")
    assert out[-1].startswith(" – abc123 ") and out[-1].endswith("not running")


def test_nothing_found_prints_reason_and_exits_one(fake, capsys):
    assert main(["--color", "never"]) == 1

    assert capsys.readouterr().out.strip() == (
        "no running Antigravity sessions; no running Claude Code sessions; "
        "no running Codex sessions; no running Grok Build sessions")


def test_window_option_reaches_every_agent(fake, capsys):
    fake.session("abc123", "/w/a", [assistant(0, context=100_000)])

    main(["abc123", "--color", "never", "--window", "200000"])

    assert "━━━━────  50%" in capsys.readouterr().out


def test_transcript_text_is_sanitised_before_printing(fake, capsys):
    fake.session("abc123", "/w/a", [{"type": "ai-title", "aiTitle": "evil\x1b[2Jtitle"},
                                    assistant(0)])

    main(["abc123", "--color", "never"])

    assert "\x1b" not in capsys.readouterr().out


def test_json_output_has_versioned_neutral_model(fake, capsys):
    fake.session("abc123", "/w/a", [assistant(0, context=100_000)])

    assert main(["abc123", "--json"]) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["schema_version"] == 1
    [session] = result["sessions"]
    assert (session["id"], session["cwd"], session["kind"]) == ("abc123", None, None)
    assert session["main"]["context_tokens"] == 100_000
    assert session["main"]["detail"]["transcript"].endswith("abc123.jsonl")


def test_json_errors_go_to_stderr(fake, capsys):
    assert main(["missing", "--json"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "no Claude Code session matching 'missing'" in captured.err


def test_watch_flag_does_not_swallow_the_target():
    args = parse_args(["-w", "."])

    assert (args.watch, args.target, args.interval) == (True, ".", 2.0)


def test_json_cannot_watch(capsys):
    with pytest.raises(SystemExit):
        parse_args(["--json", "--watch"])
    assert "--json cannot be used with --watch" in capsys.readouterr().err


def test_interval_must_be_positive(capsys):
    with pytest.raises(SystemExit):
        parse_args(["-w", "-n", "0"])
    assert "--interval must be positive" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["", "200k", "-5"])
def test_bad_window_env_is_ignored(monkeypatch, value):
    monkeypatch.setenv("AGENTS_TREE_WINDOW", value)

    assert parse_args([]).window is None


def test_version_is_the_same_everywhere():
    def version(name):
        found = re.search(r'^version = "([^"]+)"', (ROOT / name).read_text(), re.M)
        return found and found.group(1)

    assert version("pyproject.toml") == agents_tree.__version__ == version("herdr-plugin.toml")


class _Provider:
    def __init__(self, *names, missing="nothing here"):
        self.names, self.missing = names, missing

    def sessions(self, target):
        if not self.names:
            raise LookupError(self.missing)
        return [Session(id=n, title=n, main=Agent(id="main", label="main")) for n in self.names]


def test_all_providers_are_asked_and_their_sessions_merged(monkeypatch):
    monkeypatch.setattr(cli, "PROVIDERS", {"a": _Provider("a1"), "b": _Provider("b1", "b2")})

    assert [s.id for s in cli._sessions("all", None)] == ["a1", "b1", "b2"]


def test_a_provider_that_finds_nothing_is_skipped(monkeypatch):
    monkeypatch.setattr(cli, "PROVIDERS", {"a": _Provider(), "b": _Provider("b1")})

    assert [s.id for s in cli._sessions("all", None)] == ["b1"]


def test_nothing_anywhere_reports_every_provider(monkeypatch):
    monkeypatch.setattr(cli, "PROVIDERS", {"a": _Provider(missing="no a"),
                                           "b": _Provider(missing="no b")})

    with pytest.raises(LookupError, match="^no a; no b$"):
        cli._sessions("all", None)


def test_one_provider_is_asked_alone(monkeypatch):
    monkeypatch.setattr(cli, "PROVIDERS", {"a": _Provider("a1"), "b": _Provider("b1")})

    assert [s.id for s in cli._sessions("b", None)] == ["b1"]


def test_provider_defaults_to_all():
    assert parse_args([]).provider == "all"


def test_agy_provider_name_is_accepted():
    assert parse_args(["--provider", "agy"]).provider == "agy"


def _at(sid, cwd):
    return Session(id=sid, title=sid, main=Agent(id="main", label="main"), cwd=cwd)


@pytest.fixture
def around(tmp_path, monkeypatch):
    """Sessions in a repo, in its subfolder, in a sibling repo, and one with no directory;
    the shell sits in the repo's subfolder."""
    repo, sub, other = tmp_path / "repo", tmp_path / "repo" / "src", tmp_path / "other"
    for d in (sub, other):
        d.mkdir(parents=True)
    sessions = [_at("in-repo", str(repo)), _at("in-sub", str(sub)),
                _at("in-other", str(other)), _at("nowhere", None)]
    monkeypatch.setattr(cli, "_sessions", lambda provider, target: sessions)
    monkeypatch.chdir(sub)
    return tmp_path


def test_session_flag_keeps_the_sessions_of_the_current_directory(around):
    load = cli.loader(parse_args(["-s"]))

    assert [s.id for s in load()] == ["in-repo", "in-sub"]


def test_session_flag_also_keeps_sessions_below_the_current_directory(around, monkeypatch):
    monkeypatch.chdir(around)

    assert [s.id for s in cli.loader(parse_args(["--session"]))()] == [
        "in-repo", "in-sub", "in-other"]


def test_without_the_session_flag_every_session_stays(around):
    assert len(cli.loader(parse_args([]))()) == 4


def test_session_flag_with_nothing_here_says_so(around, monkeypatch, tmp_path_factory):
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    monkeypatch.chdir(elsewhere)

    with pytest.raises(LookupError, match="^no running sessions in .*elsewhere"):
        cli.loader(parse_args(["-s"]))()


def test_session_flag_and_a_target_do_not_mix(capsys):
    with pytest.raises(SystemExit):
        parse_args(["-s", "abc123"])

    assert "--session cannot be used with a target" in capsys.readouterr().err


DAY = 86400


def _waiting(sid: str, state: str, idle_secs: float | None, now: float,
             pid: int | None = None) -> Session:
    ended = None if idle_secs is None else now - idle_secs
    return Session(id=sid, title=sid, main=Agent(id="main", label="main", state=state,
                                                  ended=ended), pid=pid)


@pytest.fixture
def idle(monkeypatch):
    """Sessions waiting, blocked, working and not running, for various times."""
    now = 1_000_000_000.0
    sessions = [_waiting("waiting-5d", "waiting", 5 * DAY, now),
                _waiting("blocked-19d", "blocked", 19 * DAY, now),
                _waiting("waiting-1h", "waiting", 3600, now),
                _waiting("running", "running", None, now),
                _waiting("inactive-30d", "inactive", 30 * DAY, now)]
    monkeypatch.setattr(cli, "_sessions", lambda provider, target: sessions)
    monkeypatch.setattr(cli.time, "time", lambda: now)


def test_idle_flag_keeps_live_sessions_waiting_at_least_that_long(idle):
    assert [s.id for s in cli.loader(parse_args(["-i", "3d"]))()] == [
        "waiting-5d", "blocked-19d"]


def test_idle_flag_takes_combined_units(idle):
    assert [s.id for s in cli.loader(parse_args(["--idle", "0d30m"]))()] == [
        "waiting-5d", "blocked-19d", "waiting-1h"]


def test_idle_flag_with_nothing_that_old_says_so(idle):
    with pytest.raises(LookupError, match="^no sessions idle for that long"):
        cli.loader(parse_args(["-i", "60d"]))()


@pytest.mark.parametrize("value", ["3", "d", "3days", "3d ago", "0s", "-1d"])
def test_idle_flag_rejects_what_is_not_a_duration(value, capsys):
    with pytest.raises(SystemExit):
        parse_args(["-i", value])

    assert "--idle" in capsys.readouterr().err


NOW = 1_000_000_000.0


@pytest.fixture
def closing(monkeypatch):
    """Idle sessions to close: one with a known process, one background job without,
    one too recent. os.kill plays a process table: SIGTERM ends a process unless it is
    in `stubborn`, signal 0 tells whether it still runs."""
    state = {"calls": 0, "sessions": [
        _waiting("idle-5d", "waiting", 5 * DAY, NOW, pid=101),
        _waiting("job-19d", "blocked", 19 * DAY, NOW),
        _waiting("idle-1h", "waiting", 3600, NOW, pid=103),
        _waiting("other-6d", "waiting", 6 * DAY, NOW, pid=104)]}
    killed: list[tuple[int, int]] = []
    alive, stubborn = {101, 103, 104}, set()

    def sessions(provider, target):
        state["calls"] += 1
        return state["sessions"]

    def kill(pid, sig):
        if pid not in alive:
            raise ProcessLookupError
        if sig:
            killed.append((pid, sig))
            if pid not in stubborn:
                alive.discard(pid)

    monkeypatch.setattr(cli, "_sessions", sessions)
    monkeypatch.setattr(cli.time, "time", lambda: NOW)
    monkeypatch.setattr(cli.os, "kill", kill)
    monkeypatch.setattr(cli, "EXIT_WAIT_SECS", 0)
    state.update(killed=killed, alive=alive, stubborn=stubborn)
    return state


def _answer(monkeypatch, text):
    monkeypatch.setattr("builtins.input", lambda prompt: text)


def test_kill_closes_the_idle_sessions_with_a_known_process_after_yes(closing, monkeypatch,
                                                                       capsys):
    _answer(monkeypatch, "y")

    assert main(["-ki", "3d", "--color", "never"]) == 0

    assert closing["killed"] == [(101, cli.signal.SIGTERM), (104, cli.signal.SIGTERM)]
    out = capsys.readouterr().out
    assert "job-19d job-19d: its process is not known" in out
    assert "closed 2 of 2" in out


@pytest.mark.parametrize("answer", ["", "n", "no", "sí"])
def test_kill_closes_nothing_without_a_yes(closing, monkeypatch, answer):
    _answer(monkeypatch, answer)

    assert main(["-i", "3d", "-k", "--color", "never"]) == 1

    assert not closing["killed"]


def test_kill_closes_nothing_when_there_is_no_one_to_ask(closing, monkeypatch):
    def eof(prompt):
        raise EOFError
    monkeypatch.setattr("builtins.input", eof)

    assert main(["-ki", "3d", "--color", "never"]) == 1

    assert not closing["killed"]


def test_kill_leaves_open_a_session_that_got_busy_while_asking(closing, monkeypatch, capsys):
    def answer_while_one_wakes_up(prompt):
        closing["sessions"][0] = _waiting("idle-5d", "running", None, NOW, pid=101)
        return "y"
    monkeypatch.setattr("builtins.input", answer_while_one_wakes_up)

    main(["-ki", "3d", "--color", "never"])

    assert closing["killed"] == [(104, cli.signal.SIGTERM)]
    assert "idle-5d idle-5d: active again, left open" in capsys.readouterr().out


def test_kill_goes_on_when_a_process_is_already_gone(closing, monkeypatch, capsys):
    _answer(monkeypatch, "y")
    closing["alive"].discard(101)

    main(["-ki", "3d", "--color", "never"])

    assert closing["killed"] == [(104, cli.signal.SIGTERM)]
    out = capsys.readouterr().out
    assert "idle-5d idle-5d: already gone" in out and "closed 1 of 2" in out


@pytest.mark.parametrize(("argv", "error"), [
    (["-k"], "-k/--kill needs -i/--idle"),
    (["-ki", "3d", "--json"], "cannot be used with --watch or --json"),
    (["-ki", "3d", "-w"], "cannot be used with --watch or --json"),
])
def test_kill_option_misuse_is_refused(argv, error, capsys):
    with pytest.raises(SystemExit):
        parse_args(argv)

    assert error in capsys.readouterr().err


def _status(out: str) -> list[str]:
    """The ids in the tree printed after closing."""
    after = out.split("to resume later\n", 1)[1]
    return [line.split()[1] for line in after.splitlines() if line.startswith(" ")]


def test_kill_ends_with_every_session_left_like_a_plain_run(closing, monkeypatch, capsys):
    _answer(monkeypatch, "y")

    main(["-ki", "3d", "--color", "never"])

    assert _status(capsys.readouterr().out) == ["job-19d", "idle-1h"]


def test_kill_status_keeps_a_session_whose_process_did_not_exit(closing, monkeypatch,
                                                                capsys):
    _answer(monkeypatch, "y")
    closing["stubborn"].add(104)

    main(["-ki", "3d", "--color", "never"])

    out = capsys.readouterr().out
    assert "other-6d other-6d: still running, it did not take SIGTERM" in out
    assert "closed 1 of 2" in out
    assert _status(out) == ["job-19d", "idle-1h", "other-6d"]


def test_kill_says_so_when_nothing_is_left(closing, monkeypatch, capsys):
    _answer(monkeypatch, "y")
    closing["sessions"][:] = [closing["sessions"][0]]

    main(["-ki", "3d", "--color", "never"])

    assert capsys.readouterr().out.rstrip().endswith("no running sessions left")
