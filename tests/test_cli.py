import pytest
from conftest import assistant

from agents_tree.cli import main


def test_prints_tree_and_exits_zero(fake, capsys):
    fake.session("abc123", "/w/a", [assistant(0)])

    assert main(["abc123", "--color", "never"]) == 0

    out = capsys.readouterr().out
    assert out.splitlines()[0].split()[0] == "AGENT"
    assert out.splitlines()[1].startswith("abc123  [abc123]")
    assert "(no subagents)" in out


def test_nothing_found_prints_reason_and_exits_one(fake, capsys):
    assert main(["--color", "never"]) == 1

    assert capsys.readouterr().out.strip() == "no running Claude Code sessions"


def test_window_option_reaches_the_renderer(fake, capsys):
    fake.session("abc123", "/w/a", [assistant(0, context=100_000)])

    main(["abc123", "--color", "never", "--window", "200000"])

    assert "50 % of 200k" in capsys.readouterr().out


def test_watch_flag_does_not_swallow_the_target():
    from agents_tree.cli import parse_args

    args = parse_args(["-w", "."])

    assert (args.watch, args.target, args.interval) == (True, ".", 2.0)


def test_interval_must_be_positive(capsys):
    import pytest

    from agents_tree.cli import parse_args

    with pytest.raises(SystemExit):
        parse_args(["-w", "-n", "0"])
    assert "--interval must be positive" in capsys.readouterr().err


@pytest.mark.parametrize(("data", "keys"), [
    ("q", {"quit"}), ("\x1b", {"quit"}), ("\x03", {"quit"}), ("r", {"toggle"}),
    ("\x1b[B", set()), ("\x1b[A\x1b[A\x1b[B", set()), ("\x1bOB", set()),
    ("\x1b[<64;10;5M", set()), ("\x1b[Bq", {"quit"}),
])
def test_parse_keys(data, keys):
    from agents_tree.cli import parse_keys

    assert parse_keys(data) == keys


@pytest.mark.parametrize("value", ["", "200k", "-5"])
def test_bad_window_env_is_ignored(monkeypatch, value):
    from agents_tree.cli import parse_args

    monkeypatch.setenv("AGENTS_TREE_WINDOW", value)

    assert parse_args([]).window is None
