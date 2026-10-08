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
    assert out[0].split()[0] == "AGENT"
    assert out[1].startswith("abc123  [abc123]")
    assert out[-1].strip() == "(no subagents)"


def test_nothing_found_prints_reason_and_exits_one(fake, capsys):
    assert main(["--color", "never"]) == 1

    assert capsys.readouterr().out.strip() == (
        "no running Antigravity sessions; no running Claude Code sessions; "
        "no running Codex sessions; no running Grok Build sessions")


def test_window_option_reaches_every_agent(fake, capsys):
    fake.session("abc123", "/w/a", [assistant(0, context=100_000)])

    main(["abc123", "--color", "never", "--window", "200000"])

    assert "50 % of 200k" in capsys.readouterr().out


def test_transcript_text_is_sanitised_before_printing(fake, capsys):
    fake.session("abc123", "/w/a", [{"type": "ai-title", "aiTitle": "evil\x1b[2Jtitle"},
                                    assistant(0)])

    main(["abc123", "--color", "never"])

    assert "\x1b" not in capsys.readouterr().out


def test_watch_flag_does_not_swallow_the_target():
    args = parse_args(["-w", "."])

    assert (args.watch, args.target, args.interval) == (True, ".", 2.0)


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
