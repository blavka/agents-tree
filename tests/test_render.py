import re

import pytest

from agents_tree.model import Agent, Session
from agents_tree.render import fmt_duration, fmt_tokens, fmt_window, render

NOW = 10_000.0


def agent(label, status="completed", started=0.0, children=(), model="opus-5-5",
          effort: str | None = "high", ctx: int | None = 50_000, window=1_000_000):
    return Agent(id=label, label=label, model=model, effort=effort, context_tokens=ctx,
                 context_window=window, status=status, started=started,
                 ended=None if status == "running" else started + 60, children=list(children))


def session(agents, sid="s1", title="Work"):
    main = agent("main", status="busy", started=0.0)
    main.ended = NOW
    return Session(id=sid, provider="claude", title=title, main=main, agents=list(agents),
                   cwd="/w/app", kind="interactive")


def model_columns(text):
    """Column where each row's model text starts."""
    return {m.start() for line in text.splitlines()
            if (m := re.search(r"(opus|haiku)-", line))}


def test_rows_line_up_across_depths_and_sessions():
    text = render([
        session([agent("Explore: short", children=[agent("Plan: a much longer label here")])]),
        session([agent("x")], sid="s2"),
    ], now=NOW)

    assert len(model_columns(text)) == 1


def test_row_shows_model_effort_context_elapsed_and_status():
    text = render([session([agent("Explore: look", ctx=250_000)])], now=NOW)

    row = next(line for line in text.splitlines() if "Explore" in line)
    assert "opus-5-5 (high)" in row
    assert "250k  25 % of 1M" in row
    assert "1m00s" in row
    assert row.endswith("completed")


def test_running_agent_elapsed_counts_up_to_now():
    text = render([session([agent("busy", status="running", started=NOW - 90)])], now=NOW)

    assert "1m30s" in next(line for line in text.splitlines() if line.startswith("└─ busy"))


def test_window_override_changes_the_share():
    text = render([session([agent("a", ctx=100_000)])], now=NOW, window=200_000)

    assert "100k  50 % of 200k" in text


def test_effort_is_omitted_for_models_without_it():
    text = render([session([agent("g", model="haiku-4-5", effort=None)])], now=NOW)

    row = next(line for line in text.splitlines() if line.startswith("└─ g"))
    assert "haiku-4-5 " in row and "(" not in row


def test_tree_glyphs_mark_last_children():
    text = render([session([agent("a", children=[agent("a1"), agent("a2")]), agent("b")])],
                  now=NOW)

    labels = [re.split(r"\s{2,}opus", line)[0] for line in text.splitlines()[3:]]
    assert labels == ["├─ a", "│  ├─ a1", "│  └─ a2", "└─ b"]


def test_running_only_keeps_running_agents_and_their_parents():
    tree = [agent("done"), agent("parent", children=[agent("kid", status="running"),
                                                      agent("kid-done")])]
    text = render([session(tree)], now=NOW, running_only=True)

    assert "parent" in text and "kid" in text
    assert "done" not in text.replace("kid-done", "")
    assert "kid-done" not in text


def test_running_only_without_running_agents_says_so():
    text = render([session([agent("done")])], now=NOW, running_only=True)

    assert text.splitlines()[-1].strip() == "(no running subagents)"


def test_session_without_subagents_says_so():
    assert render([session([])], now=NOW).splitlines()[-1].strip() == "(no subagents)"


def test_fit_hides_oldest_finished_agents_first_and_never_running_ones():
    tree = [agent("old", started=1), agent("busy", status="running", started=2),
            agent("mid", started=3), agent("new", started=4)]

    text = render([session(tree)], now=NOW, max_lines=6)

    lines = text.splitlines()
    assert len(lines) <= 6
    shown = [line.split()[1] for line in lines if line.startswith(("├─", "└─"))]
    assert shown == ["busy", "new"]
    assert lines[-1].strip().startswith("… ") and "older finished agents hidden" in lines[-1]


def test_fit_counts_hidden_subtrees():
    tree = [agent("old", started=1, children=[agent("k1"), agent("k2")]), agent("new", started=5)]

    text = render([session(tree)], now=NOW, max_lines=5)

    assert "… 3 older finished agents hidden" in text


def test_narrow_width_truncates_labels_with_ellipsis():
    text = render([session([agent("general-purpose: " + "x" * 80)])], now=NOW, width=90)

    row = next(line for line in text.splitlines() if "general-purpose" in line)
    assert "…" in row
    assert len(row) <= 90


def test_color_wraps_status_in_ansi():
    text = render([session([agent("a", status="running", started=NOW)])], now=NOW, color=True)

    assert "\033[1;33mrunning\033[0m" in text


@pytest.mark.parametrize(("n", "out"), [(None, "-"), (999, "999"), (1500, "2k"),
                                        (742_000, "742k")])
def test_fmt_tokens(n, out):
    assert fmt_tokens(n) == out


def test_fmt_window():
    assert (fmt_window(1_000_000), fmt_window(200_000)) == ("1M", "200k")


@pytest.mark.parametrize(("secs", "out"), [(None, "-"), (-1, "-"), (5, "5s"), (65, "1m05s"),
                                           (3 * 3600 + 120, "3h02m")])
def test_fmt_duration(secs, out):
    assert fmt_duration(secs) == out


def test_header_names_the_columns_and_lines_up_with_them():
    text = render([session([agent("Explore: look")])], now=NOW)

    head, *rows = text.splitlines()
    assert head.split() == ["AGENT", "MODEL", "(EFFORT)", "CONTEXT", "ELAPSED", "STATUS"]
    row = next(line for line in rows if "Explore" in line)
    assert head.index("MODEL") == row.index("opus-5-5")
    assert head.index("ELAPSED") + len("ELAPSED") == row.index("1m00s") + len("1m00s")
    assert head.index("STATUS") == row.index("completed")


def test_header_can_be_left_out():
    assert not render([session([])], now=NOW, header=False).startswith("AGENT")


def test_rows_without_context_keep_columns_aligned():
    text = render([session([agent("a", ctx=None), agent("b")])], now=NOW)

    ends = {line.index("1m00s") for line in text.splitlines() if line.startswith(("├─", "└─"))}
    assert len(ends) == 1


def test_fit_leaves_room_for_the_header():
    tree = [agent(f"a{n}", started=n) for n in range(10)]

    assert len(render([session(tree)], now=NOW, max_lines=6).splitlines()) <= 6


def test_control_characters_in_transcript_text_are_neutralised():
    s = session([agent("evil: \x1b]0;pwned\x07name\nnext")], title="t\x1b[2Jx")

    text = render([s], now=NOW)

    assert "\x1b" not in text and "\x07" not in text
    assert len(text.splitlines()) == 4


def test_fit_folds_finished_children_of_a_running_agent():
    kids = [agent(f"k{n}", started=n) for n in range(10)]
    tree = [agent("boss", status="running", started=0, children=kids)]

    text = render([session(tree)], now=NOW, max_lines=7)

    assert len(text.splitlines()) <= 7
    assert "boss" in text and "k9" in text and "k0" not in text
