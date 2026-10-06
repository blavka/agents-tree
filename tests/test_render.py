import re

import pytest
from conftest import NOW, agent, session

from agents_tree.render import fmt_duration, fmt_tokens, fmt_window, render, render_lines


def model_columns(text):
    """Column where each row's model text starts."""
    return {m.start() for line in text.splitlines()
            if (m := re.search(r"(opus|haiku)-", line))}


def tree_rows(text):
    return [line for line in text.splitlines() if re.match(r"[│ ]*[├└]─", line)]


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
    assert "250k 25 % of 1M" in row
    assert "1m00s" in row
    assert row.endswith("completed")


def test_running_agent_elapsed_counts_up_to_now():
    text = render([session([agent("busy", status="running", started=NOW - 90)])], now=NOW)

    assert "1m30s" in next(line for line in text.splitlines() if line.startswith("└─ busy"))


def test_effort_is_omitted_for_models_without_it():
    text = render([session([agent("g", model="haiku-4-5", effort=None)])], now=NOW)

    row = next(line for line in text.splitlines() if line.startswith("└─ g"))
    assert "haiku-4-5 " in row and "(" not in row


def test_tree_glyphs_mark_last_children():
    text = render([session([agent("a", children=[agent("a1"), agent("a2")]), agent("b")])],
                  now=NOW)

    labels = [re.split(r"\s{2,}opus", line)[0] for line in tree_rows(text)]
    assert labels == ["├─ a", "│  ├─ a1", "│  └─ a2", "└─ b"]


def test_header_names_the_columns_and_lines_up_with_them():
    text = render([session([agent("Explore: look")])], now=NOW)

    head, *rows = text.splitlines()
    assert head.split() == ["AGENT", "MODEL", "(EFFORT)", "CONTEXT", "ELAPSED", "STATUS"]
    row = next(line for line in rows if "Explore" in line)
    assert head.index("MODEL") == row.index("opus-5-5")
    assert head.index("ELAPSED") + len("ELAPSED") == row.index("1m00s") + len("1m00s")
    assert head.index("STATUS") == row.index("completed")


def test_rows_without_context_keep_columns_aligned():
    text = render([session([agent("a", ctx=None), agent("b")])], now=NOW)

    assert len({line.index("1m00s") for line in tree_rows(text)}) == 1


def test_long_model_names_are_cut_rather_than_shifting_columns():
    text = render([session([agent("a", model="gpt-5.1-codex-max-preview-2026", effort="xhigh"),
                            agent("b")])], now=NOW)

    assert len({line.index("1m00s") for line in tree_rows(text)}) == 1
    assert "…" in tree_rows(text)[0]


def test_running_only_keeps_running_agents_and_their_parents():
    tree = [agent("done"), agent("parent", children=[agent("kid", status="running"),
                                                      agent("kid-done")])]
    text = render([session(tree)], now=NOW, running_only=True)

    labels = [row.split()[1] for row in tree_rows(text)]
    assert labels == ["parent", "kid"]


def test_running_only_without_running_agents_says_so():
    text = render([session([agent("done")])], now=NOW, running_only=True)

    assert text.splitlines()[-1].strip() == "(no running subagents)"


def test_session_without_subagents_says_so():
    assert render([session([])], now=NOW).splitlines()[-1].strip() == "(no subagents)"


def test_fit_hides_oldest_finished_agents_first_and_never_running_ones():
    tree = [agent("old", started=1), agent("busy", status="running", started=2),
            agent("mid", started=3), agent("new", started=4)]

    lines = render([session(tree)], now=NOW, max_lines=6).splitlines()

    assert len(lines) <= 6
    assert [line.split()[1] for line in tree_rows("\n".join(lines))] == ["busy", "new"]
    assert lines[-1].strip() == "… 2 older finished agents hidden"


def test_fit_counts_hidden_subtrees():
    tree = [agent("old", started=1, children=[agent("k1"), agent("k2")]), agent("new", started=5)]

    assert "… 3 older finished agents hidden" in render([session(tree)], now=NOW, max_lines=5)


def test_fit_folds_finished_children_of_a_running_agent():
    kids = [agent(f"k{n}", started=n) for n in range(10)]
    tree = [agent("boss", status="running", started=0, children=kids)]

    text = render([session(tree)], now=NOW, max_lines=7)

    assert len(text.splitlines()) <= 7
    assert "boss" in text and "k9" in text and "k0" not in text


def test_rendering_leaves_the_sessions_as_they_were():
    tree = [agent("old", started=1), agent("busy", status="running", started=2)]
    sessions = [session(tree)]

    render(sessions, now=NOW, running_only=True, max_lines=4)

    assert [a.label for a in sessions[0].agents] == ["old", "busy"]


def test_narrow_width_truncates_labels_with_ellipsis():
    text = render([session([agent("general-purpose: " + "x" * 80)])], now=NOW, width=90)

    row = next(line for line in text.splitlines() if "general-purpose" in line)
    assert "…" in row
    assert len(row) <= 90


def test_color_wraps_status_by_state():
    text = render([session([agent("a", status="running", started=NOW)])], now=NOW, color=True)

    assert "\033[1;33mrunning\033[0m" in text


def test_lines_come_with_the_agent_each_one_shows():
    lines, keys = render_lines([session([agent("a")])], now=NOW)

    assert dict(zip(keys, lines)).keys() >= {("s1", "main"), ("s1", "a")}
    assert keys[0] is None  # the header


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
