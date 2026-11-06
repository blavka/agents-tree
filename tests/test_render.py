import re

import pytest
from conftest import NOW, agent, session

from agents_tree.model import BLOCKED, RUNNING, WAITING
from agents_tree.render import fmt_duration, fmt_tokens, fmt_window, render, render_lines


BAR = re.compile("[━─]{8}")  # the context bar, not the tree's '└─'


def tree_rows(text):
    return [line for line in text.splitlines() if re.match(r"\s*[│ ]*[├└]─", line)]


def label_of(row):
    """The tree part of a row: prefix, icon and label, without the columns."""
    return re.split(r"\s{2,}[━─]", row)[0].rstrip()


def test_rows_line_up_across_depths_and_sessions():
    text = render([
        session([agent("Explore: short", children=[agent("Plan: a much longer label here")])]),
        session([agent("x")], sid="s2"),
    ], now=NOW)

    bars = {m.start() for line in text.splitlines()[1:] if (m := BAR.search(line))}
    assert len(bars) == 1


def test_session_is_one_row_under_its_directory():
    lines = render([session([])], now=NOW).splitlines()

    assert lines[1] == "app"
    assert lines[2].startswith(" ● Work ")
    assert len(lines) == 3  # no 'main' row, no '(no subagents)' note


def test_row_shows_context_bar_time_and_status():
    text = render([session([agent("Explore: look", ctx=250_000)])], now=NOW)

    row = next(line for line in text.splitlines() if "Explore" in line)
    assert "━━──────  25%" in row
    assert "1m00s" in row
    assert row.endswith("completed")


def test_context_without_a_window_shows_tokens():
    text = render([session([agent("a", ctx=42_000, window=None)])], now=NOW)

    assert re.search(r"\s42k\s", next(line for line in tree_rows(text)))


def test_one_model_is_named_once_in_the_header():
    head, *rows = render([session([agent("a")])], now=NOW).splitlines()

    assert "opus-5-5 (high)" in head
    assert not any("opus" in row for row in rows)


def test_mixed_models_get_a_column():
    text = render([session([agent("g", model="haiku-4-5", effort=None)])], now=NOW)

    head = text.splitlines()[0]
    row = next(line for line in tree_rows(text))
    assert "MODEL (EFFORT)" in head
    assert "haiku-4-5 " in row and "(" not in row
    assert head.index("MODEL") == row.index("haiku")


def test_running_agent_elapsed_counts_up_to_now():
    text = render([session([agent("busy", status="running", started=NOW - 90)])], now=NOW)

    assert "1m30s" in next(line for line in tree_rows(text) if "busy" in line)


def test_waiting_session_shows_how_long_ago_it_last_worked():
    s = session([])
    s.main.state, s.main.status, s.main.ended = WAITING, "idle", NOW - 7200

    assert "2h00m ago" in render([s], now=NOW)


def test_tree_glyphs_mark_last_children():
    text = render([session([agent("a", children=[agent("a1"), agent("a2")]), agent("b")])],
                  now=NOW)

    assert [label_of(row) for row in tree_rows(text)] == [
        "   ├─ ✓ a", "   │  ├─ ✓ a1", "   │  └─ ✓ a2", "   └─ ✓ b"]


def test_header_counts_sessions_by_state_and_names_the_columns():
    waiting = session([], sid="s2")
    waiting.main.state, waiting.main.status = WAITING, "idle"
    text = render([session([agent("Explore: look")]), waiting], now=NOW)

    head, *rows = text.splitlines()
    assert head.startswith("● 1 running  ○ 1 waiting")
    row = next(line for line in rows if "Explore" in line)
    assert (m := BAR.search(row)) and head.index("CONTEXT") == m.start()
    assert head.index("TIME") + len("TIME") == row.index("1m00s") + len("1m00s")
    assert head.index("STATUS") == row.index("completed")


def test_sessions_needing_a_person_come_first_then_working_ones():
    def main_in(state, sid, cwd):
        s = session([], sid=sid, title=sid)
        s.main.state, s.main.status, s.cwd = state, state, cwd
        return s

    text = render([main_in(WAITING, "idle-a", "/w/a"), main_in(RUNNING, "busy-b", "/w/b"),
                   main_in(BLOCKED, "asks-c", "/w/c"), main_in(WAITING, "idle-b", "/w/b")],
                  now=NOW)

    shape = [m.group(1) if (m := re.match(r" \S (\S+)", line)) else line
             for line in text.splitlines()[1:]]
    assert shape == ["c", "asks-c", "b", "busy-b", "idle-b", "a", "idle-a"]


def test_a_running_subagent_makes_an_idle_session_count_as_working():
    idle = session([agent("kid", status="running", started=NOW)], sid="s2", title="Idle")
    idle.main.state, idle.main.status, idle.cwd = WAITING, "idle", "/w/x"
    other = session([], title="Other")
    other.main.state, other.main.status = WAITING, "idle"

    lines = render([other, idle], now=NOW).splitlines()

    assert lines[1] == "x"


def test_session_rows_show_the_short_session_id_in_its_own_column():
    text = render([session([agent("a")], sid="0123456789abcdef")], now=NOW)

    head, _, session_row, subagent_row = text.splitlines()
    at = head.index("ID")
    assert session_row[at:at + 8] == "01234567" and "0123456789" not in session_row
    assert subagent_row[at:at + 8].strip() == ""


def test_a_type_every_subagent_shares_is_left_out():
    text = render([session([agent("general-purpose: one",
                                  children=[agent("general-purpose: nested")]),
                            agent("general-purpose: two")])], now=NOW)

    assert [label_of(row) for row in tree_rows(text)] == [
        "   ├─ ✓ one", "   │  └─ ✓ nested", "   └─ ✓ two"]


def test_types_stay_when_they_differ_or_there_is_one_subagent():
    mixed = render([session([agent("general-purpose: one"), agent("Explore: two")])], now=NOW)
    single = render([session([agent("web-fetch: page")])], now=NOW)

    assert "general-purpose: one" in mixed and "Explore: two" in mixed
    assert "web-fetch: page" in single


def test_a_narrow_view_drops_the_model_column_before_cutting_labels():
    def text(width):
        return render([session([agent("a long enough label to need the room",
                                      model="haiku-4-5", effort=None), agent("b")])],
                      now=NOW, width=width)

    assert "MODEL" in text(140).splitlines()[0]
    narrow = text(90)
    assert "MODEL" not in narrow.splitlines()[0] and "haiku" not in narrow
    assert "a long enough label to need the room" in narrow


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

    labels = [row.split()[2] for row in tree_rows(text)]
    assert labels == ["parent", "kid"]


def test_running_only_without_running_agents_leaves_the_session_row():
    lines = render([session([agent("done")])], now=NOW, running_only=True).splitlines()

    assert lines[-1].startswith(" ● Work")


def test_fit_hides_oldest_finished_agents_first_and_never_running_ones():
    tree = [agent("old", started=1), agent("busy", status="running", started=2),
            agent("mid", started=3), agent("new", started=4)]

    lines = render([session(tree)], now=NOW, max_lines=6).splitlines()

    assert len(lines) <= 6
    assert [line.split()[2] for line in tree_rows("\n".join(lines))] == ["busy", "new"]
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
                                           (3 * 3600 + 120, "3h02m"),
                                           (86400 + 5 * 3600 + 59, "1d05h")])
def test_fmt_duration(secs, out):
    assert fmt_duration(secs) == out
