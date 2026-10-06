import re

import pytest
from test_render import NOW, agent, session

from agents_tree.live import View, draw, handle, parse_keys
from agents_tree.model import Detail


@pytest.mark.parametrize(("data", "events"), [
    ("q", ["q"]), ("\x03", ["quit"]), ("r", ["toggle"]), ("\x1b", ["back"]),
    ("\x1b[A\x1b[B", ["up", "down"]), ("\x1bOA", ["up"]), ("jk", ["down", "up"]),
    ("\r", ["open"]), ("\x1b[C", ["open"]), ("\x1b[D", ["back"]),
    ("\x1b[5~\x1b[6~", ["pgup", "pgdn"]), ("\x1b[H\x1b[F", ["home", "end"]),
    ("\x1b[<0;12;7M", [("click", 7)]), ("\x1b[<0;12;7m", []),
    ("\x1b[<64;1;1M\x1b[<65;1;1M", ["up", "down"]),
    ("\x1b[B\x1b", ["down", "back"]), ("x\x1b[99Z", []),
])
def test_parse_keys(data, events):
    assert parse_keys(data) == events


def _drawn(view, sessions, rows=20, cols=140):
    return draw(view, sessions, None, cols=cols, rows=rows, interval=2, window=None,
                color=False, now=NOW)


def _sessions():
    a = agent("Explore: look", started=1)
    a.detail = Detail(prompt="Find the parser.\nThen report.", tools={"Read": 3, "Bash": 5},
                      last_tool="Bash", last_tool_at=NOW - 30, last_tool_pending=True,
                      last_text="Found it in parse.py.", output_tokens=1200, requests=4,
                      transcript="/tmp/agent-x.jsonl")
    return [session([a, agent("Plan: think", started=2)])]


def test_down_selects_rows_in_order_and_stops_at_the_end():
    view, sessions = View(), _sessions()
    _drawn(view, sessions)

    picked = []
    for _ in range(4):
        handle(view, "down")
        picked.append(view.selected)

    assert picked == [("s1", "main"), ("s1", "Explore: look"), ("s1", "Plan: think"),
                      ("s1", "Plan: think")]


def test_up_without_selection_starts_at_the_bottom():
    view = View()
    _drawn(view, _sessions())

    handle(view, "up")

    assert view.selected == ("s1", "Plan: think")


def test_selected_row_is_highlighted():
    view, sessions = View(selected=("s1", "Plan: think")), _sessions()

    screen = draw(view, sessions, None, cols=140, rows=20, interval=2, window=None,
                  color=True, now=NOW)

    row = next(line for line in screen if "Plan: think" in line)
    assert row.startswith("\033[30;46m")


def test_click_opens_the_detail_of_the_row_under_the_pointer():
    view, sessions = View(), _sessions()
    screen = _drawn(view, sessions)
    y = next(n for n, line in enumerate(screen, 1) if "Explore: look" in line)

    handle(view, ("click", y))

    assert view.detail and view.selected == ("s1", "Explore: look")


def test_click_on_a_header_does_nothing():
    view = View()
    _drawn(view, _sessions())

    handle(view, ("click", 1))

    assert not view.detail and view.selected is None


def test_detail_shows_prompt_tools_and_last_message():
    view = View(selected=("s1", "Explore: look"), detail=True)

    text = "\n".join(_drawn(view, _sessions(), rows=40))

    assert "Work › Explore: look" in text
    assert "4 requests · 1k output tokens" in text
    assert "Tools (8 calls)" in text and "Bash ×5, Read ×3" in text
    assert "last: Bash, 30s ago · waiting for its result" in text
    assert re.search(r"Prompt\n  Find the parser.\n  Then report.", text)
    assert "Last message\n  Found it in parse.py." in text


def test_detail_scrolls_and_clamps_at_the_end():
    sessions = _sessions()
    detail = sessions[0].agents[0].detail
    assert detail is not None
    detail.last_text = "\n".join(f"line {n}" for n in range(100))
    view = View(selected=("s1", "Explore: look"), detail=True)

    first = _drawn(view, sessions, rows=10)
    handle(view, "end")
    last = _drawn(view, sessions, rows=10)

    assert first[0].startswith("Work › Explore")
    assert "line 99" in "\n".join(last)
    assert "more lines" not in last[-1]


def test_back_leaves_the_detail_then_clears_the_selection_then_quits():
    view = View(selected=("s1", "main"), detail=True)

    assert handle(view, "back") and not view.detail
    assert handle(view, "back") and view.selected is None
    assert not handle(view, "back")


def test_q_goes_back_from_the_detail_and_quits_from_the_tree():
    view = View(selected=("s1", "main"), detail=True)

    assert handle(view, "q") and not view.detail
    assert not handle(view, "q")


def test_vanished_agent_says_so():
    view = View(selected=("s1", "gone"), detail=True)

    assert "no longer listed" in _drawn(view, _sessions())[0]


def test_footer_sits_on_the_last_line():
    screen = _drawn(View(), _sessions(), rows=12)

    assert len(screen) == 12
    assert screen[-1].startswith("agents-tree · ↑↓ select")


def test_error_is_shown_instead_of_the_tree():
    screen = draw(View(), [], "no running Claude Code sessions", cols=80, rows=5, interval=2,
                  window=None, color=False, now=NOW)

    assert screen[0] == "no running Claude Code sessions"
