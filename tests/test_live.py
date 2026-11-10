import base64
import re
import subprocess
import time

import pytest
from conftest import NOW, agent, session

from agents_tree.live import View, copy_text, draw, handle, parse_keys
from agents_tree.model import Detail
from agents_tree.render import detail_lines, span_at


@pytest.mark.parametrize(("data", "events"), [
    ("q", ["q"]), ("\x03", ["quit"]), ("r", ["toggle"]), ("fF", ["fold", "fold"]),
    ("\x1b", ["back"]),
    ("\x1b[A\x1b[B", ["up", "down"]), ("\x1bOA", ["up"]), ("jk", ["down", "up"]),
    ("\r", ["open"]), ("\x1b[C", ["open"]), ("\x1b[D", ["back"]),
    ("\x1b[5~\x1b[6~", ["pgup", "pgdn"]), ("\x1b[H\x1b[F", ["home", "end"]),
    ("\x1b[<0;12;7M", [("click", 7, 12)]), ("\x1b[<0;12;7m", []),
    ("\x1b[<35;12;7M", [("hover", 7)]),
    ("\x1b[<64;1;1M\x1b[<65;1;1M", ["up", "down"]),
    ("\x1b[B\x1b", ["down", "back"]), ("x\x1b[99Z", []),
])
def test_parse_keys(data, events):
    assert parse_keys(data) == events


def _drawn(view, sessions, rows=20, cols=140, color=False, error=None):
    return draw(view, sessions, error, cols=cols, rows=rows, interval=2, color=color, now=NOW)


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
    screen = _drawn(View(selected=("s1", "Plan: think")), _sessions(), color=True)

    row = next(line for line in screen if "Plan: think" in line)
    assert row.startswith("\033[30;46m")


def test_click_opens_the_detail_of_the_row_under_the_pointer():
    view, sessions = View(), _sessions()
    screen = _drawn(view, sessions)
    y = next(n for n, line in enumerate(screen, 1) if "Explore: look" in line)

    handle(view, ("click", y))

    assert view.detail and view.selected == ("s1", "Explore: look")


def test_hover_selects_the_row_under_the_pointer_without_opening_it():
    view, sessions = View(), _sessions()
    screen = _drawn(view, sessions)
    y = next(n for n, line in enumerate(screen, 1) if "Plan: think" in line)

    handle(view, ("hover", y))

    assert view.selected == ("s1", "Plan: think") and not view.detail
    hovered = _drawn(view, sessions, color=True)
    assert next(line for line in hovered if "Plan: think" in line).startswith("\033[30;46m")


def test_hover_off_a_row_keeps_the_selection():
    view = View(selected=("s1", "Plan: think"))
    _drawn(view, _sessions())

    handle(view, ("hover", 1))  # the header

    assert view.selected == ("s1", "Plan: think")


def test_hover_in_the_detail_changes_nothing():
    view = View(selected=("s1", "Plan: think"), detail=True)
    _drawn(view, _sessions())

    handle(view, ("hover", 3))

    assert view.detail and view.selected == ("s1", "Plan: think")


def test_click_on_the_header_does_nothing():
    view = View()
    _drawn(view, _sessions())

    handle(view, ("click", 1))

    assert not view.detail and view.selected is None


def test_tree_taller_than_the_screen_scrolls_to_keep_the_selection_visible():
    running = [agent(f"r{n}", status="running", started=n) for n in range(30)]
    view, sessions = View(), [session(running)]
    _drawn(view, sessions, rows=10)

    for _ in range(25):
        handle(view, "down")
    screen = _drawn(view, sessions, rows=10)

    assert "CONTEXT" in screen[0]  # the header stays put
    assert any(f"r{23}" in line for line in screen[1:-1])
    assert "more lines" in screen[-1]


def test_f_shows_finished_agents_folded_to_fit_the_screen():
    finished = [agent(f"done-{n}", started=n) for n in range(30)]
    view, sessions = View(), [session(finished)]

    folded = _drawn(view, sessions, rows=10)
    handle(view, "fold")
    expanded = _drawn(view, sessions, rows=10)

    assert "older finished agents hidden" in "\n".join(folded)
    assert "older finished agents hidden" not in "\n".join(expanded)
    assert "f fold finished" in expanded[-1]
    assert "done-0" in "\n".join(expanded)
    assert "more lines" in expanded[-1]


def test_detail_shows_prompt_tools_and_last_message():
    view = View(selected=("s1", "Explore: look"), detail=True)

    text = "\n".join(_drawn(view, _sessions(), rows=40))

    assert "Work › Explore: look" in text
    assert "4 requests · 1k output tokens" in text
    assert "Tools (8 calls)" in text and "Bash ×5, Read ×3" in text
    assert "last: Bash, 30s ago · waiting for its result" in text
    assert "/w/app · session s1" in text
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
    screen = _drawn(View(), [], rows=5, error="no running Claude Code sessions")

    assert screen[0] == "no running Claude Code sessions"


def test_first_frame_says_it_is_loading():
    assert _drawn(View(), None, rows=5)[0] == "loading…"


def test_detail_lines_mark_the_name_the_ids_and_the_path():
    shown = _sessions()[0]
    lines = detail_lines(shown, shown.agents[0], now=NOW, width=80)
    title, spans = lines[0]

    assert title == "Work › Explore: look"
    assert span_at(spans, 1) == ("name", "Work")
    assert span_at(spans, title.index("›") + 1) is None
    assert span_at(spans, title.index("Explore: look") + 1) == ("name", "Explore: look")
    where = next(row for row in lines if row[0].startswith("/w/app"))
    assert span_at(where[1], 1) == ("path", "/w/app")
    assert span_at(where[1], where[0].index("s1") + 1) == ("session id", "s1")
    ident = next(row for row in lines if "requests" in row[0])
    assert span_at(ident[1], ident[0].index("id ") + 1) == ("id", "Explore: look")
    colored, colored_spans = detail_lines(shown, shown.agents[0], now=NOW, width=80,
                                          color=True)[0]
    assert colored_spans == spans
    assert "Work › Explore: look" in colored


def _arm(monkeypatch):
    copied = []
    monkeypatch.setattr("agents_tree.live.copy_text", lambda text: copied.append(text) or True)
    view = View(selected=("s1", "Explore: look"), detail=True)
    return copied, view, _drawn(view, _sessions(), rows=40)


def _click(view, screen, needle, column, at):
    line = next(row for row in screen if needle in row)
    handle(view, ("click", screen.index(line) + 1, column(line)), now=at)


def test_double_click_copies_the_name_or_the_id(monkeypatch):
    copied, view, screen = _arm(monkeypatch)

    _click(view, screen, "Work ›", lambda line: line.index("Explore: look") + 1, 10)
    _click(view, screen, "Work ›", lambda line: line.index("Explore: look") + 1, 10.2)
    _click(view, screen, "Work ›", lambda line: 1, 10.3)  # a third click starts over

    assert copied == ["Explore: look"]
    assert view.flash == "copied name"
    _click(view, screen, "session s1", lambda line: line.index("s1") + 1, 12)
    _click(view, screen, "session s1", lambda line: line.index("s1") + 1, 12.2)
    assert copied[-1] == "s1" and view.flash == "copied session id"
    _click(view, screen, "id Explore: look", lambda line: line.index("id ") + 1, 14)
    _click(view, screen, "id Explore: look", lambda line: line.index("id ") + 1, 14.2)
    assert copied[-1] == "Explore: look" and view.flash == "copied id"


def test_double_click_between_values_copies_nothing(monkeypatch):
    copied, view, screen = _arm(monkeypatch)

    _click(view, screen, "Work ›", lambda line: line.index("›") + 1, 10)
    _click(view, screen, "Work ›", lambda line: line.index("›") + 1, 10.2)

    assert not copied and view.flash is None and view.detail


def test_a_failed_copy_is_reported(monkeypatch):
    monkeypatch.setattr("agents_tree.live.copy_text", lambda _text: False)
    view = View(selected=("s1", "Explore: look"), detail=True)
    screen = _drawn(view, _sessions(), rows=40)
    line = next(row for row in screen if row.startswith("Work ›"))
    y = screen.index(line) + 1

    handle(view, ("click", y, 1), now=1)
    handle(view, ("click", y, 1), now=1.2)

    assert view.flash == "copy failed" and view.detail


def test_a_slow_second_click_does_not_copy(monkeypatch):
    copied, view, screen = _arm(monkeypatch)
    line = next(row for row in screen if row.startswith("Work ›"))
    y, x = screen.index(line) + 1, 1

    handle(view, ("click", y, x), now=10)
    handle(view, ("click", y, x), now=10.6)

    assert not copied
    assert view.pointer_row == y - 1


def test_the_footer_confirms_a_copy_then_returns_to_the_hint():
    view = View(selected=("s1", "Explore: look"), detail=True, flash="copied id",
                flash_until=time.monotonic() + 30)

    assert "copied id" in _drawn(view, _sessions(), rows=12)[-1]

    view.flash_until = time.monotonic() - 1
    footer = _drawn(view, _sessions(), rows=12)[-1]
    assert "double-click copies" in footer
    assert "copied id" not in footer


def test_copy_text_writes_osc52_and_calls_wl_copy(monkeypatch, capsys):
    calls = []

    def which(name):
        return "/usr/bin/wl-copy" if name == "wl-copy" else None

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs.get("input"), kwargs.get("timeout")))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setattr("agents_tree.live.shutil.which", which)
    monkeypatch.setattr("agents_tree.live.subprocess.run", fake_run)

    assert copy_text("Explore: look")

    payload = base64.b64encode(b"Explore: look").decode("ascii")
    assert capsys.readouterr().out == f"\033]52;c;{payload}\a"
    assert calls == [(["wl-copy"], b"Explore: look", 2)]
