"""The live view: a top-like tree with a selectable row and a detail screen per agent.

Keys: ↑/↓ (j/k, the wheel) select, Enter (→, l, a click) opens the detail, Esc
(←, h, q) goes back, q quits from the tree, r toggles running-only.
"""

from __future__ import annotations

import contextlib
import copy
import os
import re
import select
import shutil
import signal
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from agents_tree.model import Session
from agents_tree.render import Key, find_agent, render_detail, render_lines

Event = str | tuple[str, int]

_TOKEN_RE = re.compile(
    r"\x1b\[<(\d+);(\d+);(\d+)([Mm])"  # SGR mouse: button;x;y, press M / release m
    r"|\x1b\[([0-9;]*)([~A-Za-z])"     # CSI: arrows, Home/End, PgUp/PgDn
    r"|\x1bO(.)"                       # SS3: arrows in application cursor mode
    r"|\x1b|.", re.S)
_CSI = {"A": "up", "B": "down", "C": "open", "D": "back", "H": "home", "F": "end"}
_TILDE = {"1": "home", "7": "home", "4": "end", "8": "end", "5": "pgup", "6": "pgdn"}
_CHARS = {"k": "up", "j": "down", "l": "open", "\r": "open", "\n": "open", "h": "back",
          "g": "home", "G": "end", " ": "pgdn", "q": "q", "Q": "q", "\x03": "quit",
          "r": "toggle", "R": "toggle", "\x1b": "back"}


def parse_keys(data: str) -> list[Event]:
    """Terminal input to events, in order. Unknown keys and sequences are dropped."""
    events: list[Event] = []
    for m in _TOKEN_RE.finditer(data):
        button, _, y, press, params, final, ss3 = m.groups()
        if button is not None:
            if press == "M" and button == "0":
                events.append(("click", int(y)))
            elif button in ("64", "65"):
                events.append("up" if button == "64" else "down")
        elif final is not None:
            event = _TILDE.get(params) if final == "~" else _CSI.get(final)
            if event:
                events.append(event)
        elif ss3 is not None:
            if ss3 in _CSI:
                events.append(_CSI[ss3])
        elif m.group(0) in _CHARS:
            events.append(_CHARS[m.group(0)])
    return events


@dataclass
class View:
    running_only: bool = False
    selected: Key | None = None
    detail: bool = False
    scroll: int = 0
    # What the last frame drew: the agent on each body line, and the body height.
    screen_keys: list[Key | None] = field(default_factory=list)
    page: int = 1


def handle(view: View, event: Event) -> bool:
    """Apply one event. Returns False when the view should close."""
    if event == "quit":
        return False
    if event == "toggle":
        view.running_only = not view.running_only
        return True
    if view.detail:
        moves = {"up": -1, "down": 1, "pgup": -view.page, "pgdn": view.page,
                 "home": -10**9, "end": 10**9}
        if event in moves:
            view.scroll = max(view.scroll + moves[event], 0)  # draw() clamps the end
        elif event in ("back", "q"):
            view.detail = False
        return True

    if event == "q":
        return False
    selectable = [k for k in view.screen_keys if k]
    if isinstance(event, tuple):
        row = event[1] - 1
        key = view.screen_keys[row] if 0 <= row < len(view.screen_keys) else None
        if key:
            view.selected, view.detail, view.scroll = key, True, 0
    elif event == "open":
        if view.selected:
            view.detail, view.scroll = True, 0
    elif event == "back":
        if not view.selected:
            return False
        view.selected = None
    elif selectable and event in ("up", "down", "pgup", "pgdn", "home", "end"):
        step = {"up": -1, "down": 1, "pgup": -view.page, "pgdn": view.page,
                "home": -len(selectable), "end": len(selectable)}[event]
        if view.selected in selectable:
            at = selectable.index(view.selected) + step
        else:
            at = 0 if step > 0 else len(selectable) - 1
        view.selected = selectable[min(max(at, 0), len(selectable) - 1)]
    return True


def draw(view: View, sessions: list[Session], error: str | None, *, cols: int, rows: int,
         interval: float, window: int | None, color: bool, now: float | None = None) -> list[str]:
    """The whole screen: body lines, then the footer on the last line."""
    room = max(rows - 1, 1)

    def dim(text: str) -> str:
        return f"\033[90m{text}\033[0m" if color else text

    view.page = max(room - 2, 1)
    view.screen_keys = []
    if error:
        body = [dim(error)]
        hint = "q quit"
    elif view.detail and view.selected:
        found = find_agent(sessions, view.selected)
        if found:
            content = render_detail(*found, now=now, window=window, color=color, width=cols)
        else:
            content = [dim("This agent is no longer listed.")]
        view.scroll = min(view.scroll, max(len(content) - room, 0))
        body = content[view.scroll:view.scroll + room]
        more = len(content) - view.scroll - len(body)
        hint = "↑↓ scroll · esc back · q quit" + (f" · {more} more lines" if more > 0 else "")
    else:
        lines, keys = render_lines(copy.deepcopy(sessions), now=now, window=window,
                                   running_only=view.running_only, color=color, width=cols,
                                   max_lines=room, selected=view.selected)
        if len(lines) > room:  # running agents alone do not fit
            lines = lines[:room - 1] + [dim(f"… {len(lines) - room + 1} more lines "
                                            "(enlarge the terminal)")]
            keys = keys[:room - 1] + [None]
        body, view.screen_keys = lines, keys
        toggle = "all" if view.running_only else "running only"
        hint = f"↑↓ select · enter detail · r {toggle} · q quit"
    footer = dim(f"agents-tree · {hint} · every {interval:g}s · " + time.strftime("%H:%M:%S"))
    return body + [""] * (room - len(body)) + [footer]


class _Redraw(Exception):
    """The terminal was resized: draw again now rather than at the next tick."""


def _on_resize(signum, frame):
    raise _Redraw


def _on_term(signum, frame):
    raise SystemExit(0)


@contextlib.contextmanager
def _screen(keys: bool, mouse: bool):
    """Alternate screen, no cursor, no autowrap, optional mouse; cbreak stdin for keys."""
    import termios
    import tty

    saved = None
    if keys:
        saved = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())
    mouse_on, mouse_off = ("\033[?1000h\033[?1006h", "\033[?1000l\033[?1006l") if mouse \
        else ("", "")
    sys.stdout.write("\033[?1049h\033[?25l\033[?7l" + mouse_on)
    sys.stdout.flush()
    try:
        yield
    finally:
        sys.stdout.write(mouse_off + "\033[?7h\033[?25h\033[?1049l")
        sys.stdout.flush()
        if saved is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, saved)


def _read_input(timeout: float, keys: bool) -> str:
    if not keys:
        time.sleep(timeout)
        return ""
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    return os.read(sys.stdin.fileno(), 1024).decode(errors="ignore") if ready else ""


def run(load: Callable[[], list[Session]], *, interval: float, window: int | None,
        color: bool, running_only: bool, mouse: bool) -> None:
    """Redraw on every key press; reload the data every `interval` seconds."""
    keys = sys.stdin.isatty()
    view = View(running_only=running_only)
    sessions: list[Session] = []
    error: str | None = None
    loaded_at: float | None = None
    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGHUP, _on_term)
    signal.signal(signal.SIGWINCH, _on_resize)
    with _screen(keys, mouse and keys):
        while True:
            try:
                if loaded_at is None or time.monotonic() - loaded_at >= interval:
                    try:
                        sessions, error = load(), None
                    except LookupError as e:
                        sessions, error = [], str(e)
                    loaded_at = time.monotonic()
                cols, rows = shutil.get_terminal_size()
                screen = draw(view, sessions, error, cols=cols, rows=rows, interval=interval,
                              window=window, color=color)
                sys.stdout.write("\033[H" + "\n".join(line + "\033[K" for line in screen)
                                 + "\033[J")
                sys.stdout.flush()
                wait = max(interval - (time.monotonic() - loaded_at), 0.0)
                events = parse_keys(_read_input(wait, keys))
            except _Redraw:
                continue
            for event in events:
                if not handle(view, event):
                    return
