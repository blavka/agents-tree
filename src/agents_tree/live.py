"""The live view: a top-like tree with a selectable row and a detail screen per agent.

Keys: ↑/↓ (j/k, the wheel) select, Enter (→, l, a click) opens the detail, Esc
(←, h, q) goes back, q quits from the tree, r toggles running-only.

Data loads on a background thread every `interval` seconds, so keys never wait
for a refresh.
"""

from __future__ import annotations

import contextlib
import os
import re
import select
import shutil
import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from agents_tree.model import Session
from agents_tree.render import Key, find_agent, render_detail, render_lines, style

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
_FAR = 10**9


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
    scroll: int = 0  # first body line shown (of the tree, or of the detail)
    # What the last frame drew: the agent on each screen line (for clicks), every
    # agent row of the tree in order (for moving the selection), and the page height.
    screen_keys: list[Key | None] = field(default_factory=list)
    row_keys: list[Key] = field(default_factory=list)
    page: int = 1


def _step(view: View, event: Event) -> int | None:
    return {"up": -1, "down": 1, "pgup": -view.page, "pgdn": view.page,
            "home": -_FAR, "end": _FAR}.get(event) if isinstance(event, str) else None


def handle(view: View, event: Event) -> bool:
    """Apply one event. Returns False when the view should close."""
    if event == "quit":
        return False
    if event == "toggle":
        view.running_only, view.scroll = not view.running_only, 0
        return True
    step = _step(view, event)
    if view.detail:
        if step is not None:
            view.scroll = max(view.scroll + step, 0)  # draw() clamps the end
        elif event in ("back", "q"):
            view.detail, view.scroll = False, 0
        return True

    if event == "q":
        return False
    selectable = view.row_keys
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
    elif selectable and step is not None:
        if view.selected in selectable:
            at = selectable.index(view.selected) + step
        else:
            at = 0 if step > 0 else len(selectable) - 1
        view.selected = selectable[min(max(at, 0), len(selectable) - 1)]
    return True


def draw(view: View, sessions: list[Session] | None, error: str | None, *, cols: int,
         rows: int, interval: float, color: bool, now: float | None = None) -> list[str]:
    """The whole screen: body lines, then the footer on the last line."""
    room = max(rows - 1, 1)
    view.page = max(room - 2, 1)
    keys: list[Key | None] = []
    more = 0
    if sessions is None:
        body, hint = [style(color, "90", "loading…")], "q quit"
    elif error:
        body, hint = [style(color, "90", error)], "q quit"
    elif view.detail and view.selected:
        found = find_agent(sessions, view.selected)
        content = (render_detail(*found, now=now, color=color, width=cols) if found
                   else [style(color, "90", "This agent is no longer listed.")])
        view.scroll = min(view.scroll, max(len(content) - room, 0))
        body = content[view.scroll:view.scroll + room]
        more = len(content) - view.scroll - len(body)
        hint = "↑↓ scroll · esc back · q quit"
    else:
        lines, line_keys = render_lines(sessions, now=now, running_only=view.running_only,
                                        color=color, width=cols, max_lines=room,
                                        selected=view.selected)
        # Running agents alone may not fit: scroll under the pinned header, keeping
        # the selected row on screen.
        header, lines, line_keys = lines[0], lines[1:], line_keys[1:]
        height = room - 1
        if view.selected in line_keys:
            at = line_keys.index(view.selected)
            view.scroll = min(max(view.scroll, at - height + 1), at)
        view.scroll = min(view.scroll, max(len(lines) - height, 0))
        body = [header, *lines[view.scroll:view.scroll + height]]
        keys = [None, *line_keys[view.scroll:view.scroll + height]]
        view.row_keys = [k for k in line_keys if k]
        more = len(lines) - view.scroll - (len(body) - 1)
        toggle = "all" if view.running_only else "running only"
        hint = f"↑↓ select · enter detail · r {toggle} · q quit"
    view.screen_keys = keys
    if more > 0:
        hint += f" · {more} more lines"
    footer = style(color, "90", f"agents-tree · {hint} · every {interval:g}s · "
                                + time.strftime("%H:%M:%S"))
    return body + [""] * (room - len(body)) + [footer]


class _Loader:
    """Calls load() every interval on a thread; wakes the UI through a pipe."""

    def __init__(self, load: Callable[[], list[Session]], interval: float) -> None:
        self.sessions: list[Session] | None = None
        self.error: str | None = None
        self.wake_r, self._wake_w = os.pipe()
        self._load, self._interval = load, interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> _Loader:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        os.close(self.wake_r)
        os.close(self._wake_w)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                sessions, error = self._load(), None
            except LookupError as e:
                sessions, error = [], str(e)
            self.sessions, self.error = sessions, error  # one reference swap each
            with contextlib.suppress(OSError):
                os.write(self._wake_w, b".")
            self._stop.wait(self._interval)

    def drain(self) -> None:
        with contextlib.suppress(OSError):
            os.read(self.wake_r, 1024)


class _Redraw(Exception):
    """The terminal was resized: draw again now."""


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
    mouse_on, mouse_off = (("\033[?1000h\033[?1006h", "\033[?1000l\033[?1006l") if mouse
                           else ("", ""))
    sys.stdout.write("\033[?1049h\033[?25l\033[?7l" + mouse_on)
    sys.stdout.flush()
    try:
        yield
    finally:
        sys.stdout.write(mouse_off + "\033[?7h\033[?25h\033[?1049l")
        sys.stdout.flush()
        if saved is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, saved)


def run(load: Callable[[], list[Session]], *, interval: float, color: bool,
        running_only: bool, mouse: bool) -> None:
    """Redraw on every key press and every loaded refresh."""
    keys = sys.stdin.isatty()
    view = View(running_only=running_only)
    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGHUP, _on_term)
    signal.signal(signal.SIGWINCH, _on_resize)
    with _screen(keys, mouse and keys), _Loader(load, interval) as loader:
        while True:
            try:
                cols, rows = shutil.get_terminal_size()
                screen = draw(view, loader.sessions, loader.error, cols=cols, rows=rows,
                              interval=interval, color=color)
                sys.stdout.write("\033[H" + "\n".join(line + "\033[K" for line in screen)
                                 + "\033[J")
                sys.stdout.flush()
                watched = [loader.wake_r, sys.stdin] if keys else [loader.wake_r]
                ready, _, _ = select.select(watched, [], [], interval)
                if loader.wake_r in ready:
                    loader.drain()
                data = (os.read(sys.stdin.fileno(), 1024).decode(errors="ignore")
                        if keys and sys.stdin in ready else "")
            except _Redraw:
                continue
            for event in parse_keys(data):
                if not handle(view, event):
                    return
