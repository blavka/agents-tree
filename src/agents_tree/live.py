"""The live view: a top-like tree with a selectable row and a detail screen per agent.

Keys: ↑/↓ (j/k, the wheel, the pointer) select, Enter (→, l, a click) opens the detail, Esc
(←, h, q) goes back, q quits from the tree, r toggles running-only, and f
shows or folds older finished agents. In the detail, a double-click copies the
name, the session id, the agent id or the path under the pointer.

Data loads on a background thread every `interval` seconds, so keys never wait
for a refresh.
"""

from __future__ import annotations

import base64
import contextlib
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import termios
import threading
import time
import tty
from collections.abc import Callable
from dataclasses import dataclass, field

from agents_tree.model import Session
from agents_tree.render import (Key, Span, detail_lines, find_agent, render_lines, span_at,
                                style)

Event = str | tuple[str, int] | tuple[str, int, int]

_TOKEN_RE = re.compile(
    r"\x1b\[<(\d+);(\d+);(\d+)([Mm])"  # SGR mouse: button;x;y, press M / release m
    r"|\x1b\[([0-9;]*)([~A-Za-z])"     # CSI: arrows, Home/End, PgUp/PgDn
    r"|\x1bO(.)"                       # SS3: arrows in application cursor mode
    r"|\x1b|.", re.S)
_CSI = {"A": "up", "B": "down", "C": "open", "D": "back", "H": "home", "F": "end"}
_TILDE = {"1": "home", "7": "home", "4": "end", "8": "end", "5": "pgup", "6": "pgdn"}
_CHARS = {"k": "up", "j": "down", "l": "open", "\r": "open", "\n": "open", "h": "back",
          "g": "home", "G": "end", " ": "pgdn", "q": "q", "Q": "q", "\x03": "quit",
          "r": "toggle", "R": "toggle", "f": "fold", "F": "fold", "\x1b": "back"}
_FAR = 10**9
_DOUBLE_CLICK = 0.4  # a third click starts over, so a triple does not copy twice
_COPIED_FOR = 1.5


def parse_keys(data: str) -> list[Event]:
    """Terminal input to events, in order. Unknown keys and sequences are dropped."""
    events: list[Event] = []
    for m in _TOKEN_RE.finditer(data):
        button, x, y, press, params, final, ss3 = m.groups()
        if button is not None:
            if press == "M" and button == "0":
                events.append(("click", int(y), int(x)))
            elif press == "M" and button == "35":  # moved, no button held
                events.append(("hover", int(y)))
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
    fold_finished: bool = True
    selected: Key | None = None
    detail: bool = False
    scroll: int = 0  # first body line shown (of the tree, or of the detail)
    # What the last frame drew: the agent on each screen line (for clicks), every
    # agent row of the tree in order (for moving the selection), and the page height.
    screen_keys: list[Key | None] = field(default_factory=list)
    row_keys: list[Key] = field(default_factory=list)
    # Detail only: per screen row, the values a double-click copies. Aligned with
    # the drawn lines, including the blank pad and the footer.
    line_spans: list[list[Span]] = field(default_factory=list)
    pointer_row: int = -1
    pointer_at: float = 0.0
    flash: str | None = None
    flash_until: float = 0.0
    page: int = 1


def _step(view: View, event: Event) -> int | None:
    return {"up": -1, "down": 1, "pgup": -view.page, "pgdn": view.page,
            "home": -_FAR, "end": _FAR}.get(event) if isinstance(event, str) else None


def handle(view: View, event: Event, *, now: float | None = None) -> bool:
    """Apply one event. Returns False when the view should close.

    `now` is the monotonic clock used for the double-click. Tests pass it.
    """
    if event == "quit":
        return False
    if event == "toggle":
        view.running_only, view.scroll = not view.running_only, 0
        return True
    if event == "fold":
        view.fold_finished, view.scroll = not view.fold_finished, 0
        return True
    step = _step(view, event)
    if view.detail:
        if step is not None:
            view.scroll = max(view.scroll + step, 0)  # draw() clamps the end
        elif event in ("back", "q"):
            _leave_detail(view)
        elif isinstance(event, tuple) and event[0] == "click":
            _copy_double_click(view, event, time.monotonic() if now is None else now)
        return True

    if event == "q":
        return False
    selectable = view.row_keys
    if isinstance(event, tuple):
        row = event[1] - 1
        key = view.screen_keys[row] if 0 <= row < len(view.screen_keys) else None
        if key:
            view.selected = key
            if event[0] == "click":
                _open_detail(view)
    elif event == "open":
        if view.selected:
            _open_detail(view)
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


def _open_detail(view: View) -> None:
    view.detail, view.scroll, view.pointer_row = True, 0, -1


def _leave_detail(view: View) -> None:
    view.detail, view.scroll, view.pointer_row, view.flash = False, 0, -1, None


def _copy_double_click(view: View, event: tuple, now: float) -> None:
    """A second click on the same detail row copies the value under the pointer."""
    row = int(event[1]) - 1
    column = int(event[2]) if len(event) > 2 else None
    if view.pointer_row != row or now - view.pointer_at > _DOUBLE_CLICK:
        view.pointer_row, view.pointer_at = row, now
        return
    view.pointer_row = -1
    if not 0 <= row < len(view.line_spans):
        return
    hit = span_at(view.line_spans[row], column)
    if hit is None or not hit[1]:
        return
    kind, payload = hit
    view.flash = f"copied {kind}" if copy_text(payload) else "copy failed"
    view.flash_until = now + _COPIED_FOR


def copy_text(text: str) -> bool:
    """Copy text via OSC 52 and, when one is installed, a desktop clipboard tool."""
    if not text:
        return False
    encoded = base64.b64encode(text.encode()).decode("ascii")
    try:
        sys.stdout.write(f"\033]52;c;{encoded}\a")
        sys.stdout.flush()
        wrote = True
    except OSError:
        wrote = False
    argv = _clipboard_argv()
    if argv is None:
        return wrote
    try:
        done = subprocess.run(argv, input=text.encode(), timeout=2, check=False,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return wrote
    return wrote or done.returncode == 0


def _clipboard_argv() -> list[str] | None:
    """wl-copy on Wayland, otherwise xclip, xsel, or wl-copy wherever it is."""
    if os.environ.get("WAYLAND_DISPLAY") and shutil.which("wl-copy"):
        return ["wl-copy"]
    if shutil.which("xclip"):
        return ["xclip", "-selection", "clipboard"]
    if shutil.which("xsel"):
        return ["xsel", "--clipboard", "--input"]
    if shutil.which("wl-copy"):
        return ["wl-copy"]
    return None


def _detail_frame(view: View, sessions: list[Session], *, cols: int, room: int,
                  color: bool, now: float | None) -> tuple[list[str], list[list[Span]], int, str]:
    """The visible detail lines, the spans on those lines, how many are below, the hint."""
    found = find_agent(sessions, view.selected) if view.selected else None
    if found:
        content = detail_lines(*found, now=now, color=color, width=cols)
    else:
        content = [(style(color, "90", "This agent is no longer listed."), [])]
    view.scroll = min(view.scroll, max(len(content) - room, 0))
    shown = content[view.scroll:view.scroll + room]
    body = [text for text, _spans in shown]
    spans = [row_spans for _text, row_spans in shown]
    more = len(content) - view.scroll - len(body)
    hint = "↑↓ scroll · double-click copies · esc back · q quit"
    return body, spans, more, hint


def draw(view: View, sessions: list[Session] | None, error: str | None, *, cols: int,
         rows: int, interval: float, color: bool, now: float | None = None) -> list[str]:
    """The whole screen: body lines, then the footer on the last line."""
    room = max(rows - 1, 1)
    view.page = max(room - 2, 1)
    keys: list[Key | None] = []
    spans: list[list[Span]] = []
    more = 0
    if sessions is None:
        body, hint = [style(color, "90", "loading…")], "q quit"
    elif error:
        body, hint = [style(color, "90", error)], "q quit"
    elif view.detail and view.selected:
        body, spans, more, hint = _detail_frame(view, sessions, cols=cols, room=room,
                                                color=color, now=now)
    else:
        lines, line_keys = render_lines(sessions, now=now, running_only=view.running_only,
                                        color=color, width=cols,
                                        max_lines=room if view.fold_finished else None,
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
        folding = "show hidden" if view.fold_finished else "fold finished"
        hint = f"↑↓ select · enter detail · r {toggle} · f {folding} · q quit"
    view.screen_keys = keys
    view.line_spans = spans + [[] for _ in range(max(room - len(spans), 0) + 1)]
    if view.detail and view.flash and time.monotonic() < view.flash_until:
        hint = view.flash
    elif more > 0:
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
    saved = None
    if keys:
        saved = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())
    mouse_on, mouse_off = (("\033[?1003h\033[?1006h", "\033[?1003l\033[?1006l") if mouse
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
