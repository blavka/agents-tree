"""Command line: one-shot print, or a top-like live view with --watch."""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import select
import shutil
import signal
import sys
import time

from agents_tree import __version__
from agents_tree.providers import PROVIDERS
from agents_tree.render import render

DESCRIPTION = "Show coding-agent sessions and their subagents as a tree: " \
              "model (effort), context use, elapsed time and status."
EPILOG = """\
targets:
  (none)        every running session on this machine
  DIR           sessions running in DIR ("." works); if none, DIR's newest session
  SESSION-ID    one session (a unique prefix is enough)

live view keys: q quit, r toggle running-only

Reads local transcripts only. Run it in its own terminal (or the herdr plugin):
started with `!` inside an agent, its output lands in that agent's context.
"""


def _positive_int(value: str) -> int:
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a whole number of tokens: {value!r}") from None
    if n <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return n


def _env_window() -> int | None:
    value = os.environ.get("AGENTS_TREE_WINDOW", "").strip()
    if not value:
        return None
    try:
        return _positive_int(value)
    except argparse.ArgumentTypeError:
        print(f"agents-tree: ignoring AGENTS_TREE_WINDOW={value!r}", file=sys.stderr)
        return None


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="agents-tree", description=DESCRIPTION, epilog=EPILOG,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", nargs="?", metavar="DIR|SESSION-ID")
    ap.add_argument("-w", "--watch", action="store_true", help="live view, like top")
    ap.add_argument("-n", "--interval", type=float, default=2.0, metavar="SECONDS",
                    help="live view refresh interval (default 2)")
    ap.add_argument("-r", "--running", action="store_true",
                    help="only running subagents (and their parents)")
    ap.add_argument("--window", type=_positive_int, metavar="TOKENS", default=_env_window(),
                    help="context window for the %% column (default: per model)")
    ap.add_argument("--provider", choices=sorted(PROVIDERS), default="claude")
    ap.add_argument("--color", choices=["auto", "always", "never"], default="auto")
    ap.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    args = ap.parse_args(argv)
    if args.interval <= 0:
        ap.error("--interval must be positive")
    return args


def frame(args: argparse.Namespace, running_only: bool, color: bool,
          width: int | None = None, max_lines: int | None = None) -> tuple[str, bool]:
    try:
        sessions = PROVIDERS[args.provider].sessions(args.target)
    except LookupError as e:
        return (f"\033[90m{e}\033[0m" if color else str(e)), False
    return render(sessions, window=args.window, running_only=running_only, color=color,
                  width=width, max_lines=max_lines), True


@contextlib.contextmanager
def _screen(keys: bool):
    """Alternate screen, no cursor, no autowrap; cbreak stdin when reading keys."""
    import termios
    import tty

    saved = None
    if keys:
        saved = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())
    sys.stdout.write("\033[?1049h\033[?25l\033[?7l")
    sys.stdout.flush()
    try:
        yield
    finally:
        sys.stdout.write("\033[?7h\033[?25h\033[?1049l")
        sys.stdout.flush()
        if saved is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, saved)


# Escape sequences (arrow keys, and the wheel scrolling the alternate screen sends as
# arrows) must not read as a lone Esc.
_ESCAPE_SEQ_RE = re.compile(r"\x1b(\[[0-9;<?]*[ -/]*[@-~]|O.)")


def parse_keys(data: str) -> set[str]:
    """Keys in one read: "quit" and/or "toggle"; escape sequences are dropped."""
    rest = _ESCAPE_SEQ_RE.sub("", data)
    keys = set()
    if any(ch in rest for ch in "qQ\x03") or rest == "\x1b":
        keys.add("quit")
    if "r" in rest or "R" in rest:
        keys.add("toggle")
    return keys


class _Redraw(Exception):
    """The terminal was resized: draw again now rather than at the next tick."""


def _on_resize(signum, frame):
    raise _Redraw


def _on_term(signum, frame):
    raise SystemExit(0)


def _read_key(timeout: float, keys: bool) -> str | None:
    if not keys:
        time.sleep(timeout)
        return None
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    return os.read(sys.stdin.fileno(), 32).decode(errors="ignore") if ready else None


def watch(args: argparse.Namespace, color: bool) -> None:
    keys = sys.stdin.isatty()
    running_only = args.running
    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGHUP, _on_term)
    signal.signal(signal.SIGWINCH, _on_resize)
    with _screen(keys):
        while True:
            try:
                cols, rows = shutil.get_terminal_size()
                room = max(rows - 2, 1)
                body, _ = frame(args, running_only, color, width=cols, max_lines=room)
                lines = body.split("\n")
                if len(lines) > room:  # running agents alone do not fit
                    more = f"… {len(lines) - room + 1} more lines (enlarge the terminal)"
                    lines = lines[: room - 1] + [f"\033[90m{more}\033[0m" if color else more]
                footer = (f"agents-tree · every {args.interval:g}s · "
                          f"r {'all' if running_only else 'running only'} · q quit · "
                          + time.strftime("%H:%M:%S"))
                footer = f"\033[90m{footer}\033[0m" if color else footer
                lines = lines + [""] * (room - len(lines)) + [footer] if rows > 2 else lines
                sys.stdout.write("\033[H" + "\n".join(line + "\033[K" for line in lines)
                                 + "\033[J")
                sys.stdout.flush()
                pressed = parse_keys(_read_key(args.interval, keys) or "")
            except _Redraw:
                continue
            if "quit" in pressed:
                return
            if "toggle" in pressed:
                running_only = not running_only


def main(argv: list[str] | None = None) -> int:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    args = parse_args(argv)
    color = args.color == "always" or (args.color == "auto" and sys.stdout.isatty()
                                       and not os.environ.get("NO_COLOR"))
    if not args.watch:
        text, ok = frame(args, args.running, color)
        print(text)
        return 0 if ok else 1
    try:
        watch(args, color)
    except KeyboardInterrupt:
        pass
    return 0
