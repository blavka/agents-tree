"""Command line: one-shot print, or a top-like live view with --watch."""

from __future__ import annotations

import argparse
import os
import signal
import sys

from agents_tree import __version__, live
from agents_tree.providers import PROVIDERS
from agents_tree.render import render

DESCRIPTION = "Show coding-agent sessions and their subagents as a tree: " \
              "model (effort), context use, elapsed time and status."
EPILOG = """\
targets:
  (none)        every running session on this machine
  DIR           sessions running in DIR ("." works); if none, DIR's newest session
  SESSION-ID    one session (a unique prefix is enough)

live view keys: ↑/↓ or j/k select, enter or a click shows the agent's detail,
  esc goes back, r toggles running-only, q quits

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
    ap.add_argument("--no-mouse", dest="mouse", action="store_false",
                    help="live view: leave the mouse to the terminal (text selection)")
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
        live.run(lambda: PROVIDERS[args.provider].sessions(args.target), interval=args.interval,
                 window=args.window, color=color, running_only=args.running, mouse=args.mouse)
    except KeyboardInterrupt:
        pass
    return 0
