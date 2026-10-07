"""Command line: one-shot print, or a top-like live view with --watch."""

from __future__ import annotations

import argparse
import os
import signal
import sys
from collections.abc import Callable

from agents_tree import __version__, live
from agents_tree.model import Session, override_window, sanitize
from agents_tree.providers import ALL, PROVIDERS
from agents_tree.render import render, style

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
    ap.add_argument("--provider", choices=[ALL, *sorted(PROVIDERS)], default=ALL,
                    help="which agent CLI to read (default: all of them)")
    ap.add_argument("--color", choices=["auto", "always", "never"], default="auto")
    ap.add_argument("--no-mouse", dest="mouse", action="store_false",
                    help="live view: leave the mouse to the terminal (text selection)")
    ap.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    args = ap.parse_args(argv)
    if args.interval <= 0:
        ap.error("--interval must be positive")
    return args


def loader(args: argparse.Namespace) -> Callable[[], list[Session]]:
    """Sessions as the views get them: sanitised once, with --window applied."""
    def load() -> list[Session]:
        sessions = _sessions(args.provider, args.target)
        sanitize(sessions)
        override_window(sessions, args.window)
        return sessions

    return load


def _sessions(provider: str, target: str | None) -> list[Session]:
    """One provider's sessions, or with ALL every provider's: those that find
    nothing are skipped, and only when none finds anything is it an error."""
    if provider != ALL:
        return PROVIDERS[provider].sessions(target)
    found: list[Session] = []
    misses = []
    for p in PROVIDERS.values():
        try:
            found += p.sessions(target)
        except LookupError as e:
            misses.append(str(e))
    if not found:
        raise LookupError("; ".join(misses))
    return found


def main(argv: list[str] | None = None) -> int:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    args = parse_args(argv)
    color = args.color == "always" or (args.color == "auto" and sys.stdout.isatty()
                                       and not os.environ.get("NO_COLOR"))
    load = loader(args)
    if not args.watch:
        try:
            sessions = load()
        except LookupError as e:
            print(style(color, "90", str(e)))
            return 1
        print(render(sessions, running_only=args.running, color=color))
        return 0
    try:
        live.run(load, interval=args.interval, color=color, running_only=args.running,
                 mouse=args.mouse)
    except KeyboardInterrupt:
        pass
    return 0
