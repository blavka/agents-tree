"""Command line: one-shot print, or a top-like live view with --watch."""

from __future__ import annotations

import argparse
import os
import re
import signal
import sys
import time
from pathlib import Path
from collections.abc import Callable

from agents_tree import __version__, live
from agents_tree.json_output import dumps as json_dumps
from agents_tree.model import BLOCKED, WAITING, Session, override_window, sanitize
from agents_tree.providers import ALL, PROVIDERS
from agents_tree.render import render, style

DESCRIPTION = "Show coding-agent sessions and their subagents as a tree: " \
              "model (effort), context use, elapsed time and status."
EPILOG = """\
targets:
  (none)        every running session on this machine
  -s            the running sessions of the current directory, its parents or subfolders
  -i 3d -k      close the sessions idle for 3 days or more (asks first)
  DIR           sessions running in DIR ("." works); if none, DIR's newest session
  SESSION-ID    one session (a unique prefix is enough)

live view keys: ↑/↓, j/k or the pointer select, enter or a click shows the agent's detail,
  esc goes back, r toggles running-only, f shows or folds older finished agents,
  q quits

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


_DURATION_RE = re.compile(r"(\d+)([smhd])")
_UNIT_SECS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def _duration(value: str) -> float:
    """'3d', '12h', '1d12h': the same units the TIME column shows."""
    text = value.strip().lower()
    parts = _DURATION_RE.findall(text)
    if not parts or "".join(n + u for n, u in parts) != text:
        raise argparse.ArgumentTypeError(f"not a duration like 3d, 12h or 1d12h: {value!r}")
    secs = sum(int(n) * _UNIT_SECS[u] for n, u in parts)
    if secs <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return secs


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
    ap.add_argument("-s", "--session", action="store_true",
                    help="only the sessions of the current directory: run in it, above it "
                         "(you are in a subfolder of their project) or below it")
    ap.add_argument("-r", "--running", action="store_true",
                    help="only running subagents (and their parents)")
    ap.add_argument("-i", "--idle", type=_duration, metavar="DURATION",
                    help="only sessions alive but waiting for at least DURATION "
                         "(3d, 12h, 1d12h): candidates to close")
    ap.add_argument("-k", "--kill", action="store_true",
                    help="with -i: close those sessions, after asking (SIGTERM; the "
                         "transcript stays, to resume later)")
    ap.add_argument("--json", action="store_true",
                    help="one JSON document for scripts (not available with --watch)")
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
    if args.session and args.target:
        ap.error("-s/--session cannot be used with a target")
    if args.json and args.watch:
        ap.error("--json cannot be used with --watch")
    if args.kill and not args.idle:
        ap.error("-k/--kill needs -i/--idle")
    if args.kill and (args.watch or args.json):
        ap.error("-k/--kill cannot be used with --watch or --json")
    return args


def loader(args: argparse.Namespace) -> Callable[[], list[Session]]:
    """Sessions as the views get them: sanitised once, with --window applied."""
    here = Path.cwd().resolve() if args.session else None

    def load() -> list[Session]:
        sessions = _sessions(args.provider, args.target)
        if here:
            sessions = [s for s in sessions if _related(s.cwd, here)]
            if not sessions:
                raise LookupError(f"no running sessions in {here}")
        if args.idle:
            sessions = [s for s in sessions if _idle_for(s, args.idle, time.time())]
            if not sessions:
                raise LookupError("no sessions idle for that long")
        sanitize(sessions)
        override_window(sessions, args.window)
        return sessions

    return load


def _related(cwd: str | None, here: Path) -> bool:
    """A session run in `here`, in a folder that contains it, or in one below it."""
    if not cwd:
        return False
    try:
        d = Path(cwd).resolve()
    except (OSError, ValueError):
        return False
    return d == here or d in here.parents or here in d.parents


def _idle_for(s: Session, secs: float, now: float) -> bool:
    """Alive (waiting or blocked) and with no activity in the last `secs`."""
    m = s.main
    return m.state in (WAITING, BLOCKED) and m.ended is not None and now - m.ended >= secs


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


EXIT_WAIT_SECS = 3.0


def _close(sessions: list[Session], load: Callable[[], list[Session]],
           color: bool) -> set[str] | None:
    """Ask, then SIGTERM the processes of the idle sessions just shown.

    The ids of the sessions whose process has exited; None when nothing was tried.
    """
    closable = [(s, s.pid) for s in sessions if s.pid]
    for s in sessions:
        if not s.pid:
            print(style(color, "90", f"{s.id[:8]} {s.title}: its process is not known, "
                                     "close it from its own CLI"))
    if not closable:
        return None
    try:
        answer = input(f"Close {len(closable)} session(s)? [y/N] ")
    except EOFError:
        answer = ""
    if answer.strip().lower() not in ("y", "yes"):
        print("nothing closed")
        return None
    # Read again: a session that got a message while we asked is no longer idle.
    try:
        still_idle = {(s.id, s.pid) for s in load()}
    except LookupError:
        still_idle = set()
    signalled = []
    for s, pid in closable:
        name = f"{s.id[:8]} {s.title}"
        if (s.id, pid) not in still_idle:
            print(f"{name}: active again, left open")
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            print(f"{name}: already gone")
            continue
        except PermissionError:
            print(f"{name}: not allowed to close it")
            continue
        signalled.append((s, pid))
    gone = _wait_gone({s.id: pid for s, pid in signalled})
    for s, _ in signalled:
        result = "closed" if s.id in gone else "still running, it did not take SIGTERM"
        print(f"{s.id[:8]} {s.title}: {result}")
    print(f"closed {len(gone)} of {len(closable)}; their transcripts stay, to resume later")
    return gone


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_gone(pids: dict[str, int]) -> set[str]:
    """The session ids whose process has exited, given SIGTERM a moment to land."""
    deadline = time.monotonic() + EXIT_WAIT_SECS
    while True:
        gone = {sid for sid, pid in pids.items() if not _alive(pid)}
        if len(gone) == len(pids) or time.monotonic() >= deadline:
            return gone
        time.sleep(0.1)


def _status_after(args: argparse.Namespace, closed: set[str], color: bool) -> None:
    """What a plain run shows now, without the sessions just closed: the provider
    may still list them for a moment after their process exited."""
    try:
        sessions = [s for s in loader(argparse.Namespace(**{**vars(args), "idle": None}))()
                    if s.id not in closed]
    except LookupError:
        sessions = []
    print()
    print(render(sessions, running_only=args.running, color=color) if sessions
          else style(color, "90", "no running sessions left"))


def main(argv: list[str] | None = None) -> int:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    args = parse_args(argv)
    color = not args.json and (args.color == "always" or
                               (args.color == "auto" and sys.stdout.isatty()
                                and not os.environ.get("NO_COLOR")))
    load = loader(args)
    if not args.watch:
        try:
            sessions = load()
        except LookupError as e:
            if args.json:
                print(f"agents-tree: {e}", file=sys.stderr)
                return 1
            print(style(color, "90", str(e)))
            return 1
        if args.json:
            print(json_dumps(sessions, running_only=args.running))
            return 0
        print(render(sessions, running_only=args.running, color=color))
        if not args.kill:
            return 0
        closed = _close(sessions, load, color)
        if closed is None:
            return 1
        _status_after(args, closed, color)
        return 0
    try:
        live.run(load, interval=args.interval, color=color, running_only=args.running,
                 mouse=args.mouse)
    except KeyboardInterrupt:
        pass
    return 0
