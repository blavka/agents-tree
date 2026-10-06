"""Draw sessions as aligned trees: label, model (effort), context, elapsed, status."""

from __future__ import annotations

import re
import time

from agents_tree.model import Agent, Session

MAX_LABEL = 70
MIN_LABEL = 16
MODEL_WIDTH = 21  # "opus-5-5 (xhigh)" and room to spare
CONTEXT_WIDTH = 19  # " 742k  74 % of 1M  "
ELAPSED_WIDTH = 7
HEADER = ("AGENT", "MODEL (EFFORT)", "CONTEXT", "ELAPSED", "STATUS")

_STATUS_COLORS = {
    "running": "1;33", "working": "1;33", "busy": "1;33", "blocked": "1;35",
    "completed": "32", "done": "32", "idle": "36",
    "failed": "31", "killed": "31", "stopped": "31",
    "stale": "90", "not running": "90",
}


_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def clean(text: str) -> str:
    """Transcript text is data: no escape sequences or line breaks reach the terminal."""
    return _CONTROL_RE.sub(" ", text)


class Style:
    def __init__(self, color: bool) -> None:
        self.color = color

    def __call__(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color and text else text


def fmt_tokens(n: int | None) -> str:
    if n is None:
        return "-"
    return f"{n / 1000:.0f}k" if n >= 1000 else str(n)


def fmt_window(n: int) -> str:
    return f"{n // 1_000_000}M" if n % 1_000_000 == 0 else fmt_tokens(n)


def fmt_duration(secs: float | None) -> str:
    if secs is None or secs < 0:
        return "-"
    secs = int(secs)
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m{secs % 60:02d}s"
    return f"{secs // 3600}h{secs % 3600 // 60:02d}m"


def keep_running(agents: list[Agent]) -> list[Agent]:
    """The running agents and the ancestors that lead to them."""
    kept = []
    for a in agents:
        if a.has_running():
            a.children = keep_running(a.children)
            kept.append(a)
    return kept


def _count(agents: list[Agent]) -> int:
    return sum(a.size() for a in agents)


def _block_lines(s: Session, note: bool) -> int:
    return 2 + _count(s.agents) + (1 if note else 0)  # header, main, agents, note


def fit(sessions: list[Session], max_lines: int) -> None:
    """Hide the oldest finished agents, at any depth, until the view fits max_lines."""
    def total() -> int:
        return (sum(_block_lines(s, bool(s.hidden) or not s.agents) for s in sessions)
                + max(len(sessions) - 1, 0))

    def finished(siblings: list[Agent], i: int):
        for a in siblings:
            if a.has_running():
                yield from finished(a.children, i)
            else:
                yield (a.started or 0, i, id(a), a, siblings)

    while total() > max_lines:
        candidates = [c for i, s in enumerate(sessions) for c in finished(s.agents, i)]
        if not candidates:
            return
        _, i, _, oldest, siblings = min(candidates, key=lambda c: c[:3])
        siblings.remove(oldest)
        sessions[i].hidden += oldest.size()


def render(sessions: list[Session], *, now: float | None = None, window: int | None = None,
           running_only: bool = False, color: bool = False, width: int | None = None,
           max_lines: int | None = None, header: bool = True) -> str:
    now = time.time() if now is None else now
    st = Style(color)

    had_agents = [bool(s.agents) for s in sessions]
    if running_only:
        for s in sessions:
            s.agents = keep_running(s.agents)
    if max_lines:
        fit(sessions, max_lines - (1 if header else 0))

    # (session index, tree prefix, agent)
    rows: list[tuple[int, str, Agent]] = []
    for i, s in enumerate(sessions):
        rows.append((i, "", s.main))

        def walk(agents: list[Agent], indent: str, i: int = i) -> None:
            for n, a in enumerate(agents):
                last = n == len(agents) - 1
                rows.append((i, indent + ("└─ " if last else "├─ "), a))
                walk(a.children, indent + ("   " if last else "│  "))

        walk(s.agents, "")

    def columns(a: Agent) -> list[str]:
        model = clean(a.model or "-")
        if a.effort:
            model = f"{model} ({clean(a.effort)})"
        win = window or a.context_window
        tokens = f"{fmt_tokens(a.context_tokens):>5}"
        if a.context_tokens and win:
            share = f"{a.context_tokens / win * 100:3.0f} % of {fmt_window(win)}"
        else:
            share = "  -"
        end = now if a.ended is None else a.ended
        elapsed = end - a.started if a.started is not None else None
        return [f"{model:<{MODEL_WIDTH}}", f"{tokens} {share}".ljust(CONTEXT_WIDTH),
                f"{fmt_duration(elapsed):>{ELAPSED_WIDTH}}"]

    cols = {id(a): columns(a) for _, _, a in rows}
    # columns, the separators around them, and the longest status we print
    cols_width = max((len("  ".join(c)) for c in cols.values()), default=0) + 4 + 11
    for _, _, a in rows:
        a.label = clean(a.label)
    needed = max((len(p) + len(a.label) for _, p, a in rows), default=0)
    label_width = min(needed, MAX_LABEL)
    if width:
        label_width = max(min(label_width, width - cols_width), MIN_LABEL)

    out: list[str] = []
    if header and rows:
        agent_h, model_h, ctx_h, elapsed_h, status_h = HEADER
        line = "  ".join([f"{agent_h:<{label_width}}", f"{model_h:<{MODEL_WIDTH}}",
                          f"{ctx_h:<{CONTEXT_WIDTH}}", f"{elapsed_h:>{ELAPSED_WIDTH}}", status_h])
        out.append(st("7", line.ljust(width)) if width else st("7", line))
    current = None
    for k, (i, prefix, a) in enumerate(rows):
        if i != current:
            current = i
            s = sessions[i]
            if current != 0:
                out.append("")
            out.append(st("1", clean(s.title)) + st("90", f"  [{clean(s.id[:8])}]")
                       + (st("34", f"  {clean(s.cwd)}") if s.cwd else "")
                       + (st("90", f"  {clean(s.kind)}") if s.kind else ""))
        room = max(label_width - len(prefix), 4)
        label = a.label if len(a.label) <= room else a.label[:room - 1] + "…"
        text = st("1", label) if a is sessions[i].main else label
        status = st(_STATUS_COLORS.get(a.status or "", "37"), clean(a.status or ""))
        out.append(prefix + text + " " * (room - len(label)) + "  "
                   + "  ".join(cols[id(a)]) + "  " + status)
        # Note lines go right after the last row of the session.
        if k == len(rows) - 1 or rows[k + 1][0] != i:
            s = sessions[i]
            if s.hidden:
                out.append(st("90", f"  … {s.hidden} older finished agents hidden"))
            elif not had_agents[i]:
                out.append(st("90", "  (no subagents)"))
            elif not s.agents:
                out.append(st("90", "  (no running subagents)"))
    return "\n".join(line.rstrip() for line in out)
