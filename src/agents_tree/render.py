"""Draw sessions as aligned trees, and one agent as a detail page.

Rendering never changes the sessions it is given: filtering (running only) and
fitting to a height decide which rows to draw, and the data stays as loaded.
"""

from __future__ import annotations

import os
import textwrap
import time

from agents_tree.model import (BLOCKED, DONE, FAILED, INACTIVE, RUNNING, STALE, WAITING, Agent,
                               Session)

MAX_LABEL = 70
MIN_LABEL = 16
MAX_MODEL = 28
HEADER = ("AGENT", "MODEL (EFFORT)", "CONTEXT", "ELAPSED", "STATUS")
SELECTED = "30;46"  # black on cyan

STATE_COLORS = {RUNNING: "1;33", WAITING: "36", BLOCKED: "1;35", DONE: "32", FAILED: "31",
                STALE: "90", INACTIVE: "90"}

# (session id, agent id): how the live view refers to a row across refreshes.
Key = tuple[str, str]


def style(color: bool, code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if color and text else text


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


def model_text(a: Agent) -> str:
    return (a.model or "-") + (f" ({a.effort})" if a.effort else "")


def share_text(a: Agent) -> str | None:
    """'25 % of 1M', or None when tokens or window are unknown."""
    if not (a.context_tokens and a.context_window):
        return None
    return f"{a.context_tokens / a.context_window * 100:.0f} % of {fmt_window(a.context_window)}"


def elapsed(a: Agent, now: float) -> float | None:
    end = now if a.ended is None else a.ended
    return end - a.started if a.started is not None else None


def state_text(color: bool, a: Agent) -> str:
    return style(color, STATE_COLORS.get(a.state, "37"), a.status or a.state)


# --- choosing rows ---------------------------------------------------------------

def _kept(agents: list[Agent], running_only: bool, hidden: set[int]) -> list[Agent]:
    return [a for a in agents
            if id(a) not in hidden and (not running_only or a.has_running())]


def _drawn(agents: list[Agent], running_only: bool) -> int:
    return sum(1 + _drawn(a.children, running_only) for a in _kept(agents, running_only, set()))


def _plan(sessions: list[Session], running_only: bool,
          max_lines: int | None) -> tuple[set[int], list[int]]:
    """Agents to leave out, as id()s, so the view fits max_lines; and per session how
    many. The oldest finished subtrees go first, at any depth; running agents stay."""
    hidden: set[int] = set()
    hidden_count = [0] * len(sessions)
    if not max_lines:
        return hidden, hidden_count
    shown = [_drawn(s.agents, running_only) for s in sessions]

    def total() -> int:
        notes = sum(1 for i in range(len(sessions)) if hidden_count[i] or not shown[i])
        return 1 + sum(2 + n for n in shown) + notes + max(len(sessions) - 1, 0)

    # Finished subtrees whose parent is drawn are disjoint: each can go on its own.
    candidates = []
    for i, s in enumerate(sessions):
        stack = _kept(s.agents, running_only, set())
        while stack:
            a = stack.pop()
            if a.has_running():
                stack.extend(_kept(a.children, running_only, set()))
            else:
                candidates.append((a.started or 0, i, len(candidates), a))
    for _, i, _, a in sorted(candidates, key=lambda c: c[:3]):
        if total() <= max_lines:
            break
        hidden.add(id(a))
        hidden_count[i] += a.size()
        shown[i] -= a.size()
    return hidden, hidden_count


# --- the tree --------------------------------------------------------------------

def render(sessions: list[Session], **kwargs) -> str:
    return "\n".join(render_lines(sessions, **kwargs)[0])


def render_lines(sessions: list[Session], *, now: float | None = None,
                 running_only: bool = False, color: bool = False, width: int | None = None,
                 max_lines: int | None = None,
                 selected: Key | None = None) -> tuple[list[str], list[Key | None]]:
    """Lines of the tree view, and for each line the agent it shows (None for others).

    The first line is the column header. With max_lines, the oldest finished agents
    are left out until the view fits; running agents always stay.
    """
    now = time.time() if now is None else now
    hidden, hidden_count = _plan(sessions, running_only, max_lines)

    rows: list[tuple[int, str, Agent]] = []  # session index, tree prefix, agent

    def walk(agents: list[Agent], indent: str, i: int) -> None:
        kept = _kept(agents, running_only, hidden)
        for n, a in enumerate(kept):
            last = n == len(kept) - 1
            rows.append((i, indent + ("└─ " if last else "├─ "), a))
            walk(a.children, indent + ("   " if last else "│  "), i)

    for i, s in enumerate(sessions):
        rows.append((i, "", s.main))
        walk(s.agents, "", i)

    models = {id(a): model_text(a) for _, _, a in rows}
    contexts = {id(a): f"{fmt_tokens(a.context_tokens):>5} {share_text(a) or '  -'}"
                for _, _, a in rows}
    model_w = min(max([len(HEADER[1]), *map(len, models.values())]), MAX_MODEL)
    context_w = max([len(HEADER[2]), *map(len, contexts.values())])
    status_w = max([len(HEADER[4]), *(len(a.status or a.state) for _, _, a in rows)])
    columns_w = model_w + context_w + 7 + status_w + 4 * 2
    label_w = min(max((len(p) + len(a.label) for _, p, a in rows), default=0), MAX_LABEL)
    if width:
        label_w = max(min(label_w, width - columns_w - 2), MIN_LABEL)

    def cells(label: str, model: str, context: str, elapsed_: str) -> str:
        model = model if len(model) <= model_w else model[:model_w - 1] + "…"
        return f"{label:<{label_w}}  {model:<{model_w}}  {context:<{context_w}}  {elapsed_:>7}"

    out: list[str] = []
    keys: list[Key | None] = []

    def emit(line: str, key: Key | None = None) -> None:
        out.append(line.rstrip())
        keys.append(key)

    emit(style(color, "7", (cells(*HEADER[:4]) + "  " + HEADER[4]).ljust(width or 0)))
    for k, (i, prefix, a) in enumerate(rows):
        s = sessions[i]
        if a is s.main:
            if i:
                emit("")
            emit(style(color, "1", s.title) + style(color, "90", f"  [{s.id[:8]}]")
                 + (style(color, "34", f"  {s.cwd}") if s.cwd else "")
                 + (style(color, "90", f"  {s.kind}") if s.kind else ""))
        key = (s.id, a.id)
        room = max(label_w - len(prefix), 4)
        label = a.label if len(a.label) <= room else a.label[:room - 1] + "…"
        line = cells(prefix + label, models[id(a)], contexts[id(a)],
                     fmt_duration(elapsed(a, now)))
        if key == selected:
            emit(style(color, SELECTED, (line + "  " + (a.status or a.state)).ljust(width or 0)),
                 key)
        else:
            if a is s.main:
                line = style(color, "1", label) + line[len(label):]
            emit(line + "  " + state_text(color, a), key)
        # Note lines go right after the last row of the session.
        if k == len(rows) - 1 or rows[k + 1][0] != i:
            if hidden_count[i]:
                emit(style(color, "90", f"  … {hidden_count[i]} older finished agents hidden"))
            elif not s.agents:
                emit(style(color, "90", "  (no subagents)"))
            elif a is s.main:
                emit(style(color, "90", "  (no running subagents)"))
    return out, keys


# --- one agent -------------------------------------------------------------------

def find_agent(sessions: list[Session], key: Key) -> tuple[Session, Agent] | None:
    for s in sessions:
        if s.id == key[0]:
            for a in s.all_agents():
                if a.id == key[1]:
                    return s, a
    return None


def render_detail(session: Session, a: Agent, *, now: float | None = None,
                  color: bool = False, width: int = 100) -> list[str]:
    """The detail view of one agent: everything it is, as wrapped lines."""
    now = time.time() if now is None else now
    d = a.detail
    out = [style(color, "1", f"{session.title} › {a.label}")]

    context = fmt_tokens(a.context_tokens)
    if share_text(a):
        context += f" ({share_text(a)})"
    out.append(" · ".join([state_text(color, a), model_text(a), f"context {context}",
                           f"elapsed {fmt_duration(elapsed(a, now))}"]))
    if not d:
        return out
    out.append(style(color, "90", f"{d.requests} requests · {fmt_tokens(d.output_tokens)} "
                                  f"output tokens · id {a.id}"))
    if d.transcript:
        out.append(style(color, "90", _home(d.transcript)))
    if d.tools:
        counts = ", ".join(f"{name} ×{n}" for name, n in
                           sorted(d.tools.items(), key=lambda kv: (-kv[1], kv[0])))
        out += ["", style(color, "1", f"Tools ({sum(d.tools.values())} calls)"),
                *_wrap(counts, width)]
        if d.last_tool:
            last = f"last: {d.last_tool}"
            if d.last_tool_at:
                last += f", {fmt_duration(now - d.last_tool_at)} ago"
            if d.last_tool_pending:
                last += style(color, "1;33", " · waiting for its result")
            out.append("  " + last)
    if d.prompt:
        out += ["", style(color, "1", d.prompt_label), *_wrap(d.prompt, width)]
    if d.last_text:
        out += ["", style(color, "1", "Last message"), *_wrap(d.last_text, width)]
    return out


def _wrap(text: str, width: int) -> list[str]:
    lines = []
    for paragraph in text.splitlines():
        lines += textwrap.wrap(paragraph.rstrip(), max(width - 4, 20)) or [""]
    return ["  " + line for line in lines]


def _home(path: str) -> str:
    home = os.path.expanduser("~")
    return "~" + path[len(home):] if path.startswith(home + os.sep) else path
