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
MODEL_HEADER, CONTEXT_HEADER, TIME_HEADER, STATUS_HEADER = (
    "MODEL (EFFORT)", "CONTEXT", "TIME", "STATUS")
BAR = 8  # context bar cells
CONTEXT_W = BAR + 5  # the bar, a space, and '100%'
NO_DIR = "(no directory)"
SELECTED = "30;46"  # black on cyan

STATE_COLORS = {RUNNING: "1;33", WAITING: "36", BLOCKED: "1;35", DONE: "32", FAILED: "31",
                STALE: "90", INACTIVE: "90"}
ICONS = {RUNNING: "●", WAITING: "○", BLOCKED: "▲", DONE: "✓", FAILED: "✗", STALE: "·",
         INACTIVE: "–"}
SUMMARY_ORDER = (BLOCKED, RUNNING, WAITING, STALE, INACTIVE, DONE, FAILED)

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
    if secs < 86400:
        return f"{secs // 3600}h{secs % 3600 // 60:02d}m"
    return f"{secs // 86400}d{secs % 86400 // 3600:02d}h"


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


def _cut(text: str, width: int) -> str:
    return text if len(text) <= width else text[:width - 1] + "…"


# --- choosing rows ---------------------------------------------------------------

def _urgency(s: Session) -> int:
    """Sessions waiting on a person first, then working ones, then idle, then the rest."""
    if s.main.state == BLOCKED:
        return 0
    if s.main.has_running() or any(a.has_running() for a in s.agents):
        return 1
    return 2 if s.main.state == WAITING else 3


def _ordered(sessions: list[Session]) -> list[Session]:
    """Grouped by directory. Groups and the sessions in them by urgency, then newest
    first; a group sorts by its most urgent session. Start time, not latest activity,
    breaks ties, so two working sessions do not swap places on every refresh."""
    def rank(s: Session) -> tuple[int, float]:
        return _urgency(s), -(s.main.started or 0)

    groups: dict[str, list[Session]] = {}
    for s in sorted(sessions, key=rank):
        groups.setdefault(s.cwd or "", []).append(s)
    return [s for group in groups.values() for s in group]


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
    groups = len({s.cwd or "" for s in sessions})

    def total() -> int:
        notes = sum(1 for n in hidden_count if n)
        return 1 + groups + sum(1 + n for n in shown) + notes

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


def _tree_rows(sessions: list[Session], running_only: bool,
               hidden: set[int]) -> list[tuple[int, str, Agent]]:
    """Session index, tree prefix and agent for every visible tree row."""
    rows: list[tuple[int, str, Agent]] = []

    def walk(agents: list[Agent], indent: str, session_index: int) -> None:
        kept = _kept(agents, running_only, hidden)
        for n, agent in enumerate(kept):
            last = n == len(kept) - 1
            rows.append((session_index, indent + ("└─ " if last else "├─ "), agent))
            walk(agent.children, indent + ("   " if last else "│  "), session_index)

    for session_index, session in enumerate(sessions):
        rows.append((session_index, " ", session.main))
        walk(session.agents, "   ", session_index)
    return rows


# --- the tree --------------------------------------------------------------------

def _summary(sessions: list[Session], model: str | None) -> str:
    """'▲ 1 blocked  ● 2 running  ○ 9 waiting', counting each session's own state."""
    counts: dict[str, int] = {}
    for s in sessions:
        counts[s.main.state] = counts.get(s.main.state, 0) + 1
    parts = [f"{ICONS[state]} {counts[state]} {state}" for state in SUMMARY_ORDER
             if state in counts]
    return "  ".join(parts + ([f"· {model}"] if model else []))


def _context_cell(a: Agent, colored: bool) -> str:
    """A thin bar and the share of the context window, colored by how full it is.
    Line glyphs, not blocks: blocks fill the whole cell and stack into a wall."""
    if not (a.context_tokens and a.context_window):
        return style(colored, "90", f"{fmt_tokens(a.context_tokens):>{CONTEXT_W}}")
    share = a.context_tokens / a.context_window
    filled = min(round(share * BAR), BAR)
    code = "32" if share < 0.5 else "33" if share < 0.8 else "31"
    return (style(colored, code, "━" * filled) + style(colored, "90", "─" * (BAR - filled))
            + style(colored, code, f" {share * 100:3.0f}%"))


def _time_text(a: Agent, is_main: bool, now: float) -> str:
    """How long it ran; for a session that is not working, how long ago it last did."""
    if is_main and a.state != RUNNING and a.ended is not None:
        return f"{fmt_duration(now - a.ended)} ago"
    return fmt_duration(elapsed(a, now))


def _project(cwd: str | None) -> str:
    return os.path.basename(cwd.rstrip(os.sep)) or cwd if cwd else NO_DIR


def render(sessions: list[Session], **kwargs) -> str:
    return "\n".join(render_lines(sessions, **kwargs)[0])


def render_lines(sessions: list[Session], *, now: float | None = None,
                 running_only: bool = False, color: bool = False, width: int | None = None,
                 max_lines: int | None = None,
                 selected: Key | None = None) -> tuple[list[str], list[Key | None]]:
    """Lines of the tree view, and for each line the agent it shows (None for others).

    The first line is the header: a count of sessions by state, then the column
    names. Sessions are grouped by directory, the ones that need attention first,
    one line each, with their subagents below. With max_lines, the oldest finished
    agents are left out until the view fits; running agents always stay.
    """
    now = time.time() if now is None else now
    sessions = _ordered(sessions)
    hidden, hidden_count = _plan(sessions, running_only, max_lines)
    rows = _tree_rows(sessions, running_only, hidden)

    labels = {id(a): (sessions[i].title + (f" [{sessions[i].kind}]"
                                           if sessions[i].kind not in (None, "interactive")
                                           else "")
                      if a is sessions[i].main else a.label)
              for i, _, a in rows}
    models = {id(a): model_text(a) for _, _, a in rows}
    # One model for everything says nothing per row: name it once, in the header.
    show_model = len(set(models.values())) > 1
    summary = _summary(sessions, None if show_model or not models
                       else next(iter(models.values())))
    times = {id(a): _time_text(a, a is sessions[i].main, now) for i, _, a in rows}

    model_w = min(max([len(MODEL_HEADER), *map(len, models.values())]), MAX_MODEL)
    time_w = max([len(TIME_HEADER), *map(len, times.values())])
    status_w = max([len(STATUS_HEADER), *(len(a.status or a.state) for _, _, a in rows)])
    columns_w = (model_w + 2 if show_model else 0) + CONTEXT_W + time_w + status_w + 3 * 2
    label_w = min(max([len(summary), *(len(p) + 2 + len(labels[id(a)]) for _, p, a in rows)]),
                  MAX_LABEL)
    if width:
        label_w = max(min(label_w, width - columns_w - 2), MIN_LABEL)

    def line(prefix: str, a: Agent, colored: bool) -> str:
        label = _cut(labels[id(a)], max(label_w - len(prefix) - 2, 4))
        icon = style(colored, STATE_COLORS.get(a.state, "37"), ICONS.get(a.state, "?"))
        pad = " " * (label_w - len(prefix) - 2 - len(label))
        if a.state in (BLOCKED, RUNNING):
            label = style(colored, "1", label)
        status = a.status or a.state
        cells = [prefix + icon + " " + label + pad]
        if show_model:
            cells.append(f"{_cut(models[id(a)], model_w):<{model_w}}")
        cells += [_context_cell(a, colored), f"{times[id(a)]:>{time_w}}",
                  style(colored, STATE_COLORS.get(a.state, "37"), f"{status:<{status_w}}")]
        return "  ".join(cells)

    head = [f"{_cut(summary, label_w):<{label_w}}"]
    if show_model:
        head.append(f"{MODEL_HEADER:<{model_w}}")
    head += [f"{CONTEXT_HEADER:<{CONTEXT_W}}", f"{TIME_HEADER:>{time_w}}", STATUS_HEADER]

    out: list[str] = []
    keys: list[Key | None] = []

    def emit(text: str, key: Key | None = None) -> None:
        out.append(text.rstrip())
        keys.append(key)

    emit(style(color, "7", "  ".join(head).ljust(width or 0)))
    for k, (i, prefix, a) in enumerate(rows):
        s = sessions[i]
        if a is s.main and (i == 0 or (sessions[i - 1].cwd or "") != (s.cwd or "")):
            emit(style(color, "1;34", _project(s.cwd)))
        key = (s.id, a.id)
        if key == selected:
            emit(style(color, SELECTED, line(prefix, a, False).ljust(width or 0)), key)
        else:
            emit(line(prefix, a, color), key)
        # The note goes right after the last row of the session.
        if (k == len(rows) - 1 or rows[k + 1][0] != i) and hidden_count[i]:
            emit(style(color, "90", f"   … {hidden_count[i]} older finished agents hidden"))
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
    # The tree shows a session by title alone: where it runs and its id live here.
    out.append(style(color, "90", " · ".join(
        [*([_home(session.cwd)] if session.cwd else []), f"session {session.id}"])))
    if not d:
        return out
    out.append(style(color, "90", f"{d.requests} requests · {fmt_tokens(d.output_tokens)} "
                                  f"output tokens · id {a.id}"))
    if d.transcript:
        out.append(style(color, "90", _home(d.transcript)))
    if d.agent_name:
        out += ["", style(color, "1", "Agent name"), *_wrap(d.agent_name, width)]
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
