"""Provider-neutral data the renderer draws: sessions and their agent trees."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Neutral states. Providers map their own markers onto these and keep their own
# word for display in Agent.status.
RUNNING = "running"  # working now
WAITING = "waiting"  # alive, waiting for its next message
BLOCKED = "blocked"  # alive, waiting for a person
DONE = "done"
FAILED = "failed"
STALE = "stale"  # no sign of life and no outcome
INACTIVE = "inactive"  # the session is not running


@dataclass
class Detail:
    """What the detail view shows beyond the tree row."""

    prompt: str | None = None
    prompt_label: str = "Prompt"
    tools: dict[str, int] = field(default_factory=dict)
    last_tool: str | None = None
    last_tool_at: float | None = None
    last_tool_pending: bool = False
    last_text: str | None = None
    output_tokens: int = 0
    requests: int = 0
    transcript: str | None = None


@dataclass
class Agent:
    id: str
    label: str
    state: str = STALE
    status: str | None = None  # the provider's word for the state, shown as is
    model: str | None = None
    effort: str | None = None
    context_tokens: int | None = None
    context_window: int | None = None
    started: float | None = None
    # None while the agent runs: the renderer measures elapsed time up to now.
    ended: float | None = None
    children: list[Agent] = field(default_factory=list)
    detail: Detail | None = None

    def has_running(self) -> bool:
        return self.state == RUNNING or any(ch.has_running() for ch in self.children)

    def size(self) -> int:
        return 1 + sum(ch.size() for ch in self.children)

    def walk(self):
        yield self
        for ch in self.children:
            yield from ch.walk()


@dataclass
class Session:
    id: str
    title: str
    main: Agent
    agents: list[Agent] = field(default_factory=list)
    cwd: str | None = None
    kind: str | None = None  # interactive / background

    def all_agents(self):
        yield self.main
        for a in self.agents:
            yield from a.walk()


_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_CONTROL_BUT_NEWLINE_RE = re.compile(r"[\x00-\x09\x0b-\x1f\x7f-\x9f]")


def _line(text: str | None) -> str | None:
    return _CONTROL_RE.sub(" ", text) if text else text


def _lines(text: str | None) -> str | None:
    return _CONTROL_BUT_NEWLINE_RE.sub(" ", text.replace("\r\n", "\n")) if text else text


def sanitize(sessions: list[Session]) -> None:
    """Transcript text is data: no escape sequences reach the terminal. Applied once,
    where sessions enter the program, so no view has to remember it."""
    for s in sessions:
        s.title, s.cwd, s.kind = _line(s.title) or "", _line(s.cwd), _line(s.kind)
        s.id = _line(s.id) or ""
        for a in s.all_agents():
            a.id, a.label = _line(a.id) or "", _line(a.label) or ""
            a.status, a.model, a.effort = _line(a.status), _line(a.model), _line(a.effort)
            d = a.detail
            if d:
                d.prompt, d.last_text = _lines(d.prompt), _lines(d.last_text)
                d.last_tool, d.transcript = _line(d.last_tool), _line(d.transcript)
                d.tools = {_line(k) or "": n for k, n in d.tools.items()}


def override_window(sessions: list[Session], window: int | None) -> None:
    """A context window the user knows better than the provider's inference."""
    if window:
        for s in sessions:
            for a in s.all_agents():
                a.context_window = window
