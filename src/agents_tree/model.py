"""Provider-neutral data the renderer draws: sessions and their agent trees."""

from __future__ import annotations

from dataclasses import dataclass, field

# Statuses an agent can be in. Providers map their own markers onto these.
RUNNING = "running"
COMPLETED = "completed"
DONE = "done"  # finished, but the provider recorded no explicit outcome
FAILED = "failed"
STALE = "stale"  # no completion marker and no activity for a while


@dataclass
class Agent:
    id: str
    label: str
    model: str | None = None
    effort: str | None = None
    context_tokens: int | None = None
    context_window: int | None = None
    status: str | None = None
    started: float | None = None
    # None while the agent runs: the renderer measures elapsed time up to now.
    ended: float | None = None
    children: list[Agent] = field(default_factory=list)

    @property
    def running(self) -> bool:
        return self.status == RUNNING

    def has_running(self) -> bool:
        return self.running or any(ch.has_running() for ch in self.children)

    def size(self) -> int:
        return 1 + sum(ch.size() for ch in self.children)


@dataclass
class Session:
    id: str
    provider: str
    title: str
    main: Agent
    agents: list[Agent] = field(default_factory=list)
    cwd: str | None = None
    kind: str | None = None  # interactive / background
    # Agents left out of the view to fit the screen.
    hidden: int = 0
