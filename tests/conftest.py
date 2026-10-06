"""Builders for a fake ~/.claude tree, shaped like the records Claude Code writes,
and for the provider-neutral model the renderer draws."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agents_tree.model import DONE, RUNNING, WAITING, Agent, Session
from agents_tree.providers import claude

T0 = datetime(2026, 10, 7, 6, 0, 0, tzinfo=timezone.utc).timestamp()
NOW = 10_000.0


# --- transcript records --------------------------------------------------------

def ts(offset: float) -> str:
    return datetime.fromtimestamp(T0 + offset, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def assistant(offset, model="claude-opus-5-5", effort: str | None = "high", context=1000,
              tool_uses=(), request: str | None = None):
    record = {
        "type": "assistant", "timestamp": ts(offset), "perTurnEffort": effort,
        "message": {
            "model": model,
            "usage": {"input_tokens": 10, "cache_read_input_tokens": context - 15,
                      "cache_creation_input_tokens": 0, "output_tokens": 5},
            "content": [{"type": "tool_use", "id": tu, "name": "Agent", "input": {}}
                        for tu in tool_uses] or [{"type": "text", "text": "ok"}],
        },
    }
    if request:
        record["requestId"] = request
    return record


def user(offset, content, **extra):
    return {"type": "user", "timestamp": ts(offset), "message": {"role": "user",
                                                               "content": content}, **extra}


def tool_result(offset, tool_use_id, content: str | list = "done"):
    return user(offset, [{"type": "tool_result", "tool_use_id": tool_use_id, "content": content}])


def notification(offset, agent_id, status="completed"):
    return {"type": "queue-operation", "operation": "enqueue", "timestamp": ts(offset),
            "content": f"<task-notification>\n<task-id>{agent_id}</task-id>\n"
                       f"<status>{status}</status>\n</task-notification>"}


def title(text):
    return {"type": "ai-title", "aiTitle": text}


class FakeClaude:
    """A ~/.claude/projects tree under tmp_path."""

    def __init__(self, root: Path) -> None:
        self.projects = root / "projects"
        self.live: list[dict] = []

    def session(self, session_id: str, cwd: str, records: list[dict]) -> Path:
        proj = self.projects / claude.project_dir(cwd).name
        proj.mkdir(parents=True, exist_ok=True)
        path = proj / f"{session_id}.jsonl"
        _write(path, records)
        return path

    def subagent(self, session: Path, agent_id: str, records: list[dict], *,
                 agent_type="general-purpose", description="Do a thing", tool_use_id=None,
                 background=False, mtime: float | None = None, **extra_meta) -> None:
        d = session.with_suffix("") / "subagents"
        d.mkdir(parents=True, exist_ok=True)
        meta = {"agentType": agent_type, "description": description,
                "toolUseId": tool_use_id, "spawnDepth": 1,
                "requestShape": "background" if background else "foreground", **extra_meta}
        (d / f"agent-{agent_id}.meta.json").write_text(json.dumps(meta))
        jl = d / f"agent-{agent_id}.jsonl"
        _write(jl, records)
        if mtime is not None:
            os.utime(jl, (mtime, mtime))


def _write(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


@pytest.fixture
def fake(tmp_path, monkeypatch):
    f = FakeClaude(tmp_path)
    monkeypatch.setattr(claude, "PROJECTS", f.projects)
    monkeypatch.setattr(claude, "live_sessions", lambda: f.live)
    claude._cache.clear()
    claude._used.clear()
    return f


# --- the neutral model ---------------------------------------------------------

_STATES = {"running": RUNNING, "completed": DONE, "busy": RUNNING, "idle": WAITING}


def agent(label, status="completed", started=0.0, children=(), model="opus-5-5",
          effort: str | None = "high", ctx: int | None = 50_000, window=1_000_000):
    return Agent(id=label, label=label, state=_STATES.get(status, DONE), status=status,
                 model=model, effort=effort, context_tokens=ctx, context_window=window,
                 started=started, ended=None if status == "running" else started + 60,
                 children=list(children))


def session(agents, sid="s1", title="Work"):
    main = agent("main", status="busy", started=0.0)
    main.ended = NOW
    return Session(id=sid, title=title, main=main, agents=list(agents), cwd="/w/app",
                   kind="interactive")
