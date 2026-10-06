"""Builders for a fake ~/.claude tree, shaped like the records Claude Code writes."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agents_tree.providers import claude

T0 = datetime(2026, 10, 7, 6, 0, 0, tzinfo=timezone.utc).timestamp()


def ts(offset: float) -> str:
    return datetime.fromtimestamp(T0 + offset, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def assistant(offset, model="claude-opus-5-5", effort: str | None = "high", context=1000,
              tool_uses=()):
    return {
        "type": "assistant", "timestamp": ts(offset), "perTurnEffort": effort,
        "message": {
            "model": model,
            "usage": {"input_tokens": 10, "cache_read_input_tokens": context - 15,
                      "cache_creation_input_tokens": 0, "output_tokens": 5},
            "content": [{"type": "tool_use", "id": tu, "name": "Agent", "input": {}}
                        for tu in tool_uses] or [{"type": "text", "text": "ok"}],
        },
    }


def tool_result(offset, tool_use_id):
    return {"type": "user", "timestamp": ts(offset),
            "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tool_use_id, "content": "done"}]}}


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
        proj = claude.project_dir(cwd)
        proj = self.projects / proj.name
        proj.mkdir(parents=True, exist_ok=True)
        path = proj / f"{session_id}.jsonl"
        _write(path, records)
        return path

    def subagent(self, session: Path, agent_id: str, records: list[dict], *,
                 agent_type="general-purpose", description="Do a thing", tool_use_id=None,
                 background=False, mtime: float | None = None) -> None:
        d = session.with_suffix("") / "subagents"
        d.mkdir(parents=True, exist_ok=True)
        meta = {"agentType": agent_type, "description": description,
                "toolUseId": tool_use_id, "spawnDepth": 1,
                "requestShape": "background" if background else "foreground"}
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
    return f
