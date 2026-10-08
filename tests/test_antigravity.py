"""Antigravity provider tests using JSONL transcript shapes."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agents_tree.model import DONE, INACTIVE, RUNNING
from agents_tree.providers import antigravity

T0 = datetime(2026, 10, 8, 6, 0, 0, tzinfo=timezone.utc).timestamp()


def ts(offset: float) -> str:
    return datetime.fromtimestamp(T0 + offset, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def user_input(offset: float, prompt: str) -> dict:
    model_note = (
        "<USER_SETTINGS_CHANGE>\n"
        "The user changed setting `Model Selection` from None to Gemini 3.8 Flash (Medium).\n"
        "</USER_SETTINGS_CHANGE>"
    )
    return {
        "step_index": 0,
        "source": "USER_EXPLICIT",
        "type": "USER_INPUT",
        "status": "DONE",
        "created_at": ts(offset),
        "content": f"<USER_REQUEST>\n{prompt}\n</USER_REQUEST>\n{model_note}",
    }


def planner_response(offset: float, tool_calls: list[dict] | None = None,
                     content: str | None = None, thinking: str | None = None) -> dict:
    d = {
        "step_index": 1,
        "source": "MODEL",
        "type": "PLANNER_RESPONSE",
        "status": "DONE",
        "created_at": ts(offset),
        "tool_calls": tool_calls or [],
    }
    if content:
        d["content"] = content
    if thinking:
        d["thinking"] = thinking
    return d


def generic(offset: float, content: str) -> dict:
    return {
        "step_index": 2,
        "source": "MODEL",
        "type": "GENERIC",
        "status": "DONE",
        "created_at": ts(offset),
        "content": content,
    }


def invoke_call(specs: list[dict]) -> dict:
    return {
        "name": "invoke_subagent",
        "args": {
            "Subagents": json.dumps(specs),
            "toolAction": "Spawning subagents",
            "toolSummary": "Launch subagents",
        },
    }


def subagents_created_content(ids: list[str]) -> str:
    parts = ["Created the following subagents:"]
    for sid in ids:
        parts.append(json.dumps({
            "conversationId": sid,
            "logAbsoluteUri": f"file:///path/to/{sid}/transcript.jsonl",
            "workspaceUris": ["file:///path/to/ws"],
        }, indent=2))
    return "\n".join(parts)


class FakeAntigravity:
    def __init__(self, root: Path) -> None:
        self.brain = root / "brain"
        self.convos = root / "conversations"
        self.brain.mkdir(parents=True, exist_ok=True)
        self.convos.mkdir(parents=True, exist_ok=True)

    def session(self, sid: str, cwd: str, records: list[dict], mtime: float | None = None) -> Path:
        p = self.brain / sid / ".system_generated" / "logs" / "transcript.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(r) + "\n" for r in records))
        mt = mtime if mtime is not None else T0 + 10
        os.utime(p, (mt, mt))

        # Write sqlite DB with trajectory_metadata_blob
        db_path = self.convos / f"{sid}.db"
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("CREATE TABLE trajectory_metadata_blob (id TEXT, data BLOB)")
        blob_data = f"prefix file://{cwd} suffix".encode("utf-8")
        c.execute("INSERT INTO trajectory_metadata_blob VALUES ('main', ?)", (blob_data,))
        conn.commit()
        conn.close()
        return p


@pytest.fixture
def fake(tmp_path, monkeypatch):
    f = FakeAntigravity(tmp_path)
    monkeypatch.setattr(antigravity, "BRAIN", f.brain)
    monkeypatch.setattr(antigravity, "CONVERSATIONS", f.convos)
    antigravity._cache.clear()
    antigravity._used.clear()
    return f


def test_read_basic_session(fake):
    fake.session("sess1", "/work/project", [
        user_input(0, "Deploy the app"),
        planner_response(5, tool_calls=[{"name": "run_command", "args": {"CommandLine": "ls"}}]),
        generic(6, "file1 file2"),
        planner_response(10, content="All done!"),
    ])

    results = antigravity.sessions("sess1")
    assert len(results) == 1
    s = results[0]
    assert s.id == "sess1"
    assert s.title == "Deploy the app"
    assert s.cwd == "/work/project"
    assert s.main.model == "Gemini 3.8 Flash (Medium)"
    assert s.main.detail is not None
    assert s.main.detail.requests == 2
    assert s.main.detail.tools == {"run_command": 1}
    assert s.main.detail.last_tool == "run_command"
    assert not s.main.detail.last_tool_pending


def test_session_with_subagents(fake):
    send_call = {"name": "send_message", "args": {"Recipient": "sess_parent"}}
    fake.session("child1", "/work/project", [
        user_input(11, "Research competitor pricing"),
        planner_response(15, tool_calls=[{"name": "search_web", "args": {"query": "prices"}}]),
        generic(16, "Results"),
        planner_response(20, tool_calls=[send_call]),
        generic(21, "Message sent"),
    ])

    # Parent session
    fake.session("sess_parent", "/work/project", [
        user_input(0, "Analyze the market"),
        planner_response(10, tool_calls=[invoke_call([{
            "Role": "Market Analyst",
            "TypeName": "research",
            "Model": "flash",
            "Prompt": "Research competitor pricing",
        }])]),
        generic(11, subagents_created_content(["child1"])),
        planner_response(30, content="Market analysis complete."),
    ])

    results = antigravity.sessions("sess_parent")
    assert len(results) == 1
    s = results[0]
    assert s.id == "sess_parent"
    assert len(s.agents) == 1
    sub = s.agents[0]
    assert sub.id == "child1"
    assert sub.label == "research: Market Analyst"
    assert sub.model == "flash"
    assert sub.state == DONE
    assert sub.status == "completed"
    assert sub.detail is not None
    assert sub.detail.tools == {"search_web": 1, "send_message": 1}


def test_find_targets(fake, monkeypatch, tmp_path):
    now = T0 + 50
    monkeypatch.setattr(time, "time", lambda: now)

    dir_a = tmp_path / "work_a"
    dir_b = tmp_path / "work_b"
    dir_a.mkdir()
    dir_b.mkdir()

    # Active session (modified recently)
    fake.session("active1", str(dir_a), [user_input(0, "Task A")], mtime=now - 10)
    # Stale/old session
    fake.session("old1", str(dir_b), [user_input(0, "Task B")], mtime=now - 500)

    # Target None -> only running
    running = antigravity.sessions(None)
    assert len(running) == 1
    assert running[0].id == "active1"
    assert running[0].main.state == RUNNING

    # Target Directory -> dir_b finds old1
    by_dir = antigravity.sessions(str(dir_b))
    assert len(by_dir) == 1
    assert by_dir[0].id == "old1"
    assert by_dir[0].main.state == INACTIVE

    # Target Prefix
    by_prefix = antigravity.sessions("act")
    assert len(by_prefix) == 1
    assert by_prefix[0].id == "active1"

    # Ambiguous prefix
    fake.session("active2", "/work/c", [user_input(0, "Task C")], mtime=now - 10)
    with pytest.raises(LookupError, match="matches 2 sessions: active1, active2"):
        antigravity.sessions("act")

    # Missing target
    with pytest.raises(LookupError, match="no Antigravity session matching 'nonexistent'"):
        antigravity.sessions("nonexistent")


def test_malformed_json_lines_ignored(fake):
    path = fake.session("sess_corrupt", "/work/a", [user_input(0, "Safe task")])
    # Append garbage line
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n{this is broken json}\n\n")

    results = antigravity.sessions("sess_corrupt")
    assert len(results) == 1
    assert results[0].title == "Safe task"
