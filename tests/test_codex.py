"""Codex provider tests using JSONL shapes captured from the disposable demo."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agents_tree.model import DONE, FAILED, INACTIVE, RUNNING, STALE, WAITING
from agents_tree.providers import codex

T0 = datetime(2026, 10, 7, 6, 0, 0, tzinfo=timezone.utc).timestamp()


def ts(offset: float) -> str:
    return datetime.fromtimestamp(T0 + offset, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def meta(thread, root, cwd, originator="codex_tui", source="user"):
    return {"type": "session_meta", "timestamp": ts(0), "payload": {
        "id": thread, "session_id": root, "cwd": cwd, "originator": originator,
        "thread_source": source,
    }}


def turn(offset, model="gpt-6.1-sol", effort="medium"):
    return {"type": "turn_context", "timestamp": ts(offset),
            "payload": {"model": model, "effort": effort}}


def message(offset, role, text):
    return {"type": "response_item", "timestamp": ts(offset), "payload": {
        "type": "message", "role": role,
        "content": [{"type": "input_text", "text": text}],
    }}


def agent_message(offset, text):
    return {"type": "response_item", "timestamp": ts(offset), "payload": {
        "type": "agent_message", "content": [{"type": "output_text", "text": text}],
    }}


def tool(offset, name, call_id="call-1"):
    return {"type": "response_item", "timestamp": ts(offset), "payload": {
        "type": "custom_tool_call", "name": name, "call_id": call_id,
    }}


def tool_done(offset, call_id="call-1"):
    return {"type": "response_item", "timestamp": ts(offset), "payload": {
        "type": "custom_tool_call_output", "call_id": call_id,
    }}


def usage(offset, response="r1", total=1000, output=20):
    return {"type": "token_usage_record", "timestamp": ts(offset), "payload": {
        "response_id": response,
        "usage": {"total_tokens": total, "output_tokens": output,
                  "reasoning_output_tokens": 5},
    }}


def activity(offset, child, path, kind):
    return {"type": "event_msg", "timestamp": ts(offset), "payload": {
        "type": "item_completed", "item": {
            "type": "SubAgentActivity", "kind": kind, "agent_thread_id": child,
            "agent_path": path,
        },
    }}


def task_started(offset, window=258400):
    return {"type": "event_msg", "timestamp": ts(offset), "payload": {
        "type": "task_started", "model_context_window": window,
    }}


class FakeCodex:
    def __init__(self, root: Path) -> None:
        self.sessions = root / "sessions"

    def transcript(self, thread, root, cwd, records, *, source="user", originator="codex_tui",
                   day="07"):
        path = self.sessions / "2026" / "10" / day / f"rollout-demo-{thread}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        all_records = [meta(thread, root, cwd, originator, source), *records]
        path.write_text("".join(json.dumps(r) + "\n" for r in all_records))
        os.utime(path, (T0 + 10, T0 + 10))
        return path


@pytest.fixture
def fake(tmp_path, monkeypatch):
    f = FakeCodex(tmp_path)
    monkeypatch.setattr(codex, "SESSIONS", f.sessions)
    codex._cache.clear()
    codex._meta_cache.clear()
    codex._used.clear()
    return f


def test_load_reads_main_details_and_a_completed_subagent(fake):
    root = fake.transcript("root", "root", "/w/app", [
        task_started(0), message(1, "user", "fix the login"), tool(2, "exec"), tool_done(3),
        agent_message(4, "Looking."), usage(5),
        activity(6, "child", "/root/map_modules", "started"),
        activity(9, "child", "/root/map_modules", "completed"),
    ])
    fake.transcript("child", "root", "/w/app", [
        turn(7, "gpt-6-luna", "low"), agent_message(8, "Mapped it."), usage(8, "r2", 50),
    ])

    session = codex.load(root, live=True, now=T0 + 20)

    assert session.title == "fix the login"
    assert (session.main.model, session.main.effort) == (None, None)
    assert (session.main.context_tokens, session.main.context_window) == (1000, 258400)
    assert session.main.detail and session.main.detail.tools == {"exec": 1}
    assert session.main.detail.last_text == "Looking."
    assert session.main.detail.requests == 1
    agent = session.agents[0]
    assert (agent.id, agent.label, agent.state, agent.status) == (
        "child", "agent: map_modules", DONE, "completed")
    assert (agent.model, agent.effort, agent.context_tokens) == ("gpt-6-luna", "low", 50)
    assert agent.detail and agent.detail.prompt is None
    assert agent.detail.last_text == "Mapped it."


def test_nested_agents_follow_the_transcript_that_spawned_them(fake):
    root = fake.transcript("root", "root", "/w/app", [
        activity(1, "parent", "/root/review", "started"),
        activity(8, "parent", "/root/review", "completed"),
    ])
    fake.transcript("parent", "root", "/w/app", [
        activity(2, "child", "/root/review/callers", "started"),
        activity(7, "child", "/root/review/callers", "completed"),
    ], source="subagent")
    fake.transcript("child", "root", "/w/app", [agent_message(4, "Done")], source="subagent")

    session = codex.load(root, now=T0 + 20)

    assert [agent.label for agent in session.agents] == ["agent: review"]
    assert [agent.label for agent in session.agents[0].children] == ["agent: callers"]


@pytest.mark.parametrize(("outcome", "age", "live", "state", "status"), [
    ("completed", 1, False, DONE, "completed"),
    ("failed", 1, False, FAILED, "failed"),
    (None, 1, True, RUNNING, "running"),
    (None, codex.STALE_AFTER_SECS + 5, True, WAITING, "waiting"),
    (None, codex.STALE_AFTER_SECS + 5, False, STALE, "stale"),
])
def test_subagent_states(fake, outcome, age, live, state, status):
    records = [activity(1, "child", "/root/child", "started")]
    if outcome:
        records.append(activity(2, "child", "/root/child", outcome))
    root = fake.transcript("root", "root", "/w/app", records)
    child = fake.transcript("child", "root", "/w/app", [], source="subagent")
    now = time.time()
    os.utime(child, (now - age, now - age))

    agent = codex.load(root, live=live, now=now).agents[0]

    assert (agent.state, agent.status) == (state, status)
    assert (agent.ended is None) == (state == RUNNING)


def test_inactive_main_and_background_kind(fake):
    root = fake.transcript("root", "root", "/w/app", [], originator="codex_exec")

    session = codex.load(root)

    assert (session.main.state, session.main.status, session.kind) == (
        INACTIVE, "not running", "background")


def test_running_main_and_resumed_child_have_no_end_time(fake):
    root = fake.transcript("root", "root", "/w/app", [
        activity(1, "child", "/root/child", "started"),
        activity(2, "child", "/root/child", "completed"),
        activity(3, "child", "/root/child", "started"),
    ])
    child = fake.transcript("child", "root", "/w/app", [], source="subagent")
    now = time.time()
    os.utime(child, (now, now))

    session = codex.load(root, live=True, now=now)

    assert session.main.ended is None
    assert session.agents[0].state == RUNNING
    assert session.agents[0].ended is None


def test_malformed_lines_and_partial_appends_are_skipped(fake):
    path = fake.transcript("root", "root", "/w/app", [usage(1, total=100)])
    with path.open("a") as file:
        file.write("not json\n[1, 2]\n")
        line = json.dumps(usage(2, "r2", 900))
        file.write(line[:30])

    assert codex.load(path).main.context_tokens == 100

    with path.open("a") as file:
        file.write(line[30:] + "\n")
    assert codex.load(path).main.context_tokens == 900


def test_damaged_utf8_does_not_hide_other_sessions(fake):
    bad = fake.sessions / "2026" / "10" / "07" / "rollout-bad.jsonl"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b'\xff\xfebroken\n')
    good = fake.transcript("root", "root", "/w/app", [])

    assert codex._records()["root"][0] == good


def test_stale_pending_tool_is_not_running_when_its_session_is_inactive(fake):
    root = fake.transcript("root", "root", "/w/app", [
        activity(1, "child", "/root/child", "started"),
    ])
    child = fake.transcript("child", "root", "/w/app", [tool(2, "exec")], source="subagent")
    now = time.time()
    os.utime(child, (now - codex.STALE_AFTER_SECS - 1, now - codex.STALE_AFTER_SECS - 1))

    assert codex.load(root, live=False, now=now).agents[0].state == STALE


def test_find_handles_live_directory_fallback_and_id_prefix(fake, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    old = fake.transcript("old-session", "old-session", str(work), [], day="06")
    new = fake.transcript("abcdef12", "abcdef12", str(work), [], day="07")
    now = time.time()
    os.utime(old, (now - 1000, now - 1000))
    os.utime(new, (now - 1, now - 1))

    records = codex._records()
    assert [(meta.thread_id, live) for _, meta, live in codex.find(str(work), records, now)] == [
        ("abcdef12", True)]
    assert codex.find("abcdef", records, now)[0][1].thread_id == "abcdef12"

    os.utime(new, (now - 500, now - 500))
    assert codex.find(str(work), records, now)[0][1].thread_id == "abcdef12"


def test_find_ambiguous_prefix_and_missing_target(fake):
    fake.transcript("abc1", "abc1", "/w/a", [])
    fake.transcript("abc2", "abc2", "/w/b", [])

    with pytest.raises(LookupError, match="matches 2 sessions: abc1, abc2"):
        codex.find("abc")
    with pytest.raises(LookupError, match="no Codex session matching"):
        codex.find("nope")


def test_sessions_forget_transcripts_they_no_longer_read(fake, tmp_path):
    old = fake.transcript("old", "old", "/w/old", [agent_message(1, "old")], day="06")
    codex.read_transcript(old)
    work = tmp_path / "work"
    work.mkdir()
    live = fake.transcript("live", "live", str(work), [agent_message(1, "new")])
    os.utime(live, None)

    codex.sessions(str(work))

    assert old not in codex._cache
