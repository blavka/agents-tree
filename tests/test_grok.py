"""Grok Build provider: fixture-driven, mirrors test_claude.py coverage."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import pytest

from agents_tree.model import DONE, FAILED, INACTIVE, RUNNING, STALE, WAITING
from agents_tree.providers import grok

T0 = datetime(2026, 10, 7, 6, 0, 0, tzinfo=timezone.utc).timestamp()


def ts(offset: float) -> str:
    return datetime.fromtimestamp(T0 + offset, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def envelope(offset, update: dict, session_id="s1"):
    return {"timestamp": int(T0 + offset), "method": "session/update",
            "params": {"sessionId": session_id, "update": update}}


def user_chunk(offset, text, session_id="s1"):
    return envelope(offset, {"sessionUpdate": "user_message_chunk",
                             "content": {"type": "text", "text": text}}, session_id)


def agent_chunk(offset, text, session_id="s1"):
    return envelope(offset, {"sessionUpdate": "agent_message_chunk",
                             "content": {"type": "text", "text": text}}, session_id)


def tool_call(offset, tool_id, title="Read", status="pending", session_id="s1"):
    return envelope(offset, {"sessionUpdate": "tool_call", "toolCallId": tool_id,
                             "title": title, "kind": "read", "status": status}, session_id)


def tool_done(offset, tool_id, session_id="s1"):
    return envelope(offset, {"sessionUpdate": "tool_call_update", "toolCallId": tool_id,
                             "status": "completed"}, session_id)


def usage_line(offset, *, input_tokens=100, output_tokens=20, message_id="m1",
               model="grok-4.6", session_id="s1"):
    return envelope(offset, {"sessionUpdate": "usage", "messageId": message_id,
                             "usage": {"inputTokens": input_tokens,
                                       "cachedReadTokens": 880,
                                       "outputTokens": output_tokens},
                             "modelId": model}, session_id)


class FakeGrok:
    """A ~/.grok/sessions tree under tmp_path."""

    def __init__(self, root: Path) -> None:
        self.home = root
        self.sessions = root / "sessions"
        self.live: list[dict] | None = None  # None → derive from files via provider

    def _cwd_dir(self, cwd: str) -> Path:
        d = self.sessions / quote(cwd, safe="")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def session(self, session_id: str, cwd: str, updates: list[dict], *,
                title="Fix login", model="grok-4.6", effort="high",
                context_window=2_000_000, session_kind=None, parent=None,
                usage: dict | None = None, updated_at: float | None = None) -> Path:
        d = self._cwd_dir(cwd) / session_id
        d.mkdir(parents=True, exist_ok=True)
        summary = {
            "info": {"id": session_id, "cwd": cwd},
            "generated_title": title,
            "session_summary": title,
            "created_at": ts(0),
            "updated_at": ts(updated_at if updated_at is not None else 10),
            "last_active_at": ts(updated_at if updated_at is not None else 10),
            "current_model_id": model,
            "reasoning_effort": effort,
            "context_window": context_window,
            "num_messages": len(updates),
            "num_chat_messages": len(updates),
        }
        if session_kind:
            summary["session_kind"] = session_kind
        if parent:
            summary["parent_session_id"] = parent
        (d / "summary.json").write_text(json.dumps(summary))
        (d / "updates.jsonl").write_text("".join(json.dumps(u) + "\n" for u in updates))
        if usage is not None:
            (d / "usage.json").write_text(json.dumps(usage))
        # Make updates look fresh relative to "now" in tests that pass now=T0+…
        mtime = T0 + (updated_at if updated_at is not None else 10)
        os.utime(d / "updates.jsonl", (mtime, mtime))
        os.utime(d / "summary.json", (mtime, mtime))
        return d

    def subagent(self, parent: Path, subagent_id: str, *, child_id: str | None = None,
                 child_cwd: str | None = None, updates: list[dict] | None = None,
                 subagent_type="general-purpose", description="Do a thing",
                 prompt="Look for the bug.", status="running",
                 model="grok-4.6", effort="high", mtime: float | None = None,
                 started_offset=1.0, completed_offset: float | None = None) -> Path:
        parent_summary = json.loads((parent / "summary.json").read_text())
        parent_id = parent_summary["info"]["id"]
        parent_cwd = str(parent_summary["info"]["cwd"])
        child_id = child_id or f"child-{subagent_id}"
        child_cwd = child_cwd or parent_cwd
        child_updates = updates if updates is not None else [
            user_chunk(started_offset, prompt, child_id),
            agent_chunk(started_offset + 1, "ok", child_id),
            usage_line(started_offset + 1, message_id="c1", model=model, session_id=child_id),
        ]
        child = self.session(child_id, child_cwd, child_updates, title=description,
                             model=model, effort=effort, session_kind="subagent",
                             parent=parent_id, updated_at=started_offset + 2)
        meta_dir = parent / "subagents" / subagent_id
        meta_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "subagent_id": subagent_id,
            "parent_session_id": parent_id,
            "child_session_id": child_id,
            "subagent_type": subagent_type,
            "description": description,
            "prompt": prompt,
            "status": status,
            "started_at": ts(started_offset),
            "child_cwd": child_cwd,
            "effective_model_id": model,
        }
        if completed_offset is not None:
            meta["completed_at"] = ts(completed_offset)
            meta["duration_ms"] = int((completed_offset - started_offset) * 1000)
        (meta_dir / "meta.json").write_text(json.dumps(meta))
        touch = mtime if mtime is not None else T0 + started_offset + 2
        os.utime(child / "updates.jsonl", (touch, touch))
        os.utime(meta_dir / "meta.json", (touch, touch))
        return child


@pytest.fixture
def fake(tmp_path, monkeypatch):
    f = FakeGrok(tmp_path)
    monkeypatch.setattr(grok, "GROK_HOME", f.home)
    monkeypatch.setattr(grok, "SESSIONS", f.sessions)
    grok._cache.clear()
    grok._used.clear()
    grok._live = grok._LiveCache()

    real_query = grok._query_live

    def query():
        if f.live is not None:
            return list(f.live)
        return real_query()

    monkeypatch.setattr(grok, "_query_live", query)
    return f


def test_load_reads_model_effort_context_and_title(fake):
    path = fake.session("s1", "/w/app", [
        user_chunk(0, "fix the login"),
        agent_chunk(1, "Looking."),
        usage_line(1, input_tokens=100, output_tokens=20),
    ])

    s = grok.load(path, now=T0 + 10)

    assert s.title == "Fix login"
    assert (s.main.model, s.main.effort) == ("grok-4.6", "high")
    assert s.main.context_tokens == 1000  # 100 + 880 + 20
    assert s.main.context_window == 2_000_000


def test_short_model_strips_xai_prefix_and_default_window(fake):
    path = fake.session("s1", "/w/app", [
        usage_line(0, model="xai/grok-3-mini"),
    ], model="xai/grok-3-mini", effort=None, context_window=None)

    main = grok.load(path, now=T0 + 10).main

    assert main.model == "grok-3-mini"
    assert main.context_window == 128_000


def test_title_falls_back_to_session_id_prefix(fake):
    path = fake.session("0123456789abcdef", "/w/app", [agent_chunk(0, "hi")], title="")
    # Empty generated title → rewrite summary without title fields
    summary = json.loads((path / "summary.json").read_text())
    summary["generated_title"] = ""
    summary["session_summary"] = ""
    (path / "summary.json").write_text(json.dumps(summary))

    assert grok.load(path).title == "01234567"


@pytest.mark.parametrize(("status", "age", "state", "shown"), [
    ("completed", 0, DONE, "completed"),
    ("failed", 0, FAILED, "failed"),
    ("cancelled", 0, FAILED, "cancelled"),
    ("running", 0, RUNNING, "running"),
    ("running", grok.STALE_AFTER_SECS + 60, STALE, "stale"),
])
def test_subagent_state_from_meta(fake, status, age, state, shown):
    path = fake.session("s1", "/w/app", [user_chunk(0, "go"), agent_chunk(1, "ok")])
    now = time.time()
    fake.subagent(path, "a1", status=status, mtime=now - age,
                  completed_offset=5.0 if status != "running" else None)

    agent = grok.load(path, {"sessionId": "s1", "status": "busy"}, now=now).agents[0]

    assert (agent.state, agent.status) == (state, shown)
    assert (agent.ended is None) == (state == RUNNING)


@pytest.mark.parametrize(("live_status", "state"), [
    ("busy", RUNNING), ("working", RUNNING), ("idle", WAITING), ("done", DONE),
    ("something new", WAITING),
])
def test_main_state_from_the_live_session(fake, live_status, state):
    path = fake.session("s1", "/w/app", [agent_chunk(0, "hi")])

    main = grok.load(path, {"sessionId": "s1", "status": live_status}).main

    assert (main.state, main.status) == (state, live_status)


def test_session_that_is_not_running_is_inactive(fake):
    main = grok.load(fake.session("s1", "/w/app", [agent_chunk(0, "hi")])).main

    assert (main.state, main.status) == (INACTIVE, "not running")


def test_subagent_label_and_prompt(fake):
    path = fake.session("s1", "/w/app", [
        user_chunk(0, "parent"), tool_call(1, "tu1", "spawn_subagent"),
    ])
    fake.subagent(path, "a1", subagent_type="explore", description="Study code",
                  prompt="Read the auth module.", status="completed",
                  completed_offset=5)

    s = grok.load(path, now=T0 + 10)

    assert [a.label for a in s.agents] == ["explore: Study code"]
    assert s.agents[0].detail is not None
    assert s.agents[0].detail.prompt == "Read the auth module."


def test_nested_subagent_hangs_under_the_agent_that_spawned_it(fake):
    """Nested spawn: meta under the child's subagents/, like Claude's spawnDepth."""
    path = fake.session("s1", "/w/app", [
        user_chunk(0, "parent"), tool_call(1, "tu1", "spawn_subagent"),
    ])
    review = fake.subagent(
        path, "review", subagent_type="general-purpose",
        description="Review checkout path", status="running", started_offset=1)
    fake.subagent(
        review, "find", subagent_type="explore",
        description="Find reserve and release callers", status="running",
        started_offset=2,
        updates=[
            user_chunk(2, "Find every caller", "child-find"),
            agent_chunk(3, "Found three.", "child-find"),
        ],
        child_id="child-find")

    s = grok.load(path, now=T0 + 10)

    assert [a.label for a in s.agents] == ["general-purpose: Review checkout path"]
    assert [a.label for a in s.agents[0].children] == [
        "explore: Find reserve and release callers"]
    assert s.agents[0].children[0].id == "find"


def test_nested_cycle_through_child_sessions_does_not_lose_agents(fake):
    """A child that somehow lists its parent again must still show both agents."""
    path = fake.session("s1", "/w/app", [agent_chunk(0, "hi")])
    review = fake.subagent(path, "review", description="Review", status="completed",
                           completed_offset=5, started_offset=1, child_id="child-review")
    # Fabricate a meta under the child that points back at the top session.
    meta_dir = review / "subagents" / "loop"
    meta_dir.mkdir(parents=True)
    (meta_dir / "meta.json").write_text(json.dumps({
        "subagent_id": "loop",
        "parent_session_id": "child-review",
        "child_session_id": "s1",
        "child_cwd": "/w/app",
        "subagent_type": "explore",
        "description": "loop",
        "status": "completed",
        "started_at": ts(3),
        "completed_at": ts(4),
    }))

    s = grok.load(path, now=T0 + 10)

    assert sum(a.size() for a in s.agents) == 2
    assert [a.id for a in s.agents] == ["review"]
    assert [a.id for a in s.agents[0].children] == ["loop"]


def test_plugin_agent_type_drops_its_plugin_prefix(fake):
    path = fake.session("s1", "/w/app", [agent_chunk(0, "hi")])
    fake.subagent(path, "a1", subagent_type="systems-programming:golang-pro",
                  description="Review", status="completed", completed_offset=5)

    assert grok.load(path, now=T0 + 10).agents[0].label == "golang-pro: Review"


def test_agents_are_ordered_by_start(fake):
    path = fake.session("s1", "/w/app", [agent_chunk(0, "hi")])
    fake.subagent(path, "late", description="late", started_offset=20, status="completed",
                  completed_offset=25)
    fake.subagent(path, "early", description="early", started_offset=10, status="completed",
                  completed_offset=15)

    labels = [a.label for a in grok.load(path, now=T0 + 30).agents]

    assert labels == ["general-purpose: early", "general-purpose: late"]


def test_garbage_lines_are_skipped(fake):
    path = fake.session("s1", "/w/app", [usage_line(0)])
    with (path / "updates.jsonl").open("a") as f:
        f.write("not json\n[1, 2]\n{\"method\": \"session/update\", \"params\": 3}\n")

    assert grok.load(path).main.context_tokens == 1000


def test_appended_lines_are_read_on_top_of_the_earlier_parse(fake):
    path = fake.session("s1", "/w/app", [
        usage_line(0, input_tokens=10, output_tokens=5, message_id="a"),
    ])
    # Force context from first parse
    assert grok.load(path).main.context_tokens == 895  # 10+880+5

    with (path / "updates.jsonl").open("a") as f:
        line = usage_line(5, input_tokens=100, output_tokens=20, message_id="b")
        f.write(json.dumps(line) + "\n")

    assert grok.load(path).main.context_tokens == 1000


def test_a_line_still_being_written_waits_for_its_end(fake):
    path = fake.session("s1", "/w/app", [usage_line(0, message_id="a")])
    line = json.dumps(usage_line(5, input_tokens=500, output_tokens=50, message_id="b"))
    with (path / "updates.jsonl").open("a") as f:
        f.write(line[:40])

    assert grok.load(path).main.context_tokens == 1000

    with (path / "updates.jsonl").open("a") as f:
        f.write(line[40:] + "\n")

    assert grok.load(path).main.context_tokens == 1430  # 500+880+50


def test_sessions_forget_transcripts_they_no_longer_read(fake):
    old = fake.session("old", "/w/a", [agent_chunk(0, "hi")], updated_at=0)
    grok.read_updates(old / "updates.jsonl")
    fake.session("s1", "/w/a", [agent_chunk(0, "hi")], updated_at=100)
    fake.live = [{"sessionId": "s1", "cwd": "/w/a", "kind": "interactive", "status": "busy"}]

    grok.sessions(None)

    assert (old / "updates.jsonl") not in grok._cache


def test_find_without_target_lists_live_sessions(fake):
    fake.session("s1", "/w/a", [agent_chunk(0, "hi")], updated_at=100)
    fake.session("s2", "/w/b", [agent_chunk(0, "hi")], updated_at=0)
    # Force s1 live via explicit list; s2 absent
    fake.live = [{"sessionId": "s1", "cwd": "/w/a", "kind": "interactive", "status": "busy"}]

    found = grok.find(None)

    assert [(p.name, cwd) for p, _, cwd in found] == [("s1", "/w/a")]


def test_find_without_live_sessions_raises(fake):
    fake.live = []
    with pytest.raises(LookupError, match="no running"):
        grok.find(None)


def test_find_directory_prefers_sessions_running_there(fake, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    fake.session("old", str(work), [agent_chunk(0, "hi")], updated_at=0)
    fake.session("live", str(work), [agent_chunk(0, "hi")], updated_at=100)
    fake.live = [{"sessionId": "live", "cwd": str(work), "kind": "interactive", "status": "idle"}]

    assert [p.name for p, _, _ in grok.find(str(work))] == ["live"]


def test_find_directory_falls_back_to_newest_session(fake, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    old = fake.session("old", str(work), [agent_chunk(0, "hi")], updated_at=0)
    new = fake.session("new", str(work), [agent_chunk(0, "hi")], updated_at=100)
    os.utime(old, (T0, T0))
    os.utime(new, (T0 + 100, T0 + 100))
    fake.live = []

    [(path, live, cwd)] = grok.find(str(work))

    assert (path.name, live, cwd) == ("new", None, str(work))


def test_find_directory_without_sessions_raises(fake, tmp_path):
    with pytest.raises(LookupError, match="no Grok Build sessions in"):
        grok.find(str(tmp_path))


def test_find_session_by_id_prefix(fake):
    fake.session("abcdef12-3456", "/w/a", [agent_chunk(0, "hi")])

    [(path, live, _)] = grok.find("abcdef")

    assert (path.name, live) == ("abcdef12-3456", None)


def test_find_unknown_session_raises(fake):
    with pytest.raises(LookupError, match="no Grok Build session matching"):
        grok.find("nope")


def test_ambiguous_prefix_names_the_candidates(fake):
    fake.session("abc1", "/w/a", [agent_chunk(0, "hi")])
    fake.session("abc2", "/w/b", [agent_chunk(0, "hi")])

    with pytest.raises(LookupError, match="matches 2 sessions: abc1, abc2"):
        grok.find("abc")


def test_hidden_subagent_sessions_are_not_listed_at_top_level(fake, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    parent = fake.session("parent", str(work), [agent_chunk(0, "hi")], updated_at=0)
    fake.subagent(parent, "a1", status="completed", completed_offset=5)
    fake.live = []

    [(path, _, _)] = grok.find(str(work))

    assert path.name == "parent"


def test_encode_cwd_matches_urlencoding(fake):
    assert grok.encode_cwd("/w/app") == "%2Fw%2Fapp"
    assert grok.decode_cwd_dir(fake.sessions / "%2Fw%2Fapp") == "/w/app"


def test_long_cwd_recovers_via_dot_cwd_file(fake, tmp_path):
    # Simulate hash-encoded dirname with .cwd sidecar (no blake3 needed to read).
    long_cwd = "/w/" + "中" * 40
    group = fake.sessions / "workspace-abcdef0123456789"
    group.mkdir(parents=True)
    (group / ".cwd").write_text(long_cwd)
    session = group / "s1"
    session.mkdir()
    summary = {"info": {"id": "s1", "cwd": long_cwd}, "generated_title": "Long",
               "created_at": ts(0), "updated_at": ts(1), "current_model_id": "grok-4.6"}
    (session / "summary.json").write_text(json.dumps(summary))
    (session / "updates.jsonl").write_text(json.dumps(agent_chunk(0, "hi")) + "\n")

    assert grok.decode_cwd_dir(group) == long_cwd
    assert grok.session_dir_by_id("s1", long_cwd) == session


def test_detail_collects_prompt_tools_and_last_message(fake):
    path = fake.session("s1", "/w/app", [
        user_chunk(0, "first ask"),
        agent_chunk(1, "working"),
        tool_call(2, "t1", "Bash"),
        tool_call(3, "t2", "Read `/tmp/x`"),
        tool_done(4, "t1"),
        tool_done(5, "t2"),
        agent_chunk(6, "All done."),
        usage_line(6, message_id="r1", output_tokens=40),
        usage_line(7, message_id="r2", output_tokens=5),
    ])
    fake.subagent(path, "a1", status="completed", completed_offset=8, prompt="Look for the bug.",
                  updates=[
                      user_chunk(1, "Look for the bug.", "child-a1"),
                      tool_call(2, "b1", "Bash", session_id="child-a1"),
                      tool_call(3, "b2", "Read", session_id="child-a1"),
                      tool_done(4, "b1", "child-a1"),
                      tool_done(5, "b2", "child-a1"),
                      agent_chunk(6, "All done.", "child-a1"),
                      usage_line(6, message_id="c1", output_tokens=40, session_id="child-a1"),
                      usage_line(7, message_id="c2", output_tokens=5, session_id="child-a1"),
                  ])

    s = grok.load(path, now=T0 + 20)
    d = s.agents[0].detail
    main_d = s.main.detail

    assert d is not None
    assert d.prompt == "Look for the bug."
    assert d.tools == {"Bash": 1, "Read": 1}
    assert (d.last_tool, d.last_tool_pending) == ("Read", False)
    assert d.last_text == "All done."
    assert d.requests == 2
    assert main_d is not None and main_d.prompt_label == "Last prompt"
    assert main_d.prompt == "first ask"


def test_agent_waiting_on_a_long_tool_call_stays_running(fake):
    path = fake.session("s1", "/w/app", [agent_chunk(0, "hi")])
    now = T0 + 1000
    fake.subagent(path, "a1", status="running", mtime=T0, updates=[
        user_chunk(1, "run tests", "child-a1"),
        tool_call(2, "bash1", "Bash", session_id="child-a1"),
    ])

    s = grok.load(path, {"sessionId": "s1", "status": "busy"}, now=now)

    assert s.agents[0].state == RUNNING
    assert s.agents[0].detail and s.agents[0].detail.last_tool_pending


def test_usage_json_fills_context_when_updates_lack_usage(fake):
    path = fake.session("s1", "/w/app", [user_chunk(0, "hi"), agent_chunk(1, "ok")], usage={
        "sessionId": "s1",
        "session": {"inputTokens": 50, "cachedReadTokens": 900, "outputTokens": 30,
                    "totalTokens": 80, "modelCalls": 2},
        "turns": [{"turnNumber": 1, "inputTokens": 50, "cachedReadTokens": 900,
                   "outputTokens": 30, "totalTokens": 80}],
    })

    main = grok.load(path).main

    # totalTokens, not input + cache + output (cache sits inside input).
    assert main.context_tokens == 80
    assert main.context_window == 2_000_000
    assert main.detail and main.detail.requests >= 1


def _with_total(offset, total, prompt, session_id="s1"):
    event = envelope(offset, {"sessionUpdate": "agent_message_chunk",
                              "content": {"type": "text", "text": "x"}}, session_id)
    event["params"]["_meta"] = {"totalTokens": total, "promptId": prompt}
    return event


def test_latest_request_total_is_not_replaced_by_the_usage_sum(fake):
    path = fake.session("s1", "/w/app", [
        _with_total(1, 1795, "p1"),
        _with_total(2, 35928, "p1"),
    ], model="grok-4.7", usage={
        "session": {"inputTokens": 796101, "cachedReadTokens": 2640000,
                    "outputTokens": 14653, "reasoningTokens": 11938,
                    "totalTokens": 2158385, "modelCalls": 16,
                    "primaryModelId": "grok-4.7-build"},
        "turns": [{"turnNumber": 3, "inputTokens": 796101, "cachedReadTokens": 2640000,
                   "outputTokens": 14653, "reasoningTokens": 11938,
                   "totalTokens": 2158385, "primaryModelId": "grok-4.7-build"}],
    })

    main = grok.load(path, {"sessionId": "s1", "status": "busy"}, now=T0 + 10).main

    assert main.context_tokens == 35928
    assert main.context_window == 2_000_000
    assert main.model == "grok-4.7"


def test_running_subagent_keeps_live_context_and_summary_model(fake):
    path = fake.session("s1", "/w/app", [_with_total(0, 31079, "parent")], model="grok-4.7")
    child = fake.subagent(
        path, "a1", description="Write cart and inventory tests", status="running",
        model="grok-4.7", updates=[
            _with_total(1, 2679, "child", "child-a1"),
            _with_total(2, 29421, "child", "child-a1"),
        ])
    (child / "usage.json").write_text(json.dumps({
        "session": {"primaryModelId": "grok-4.7-build", "totalTokens": 999999,
                    "inputTokens": 900000, "cachedReadTokens": 800000,
                    "outputTokens": 1000, "reasoningTokens": 1000, "modelCalls": 8},
        "turns": [{"totalTokens": 999999, "inputTokens": 900000,
                   "cachedReadTokens": 800000, "outputTokens": 1000,
                   "reasoningTokens": 1000, "primaryModelId": "grok-4.7-build"}],
    }))

    agent = grok.load(path, {"sessionId": "s1", "status": "busy"}, now=T0 + 10).agents[0]

    assert agent.state == RUNNING
    assert agent.context_tokens == 29421
    assert agent.context_window == 2_000_000
    assert agent.model == "grok-4.7"


def test_live_detection_from_recent_updates(fake):
    fake.session("s1", "/w/app", [agent_chunk(0, "hi")], updated_at=0)
    # Make updates look "now"
    path = grok.SESSIONS / quote("/w/app", safe="") / "s1" / "updates.jsonl"
    now = time.time()
    os.utime(path, (now, now))
    fake.live = None  # use real query
    grok._live = grok._LiveCache()

    rows = grok.live_sessions()

    assert any(r["sessionId"] == "s1" for r in rows)


def test_open_sessions_are_listed_even_when_the_transcript_is_quiet(fake):
    old = fake.session("quiet-a", "/w/a", [agent_chunk(0, "hi")], updated_at=-1_000_000)
    fake.session("quiet-b", "/w/b", [agent_chunk(0, "hi")], updated_at=-1_000_000)
    fake.session("gone", "/w/c", [agent_chunk(0, "hi")], updated_at=-1_000_000)
    (fake.home / "active_sessions.json").write_text(json.dumps([
        {"session_id": "quiet-a", "pid": os.getpid(), "cwd": "/w/a"},
        {"session_id": "quiet-b", "pid": os.getpid(), "cwd": "/w/b"},
        {"session_id": "gone", "pid": 2**31 - 1, "cwd": "/w/c"},
    ]))
    # The quiet files must not look recent on their own.
    for path in (old / "updates.jsonl",):
        os.utime(path, (1, 1))
    fake.live = None
    grok._live = grok._LiveCache()

    rows = {r["sessionId"]: r["status"] for r in grok.live_sessions()}

    assert rows == {"quiet-a": "idle", "quiet-b": "idle"}


def test_live_detection_from_running_subagent(fake):
    path = fake.session("s1", "/w/app", [agent_chunk(0, "hi")], updated_at=0)
    # Parent updates are old; child meta says running.
    os.utime(path / "updates.jsonl", (T0, T0))
    fake.subagent(path, "a1", status="running", mtime=time.time())
    fake.live = None
    grok._live = grok._LiveCache()

    rows = grok.live_sessions()

    assert any(r["sessionId"] == "s1" and r["status"] == "busy" for r in rows)
