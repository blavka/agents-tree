import json
import os
import time

import pytest
from conftest import T0, assistant, notification, title, tool_result

from agents_tree.providers import claude


def test_load_reads_model_effort_context_and_title(fake):
    path = fake.session("s1", "/w/app", [
        title("Fix login"),
        assistant(0, model="claude-opus-5-5", effort="xhigh", context=120_000),
    ])

    s = claude.load(path, now=T0 + 10)

    assert s.title == "Fix login"
    assert (s.main.model, s.main.effort) == ("opus-5-5", "xhigh")
    assert s.main.context_tokens == 120_000
    assert s.main.context_window == 1_000_000


def test_load_strips_date_suffix_and_handles_models_without_effort(fake):
    path = fake.session("s1", "/w/app", [assistant(0, tool_uses=["tu1"])])
    fake.subagent(path, "a1", [assistant(1, model="claude-haiku-4-5-20251001", effort=None)],
                  tool_use_id="tu1")

    agent = claude.load(path, now=T0 + 10).agents[0]

    assert (agent.model, agent.effort, agent.context_window) == ("haiku-4-5", None, 200_000)


def test_title_falls_back_to_session_id_prefix(fake):
    path = fake.session("0123456789abcdef", "/w/app", [assistant(0)])

    assert claude.load(path).title == "01234567"


@pytest.mark.parametrize(("records_in_main", "background", "age", "expected"), [
    ([notification(5, "a1", "completed")], True, 0, "completed"),
    ([notification(5, "a1", "failed")], True, 0, "failed"),
    ([tool_result(5, "tu1")], False, 0, "done"),
    # A background agent's immediate "launched" tool_result does not mean it finished.
    ([tool_result(5, "tu1")], True, 0, "running"),
    ([], True, 0, "running"),
    ([], True, claude.STALE_AFTER_SECS + 60, "stale"),
])
def test_subagent_status(fake, records_in_main, background, age, expected):
    path = fake.session("s1", "/w/app", [assistant(0, tool_uses=["tu1"]), *records_in_main])
    now = time.time()
    fake.subagent(path, "a1", [assistant(1)], tool_use_id="tu1", background=background,
                  mtime=now - age)

    agent = claude.load(path, now=now).agents[0]

    assert agent.status == expected
    assert (agent.ended is None) == (expected == "running")


def test_nested_subagent_hangs_under_the_agent_that_spawned_it(fake):
    path = fake.session("s1", "/w/app", [assistant(0, tool_uses=["tu1"])])
    fake.subagent(path, "parent", [assistant(1, tool_uses=["tu2"])], tool_use_id="tu1",
                  description="Plan")
    fake.subagent(path, "child", [assistant(2)], tool_use_id="tu2", agent_type="Explore",
                  description="Study code")

    s = claude.load(path, now=T0 + 10)

    assert [a.label for a in s.agents] == ["general-purpose: Plan"]
    assert [a.label for a in s.agents[0].children] == ["Explore: Study code"]


def test_plugin_agent_type_drops_its_plugin_prefix(fake):
    path = fake.session("s1", "/w/app", [assistant(0, tool_uses=["tu1"])])
    fake.subagent(path, "a1", [assistant(1)], tool_use_id="tu1",
                  agent_type="systems-programming:golang-pro", description="Review")

    assert claude.load(path, now=T0 + 10).agents[0].label == "golang-pro: Review"


def test_agents_are_ordered_by_start(fake):
    path = fake.session("s1", "/w/app", [assistant(0, tool_uses=["tu1", "tu2"])])
    fake.subagent(path, "late", [assistant(20)], tool_use_id="tu2", description="late")
    fake.subagent(path, "early", [assistant(10)], tool_use_id="tu1", description="early")

    labels = [a.label for a in claude.load(path, now=T0 + 30).agents]

    assert labels == ["general-purpose: early", "general-purpose: late"]


def test_garbage_lines_are_skipped(fake):
    path = fake.session("s1", "/w/app", [assistant(0, context=5000)])
    with path.open("a") as f:
        f.write("not json\n[1, 2]\n{\"type\": \"assistant\", \"message\": 3}\n")

    assert claude.load(path).main.context_tokens == 5000


def test_changed_transcript_is_read_again(fake):
    path = fake.session("s1", "/w/app", [assistant(0, context=1000)])
    assert claude.load(path).main.context_tokens == 1000

    with path.open("a") as f:
        f.write(__import__("json").dumps(assistant(5, context=9000)) + "\n")
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))

    assert claude.load(path).main.context_tokens == 9000


def test_find_without_target_lists_live_sessions(fake):
    fake.session("s1", "/w/a", [assistant(0)])
    fake.session("s2", "/w/b", [assistant(0)])
    fake.live = [{"sessionId": "s1", "cwd": "/w/a", "kind": "interactive", "status": "busy"},
                 {"sessionId": "gone", "cwd": "/w/c", "kind": "background", "status": "blocked"}]

    found = claude.find(None)

    assert [p.stem for p, _ in found] == ["s1"]


def test_find_without_live_sessions_raises(fake):
    with pytest.raises(LookupError, match="no running"):
        claude.find(None)


def test_find_directory_prefers_sessions_running_there(fake, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    fake.session("old", str(work), [assistant(0)])
    fake.session("live", str(work), [assistant(0)])
    fake.live = [{"sessionId": "live", "cwd": str(work), "kind": "interactive", "status": "idle"}]

    assert [p.stem for p, _ in claude.find(str(work))] == ["live"]


def test_find_directory_falls_back_to_newest_session(fake, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    old = fake.session("old", str(work), [assistant(0)])
    new = fake.session("new", str(work), [assistant(0)])
    os.utime(old, (T0, T0))
    os.utime(new, (T0 + 100, T0 + 100))

    [(path, live)] = claude.find(str(work))

    assert path.stem == "new"
    assert live is not None and live["status"] == "not running"


def test_find_directory_without_sessions_raises(fake, tmp_path):
    with pytest.raises(LookupError, match="no Claude Code sessions in"):
        claude.find(str(tmp_path))


def test_find_session_by_id_prefix(fake):
    fake.session("abcdef12-3456", "/w/a", [assistant(0)])

    [(path, live)] = claude.find("abcdef")

    assert path.stem == "abcdef12-3456"
    assert live == {"status": "not running"}


def test_find_unknown_session_raises(fake):
    with pytest.raises(LookupError, match="no Claude Code session matching"):
        claude.find("nope")


def test_load_takes_cwd_kind_and_status_from_the_live_entry(fake):
    path = fake.session("s1", "/w/a", [assistant(0)])

    s = claude.load(path, {"sessionId": "s1", "cwd": "/w/a", "kind": "background",
                           "status": "working"})

    assert (s.cwd, s.kind, s.main.status) == ("/w/a", "background", "working")


@pytest.mark.parametrize(("model", "window"), [
    ("claude-opus-5-5", 1_000_000), ("claude-sonnet-5-5", 1_000_000),
    ("claude-fable-5-1", 1_000_000), ("claude-haiku-4-5-20251001", 200_000),
    ("claude-sonnet-4-5", 200_000), (None, 200_000),
])
def test_context_window_by_model(model, window):
    assert claude.context_window(model) == window


# --- regressions from review -------------------------------------------------

def _two_agent_session(fake, main_extra=(), agent_records=None, *, meta=None, mtime=None):
    path = fake.session("s1", "/w/app", [assistant(0, tool_uses=["tu1"]), *main_extra])
    fake.subagent(path, "a1", agent_records or [assistant(1)], tool_use_id="tu1",
                  background=True, mtime=mtime)
    if meta:
        meta_path = path.with_suffix("") / "subagents" / "agent-a1.meta.json"
        data = json.loads(meta_path.read_text())
        data.update(meta)
        meta_path.write_text(json.dumps(data))
    return path


def test_agent_resumed_after_its_notification_is_running(fake):
    now = T0 + 200
    path = _two_agent_session(fake, [notification(5, "a1")],
                              [assistant(1), assistant(190)], mtime=now - 10)

    assert claude.load(path, now=now).agents[0].status == "running"


def test_newest_notification_wins_whatever_file_it_is_in(fake):
    path = fake.session("s1", "/w/app", [assistant(0, tool_uses=["tu1", "tu2"]),
                                         notification(50, "child", "completed")])
    fake.subagent(path, "parent", [assistant(1, tool_uses=["tu3"]),
                                   notification(20, "child", "failed")], tool_use_id="tu1")
    fake.subagent(path, "child", [assistant(2)], tool_use_id="tu3", mtime=T0)

    s = claude.load(path, now=T0 + 1000)

    assert s.agents[0].children[0].status == "completed"


@pytest.mark.parametrize(("live", "expected"), [
    ({"sessionId": "s1", "status": "working"}, "idle"),
    (None, "stale"),
])
def test_teammate_without_notification_is_idle_while_the_session_lives(fake, live, expected):
    path = _two_agent_session(fake, meta={"taskKind": "in_process_teammate", "teamName": "t",
                                          "toolUseId": None}, mtime=T0)

    assert claude.load(path, live, now=T0 + 1000).agents[0].status == expected


def test_agent_waiting_on_a_long_tool_call_stays_running(fake):
    path = _two_agent_session(fake, agent_records=[assistant(1, tool_uses=["bash1"])], mtime=T0)

    s = claude.load(path, {"sessionId": "s1", "status": "busy"}, now=T0 + 1000)

    assert s.agents[0].status == "running"


def test_async_launch_answer_does_not_finish_an_old_style_agent(fake):
    launched = {"type": "user", "timestamp": "2026-10-07T06:00:02Z", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "tu1",
         "content": [{"type": "text", "text": "Async agent launched successfully."}]}]}}
    path = _two_agent_session(fake, [launched], meta={"requestShape": None}, mtime=T0)

    assert claude.load(path, now=T0 + 1000).agents[0].status == "stale"


def test_glob_characters_in_a_session_target_do_not_crash(fake):
    fake.session("abc", "/w/a", [assistant(0)])

    assert claude.transcript_path("*") is None
    assert claude.transcript_path("[a") is None


def test_ambiguous_prefix_names_the_candidates(fake):
    fake.session("abc1", "/w/a", [assistant(0)])
    fake.session("abc2", "/w/b", [assistant(0)])

    with pytest.raises(LookupError, match="matches 2 sessions: abc1, abc2"):
        claude.find("abc")


def test_parent_agent_id_wins_and_cycles_do_not_lose_agents(fake):
    path = fake.session("s1", "/w/app", [assistant(0)])
    fake.subagent(path, "x", [assistant(1)], description="x")
    fake.subagent(path, "y", [assistant(2)], description="y")
    for aid, parent in (("x", "y"), ("y", "x")):
        meta_path = path.with_suffix("") / "subagents" / f"agent-{aid}.meta.json"
        data = json.loads(meta_path.read_text())
        data["parentAgentId"] = parent
        meta_path.write_text(json.dumps(data))

    s = claude.load(path, now=T0 + 10)

    assert sum(a.size() for a in s.agents) == 2


def test_custom_title_beats_the_generated_one_and_live_name_fills_in(fake):
    named = fake.session("s1", "/w/a", [title("Generated"), {"type": "custom-title",
                                                             "customTitle": "Mine"}])
    plain = fake.session("s2", "/w/a", [assistant(0)])

    assert claude.load(named).title == "Mine"
    assert claude.load(plain, {"sessionId": "s2", "name": "wimber-fb"}).title == "wimber-fb"


def test_context_beyond_the_inferred_window_means_a_long_window(fake):
    path = fake.session("s1", "/w/a", [assistant(0, model="claude-sonnet-4-5", context=400_000)])

    assert claude.load(path).main.context_window == 1_000_000


def test_odd_field_types_are_ignored(fake):
    odd = {"type": "assistant", "timestamp": 5, "message": {
        "model": 42, "usage": [1, 2], "content": [{"type": "tool_use", "id": 7}]}}
    path = fake.session("s1", "/w/a", [assistant(0, context=3000), odd])

    s = claude.load(path)

    assert (s.main.model, s.main.context_tokens) == ("opus-5-5", 3000)


def test_detail_collects_prompt_tools_last_message_and_deduplicated_usage(fake):
    path = fake.session("s1", "/w/app", [assistant(0, tool_uses=["tu1"])])
    first = assistant(2, tool_uses=["b1"])
    first["requestId"] = "req1"
    second = assistant(2, tool_uses=["b2"])  # same response, second content block
    second["requestId"] = "req1"
    second["message"]["content"][0]["name"] = "Read"
    reply = assistant(4)
    reply["requestId"] = "req2"
    reply["message"]["content"] = [{"type": "text", "text": "All done."}]
    reply["message"]["usage"]["output_tokens"] = 40
    fake.subagent(path, "a1", [
        {"type": "user", "timestamp": "2026-10-07T06:00:01Z",
         "message": {"role": "user", "content": "Look for the bug."}},
        {"type": "user", "isMeta": True, "message": {"content": "<system-reminder>x"}},
        first, second, tool_result(3, "b1"), tool_result(3, "b2"), reply,
    ], tool_use_id="tu1")

    d = claude.load(path, now=T0 + 10).agents[0].detail

    assert d is not None
    assert d.prompt == "Look for the bug."
    assert d.tools == {"Agent": 1, "Read": 1}
    assert (d.last_tool, d.last_tool_pending) == ("Read", False)
    assert d.last_text == "All done."
    assert (d.requests, d.output_tokens) == (2, 45)
    assert d.transcript and d.transcript.endswith("agent-a1.jsonl")


def test_main_detail_shows_the_last_user_prompt(fake):
    path = fake.session("s1", "/w/app", [
        {"type": "user", "message": {"content": "first ask"}},
        assistant(1),
        {"type": "user", "message": {"content": [{"type": "text", "text": "second ask"}]}},
    ])

    d = claude.load(path).main.detail

    assert d is not None and (d.prompt_label, d.prompt) == ("Last prompt", "second ask")
