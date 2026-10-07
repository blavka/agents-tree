"""Codex: JSONL transcripts under ``~/.codex/sessions`` and their subagents.

Codex writes one transcript per thread, arranged by date::

  ~/.codex/sessions/YYYY/MM/DD/rollout-<time>-<thread-id>.jsonl

Each transcript starts with ``session_meta``.  Its ``session_id`` identifies the
root conversation while ``id`` identifies this thread.  The root transcript
records ``SubAgentActivity`` items with the child thread id and agent path;
the same items in a child transcript make nesting explicit.  Prompt text sent
to a spawned agent is encrypted in current Codex versions, so this provider
intentionally does not try to decode it.

These files are an internal Codex format.  This provider reads them only; it
does not connect to Codex's app-server or a running session.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from agents_tree.model import (DONE, FAILED, INACTIVE, RUNNING, STALE, WAITING, Agent,
                               Detail, Session)

CODEX_HOME = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
SESSIONS = CODEX_HOME / "sessions"

STALE_AFTER_SECS = 120
_TEXT_LIMIT = 20_000


def _parse_ts(value) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _int(value) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _content_text(content) -> str | None:
    if not isinstance(content, list):
        return None
    parts = [block["text"].strip() for block in content
             if isinstance(block, dict) and isinstance(block.get("text"), str)
             and block.get("type") in ("input_text", "output_text", "text")
             and block["text"].strip()]
    return "\n".join(parts)[:_TEXT_LIMIT] if parts else None


@dataclass(slots=True)
class Meta:
    thread_id: str
    session_id: str
    cwd: str | None
    originator: str | None


@dataclass(slots=True)
class Activity:
    thread_id: str
    path: str
    started: float | None = None
    ended: float | None = None
    outcome: str | None = None


@dataclass(slots=True)
class Transcript:
    model: str | None = None
    effort: str | None = None
    context: int | None = None
    context_window: int | None = None
    first: float | None = None
    last: float | None = None
    first_prompt: str | None = None
    last_prompt: str | None = None
    last_text: str | None = None
    tool_uses: set[str] = field(default_factory=set)
    tool_results: set[str] = field(default_factory=set)
    tool_counts: dict[str, int] = field(default_factory=dict)
    last_tool: str | None = None
    last_tool_at: float | None = None
    last_tool_id: str | None = None
    outputs: dict[str, int] = field(default_factory=dict)
    activities: dict[str, Activity] = field(default_factory=dict)

    @property
    def pending_tool(self) -> bool:
        return bool(self.tool_uses - self.tool_results)


@dataclass(slots=True)
class _Cached:
    inode: int
    offset: int
    transcript: Transcript


@dataclass(slots=True)
class _MetaCached:
    inode: int
    size: int
    meta: Meta | None


_cache: dict[Path, _Cached] = {}
_meta_cache: dict[Path, _MetaCached] = {}
_used: set[Path] = set()


def _set_time(t: Transcript, ts: float | None) -> None:
    if ts is not None:
        t.first = t.first or ts
        t.last = ts


def _activity(t: Transcript, item: dict, ts: float | None) -> None:
    if item.get("type") != "SubAgentActivity":
        return
    aid, path = item.get("agent_thread_id"), item.get("agent_path")
    if not isinstance(aid, str) or not aid or not isinstance(path, str) or not path:
        return
    activity = t.activities.setdefault(aid, Activity(aid, path))
    kind = item.get("kind")
    if kind == "started":
        # A resumed child can emit another start after an earlier completion.
        # Its newest lifecycle is the one the tree needs to show.
        activity.started = ts or activity.started
        activity.ended, activity.outcome = None, None
    elif kind in ("completed", "failed", "interrupted", "cancelled"):
        activity.ended = ts or activity.ended
        activity.outcome = kind


def _read_usage(t: Transcript, payload: dict) -> None:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return
    total = _int(usage.get("total_tokens"))
    if total:
        t.context = total
    response = payload.get("response_id")
    if isinstance(response, str) and response:
        t.outputs[response] = max(
            t.outputs.get(response, 0),
            _int(usage.get("output_tokens")),
        )


def _read_event(t: Transcript, payload: dict, ts: float | None) -> None:
    message_kind = payload.get("type")
    if message_kind == "task_started":
        t.context_window = _int(payload.get("model_context_window")) or t.context_window
    elif message_kind == "token_count":
        info = payload.get("info")
        if isinstance(info, dict):
            t.context_window = _int(info.get("model_context_window")) or t.context_window
            usage = info.get("last_token_usage")
            if isinstance(usage, dict):
                total = _int(usage.get("total_tokens"))
                if total:
                    t.context = total
    elif message_kind in ("item_started", "item_completed"):
        item = payload.get("item")
        if isinstance(item, dict):
            _activity(t, item, ts)


def _read_response(t: Transcript, payload: dict, ts: float | None) -> None:
    item_type = payload.get("type")
    if item_type == "message":
        text = _content_text(payload.get("content"))
        if payload.get("role") == "user" and text and not text.startswith("<"):
            t.first_prompt = t.first_prompt or text
            t.last_prompt = text
        return
    if item_type == "agent_message":
        text = _content_text(payload.get("content"))
        if text:
            t.last_text = text
        return
    if item_type in ("function_call", "custom_tool_call"):
        name = payload.get("name")
        call_id = payload.get("call_id")
        if isinstance(name, str) and name:
            t.tool_counts[name] = t.tool_counts.get(name, 0) + 1
            t.last_tool, t.last_tool_at = name, ts
        if isinstance(call_id, str) and call_id:
            t.tool_uses.add(call_id)
            t.last_tool_id = call_id
        return
    if item_type in ("function_call_output", "custom_tool_call_output"):
        call_id = payload.get("call_id")
        if isinstance(call_id, str) and call_id:
            t.tool_results.add(call_id)


def _read_line(t: Transcript, line: str) -> None:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return
    if not isinstance(event, dict):
        return
    ts = _parse_ts(event.get("timestamp"))
    _set_time(t, ts)
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return
    kind = event.get("type")
    if kind == "turn_context":
        model, effort = payload.get("model"), payload.get("effort")
        if isinstance(model, str) and model:
            t.model = model
        if isinstance(effort, str) and effort:
            t.effort = effort
    elif kind == "token_usage_record":
        _read_usage(t, payload)
    elif kind == "event_msg":
        _read_event(t, payload, ts)
    elif kind == "response_item":
        _read_response(t, payload, ts)


def read_transcript(path: Path) -> Transcript:
    """Parse appended complete JSONL records on top of the cached transcript."""
    _used.add(path)
    try:
        st = path.stat()
    except OSError:
        return Transcript()
    hit = _cache.get(path)
    if not hit or hit.inode != st.st_ino or st.st_size < hit.offset:
        hit = _cache[path] = _Cached(st.st_ino, 0, Transcript())
    if st.st_size > hit.offset:
        try:
            with open(path, "rb") as file:
                file.seek(hit.offset)
                data = file.read(st.st_size - hit.offset)
        except OSError:
            return hit.transcript
        end = data.rfind(b"\n") + 1
        for line in data[:end].splitlines():
            _read_line(hit.transcript, line.decode("utf-8", errors="replace"))
        hit.offset += end
    return hit.transcript


def forget_unused() -> None:
    for path in set(_cache) - _used:
        del _cache[path]


def read_meta(path: Path) -> Meta | None:
    """Read only the leading session_meta line, cached until the file changes."""
    try:
        st = path.stat()
    except OSError:
        return None
    hit = _meta_cache.get(path)
    if hit and hit.inode == st.st_ino and hit.size == st.st_size:
        return hit.meta
    meta = None
    try:
        with open(path, encoding="utf-8", errors="replace") as file:
            for line in file:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                payload = event.get("payload")
                if event.get("type") != "session_meta" or not isinstance(payload, dict):
                    continue
                thread, root = payload.get("id"), payload.get("session_id")
                if isinstance(thread, str) and thread and isinstance(root, str) and root:
                    cwd = payload.get("cwd")
                    originator = payload.get("originator")
                    meta = Meta(thread, root, cwd if isinstance(cwd, str) else None,
                                originator if isinstance(originator, str) else None)
                    break
    except OSError:
        pass
    _meta_cache[path] = _MetaCached(st.st_ino, st.st_size, meta)
    return meta


Record = tuple[Path, Meta]


def _records() -> dict[str, Record]:
    try:
        paths = SESSIONS.rglob("*.jsonl")
    except OSError:
        return {}
    records = {}
    for path in paths:
        meta = read_meta(path)
        if meta:
            records[meta.thread_id] = (path, meta)
    return records


def _same_dir(a: str | None, directory: Path) -> bool:
    try:
        return bool(a) and Path(a).resolve() == directory
    except (OSError, ValueError):
        return False


def _roots(records: dict[str, Record]) -> list[Record]:
    return [(path, meta) for path, meta in records.values()
            if meta.thread_id == meta.session_id]


def _live_roots(records: dict[str, Record], now: float) -> set[str]:
    live = set()
    for path, meta in records.values():
        if now - _mtime(path) < STALE_AFTER_SECS:
            live.add(meta.session_id)
    return live


def _match_id(target: str, roots: list[Record]) -> Record | None:
    matches = [(path, meta) for path, meta in roots
               if meta.thread_id == target or meta.thread_id.startswith(target)]
    exact = [(path, meta) for path, meta in matches if meta.thread_id == target]
    if exact:
        matches = exact
    if len(matches) > 1:
        shown = ", ".join(meta.thread_id[:13] for _, meta in matches[:5])
        raise LookupError(f"'{target}' matches {len(matches)} sessions: {shown}")
    return matches[0] if matches else None


Found = list[tuple[Path, Meta, bool]]


def find(target: str | None, records: dict[str, Record] | None = None,
         now: float | None = None) -> Found:
    """Find root Codex transcripts for the standard target forms."""
    records = _records() if records is None else records
    now = time.time() if now is None else now
    roots = _roots(records)
    live = _live_roots(records, now)
    if target is None:
        found = [(path, meta, True) for path, meta in roots if meta.session_id in live]
        if not found:
            raise LookupError("no running Codex sessions")
        return found
    if Path(target).is_dir():
        directory = Path(target).resolve()
        pool = [(path, meta) for path, meta in roots if _same_dir(meta.cwd, directory)]
        if not pool:
            raise LookupError(f"no Codex sessions in {directory}")
        active = [(path, meta, True) for path, meta in pool if meta.session_id in live]
        if active:
            return active
        path, meta = max(pool, key=lambda row: _mtime(row[0]))
        return [(path, meta, False)]
    found = _match_id(target, roots)
    if not found:
        raise LookupError(f"no Codex session matching '{target}' (and no such directory)")
    path, meta = found
    return [(path, meta, meta.session_id in live)]


def _label(path: str) -> str:
    return f"agent: {path.rsplit('/', 1)[-1]}"


def _window(t: Transcript) -> int:
    return t.context_window or 200_000


def _agent_from(aid: str, label: str, t: Transcript, state: str, status: str,
                transcript: Path, *, started: float | None = None,
                ended: float | None = None) -> Agent:
    detail = Detail(
        prompt=t.last_prompt if aid == "main" else None,
        prompt_label="Last prompt" if aid == "main" else "Prompt",
        tools=dict(t.tool_counts), last_tool=t.last_tool, last_tool_at=t.last_tool_at,
        last_tool_pending=bool(t.last_tool_id and t.last_tool_id not in t.tool_results),
        last_text=t.last_text, output_tokens=sum(t.outputs.values()), requests=len(t.outputs),
        transcript=str(transcript),
    )
    return Agent(
        id=aid, label=label, state=state, status=status, model=t.model, effort=t.effort,
        context_tokens=t.context, context_window=_window(t),
        started=started if started is not None else t.first,
        ended=ended, detail=detail,
    )


def _child_state(activity: Activity, t: Transcript, path: Path, now: float,
                 session_live: bool) -> tuple[str, str]:
    if activity.outcome == "completed":
        return DONE, "completed"
    if activity.outcome in ("failed", "interrupted", "cancelled"):
        return FAILED, activity.outcome
    if now - _mtime(path) < STALE_AFTER_SECS or (session_live and t.pending_tool):
        return RUNNING, "running"
    if session_live:
        return WAITING, "waiting"
    return STALE, "stale"


def _children(parent: Transcript, records: dict[str, Record], now: float,
              session_live: bool, seen: set[str]) -> list[Agent]:
    agents = []
    for activity in sorted(parent.activities.values(), key=lambda a: a.started or 0):
        if activity.thread_id in seen:
            continue
        record = records.get(activity.thread_id)
        if not record:
            continue
        path, _ = record
        child_t = read_transcript(path)
        state, status = _child_state(activity, child_t, path, now, session_live)
        agent = _agent_from(activity.thread_id, _label(activity.path), child_t, state, status,
                            path, started=activity.started,
                            ended=activity.ended or (child_t.last if state != RUNNING else None))
        agent.children = _children(child_t, records, now, session_live, seen | {activity.thread_id})
        agents.append(agent)
    return agents


def _title(t: Transcript, meta: Meta) -> str:
    if t.first_prompt:
        return t.first_prompt.splitlines()[0][:80]
    return meta.thread_id[:8]


def load(path: Path, meta: Meta | None = None, *, records: dict[str, Record] | None = None,
         live: bool = False, now: float | None = None) -> Session:
    """Build a neutral Session from one root transcript and its child threads."""
    now = time.time() if now is None else now
    meta = meta or read_meta(path)
    if meta is None:
        raise LookupError(f"not a Codex transcript: {path}")
    records = _records() if records is None else records
    main_t = read_transcript(path)
    roots = _children(main_t, records, now, live, {meta.thread_id})
    state, status = (RUNNING, "running") if live else (INACTIVE, "not running")
    main = _agent_from("main", "main", main_t, state, status, path)
    main.ended = None if live else main_t.last
    kind = "background" if meta.originator == "codex_exec" else "interactive"
    return Session(id=meta.thread_id, title=_title(main_t, meta), main=main, agents=roots,
                   cwd=meta.cwd, kind=kind)


def sessions(target: str | None) -> list[Session]:
    now = time.time()
    _used.clear()
    records = _records()
    loaded = [load(path, meta, records=records, live=live, now=now)
              for path, meta, live in find(target, records, now)]
    forget_unused()
    return loaded
