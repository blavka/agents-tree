"""Antigravity: JSONL transcripts under ~/.gemini/antigravity/brain and their subagents.

Layout read here:
  ~/.gemini/antigravity/brain/<session-id>/.system_generated/logs/transcript.jsonl
  ~/.gemini/antigravity/conversations/<session-id>.db (metadata and workspace)

Subagents are spawned via invoke_subagent tool calls in the parent transcript,
and each subagent keeps its own transcript under brain/<subagent-id>.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from agents_tree.model import (DONE, INACTIVE, RUNNING, STALE, Agent,
                               Detail, Session)

ANTIGRAVITY_DIR = Path(
    os.environ.get("ANTIGRAVITY_HOME") or Path.home() / ".gemini" / "antigravity"
)
BRAIN = ANTIGRAVITY_DIR / "brain"
CONVERSATIONS = ANTIGRAVITY_DIR / "conversations"

STALE_AFTER_SECS = 120
_TEXT_LIMIT = 20_000
_PROMPT_TAG_RE = re.compile(r"<USER_REQUEST>(.*?)</USER_REQUEST>", re.DOTALL)
_SUBAGENT_CONVO_RE = re.compile(r'"conversationId":\s*"([^"]+)"')


def _parse_ts(value) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def context_window(model: str | None) -> int:
    """Default context window for Gemini models."""
    m = (model or "").lower()
    if "pro" in m:
        return 2_000_000
    return 1_000_000


@dataclass
class SubagentSpec:
    conversation_id: str
    role: str | None = None
    type_name: str | None = None
    model: str | None = None
    prompt: str | None = None


@dataclass
class Transcript:
    first_ts: float | None = None
    last_ts: float | None = None
    first_prompt: str | None = None
    last_prompt: str | None = None
    last_text: str | None = None
    last_tool: str | None = None
    last_tool_at: float | None = None
    last_tool_pending: bool = False
    model: str | None = None
    tool_counts: dict[str, int] = field(default_factory=dict)
    requests: int = 0
    workspace: str | None = None
    subagents: list[SubagentSpec] = field(default_factory=list)
    pending_subagent_specs: list[dict] = field(default_factory=list)
    has_completed: bool = False


@dataclass
class _Cached:
    inode: int
    offset: int
    transcript: Transcript


_cache: dict[Path, _Cached] = {}
_used: set[Path] = set()


def _extract_prompt(text: str | None) -> str | None:
    if not text:
        return None
    m = _PROMPT_TAG_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _parse_subagent_specs(raw_arg: str | list | None) -> list[dict]:
    """Parse subagent specs from invoke_subagent, tolerating truncated strings."""
    if isinstance(raw_arg, list):
        return [s for s in raw_arg if isinstance(s, dict)]
    if not isinstance(raw_arg, str):
        return []
    try:
        data = json.loads(raw_arg)
        if isinstance(data, list):
            return [s for s in data if isinstance(s, dict)]
    except json.JSONDecodeError:
        pass

    roles = re.findall(r'"Role":\s*"([^"]+)"', raw_arg)
    types = re.findall(r'"TypeName":\s*"([^"]+)"', raw_arg)
    models = re.findall(r'"Model":\s*"([^"]+)"', raw_arg)
    count = max(len(roles), len(types), len(models), 1)
    specs = []
    for i in range(count):
        specs.append({
            "Role": roles[i] if i < len(roles) else None,
            "TypeName": types[i] if i < len(types) else None,
            "Model": models[i] if i < len(models) else None,
        })
    return specs


def _handle_content(t: Transcript, content: str) -> None:
    if "Model Selection" in content:
        m = re.search(r"Model Selection` from \S+ to ([^\n]+)\.", content)
        if m:
            t.model = m.group(1).strip()
    if not t.workspace and "/Workspace" in content:
        m = re.search(r"(/[\w./-]+Workspace[\w./-]*)", content)
        if m:
            t.workspace = m.group(1)


def _handle_user_input(t: Transcript, content: str | None) -> None:
    prompt = _extract_prompt(content)
    if prompt:
        if t.first_prompt is None:
            t.first_prompt = prompt
        t.last_prompt = prompt
    t.last_tool_pending = False


def _handle_planner_response(t: Transcript, data: dict, ts_val: float | None) -> None:
    t.requests += 1
    tool_calls = data.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            name = tc.get("name")
            if isinstance(name, str):
                t.tool_counts[name] = t.tool_counts.get(name, 0) + 1
                t.last_tool = name
                t.last_tool_at = ts_val
                t.last_tool_pending = True

                if name == "invoke_subagent":
                    args = tc.get("args") or {}
                    t.pending_subagent_specs.extend(_parse_subagent_specs(args.get("Subagents")))
                elif name == "send_message":
                    t.has_completed = True
    else:
        t.last_tool_pending = False

    thinking = data.get("thinking")
    content = data.get("content")
    if isinstance(thinking, str) and thinking:
        t.last_text = thinking[-_TEXT_LIMIT:]
    elif isinstance(content, str) and content:
        t.last_text = content[-_TEXT_LIMIT:]


def _handle_generic(t: Transcript, content: str | None) -> None:
    t.last_tool_pending = False
    if not isinstance(content, str):
        return
    t.last_text = content[-_TEXT_LIMIT:]
    existing_ids = {s.conversation_id for s in t.subagents}
    subagent_detected = (
        "Created the following subagents:" in content or
        ("You have" in content and "active subagent" in content)
    )
    if subagent_detected:
        found_ids = _SUBAGENT_CONVO_RE.findall(content)
        for cid in found_ids:
            if cid in existing_ids:
                continue
            existing_ids.add(cid)
            spec = t.pending_subagent_specs.pop(0) if t.pending_subagent_specs else {}
            t.subagents.append(
                SubagentSpec(
                    conversation_id=cid,
                    role=spec.get("Role"),
                    type_name=spec.get("TypeName"),
                    model=spec.get("Model"),
                    prompt=spec.get("Prompt"),
                )
            )


def _read_line(t: Transcript, raw: str) -> None:
    if not raw.strip():
        return
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return
    if not isinstance(data, dict):
        return

    ts_val = _parse_ts(data.get("created_at"))
    if ts_val is not None:
        if t.first_ts is None:
            t.first_ts = ts_val
        t.last_ts = ts_val

    content = data.get("content")
    if isinstance(content, str):
        _handle_content(t, content)

    step_type = data.get("type")
    if step_type == "USER_INPUT":
        _handle_user_input(t, content if isinstance(content, str) else None)
    elif step_type == "PLANNER_RESPONSE":
        _handle_planner_response(t, data, ts_val)
    elif step_type == "GENERIC":
        _handle_generic(t, content if isinstance(content, str) else None)


def read_transcript(path: Path) -> Transcript:
    """Read transcript incrementally, caching previously parsed lines."""
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


def _read_workspace_from_db(conv_id: str) -> str | None:
    db_path = CONVERSATIONS / f"{conv_id}.db"
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        c = conn.cursor()
        c.execute("SELECT data FROM trajectory_metadata_blob WHERE id = 'main' LIMIT 1")
        row = c.fetchone()
        conn.close()
        if row and isinstance(row[0], (bytes, bytearray)):
            m = re.search(rb"file://([^\s\x00-\x1f\x7f-\xff\"'<>]+)", row[0])
            if m:
                return m.group(1).decode("utf-8", errors="replace")
    except (sqlite3.Error, OSError):
        pass
    return None


@dataclass
class ConvoMeta:
    session_id: str
    transcript_path: Path
    cwd: str | None = None
    mtime: float = 0.0


def _discover_sessions() -> dict[str, ConvoMeta]:
    """Scan brain directory for all transcripts."""
    result = {}
    if not BRAIN.is_dir():
        return result
    try:
        entries = os.scandir(BRAIN)
    except OSError:
        return result

    for entry in entries:
        if not entry.is_dir():
            continue
        session_id = entry.name
        transcript_path = Path(entry.path) / ".system_generated" / "logs" / "transcript.jsonl"
        if transcript_path.is_file():
            cwd = _read_workspace_from_db(session_id)
            mt = _mtime(transcript_path)
            result[session_id] = ConvoMeta(session_id, transcript_path, cwd, mt)
    return result


def _agent_from(aid: str, label: str, t: Transcript, state: str, status: str,
                transcript: Path, *, model: str | None = None,
                prompt: str | None = None) -> Agent:
    detail = Detail(
        prompt=prompt or t.first_prompt or t.last_prompt,
        prompt_label="Prompt",
        tools=dict(t.tool_counts),
        last_tool=t.last_tool,
        last_tool_at=t.last_tool_at,
        last_tool_pending=t.last_tool_pending,
        last_text=t.last_text,
        requests=t.requests,
        transcript=str(transcript),
    )
    eff_model = model or t.model or "Gemini"
    return Agent(
        id=aid,
        label=label,
        state=state,
        status=status,
        model=eff_model,
        context_window=context_window(eff_model),
        started=t.first_ts,
        ended=t.last_ts if state != RUNNING else None,
        detail=detail,
    )


def _build_subagents(parent_t: Transcript, convos: dict[str, ConvoMeta],
                     now: float, session_live: bool, seen: set[str]) -> list[Agent]:
    agents = []
    for spec in parent_t.subagents:
        cid = spec.conversation_id
        if cid in seen:
            continue
        meta = convos.get(cid)
        default_path = BRAIN / cid / ".system_generated" / "logs" / "transcript.jsonl"
        transcript_path = meta.transcript_path if meta else default_path
        child_t = read_transcript(transcript_path)

        mt = meta.mtime if meta else _mtime(transcript_path)
        is_live = (now - mt < STALE_AFTER_SECS) or (session_live and child_t.last_tool_pending)

        if is_live:
            state, status = RUNNING, "running"
        elif child_t.has_completed:
            state, status = DONE, "completed"
        elif mt > 0:
            state, status = DONE, "done"
        else:
            state, status = STALE, "stale"

        role = spec.role
        if not role and child_t.first_prompt:
            role = child_t.first_prompt.splitlines()[0][:40]
        label = f"{spec.type_name or 'subagent'}: {role or cid[:8]}"

        agent = _agent_from(
            cid,
            label,
            child_t,
            state,
            status,
            transcript_path,
            model=spec.model or child_t.model,
            prompt=spec.prompt or child_t.first_prompt,
        )
        agent.children = _build_subagents(child_t, convos, now, session_live, seen | {cid})
        agents.append(agent)
    return agents


def _load_session(meta: ConvoMeta, convos: dict[str, ConvoMeta],
                  now: float, is_live: bool) -> Session:
    t = read_transcript(meta.transcript_path)
    cwd = meta.cwd or t.workspace

    state = RUNNING if is_live else INACTIVE
    status = "running" if is_live else "not running"

    main_agent = _agent_from(
        "main",
        "main",
        t,
        state,
        status,
        meta.transcript_path,
    )

    agents = _build_subagents(t, convos, now, is_live, {meta.session_id})

    title = t.first_prompt or meta.session_id[:8]
    title = title.splitlines()[0][:80]

    return Session(
        id=meta.session_id,
        title=title,
        main=main_agent,
        agents=agents,
        cwd=cwd,
        kind="interactive",
    )


def sessions(target: str | None) -> list[Session]:
    """Find and build Antigravity sessions matching target."""
    _used.clear()
    now = time.time()
    convos = _discover_sessions()

    # Determine which sessions are subagents to avoid treating them as root sessions
    subagent_ids = set()
    for meta in convos.values():
        t = read_transcript(meta.transcript_path)
        for s in t.subagents:
            subagent_ids.add(s.conversation_id)

    roots = {sid: meta for sid, meta in convos.items() if sid not in subagent_ids}

    # Live sessions: transcript modified in last 120s
    live_ids = {sid for sid, meta in roots.items() if (now - meta.mtime) < STALE_AFTER_SECS}

    matched: list[tuple[ConvoMeta, bool]] = []

    if target is None:
        matched = [(roots[sid], True) for sid in live_ids]
        if not matched:
            raise LookupError("no running Antigravity sessions")
    elif Path(target).is_dir():
        target_dir = Path(target).resolve()
        pool = []
        for meta in roots.values():
            if meta.cwd and Path(meta.cwd).resolve() == target_dir:
                pool.append(meta)
        if not pool:
            raise LookupError(f"no Antigravity sessions in {target_dir}")
        active = [(m, True) for m in pool if m.session_id in live_ids]
        if active:
            matched = active
        else:
            newest = max(pool, key=lambda m: m.mtime)
            matched = [(newest, False)]
    else:
        # Match by ID or unique prefix
        exact = [m for m in roots.values() if m.session_id == target]
        if exact:
            matched = [(exact[0], exact[0].session_id in live_ids)]
        else:
            prefixes = [m for m in roots.values() if m.session_id.startswith(target)]
            if len(prefixes) > 1:
                prefixes.sort(key=lambda m: m.session_id)
                shown = ", ".join(m.session_id[:13] for m in prefixes[:5])
                raise LookupError(f"'{target}' matches {len(prefixes)} sessions: {shown}")
            if len(prefixes) == 1:
                matched = [(prefixes[0], prefixes[0].session_id in live_ids)]
            else:
                raise LookupError(
                    f"no Antigravity session matching '{target}' (and no such directory)"
                )

    result = [_load_session(meta, convos, now, is_live) for meta, is_live in matched]
    forget_unused()
    return result
