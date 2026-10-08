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
from urllib.parse import unquote

from agents_tree.model import (DONE, FAILED, INACTIVE, RUNNING, STALE, WAITING, Agent,
                               Detail, Session)

def _get_antigravity_dir() -> Path:
    if os.environ.get("ANTIGRAVITY_HOME"):
        return Path(os.environ["ANTIGRAVITY_HOME"])
    for cand in (
        Path.home() / ".gemini" / "antigravity-cli",
        Path.home() / ".gemini" / "antigravity",
    ):
        if (cand / "brain").is_dir():
            return cand
    return Path.home() / ".gemini" / "antigravity"


ANTIGRAVITY_DIR = _get_antigravity_dir()
BRAIN = ANTIGRAVITY_DIR / "brain"
CONVERSATIONS = ANTIGRAVITY_DIR / "conversations"

STALE_AFTER_SECS = 120
_TEXT_LIMIT = 20_000
_PROMPT_TAG_RE = re.compile(r"<USER_REQUEST>(.*?)</USER_REQUEST>", re.DOTALL)
_SUBAGENT_CONVO_RE = re.compile(r'"conversationId":\s*"([^"]+)"')
_FAILED_STATUSES = {"error", "failed", "cancelled", "canceled", "interrupted", "aborted"}


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
    effort: str | None = None
    tool_counts: dict[str, int] = field(default_factory=dict)
    requests: int = 0
    context: int | None = None
    output_tokens: int = 0
    workspace: str | None = None
    subagents: list[SubagentSpec] = field(default_factory=list)
    pending_subagent_specs: list[dict] = field(default_factory=list)
    has_completed: bool = False
    last_status: str | None = None


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
        m = re.search(r"Model Selection` from \S+ to (.+?)(?=\.\s*(?:[A-Z<]|$))", content)
        if m:
            raw_model = m.group(1).strip().rstrip(".")
            eff_match = re.search(r"^(.*?)\s*\(([^)]+)\)$", raw_model)
            if eff_match:
                t.model = eff_match.group(1).strip()
                t.effort = eff_match.group(2).strip()
            else:
                t.model = raw_model
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
    input_tokens = data.get("input_tokens")
    cache_tokens = data.get("cache_read_tokens")
    if isinstance(input_tokens, int) or isinstance(cache_tokens, int):
        total_in = (input_tokens if isinstance(input_tokens, int) else 0) + (
            cache_tokens if isinstance(cache_tokens, int) else 0
        )
        if total_in > 0:
            t.context = total_in
    out_tokens = data.get("output_tokens")
    if isinstance(out_tokens, int):
        t.output_tokens += out_tokens

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

    status = data.get("status")
    if isinstance(status, str) and status:
        t.last_status = status.lower()

    ts_val = _parse_ts(data.get("created_at"))
    if ts_val is not None:
        if t.first_ts is None:
            t.first_ts = ts_val
        t.last_ts = ts_val

    content = data.get("content")
    step_type = data.get("type")
    if step_type == "USER_INPUT":
        _handle_user_input(t, content if isinstance(content, str) else None)
        if isinstance(content, str):
            _handle_content(t, content)
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
                return unquote(m.group(1).decode("utf-8", errors="replace"))
    except (sqlite3.Error, OSError):
        pass
    return None


def _read_model_from_db(conv_id: str) -> tuple[str | None, str | None]:
    db_path = CONVERSATIONS / f"{conv_id}.db"
    if not db_path.exists():
        return None, None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        c = conn.cursor()
        c.execute("SELECT data FROM gen_metadata LIMIT 1")
        row = c.fetchone()
        conn.close()
        if row and isinstance(row[0], (bytes, bytearray)):
            m = re.search(
                rb"(?:gemini-[0-9a-zA-Z.-]+|claude-[0-9a-zA-Z.-]+|gpt-[0-9a-zA-Z.-]+)",
                row[0],
            )
            if m:
                raw = m.group(0).decode("utf-8", errors="replace")
                for eff in ("low", "medium", "high", "xhigh", "tiered"):
                    if raw.endswith(f"-{eff}"):
                        return raw[:-len(eff) - 1], eff
                return raw, None
    except (sqlite3.Error, OSError):
        pass
    return None, None


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
                effort: str | None = None,
                prompt: str | None = None) -> Agent:
    detail = Detail(
        prompt=prompt or t.first_prompt or t.last_prompt,
        prompt_label="Prompt",
        tools=dict(t.tool_counts),
        last_tool=t.last_tool,
        last_tool_at=t.last_tool_at,
        last_tool_pending=t.last_tool_pending,
        last_text=t.last_text,
        output_tokens=t.output_tokens,
        requests=t.requests,
        transcript=str(transcript),
    )
    eff_model = model or t.model or "Gemini"
    eff_effort = effort or t.effort
    return Agent(
        id=aid,
        label=label,
        state=state,
        status=status,
        model=eff_model,
        effort=eff_effort,
        context_tokens=t.context,
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

        if child_t.last_status in _FAILED_STATUSES:
            state, status = FAILED, child_t.last_status
        elif is_live:
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

        db_model, db_effort = _read_model_from_db(cid)
        agent = _agent_from(
            cid,
            label,
            child_t,
            state,
            status,
            transcript_path,
            model=db_model or spec.model or child_t.model,
            effort=db_effort or child_t.effort,
            prompt=spec.prompt or child_t.first_prompt,
        )
        agent.children = _build_subagents(child_t, convos, now, session_live, seen | {cid})
        agents.append(agent)
    return agents


def _session_is_live(session_id: str, convos: dict[str, ConvoMeta], now: float,
                     memo: dict[str, bool], seen: set[str] | None = None) -> bool:
    """Whether a session or any of its descendants still has recent activity."""
    if session_id in memo:
        return memo[session_id]
    if seen and session_id in seen:
        return False
    meta = convos.get(session_id)
    if not meta:
        return False
    transcript = read_transcript(meta.transcript_path)
    if transcript.last_status in _FAILED_STATUSES:
        memo[session_id] = False
        return False
    if now - meta.mtime < STALE_AFTER_SECS:
        memo[session_id] = True
        return True
    descendants = (seen or set()) | {session_id}
    live = any(_session_is_live(child.conversation_id, convos, now, memo, descendants)
               for child in transcript.subagents)
    memo[session_id] = live
    return live


def _load_session(meta: ConvoMeta, convos: dict[str, ConvoMeta],
                  now: float, is_live: bool) -> Session:
    t = read_transcript(meta.transcript_path)
    cwd = meta.cwd or t.workspace

    main_live = now - meta.mtime < STALE_AFTER_SECS
    if main_live:
        state, status = RUNNING, "running"
    elif is_live:
        state, status = WAITING, "waiting for subagent"
    else:
        state, status = INACTIVE, "not running"

    db_model, db_effort = _read_model_from_db(meta.session_id)
    main_agent = _agent_from(
        "main",
        "main",
        t,
        state,
        status,
        meta.transcript_path,
        model=t.model or db_model,
        effort=t.effort or db_effort,
    )

    agents = _build_subagents(t, convos, now, main_live, {meta.session_id})

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

    # A quiet parent remains live while one of its subagents continues working.
    live_memo: dict[str, bool] = {}
    live_ids = {sid for sid in roots if _session_is_live(sid, convos, now, live_memo)}

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
