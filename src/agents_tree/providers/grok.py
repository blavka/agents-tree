"""Grok Build: sessions under ~/.grok/sessions and their subagents.

Layout read here (internal to Grok Build; every field is optional):

  ~/.grok/sessions/<url-encoded-cwd>/<session-id>/
      summary.json          title, model, effort, context_window, kind, parent
      updates.jsonl         ACP session/update stream (conversation + tools)
      usage.json            optional turn/session token totals
      subagents/<id>/meta.json
          {subagent_id, child_session_id, parent_session_id, subagent_type,
           description, prompt, status, started_at, completed_at, ...}

  Child sessions live in the normal sessions tree (session_kind "subagent");
  the parent's subagents/<id>/meta.json links to them.

Running sessions: recent updates.jsonl activity, or a subagent still marked
running. Override the home with GROK_HOME (same as Grok Build).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, unquote

from agents_tree.model import (DONE, FAILED, INACTIVE, RUNNING, STALE, WAITING, Agent,
                               Detail, Session)

GROK_HOME = Path(os.environ.get("GROK_HOME") or Path.home() / ".grok")
SESSIONS = GROK_HOME / "sessions"

STALE_AFTER_SECS = 120
LIVE_TTL_SECS = 5.0
_TEXT_LIMIT = 20_000

_OUTCOME_STATES = {"completed": DONE, "failed": FAILED, "cancelled": FAILED, "killed": FAILED}
_LIVE_STATES = {"busy": RUNNING, "working": RUNNING, "idle": WAITING, "blocked": WAITING,
                "done": DONE}

# Hidden from top-level listings; they appear under their parent.
_SUBAGENT_KINDS = ("subagent", "subagent_fork", "subagent_resume")


def context_window(model: str | None, recorded: int | None = None) -> int:
    """Prefer the window Grok persisted; else infer from the model family."""
    if recorded and recorded > 0:
        return recorded
    name = (model or "").lower()
    if "grok-4" in name or name.startswith("grok-code"):
        return 2_000_000
    return 128_000


def short_model(model: str | None) -> str | None:
    if not model:
        return None
    return model.removeprefix("xai/")


def encode_cwd(cwd: str) -> str:
    """Match Grok Build's short-path encoding (URL-encode the whole path)."""
    return quote(cwd, safe="")


def decode_cwd_dir(dir_path: Path) -> str | None:
    """Recover the cwd from a sessions group directory name (or its .cwd file)."""
    name = dir_path.name
    try:
        decoded = unquote(name)
    except (ValueError, UnicodeDecodeError):
        decoded = ""
    if decoded.startswith("/") or (len(decoded) > 1 and decoded[1] == ":"):
        return decoded
    try:
        text = (dir_path / ".cwd").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def _parse_ts(value) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # Grok envelopes use unix seconds; some fields use millis.
        ts = float(value)
        return ts / 1000.0 if ts > 1e12 else ts
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


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _model_id(value) -> str | None:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict):
        for key in ("id", "0", "model"):
            inner = value.get(key)
            if isinstance(inner, str) and inner:
                return inner
    return None


# --- reading updates.jsonl ----------------------------------------------------

@dataclass(slots=True)
class Transcript:
    """Folded view of one updates.jsonl (or a child session's)."""

    model: str | None = None
    effort: str | None = None
    context: int | None = None
    context_window: int | None = None
    first: float | None = None
    last: float | None = None
    title: str | None = None
    tool_uses: set[str] = field(default_factory=set)
    tool_results: set[str] = field(default_factory=set)
    tool_counts: dict[str, int] = field(default_factory=dict)
    last_tool: str | None = None
    last_tool_at: float | None = None
    last_tool_id: str | None = None
    first_prompt: str | None = None
    last_prompt: str | None = None
    last_text: str | None = None
    outputs: dict[str, int] = field(default_factory=dict)  # request/message id -> output tokens
    pending_statuses: dict[str, str] = field(default_factory=dict)

    @property
    def pending_tool(self) -> bool:
        return bool(self.tool_uses - self.tool_results)


@dataclass(slots=True)
class _Cached:
    inode: int
    offset: int
    transcript: Transcript


_cache: dict[Path, _Cached] = {}
_used: set[Path] = set()


def read_updates(path: Path) -> Transcript:
    """Parse updates.jsonl, reading only what was appended since the last call."""
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
            with open(path, "rb") as f:
                f.seek(hit.offset)
                data = f.read(st.st_size - hit.offset)
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


def _unwrap_update(e: dict) -> tuple[float | None, dict | None]:
    """Return (timestamp, update-dict) from an envelope or a bare ACP notification."""
    ts = _parse_ts(e.get("timestamp"))
    if isinstance(e.get("method"), str) and isinstance(e.get("params"), dict):
        params = e["params"]
        update = params.get("update")
        return ts, update if isinstance(update, dict) else None
    # Legacy: bare {"sessionId", "update": {...}}
    update = e.get("update")
    if isinstance(update, dict):
        return ts, update
    # Bare update object with sessionUpdate
    if "sessionUpdate" in e or "session_update" in e:
        return ts, e
    return ts, None


def _update_kind(update: dict) -> str:
    kind = update.get("sessionUpdate") or update.get("session_update") or ""
    return kind if isinstance(kind, str) else ""


def _content_text(content) -> str | None:
    if isinstance(content, str):
        text = content
    elif isinstance(content, dict):
        if content.get("type") == "text" and isinstance(content.get("text"), str):
            text = content["text"]
        else:
            return None
    elif isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text")
                if isinstance(t, str) and t.strip():
                    parts.append(t.strip())
        text = "\n".join(parts)
    else:
        return None
    text = text.strip()
    return text[:_TEXT_LIMIT] if text else None


def _tool_name(update: dict) -> str:
    for key in ("title", "toolName", "tool_name", "kind", "name"):
        value = update.get(key)
        if isinstance(value, str) and value.strip():
            # Titles look like `Read `/tmp/foo.rs``; keep the leading word.
            name = value.strip().split(None, 1)[0].strip("`")
            return name or value.strip()
    return "tool"


def _read_line(t: Transcript, line: str) -> None:
    try:
        e = json.loads(line)
    except json.JSONDecodeError:
        return
    if not isinstance(e, dict):
        return
    ts, update = _unwrap_update(e)
    if ts:
        t.first = t.first or ts
        t.last = ts
    if not isinstance(update, dict):
        return
    kind = _update_kind(update)
    if kind in ("user_message_chunk", "user_message"):
        text = _content_text(update.get("content"))
        if text and not text.startswith("<"):
            t.first_prompt = t.first_prompt or text
            t.last_prompt = text
        return
    if kind in ("agent_message_chunk", "agent_message", "agent_thought_chunk"):
        if kind != "agent_thought_chunk":
            text = _content_text(update.get("content"))
            if text:
                t.last_text = text
        return
    if kind == "tool_call":
        tool_id = update.get("toolCallId") or update.get("tool_call_id")
        name = _tool_name(update)
        if isinstance(tool_id, str) and tool_id:
            t.tool_uses.add(tool_id)
            t.last_tool_id = tool_id
            status = update.get("status")
            if isinstance(status, str):
                t.pending_statuses[tool_id] = status
                if status in ("completed", "failed", "cancelled"):
                    t.tool_results.add(tool_id)
        t.tool_counts[name] = t.tool_counts.get(name, 0) + 1
        t.last_tool, t.last_tool_at = name, ts
        return
    if kind in ("tool_call_update", "tool_result"):
        tool_id = update.get("toolCallId") or update.get("tool_call_id")
        if isinstance(tool_id, str) and tool_id:
            status = update.get("status")
            if status is None or status in ("completed", "failed", "cancelled", "done"):
                t.tool_results.add(tool_id)
            elif isinstance(status, str):
                t.pending_statuses[tool_id] = status
        return
    if kind == "usage":
        _read_usage_update(t, update, ts)
        return
    # xAI / alternate shapes: model id on the update
    model = _model_id(update.get("modelId") or update.get("model_id") or update.get("model"))
    if model:
        t.model = model


def _read_usage_update(t: Transcript, update: dict, ts: float | None) -> None:
    usage = update.get("usage")
    if not isinstance(usage, dict):
        usage = update
    total = sum(_int(usage.get(k)) for k in (
        "input_tokens", "inputTokens", "cache_creation_input_tokens", "cacheCreationInputTokens",
        "cache_read_input_tokens", "cached_read_tokens", "cachedReadTokens", "cacheReadInputTokens",
        "output_tokens", "outputTokens", "reasoning_tokens", "reasoningTokens",
    ))
    if not total:
        total = _int(usage.get("total_tokens") or usage.get("totalTokens"))
    if total:
        t.context = total
    out = _int(usage.get("output_tokens") or usage.get("outputTokens"))
    msg_id = update.get("messageId") or update.get("message_id") or update.get("requestId")
    if isinstance(msg_id, str) and msg_id:
        t.outputs[msg_id] = max(t.outputs.get(msg_id, 0), out)
    elif out:
        # No id: count each usage line as its own request.
        key = f"@{ts or 0}:{len(t.outputs)}"
        t.outputs[key] = out
    model = _model_id(update.get("modelId") or update.get("model_id") or usage.get("model"))
    if model:
        t.model = model


def _apply_usage_file(t: Transcript, path: Path) -> None:
    data = _read_json(path)
    session = data.get("session") if isinstance(data.get("session"), dict) else data
    if not isinstance(session, dict):
        return
    total = sum(_int(session.get(k)) for k in (
        "inputTokens", "input_tokens", "cachedReadTokens", "cached_read_tokens",
        "cacheCreationTokens", "cache_creation_tokens", "outputTokens", "output_tokens",
        "reasoningTokens", "reasoning_tokens",
    ))
    if not total:
        total = _int(session.get("totalTokens") or session.get("total_tokens"))
    if total:
        t.context = total
    turns = data.get("turns")
    if isinstance(turns, list) and turns:
        last = turns[-1]
        if isinstance(last, dict):
            turn_total = sum(_int(last.get(k)) for k in (
                "inputTokens", "input_tokens", "cachedReadTokens", "cached_read_tokens",
                "cacheCreationTokens", "cache_creation_tokens", "outputTokens", "output_tokens",
                "reasoningTokens", "reasoning_tokens",
            )) or _int(last.get("totalTokens") or last.get("total_tokens"))
            if turn_total:
                t.context = turn_total
    model = _model_id(session.get("primaryModelId") or session.get("primary_model_id"))
    if model and not t.model:
        t.model = model
    calls = _int(session.get("modelCalls") or session.get("model_calls"))
    out = _int(session.get("outputTokens") or session.get("output_tokens"))
    if calls and not t.outputs:
        t.outputs["usage"] = out
        # Fabricate distinct request slots so detail.requests mirrors call count.
        for i in range(max(0, calls - 1)):
            t.outputs[f"usage-{i}"] = 0


# --- summary / discovery ------------------------------------------------------

@dataclass(slots=True)
class Summary:
    session_id: str
    cwd: str | None = None
    title: str | None = None
    model: str | None = None
    effort: str | None = None
    context_window: int | None = None
    session_kind: str | None = None
    parent_session_id: str | None = None
    created_at: float | None = None
    updated_at: float | None = None
    last_active_at: float | None = None
    hidden: bool = False
    agent_name: str | None = None


def read_summary(path: Path) -> Summary | None:
    data = _read_json(path)
    if not data:
        return None
    raw_info = data.get("info")
    info: dict = raw_info if isinstance(raw_info, dict) else {}
    sid = info.get("id")
    if isinstance(sid, dict):
        sid = sid.get("0") or sid.get("id")
    if not isinstance(sid, str) or not sid:
        sid = path.parent.name
    raw_cwd = info.get("cwd")
    cwd = raw_cwd if isinstance(raw_cwd, str) else None
    generated = data.get("generated_title")
    summary_text = data.get("session_summary")
    title = None
    if isinstance(generated, str) and generated.strip():
        title = generated.strip()
    elif isinstance(summary_text, str) and summary_text.strip():
        title = summary_text.strip()
    window = data.get("context_window")
    kind = data.get("session_kind")
    parent = data.get("parent_session_id")
    hidden = data.get("hidden")
    if hidden is None and isinstance(kind, str):
        hidden = kind.startswith("subagent")
    agent = data.get("agent_name")
    effort = data.get("reasoning_effort")
    return Summary(
        session_id=sid,
        cwd=cwd,
        title=title,
        model=_model_id(data.get("current_model_id")),
        effort=effort if isinstance(effort, str) else None,
        context_window=_int(window) or None,
        session_kind=kind if isinstance(kind, str) else None,
        parent_session_id=parent if isinstance(parent, str) else None,
        created_at=_parse_ts(data.get("created_at")),
        updated_at=_parse_ts(data.get("updated_at")),
        last_active_at=_parse_ts(data.get("last_active_at")),
        hidden=bool(hidden),
        agent_name=agent if isinstance(agent, str) else None,
    )


def _cwd_dirs() -> list[Path]:
    try:
        return [p for p in SESSIONS.iterdir() if p.is_dir()]
    except OSError:
        return []


def _session_dirs_in(cwd_dir: Path) -> list[Path]:
    try:
        return [p for p in cwd_dir.iterdir() if p.is_dir() and (p / "summary.json").is_file()]
    except OSError:
        return []


def iter_sessions(*, include_hidden: bool = False
                  ) -> list[tuple[Path, Summary, str | None]]:
    """(session_dir, summary, cwd) for every persisted session."""
    found = []
    for cwd_dir in _cwd_dirs():
        cwd = decode_cwd_dir(cwd_dir)
        for session_dir in _session_dirs_in(cwd_dir):
            summary = read_summary(session_dir / "summary.json")
            if not summary:
                continue
            if summary.hidden and not include_hidden:
                continue
            if summary.session_kind in _SUBAGENT_KINDS and not include_hidden:
                continue
            found.append((session_dir, summary, summary.cwd or cwd))
    return found


def session_dir_by_id(session_id: str, cwd: str | None = None) -> Path | None:
    """A session directory by id or unique prefix. Raises LookupError if ambiguous."""
    if cwd:
        try:
            direct = SESSIONS / encode_cwd(cwd) / session_id
            if (direct / "summary.json").is_file():
                return direct
        except OSError:
            pass  # Encoded name past the filesystem limit; fall through to .cwd scan.
        # Long cwd: scan group dirs whose decoded path matches.
        resolved = None
        try:
            resolved = str(Path(cwd).resolve())
        except (OSError, ValueError):
            resolved = cwd
        for cwd_dir in _cwd_dirs():
            decoded = decode_cwd_dir(cwd_dir)
            if decoded not in (cwd, resolved):
                continue
            direct = cwd_dir / session_id
            try:
                if (direct / "summary.json").is_file():
                    return direct
            except OSError:
                continue
    hits = []
    for cwd_dir in _cwd_dirs():
        for session_dir in _session_dirs_in(cwd_dir):
            if session_dir.name == session_id or session_dir.name.startswith(session_id):
                hits.append(session_dir)
    exact = [p for p in hits if p.name == session_id]
    if exact:
        hits = exact
    names = sorted({p.name for p in hits})
    if len(names) > 1:
        shown = ", ".join(n[:13] for n in names[:5]) + (" …" if len(names) > 5 else "")
        raise LookupError(f"'{session_id}' matches {len(names)} sessions: {shown}")
    return max(hits, key=_mtime) if hits else None


def _same_dir(a: str | None, resolved: Path) -> bool:
    try:
        return bool(a) and Path(a).resolve() == resolved  # type: ignore[arg-type]
    except (OSError, ValueError):
        return False


# --- live sessions ------------------------------------------------------------

@dataclass(slots=True)
class _LiveCache:
    at: float = float("-inf")
    rows: list[dict] = field(default_factory=list)


_live = _LiveCache()


def live_sessions() -> list[dict]:
    if time.monotonic() - _live.at >= LIVE_TTL_SECS:
        _live.rows, _live.at = _query_live(), time.monotonic()
    return _live.rows


def _query_live() -> list[dict]:
    now = time.time()
    rows = []
    for session_dir, summary, cwd in iter_sessions():
        status = _live_status(session_dir, summary, now)
        if status is None:
            continue
        kind = "headless" if summary.session_kind == "headless" else "interactive"
        rows.append({"sessionId": summary.session_id, "cwd": cwd, "kind": kind,
                     "status": status, "name": summary.title})
    return rows


def _live_status(session_dir: Path, summary: Summary, now: float) -> str | None:
    """None when the session does not look running; else a status word."""
    updates = session_dir / "updates.jsonl"
    recent = now - _mtime(updates) < STALE_AFTER_SECS if updates.exists() else False
    if not recent and summary.last_active_at and now - summary.last_active_at < STALE_AFTER_SECS:
        recent = True
    running_child = False
    for meta_path in _subagent_meta_paths(session_dir):
        meta = _read_json(meta_path)
        if meta.get("status") == "running":
            running_child = True
            break
    if not recent and not running_child:
        return None
    if running_child or recent:
        # Prefer busy when the transcript is still moving.
        if recent and now - _mtime(updates) < 15:
            return "busy"
        return "idle" if not running_child else "busy"
    return None


# --- building the tree --------------------------------------------------------

Found = list[tuple[Path, "dict | None", "str | None"]]


def _live_found(live: list[dict]) -> Found:
    found: Found = []
    for s in live:
        try:
            path = session_dir_by_id(s["sessionId"], s.get("cwd"))
        except LookupError:
            continue
        if path:
            found.append((path, s, s.get("cwd")))
    return found


def find(target: str | None) -> Found:
    """Session directories to show for a target.

    target None: every running session. A directory: sessions running there,
    else that directory's newest top-level session. Anything else: a session id
    (prefix). Raises LookupError when nothing matches.
    """
    if target is None:
        found = _live_found(live_sessions())
        if not found:
            raise LookupError("no running Grok Build sessions")
        return found

    if Path(target).is_dir():
        d = Path(target).resolve()
        found = _live_found([s for s in live_sessions() if _same_dir(s.get("cwd"), d)])
        if found:
            return found
        pool = [(p, s, cwd) for p, s, cwd in iter_sessions() if _same_dir(cwd, d)]
        if not pool:
            raise LookupError(f"no Grok Build sessions in {d}")
        newest = max(pool, key=lambda row: _mtime(row[0]))
        return [(newest[0], None, newest[2])]

    path = session_dir_by_id(target)
    if not path:
        raise LookupError(f"no Grok Build session matching '{target}' (and no such directory)")
    summary = read_summary(path / "summary.json")
    live = next((s for s in live_sessions() if s["sessionId"] == path.name), None)
    cwd = (live or {}).get("cwd") if live else (summary.cwd if summary else None)
    if not cwd:
        cwd = decode_cwd_dir(path.parent)
    return [(path, live, cwd)]


def _subagent_meta_paths(session_dir: Path) -> list[Path]:
    root = session_dir / "subagents"
    if not root.is_dir():
        return []
    paths = []
    try:
        for child in sorted(root.iterdir()):
            meta = child / "meta.json"
            if meta.is_file():
                paths.append(meta)
    except OSError:
        return []
    return paths


def _agent_from(aid: str, label: str, t: Transcript, state: str, status: str | None,
                window: int | None, transcript: Path, *,
                started: float | None = None, ended: float | None = None,
                model: str | None = None, effort: str | None = None) -> Agent:
    main = aid == "main"
    detail = Detail(
        prompt=t.last_prompt if main else (t.first_prompt or t.last_prompt),
        prompt_label="Last prompt" if main else "Prompt",
        tools=dict(t.tool_counts), last_tool=t.last_tool, last_tool_at=t.last_tool_at,
        last_tool_pending=bool(t.last_tool_id and t.last_tool_id not in t.tool_results),
        last_text=t.last_text, output_tokens=sum(t.outputs.values()), requests=len(t.outputs),
        transcript=str(transcript),
    )
    return Agent(
        id=aid, label=label, state=state, status=status,
        model=short_model(model or t.model), effort=effort or t.effort,
        context_tokens=t.context, context_window=window,
        started=started if started is not None else t.first,
        ended=ended, detail=detail,
    )


def _subagent_state(meta: dict, t: Transcript, mtime: float, now: float,
                    session_live: bool) -> tuple[str, str]:
    status_word = meta.get("status") if isinstance(meta.get("status"), str) else ""
    if status_word in _OUTCOME_STATES:
        return _OUTCOME_STATES[status_word], status_word
    recent = now - mtime < STALE_AFTER_SECS
    if status_word == "running" or recent or (session_live and t.pending_tool):
        if recent or t.pending_tool or status_word == "running":
            if status_word == "running" and not recent and not t.pending_tool:
                return STALE, "stale"
            return RUNNING, "running"
    if session_live:
        return WAITING, "idle"
    return STALE, "stale"


def _apply_summary(t: Transcript, summary: Summary) -> None:
    if summary.model and not t.model:
        t.model = summary.model
    if summary.effort and not t.effort:
        t.effort = summary.effort
    if summary.context_window:
        t.context_window = summary.context_window
    if summary.created_at and not t.first:
        t.first = summary.created_at
    if summary.updated_at and (not t.last or summary.updated_at > t.last):
        t.last = summary.updated_at


def _str_field(meta: dict, key: str) -> str | None:
    value = meta.get(key)
    return value if isinstance(value, str) and value else None


def _load_subagent(meta_path: Path, now: float, session_live: bool) -> tuple[Agent, Transcript]:
    meta = _read_json(meta_path)
    aid = _str_field(meta, "subagent_id") or meta_path.parent.name
    child_id = _str_field(meta, "child_session_id")
    child_cwd = _str_field(meta, "child_cwd")
    child_dir = session_dir_by_id(child_id, child_cwd) if child_id else None
    if child_dir:
        child_updates = child_dir / "updates.jsonl"
    else:
        child_updates = meta_path.with_name("updates.jsonl")
    t = read_updates(child_updates) if child_updates.exists() else Transcript()
    if child_dir:
        _apply_usage_file(t, child_dir / "usage.json")
        child_summary = read_summary(child_dir / "summary.json")
        if child_summary:
            _apply_summary(t, child_summary)
    model_from_meta = _str_field(meta, "effective_model_id")
    if model_from_meta and not t.model:
        t.model = model_from_meta
    prompt = _str_field(meta, "prompt")
    if prompt and not t.first_prompt:
        t.first_prompt = prompt[:_TEXT_LIMIT]
    started = _parse_ts(meta.get("started_at")) or t.first
    ended_meta = _parse_ts(meta.get("completed_at"))
    mtime = _mtime(child_updates) if child_updates.exists() else _mtime(meta_path)
    state, status = _subagent_state(meta, t, mtime, now, session_live)
    agent_type = str(meta.get("subagent_type") or "?").rsplit(":", 1)[-1]
    description = _str_field(meta, "description") or aid
    window = context_window(t.model, t.context_window)
    if t.context and t.context > window:
        window = max(window, t.context)
    transcript = child_updates if child_updates.exists() else meta_path
    agent = _agent_from(
        aid, f"{agent_type}: {description}", t, state, status, window, transcript,
        started=started, ended=None if state == RUNNING else (ended_meta or t.last),
        model=t.model, effort=t.effort,
    )
    return agent, t


def _window_for(main_t: Transcript, children: list[Transcript],
                summary: Summary) -> int:
    window = context_window(main_t.model, main_t.context_window or summary.context_window)
    if main_t.context and main_t.context > window:
        window = max(window, main_t.context)
    for t in children:
        if t.model == main_t.model and t.context and t.context > window:
            window = max(window, t.context)
    return window


def load(path: Path, live: dict | None = None, cwd: str | None = None,
         now: float | None = None) -> Session:
    """One top-level session. live is its live-sessions entry, None when inactive."""
    now = time.time() if now is None else now
    summary = read_summary(path / "summary.json") or Summary(session_id=path.name)
    updates_path = path / "updates.jsonl"
    main_t = read_updates(updates_path)
    _apply_usage_file(main_t, path / "usage.json")
    _apply_summary(main_t, summary)

    agents: dict[str, Agent] = {}
    child_transcripts: list[Transcript] = []
    for meta_path in _subagent_meta_paths(path):
        agent, t = _load_subagent(meta_path, now, live is not None)
        agents[agent.id] = agent
        child_transcripts.append(t)
    roots = sorted(agents.values(), key=lambda a: a.started or 0)

    if live is None:
        state, status = INACTIVE, "not running"
    else:
        status = live.get("status")
        state = _LIVE_STATES.get(status, WAITING) if isinstance(status, str) else WAITING

    window = _window_for(main_t, child_transcripts, summary)
    main = _agent_from("main", "main", main_t, state, status, window, updates_path,
                       model=main_t.model or summary.model,
                       effort=main_t.effort or summary.effort)
    main.ended = main_t.last
    title = summary.title or (live or {}).get("name") or path.name[:8]
    kind = (live or {}).get("kind")
    if not kind:
        kind = "headless" if summary.session_kind == "headless" else "interactive"
    return Session(id=summary.session_id or path.name, title=title, main=main, agents=roots,
                   cwd=cwd or summary.cwd, kind=kind)


def sessions(target: str | None) -> list[Session]:
    now = time.time()
    _used.clear()
    loaded = [load(path, live, cwd, now) for path, live, cwd in find(target)]
    forget_unused()
    return loaded
