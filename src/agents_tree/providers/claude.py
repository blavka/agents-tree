"""Claude Code: sessions under ~/.claude/projects and their subagents.

Layout read here (none of it is a documented interface, so every field is optional):

  ~/.claude/projects/<project>/<session-id>.jsonl          main transcript
  ~/.claude/projects/<project>/<session-id>/**/agent-<id>.jsonl
  ~/.claude/projects/<project>/<session-id>/**/agent-<id>.meta.json
      {"agentType", "description", "toolUseId", "parentAgentId",
       "requestShape": "background"|"foreground", "taskKind": "in_process_teammate", ...}

Running sessions come from `claude agents --json`, falling back to the pid files
in ~/.claude/sessions/.
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from agents_tree.model import (BLOCKED, DONE, FAILED, INACTIVE, RUNNING, STALE, WAITING, Agent,
                               Detail, Session)

CLAUDE_DIR = Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")
PROJECTS = CLAUDE_DIR / "projects"

# A subagent with no completion marker whose transcript has not changed for this
# long, and that is not waiting on a tool call, is reported as stale.
STALE_AFTER_SECS = 120
# `claude agents --json` costs a node start (~0.2 s CPU); reuse its answer this long.
LIVE_TTL_SECS = 5.0

_NOTIFICATION_RE = re.compile(r"<task-notification>(.*?)</task-notification>", re.S)
_TASK_ID_RE = re.compile(r"<task-id>([^<]+)</task-id>")
_STATUS_RE = re.compile(r"<status>([^<]+)</status>")
_DATE_SUFFIX_RE = re.compile(r"-\d{8}$")
_USAGE_KEYS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
               "output_tokens")
# The Agent tool answers a background launch at once; that answer is not a result.
_ASYNC_LAUNCH_PREFIX = "Async agent launched"
# Long texts are kept only this far for the detail view.
_TEXT_LIMIT = 20_000

# Claude Code's words for a session or an agent outcome, as neutral states.
_LIVE_STATES = {"busy": RUNNING, "working": RUNNING, "idle": WAITING, "blocked": BLOCKED,
                "done": DONE}
_OUTCOME_STATES = {"completed": DONE, "failed": FAILED, "killed": FAILED, "stopped": FAILED}


def context_window(model: str | None) -> int:
    """Claude Code does not record the window size; infer it from the model family."""
    if re.search(r"(opus|sonnet|fable|mythos)-5", model or ""):
        return 1_000_000
    return 200_000


def short_model(model: str | None) -> str | None:
    if not model:
        return None
    return _DATE_SUFFIX_RE.sub("", model.removeprefix("claude-"))


def _parse_ts(value) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _int(value) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


# --- reading transcripts ------------------------------------------------------

@dataclass(slots=True)
class Transcript:
    """What one .jsonl transcript says about its agent. Every field folds over the
    lines in order, so appended lines can be read on top of an earlier parse."""

    model: str | None = None
    effort: str | None = None
    context: int | None = None
    first: float | None = None
    last: float | None = None
    title: str | None = None
    custom_title: str | None = None
    tool_uses: set[str] = field(default_factory=set)
    tool_results: set[str] = field(default_factory=set)
    # task id -> (timestamp of the notification, status); the newest one wins.
    task_status: dict[str, tuple[float, str]] = field(default_factory=dict)
    tool_counts: dict[str, int] = field(default_factory=dict)
    last_tool: str | None = None
    last_tool_at: float | None = None
    last_tool_id: str | None = None
    first_prompt: str | None = None
    last_prompt: str | None = None
    # When a message last arrived (a prompt, or another agent's message waking it up).
    last_inbound_at: float | None = None
    last_text: str | None = None
    # One API response is written as several lines sharing a requestId.
    outputs: dict[str, int] = field(default_factory=dict)

    @property
    def pending_tool(self) -> bool:
        """A tool call of this agent has no result yet (a long command, a prompt)."""
        return bool(self.tool_uses - self.tool_results)


@dataclass(slots=True)
class _Cached:
    inode: int
    offset: int  # bytes parsed so far, always at a line end
    transcript: Transcript


_cache: dict[Path, _Cached] = {}
_used: set[Path] = set()


def read_transcript(path: Path) -> Transcript:
    """Parse a transcript, reading only what was appended since the last call."""
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
        end = data.rfind(b"\n") + 1  # a line still being written waits for the next call
        for line in data[:end].splitlines():
            _read_line(hit.transcript, line.decode("utf-8", errors="replace"))
        hit.offset += end
    return hit.transcript


def forget_unused() -> None:
    """Drop cached transcripts that were not read since _used was last cleared."""
    for path in set(_cache) - _used:
        del _cache[path]


def _put_note(notes: dict[str, tuple[float, str]], task_id: str, note: tuple[float, str]) -> None:
    if task_id not in notes or notes[task_id][0] <= note[0]:
        notes[task_id] = note


def _read_line(t: Transcript, line: str) -> None:
    try:
        e = json.loads(line)
    except json.JSONDecodeError:
        return
    if not isinstance(e, dict):
        return
    ts = _parse_ts(e.get("timestamp"))
    if ts:
        t.first = t.first or ts
        t.last = ts
    if "<task-notification>" in line:
        for body in _NOTIFICATION_RE.findall(line.replace("\\n", "\n")):
            task_id, status = _TASK_ID_RE.search(body), _STATUS_RE.search(body)
            if task_id and status:
                _put_note(t.task_status, task_id.group(1), (ts or 0.0, status.group(1)))

    kind = e.get("type")
    if kind == "ai-title" and isinstance(e.get("aiTitle"), str):
        t.title = e["aiTitle"]
        return
    if kind == "custom-title" and isinstance(e.get("customTitle"), str):
        t.custom_title = e["customTitle"]
        return
    msg = e.get("message")
    if not isinstance(msg, dict):
        return
    content = msg.get("content")
    if kind == "user":
        _read_user(t, e, content, ts)
    elif kind == "assistant":
        _read_assistant(t, e, msg, content, ts)


def _read_user(t: Transcript, e: dict, content, ts: float | None) -> None:
    if isinstance(content, str) and content.strip():
        t.last_inbound_at = ts or t.last_inbound_at
    if not e.get("isMeta"):
        prompt = _prompt_text(content)
        if prompt:
            t.first_prompt = t.first_prompt or prompt
            t.last_prompt = prompt
    if isinstance(content, list):
        t.tool_results.update(b["tool_use_id"] for b in content
                              if isinstance(b, dict) and b.get("type") == "tool_result"
                              and isinstance(b.get("tool_use_id"), str)
                              and not _result_text(b).startswith(_ASYNC_LAUNCH_PREFIX))


def _read_assistant(t: Transcript, e: dict, msg: dict, content, ts: float | None) -> None:
    model = msg.get("model")
    if isinstance(model, str) and model and model != "<synthetic>":
        t.model = model
        effort = e.get("perTurnEffort")
        t.effort = effort if isinstance(effort, str) else None
    raw_usage = msg.get("usage")
    usage: dict = raw_usage if isinstance(raw_usage, dict) else {}
    total = sum(_int(usage.get(k)) for k in _USAGE_KEYS)
    if total:
        t.context = total
    request = e.get("requestId")
    if isinstance(request, str):
        t.outputs[request] = max(t.outputs.get(request, 0), _int(usage.get("output_tokens")))
    if not isinstance(content, list):
        return
    texts = []
    for b in content:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text" and isinstance(b.get("text"), str) and b["text"].strip():
            texts.append(b["text"].strip())
        elif b.get("type") == "tool_use" and isinstance(b.get("name"), str):
            tool_id = b.get("id") if isinstance(b.get("id"), str) else None
            if tool_id:
                t.tool_uses.add(tool_id)
            t.tool_counts[b["name"]] = t.tool_counts.get(b["name"], 0) + 1
            t.last_tool, t.last_tool_at, t.last_tool_id = b["name"], ts, tool_id
    if texts:
        t.last_text = "\n\n".join(texts)[:_TEXT_LIMIT]


def _prompt_text(content) -> str | None:
    """A message the user (or the spawning agent) wrote; not tool results or wrappers."""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "\n".join(b["text"] for b in content if isinstance(b, dict)
                         and b.get("type") == "text" and isinstance(b.get("text"), str))
    else:
        return None
    text = text.strip()
    if not text or text.startswith("<"):
        return None
    return text[:_TEXT_LIMIT]


def _result_text(block: dict) -> str:
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content and isinstance(content[0], dict):
        text = content[0].get("text")
        return text if isinstance(text, str) else ""
    return ""


# --- finding sessions -------------------------------------------------------

def _live_row(r: dict, status) -> dict:
    return {"sessionId": r["sessionId"], "cwd": r.get("cwd"), "kind": r.get("kind"),
            "status": status, "name": r.get("name")}


@dataclass(slots=True)
class _LiveCache:
    at: float = float("-inf")
    rows: list[dict] = field(default_factory=list)


_live = _LiveCache()


def live_sessions() -> list[dict]:
    """Running sessions as [{sessionId, cwd, kind, status, name}]."""
    if time.monotonic() - _live.at >= LIVE_TTL_SECS:
        _live.rows, _live.at = _query_live(), time.monotonic()
    return _live.rows


def _query_live() -> list[dict]:
    try:
        out = subprocess.run(["claude", "agents", "--json"], capture_output=True,
                             text=True, timeout=10, check=False).stdout
        rows = json.loads(out)
        if isinstance(rows, list):
            return [_live_row(r, r.get("state") or r.get("status")) for r in rows
                    if isinstance(r, dict) and isinstance(r.get("sessionId"), str)]
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        pass
    rows = []
    for f in (CLAUDE_DIR / "sessions").glob("*.json"):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
            pid = int(r["pid"])
            if pid <= 0 or not isinstance(r["sessionId"], str):
                continue
            os.kill(pid, 0)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        rows.append(_live_row(r, r.get("status")))
    return rows


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def project_dir(directory: str) -> Path:
    return PROJECTS / re.sub(r"[^A-Za-z0-9]", "-", directory)


def transcript_path(session_id: str, cwd: str | None = None) -> Path | None:
    """The transcript of a session id or a unique prefix of one; with the session's
    cwd, its project directory is tried before every other.

    Raises LookupError when a prefix matches several sessions.
    """
    if cwd:
        direct = project_dir(cwd) / f"{session_id}.jsonl"
        if direct.exists():
            return direct
    hits = list(PROJECTS.glob(f"*/{glob.escape(session_id)}*.jsonl"))
    exact = [p for p in hits if p.stem == session_id]
    if exact:
        hits = exact
    stems = sorted({p.stem for p in hits})
    if len(stems) > 1:
        shown = ", ".join(s[:13] for s in stems[:5]) + (" …" if len(stems) > 5 else "")
        raise LookupError(f"'{session_id}' matches {len(stems)} sessions: {shown}")
    return max(hits, key=_mtime) if hits else None


def _same_dir(a: str | None, resolved: Path) -> bool:
    try:
        return bool(a) and Path(a).resolve() == resolved  # type: ignore[arg-type]
    except (OSError, ValueError):
        return False


# (transcript, live entry or None when the session is not running, its cwd)
Found = list[tuple[Path, "dict | None", "str | None"]]


def _live_found(live: list[dict]) -> Found:
    found: Found = []
    for s in live:
        try:
            path = transcript_path(s["sessionId"], s.get("cwd"))
        except LookupError:
            continue
        if path:
            found.append((path, s, s.get("cwd")))
    return found


def find(target: str | None) -> Found:
    """Transcripts to show for a target.

    target None: every running session. A directory: the sessions running there,
    else that directory's newest session. Anything else: a session id (prefix).
    Raises LookupError with a message for the user when nothing matches.
    """
    if target is None:
        found = _live_found(live_sessions())
        if not found:
            raise LookupError("no running Claude Code sessions")
        return found

    if Path(target).is_dir():
        d = Path(target).resolve()
        found = _live_found([s for s in live_sessions() if _same_dir(s["cwd"], d)])
        if found:
            return found
        proj = project_dir(str(d))
        pool = list(proj.glob("*.jsonl")) if proj.is_dir() else []
        if not pool:
            raise LookupError(f"no Claude Code sessions in {d}")
        return [(max(pool, key=_mtime), None, str(d))]

    path = transcript_path(target)
    if not path:
        raise LookupError(f"no Claude Code session matching '{target}' (and no such directory)")
    live = next((s for s in live_sessions() if s["sessionId"] == path.stem), None)
    return [(path, live, live.get("cwd") if live else None)]


# --- building the tree --------------------------------------------------------

def _agent_from(aid: str, label: str, t: Transcript, state: str, status: str | None,
                window: int | None, transcript: Path) -> Agent:
    main = aid == "main"
    detail = Detail(
        prompt=t.last_prompt if main else t.first_prompt,
        prompt_label="Last prompt" if main else "Prompt",
        tools=dict(t.tool_counts), last_tool=t.last_tool, last_tool_at=t.last_tool_at,
        last_tool_pending=bool(t.last_tool_id and t.last_tool_id not in t.tool_results),
        last_text=t.last_text, output_tokens=sum(t.outputs.values()), requests=len(t.outputs),
        transcript=str(transcript),
    )
    return Agent(
        id=aid, label=label, state=state, status=status, model=short_model(t.model),
        effort=t.effort, context_tokens=t.context, context_window=window,
        started=t.first, ended=None if state == RUNNING else t.last, detail=detail,
    )


def _subagent_state(meta: dict, t: Transcript, mtime: float, now: float,
                    note: tuple[float, str] | None, tool_results: set[str],
                    session_live: bool) -> tuple[str, str]:
    """(state, the word shown for it)."""
    recent = now - mtime < STALE_AFTER_SECS
    # A notification is the outcome, unless a message woke the agent up again since.
    if note and not (recent and (t.last_inbound_at or 0) > note[0]):
        return _OUTCOME_STATES.get(note[1], DONE), note[1]
    if meta.get("requestShape") != "background" and meta.get("toolUseId") in tool_results:
        return DONE, "done"
    if recent or (session_live and t.pending_tool):
        return RUNNING, "running"
    if session_live and (meta.get("taskKind") == "in_process_teammate" or meta.get("teamName")):
        return WAITING, "idle"  # a teammate waiting for its next message
    return STALE, "stale"


def _windows(transcripts: list[Transcript]) -> dict[str | None, int]:
    """Context window per model for one session: inferred from the family, and the
    long window wherever any transcript of the session went past the inferred one."""
    windows: dict[str | None, int] = {}
    for t in transcripts:
        if t.model:
            window = context_window(t.model)
            if t.context and t.context > window:
                window = 1_000_000
            windows[t.model] = max(windows.get(t.model, 0), window)
    return windows


def load(path: Path, live: dict | None = None, cwd: str | None = None,
         now: float | None = None) -> Session:
    """One session. live is its `claude agents` entry, None when it is not running."""
    now = time.time() if now is None else now
    main_t = read_transcript(path)

    metas = []
    for meta_path in sorted(path.with_suffix("").glob("**/*.meta.json")):
        stem = meta_path.name.removesuffix(".meta.json")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        jl = meta_path.with_name(stem + ".jsonl")
        metas.append((stem.removeprefix("agent-"), meta if isinstance(meta, dict) else {},
                      read_transcript(jl), _mtime(jl), jl))

    # Completion markers live in whichever transcript spawned the agent.
    notes = dict(main_t.task_status)
    tool_results = set(main_t.tool_results)
    for _, _, t, _, _ in metas:
        for task_id, note in t.task_status.items():
            _put_note(notes, task_id, note)
        tool_results |= t.tool_results
    windows = _windows([main_t, *(t for _, _, t, _, _ in metas)])

    agents: dict[str, Agent] = {}
    parent_of: dict[str, str | None] = {}
    for aid, meta, t, mtime, jl in metas:
        state, status = _subagent_state(meta, t, mtime, now, notes.get(aid), tool_results,
                                        live is not None)
        agent_type = str(meta.get("agentType") or "?").rsplit(":", 1)[-1]
        label = f"{agent_type}: {meta.get('description') or aid}"
        agents[aid] = _agent_from(aid, label, t, state, status, windows.get(t.model), jl)
        tool_use = meta.get("toolUseId")
        parent = meta.get("parentAgentId")
        if not (isinstance(parent, str) and parent != aid):
            parent = next((pid for pid, _, pt, _, _ in metas
                           if pid != aid and tool_use and tool_use in pt.tool_uses), None)
        parent_of[aid] = parent

    roots = []
    for aid, agent in agents.items():
        pid = _parent_without_cycle(aid, parent_of, agents)
        (agents[pid].children if pid else roots).append(agent)
    for agent in agents.values():
        agent.children.sort(key=lambda a: a.started or 0)
    roots.sort(key=lambda a: a.started or 0)

    if live is None:
        state, status = INACTIVE, "not running"
    else:
        status = live.get("status")
        state = _LIVE_STATES.get(status, WAITING) if isinstance(status, str) else WAITING
    main = _agent_from("main", "main", main_t, state, status, windows.get(main_t.model), path)
    main.ended = main_t.last  # the main loop's span is its transcript's, live or not
    title = (main_t.custom_title or main_t.title or (live or {}).get("name")
             or path.stem[:8])
    return Session(id=path.stem, title=title, main=main, agents=roots, cwd=cwd,
                   kind=(live or {}).get("kind"))


def _parent_without_cycle(aid: str, parent_of: dict[str, str | None],
                          agents: dict[str, Agent]) -> str | None:
    pid = parent_of.get(aid)
    if pid not in agents:
        return None
    seen = {aid}
    step: str | None = pid
    while step:
        if step in seen:
            return None  # a cycle: show the agent at the top level rather than lose it
        seen.add(step)
        nxt = parent_of.get(step)
        step = nxt if nxt in agents else None
    return pid


def sessions(target: str | None) -> list[Session]:
    """Sessions for a target. The transcript cache keeps only what this call read."""
    now = time.time()
    _used.clear()
    loaded = [load(path, live, cwd, now) for path, live, cwd in find(target)]
    forget_unused()
    return loaded
