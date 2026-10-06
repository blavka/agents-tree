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
from datetime import datetime
from pathlib import Path

from agents_tree.model import DONE, RUNNING, STALE, Agent, Session

NAME = "claude"
CLAUDE_DIR = Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")
PROJECTS = CLAUDE_DIR / "projects"
NOT_RUNNING = "not running"
IDLE = "idle"

# A subagent with no completion marker whose transcript has not changed for this
# long, and that is not waiting on a tool call, is reported as stale.
STALE_AFTER_SECS = 120
# A notification is overridden by later activity only when that activity comes
# this much after it (the agent writes its last lines around the notification).
RESUMED_AFTER_SECS = 5

_NOTIFICATION_RE = re.compile(r"<task-notification>(.*?)</task-notification>", re.S)
_TASK_ID_RE = re.compile(r"<task-id>([^<]+)</task-id>")
_STATUS_RE = re.compile(r"<status>([^<]+)</status>")
_DATE_SUFFIX_RE = re.compile(r"-\d{8}$")
_USAGE_KEYS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
               "output_tokens")
# The Agent tool answers a background launch at once; that answer is not a result.
_ASYNC_LAUNCH_PREFIX = "Async agent launched"


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


class Transcript:
    """What one .jsonl transcript says about its agent."""

    __slots__ = ("model", "effort", "context", "first", "last", "title", "custom_title",
                 "tool_uses", "tool_results", "task_status")

    def __init__(self) -> None:
        self.model: str | None = None
        self.effort: str | None = None
        self.context: int | None = None
        self.first: float | None = None
        self.last: float | None = None
        self.title: str | None = None
        self.custom_title: str | None = None
        self.tool_uses: set[str] = set()
        self.tool_results: set[str] = set()
        # task id -> (timestamp of the notification, status); the newest one wins.
        self.task_status: dict[str, tuple[float, str]] = {}

    @property
    def pending_tool(self) -> bool:
        """A tool call of this agent has no result yet (a long command, a prompt)."""
        return bool(self.tool_uses - self.tool_results)


_cache: dict[Path, tuple[tuple[int, int], Transcript]] = {}
_CACHE_LIMIT = 4096


def read_transcript(path: Path) -> Transcript:
    """Parse a transcript, reusing the last parse while the file is unchanged."""
    try:
        st = path.stat()
    except OSError:
        return Transcript()
    key = (st.st_mtime_ns, st.st_size)
    hit = _cache.get(path)
    if hit and hit[0] == key:
        return hit[1]

    t = Transcript()
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                _read_line(t, line)
    except OSError:
        return t
    if len(_cache) >= _CACHE_LIMIT:
        _cache.clear()
    _cache[path] = (key, t)
    return t


def _int(value) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


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
        _read_notifications(t, line, ts or 0.0)

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
    if kind == "assistant":
        model = msg.get("model")
        if isinstance(model, str) and model and model != "<synthetic>":
            t.model = model
            effort = e.get("perTurnEffort")
            t.effort = effort if isinstance(effort, str) else None
        usage = msg.get("usage")
        if isinstance(usage, dict):
            total = sum(_int(usage.get(k)) for k in _USAGE_KEYS)
            if total:
                t.context = total
        if isinstance(content, list):
            t.tool_uses.update(b["id"] for b in content
                               if isinstance(b, dict) and b.get("type") == "tool_use"
                               and isinstance(b.get("id"), str))
    elif kind == "user" and isinstance(content, list):
        t.tool_results.update(b["tool_use_id"] for b in content
                              if isinstance(b, dict) and b.get("type") == "tool_result"
                              and isinstance(b.get("tool_use_id"), str)
                              and not _result_text(b).startswith(_ASYNC_LAUNCH_PREFIX))


def _result_text(block: dict) -> str:
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content and isinstance(content[0], dict):
        text = content[0].get("text")
        return text if isinstance(text, str) else ""
    return ""


def _read_notifications(t: Transcript, line: str, ts: float) -> None:
    for body in _NOTIFICATION_RE.findall(line.replace("\\n", "\n")):
        task_id, status = _TASK_ID_RE.search(body), _STATUS_RE.search(body)
        if not (task_id and status):
            continue
        previous = t.task_status.get(task_id.group(1))
        if previous is None or previous[0] <= ts:
            t.task_status[task_id.group(1)] = (ts, status.group(1))


# --- finding sessions -------------------------------------------------------

def live_sessions() -> list[dict]:
    """Running sessions as [{sessionId, cwd, kind, status, name}]."""
    try:
        out = subprocess.run(["claude", "agents", "--json"], capture_output=True,
                             text=True, timeout=10).stdout
        rows = json.loads(out)
        if isinstance(rows, list):
            return [{"sessionId": r["sessionId"], "cwd": r.get("cwd"), "kind": r.get("kind"),
                     "status": r.get("state") or r.get("status"), "name": r.get("name")}
                    for r in rows if isinstance(r, dict) and isinstance(r.get("sessionId"), str)]
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        pass
    rows = []
    for f in (CLAUDE_DIR / "sessions").glob("*.json"):
        try:
            r = json.loads(f.read_text())
            pid = int(r["pid"])
            if pid <= 0 or not isinstance(r["sessionId"], str):
                continue
            os.kill(pid, 0)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        rows.append({"sessionId": r["sessionId"], "cwd": r.get("cwd"),
                     "kind": r.get("kind"), "status": r.get("status"), "name": r.get("name")})
    return rows


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def transcript_path(session_id: str) -> Path | None:
    """The transcript of a session id or a unique prefix of one.

    Raises LookupError when a prefix matches several sessions.
    """
    hits = list(PROJECTS.glob(f"*/{glob.escape(session_id)}*.jsonl"))
    exact = [p for p in hits if p.stem == session_id]
    if exact:
        hits = exact
    stems = sorted({p.stem for p in hits})
    if len(stems) > 1:
        shown = ", ".join(s[:13] for s in stems[:5]) + (" …" if len(stems) > 5 else "")
        raise LookupError(f"'{session_id}' matches {len(stems)} sessions: {shown}")
    return max(hits, key=_mtime) if hits else None


def project_dir(directory: str) -> Path:
    return PROJECTS / re.sub(r"[^A-Za-z0-9]", "-", directory)


def _same_dir(a: str | None, b: str) -> bool:
    try:
        return bool(a) and Path(a).resolve() == Path(b).resolve()  # type: ignore[arg-type]
    except (OSError, ValueError):
        return False


Found = list[tuple[Path, "dict | None"]]


def _live_found(sessions: list[dict]) -> Found:
    found: Found = []
    for s in sessions:
        try:
            path = transcript_path(s["sessionId"])
        except LookupError:
            continue
        if path:
            found.append((path, s))
    return found


def find(target: str | None) -> Found:
    """Transcripts to show for a target, each with its live-session entry.

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
        d = str(Path(target).resolve())
        found = _live_found([s for s in live_sessions() if _same_dir(s["cwd"], d)])
        if found:
            return found
        proj = project_dir(d)
        pool = list(proj.glob("*.jsonl")) if proj.is_dir() else []
        if not pool:
            raise LookupError(f"no Claude Code sessions in {d}")
        return [(max(pool, key=_mtime), {"cwd": d, "status": NOT_RUNNING})]

    path = transcript_path(target)
    if not path:
        raise LookupError(f"no Claude Code session matching '{target}' (and no such directory)")
    live = next((s for s in live_sessions() if s["sessionId"] == path.stem), None)
    return [(path, live or {"status": NOT_RUNNING})]


# --- building the tree --------------------------------------------------------

def _agent_from(aid: str, label: str, t: Transcript, status: str | None) -> Agent:
    window = context_window(t.model) if t.model else None
    if window and t.context and t.context > window:
        window = 1_000_000  # a model running with the long-context window
    return Agent(
        id=aid, label=label, model=short_model(t.model), effort=t.effort,
        context_tokens=t.context, context_window=window,
        status=status, started=t.first, ended=None if status == RUNNING else t.last,
    )


def _status(aid: str, meta: dict, t: Transcript, mtime: float | None, now: float,
            notes: dict[str, tuple[float, str]], tool_results: set[str],
            session_live: bool) -> str:
    recent = mtime is not None and now - mtime < STALE_AFTER_SECS
    note = notes.get(aid)
    if note:
        note_ts, note_status = note
        resumed = recent and t.last is not None and t.last > note_ts + RESUMED_AFTER_SECS
        if not resumed:
            return note_status
    if meta.get("requestShape") != "background" and meta.get("toolUseId") in tool_results:
        return DONE
    if recent or (session_live and t.pending_tool):
        return RUNNING
    if session_live and (meta.get("taskKind") == "in_process_teammate" or meta.get("teamName")):
        return IDLE  # a teammate waiting for its next message
    return STALE


def load(path: Path, live: dict | None = None, now: float | None = None) -> Session:
    now = time.time() if now is None else now
    live = live or {}
    session_live = bool(live) and live.get("status") != NOT_RUNNING
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
                      read_transcript(jl), _mtime(jl) or None))

    # Completion markers live in whichever transcript spawned the agent.
    notes = dict(main_t.task_status)
    tool_results = set(main_t.tool_results)
    for _, _, t, _ in metas:
        for task_id, note in t.task_status.items():
            if task_id not in notes or notes[task_id][0] <= note[0]:
                notes[task_id] = note
        tool_results |= t.tool_results

    agents: dict[str, Agent] = {}
    parent_of: dict[str, str | None] = {}
    for aid, meta, t, mtime in metas:
        status = _status(aid, meta, t, mtime, now, notes, tool_results, session_live)
        agent_type = str(meta.get("agentType") or "?").rsplit(":", 1)[-1]
        label = f"{agent_type}: {meta.get('description') or aid}"
        agents[aid] = _agent_from(aid, label, t, status)
        tool_use = meta.get("toolUseId")
        parent = meta.get("parentAgentId")
        if not (isinstance(parent, str) and parent != aid):
            parent = next((pid for pid, _, pt, _ in metas
                           if pid != aid and tool_use and tool_use in pt.tool_uses), None)
        parent_of[aid] = parent

    roots = []
    for aid, agent in agents.items():
        pid = _parent_without_cycle(aid, parent_of, agents)
        (agents[pid].children if pid else roots).append(agent)
    for agent in agents.values():
        agent.children.sort(key=lambda a: a.started or 0)
    roots.sort(key=lambda a: a.started or 0)

    main = _agent_from("main", "main", main_t, live.get("status"))
    main.ended = main_t.last  # the main loop's span is its transcript's, live or not
    title = main_t.custom_title or main_t.title or live.get("name") or path.stem[:8]
    return Session(id=path.stem, provider=NAME, title=title, main=main, agents=roots,
                   cwd=live.get("cwd"), kind=live.get("kind"))


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
    now = time.time()
    return [load(path, live, now) for path, live in find(target)]
