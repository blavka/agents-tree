"""Provider-neutral JSON output for scripts and other machine consumers."""

from __future__ import annotations

import json

from agents_tree.model import Agent, Detail, Session

SCHEMA_VERSION = 1


def _detail(detail: Detail | None) -> dict | None:
    if detail is None:
        return None
    return {
        "prompt": detail.prompt,
        "prompt_label": detail.prompt_label,
        "tools": detail.tools,
        "last_tool": detail.last_tool,
        "last_tool_at": detail.last_tool_at,
        "last_tool_pending": detail.last_tool_pending,
        "last_text": detail.last_text,
        "output_tokens": detail.output_tokens,
        "requests": detail.requests,
        "transcript": detail.transcript,
    }


def _agent(agent: Agent, running_only: bool) -> dict:
    children = agent.children
    if running_only:
        children = [child for child in children if child.has_running()]
    return {
        "id": agent.id,
        "label": agent.label,
        "state": agent.state,
        "status": agent.status,
        "model": agent.model,
        "effort": agent.effort,
        "context_tokens": agent.context_tokens,
        "context_window": agent.context_window,
        "started": agent.started,
        "ended": agent.ended,
        "children": [_agent(child, running_only) for child in children],
        "detail": _detail(agent.detail),
    }


def data(sessions: list[Session], *, running_only: bool = False) -> dict:
    """The versioned JSON-compatible representation of sessions."""
    return {
        "schema_version": SCHEMA_VERSION,
        "sessions": [{
            "id": session.id,
            "title": session.title,
            "cwd": session.cwd,
            "kind": session.kind,
            "main": _agent(session.main, running_only),
            "agents": [_agent(agent, running_only) for agent in session.agents
                       if not running_only or agent.has_running()],
        } for session in sessions],
    }


def dumps(sessions: list[Session], *, running_only: bool = False) -> str:
    """One stable, human-readable JSON document."""
    return json.dumps(data(sessions, running_only=running_only), ensure_ascii=False,
                      indent=2, sort_keys=True)
