# Working on agents-tree

agents-tree shows coding-agent sessions and their subagents as a tree. It reads
the files an agent CLI keeps on disk; it never talks to a running session. Claude
Code, Codex and Grok Build are supported. Read this before changing code.

## Layout

| Path | What it is |
| --- | --- |
| `src/agents_tree/providers/` | One module per agent CLI. Everything specific to a CLI lives here, nowhere else. |
| `src/agents_tree/model.py` | The neutral data every provider returns: `Session`, `Agent`, `Detail`, and the states. |
| `src/agents_tree/render.py` | Draws sessions as a tree and one agent as a detail page. Never changes its input. |
| `src/agents_tree/live.py` | The `-w` live view: keys, mouse, scrolling, a loader thread. |
| `src/agents_tree/cli.py` | Arguments, and `loader()`: provider → `sanitize()` → `--window` override. |
| `src/agents_tree/herdr.py`, `herdr/`, `herdr-plugin.toml` | The herdr plugin. It runs the source in `src/` directly, no install. |
| `tests/` | pytest. `conftest.py` builds fake transcript trees and neutral model objects. |
| `docs/demo/` | How to stage a throwaway demo project, run a real agent in it, and check agents-tree against it. |

`model.py`, `render.py` and `live.py` must stay provider-neutral: no CLI names,
no CLI-specific status words, no file layouts.

## Commands

Everything runs through [uv](https://docs.astral.sh/uv/); CI runs the same.

```bash
uv sync
uv run pytest
uv run pylint src
uv run pylint tests --disable=redefined-outer-name,protected-access,unused-argument
uv run pyright
PYTHONPATH=src python3 -m agents_tree --help   # what the herdr plugin runs
```

All four checks must pass before a commit: CI (`.github/workflows/test.yaml`)
fails otherwise. Python 3.10 is the floor, so no syntax or stdlib newer than
that (`tomllib`, for one, is 3.11). The package has no runtime dependencies;
keep it that way.

## Adding a provider (Codex, Grok, ...)

1. **Look at real files first.** Find where the CLI keeps its sessions and how a
   subagent shows up in them. Capture two or three real transcripts from a demo
   run (see `docs/demo/`), never from someone's real work, and build the parser
   and the test fixtures from those. Do not guess field names.
2. **Write `src/agents_tree/providers/<name>.py`** with
   `sessions(target: str | None) -> list[Session]`:
   - `target` is `None` (every running session of this CLI), an existing
     directory (sessions running there, else that directory's newest session),
     or a session id or unique prefix of one.
   - Raise `LookupError` with a message for the user when nothing matches.
   - Return raw text; `cli.loader()` sanitises it once for every provider.
3. **Fill the model** (`model.py`):
   - `Session.main` is the session's own loop, `Session.agents` the top-level
     subagents, nested ones in `Agent.children`.
   - `Agent.id` must be unique within its session; the live view keys rows by
     `(session id, agent id)`.
   - `Agent.state` is one of `RUNNING`, `WAITING`, `BLOCKED`, `DONE`, `FAILED`,
     `STALE`, `INACTIVE`; `Agent.status` is the CLI's own word for it, shown as is.
   - `Agent.ended` is `None` while the agent runs; elapsed time counts to now.
   - `context_window` is the provider's best knowledge; the user can override it.
   - `Agent.detail` feeds the detail page: prompt, tool counts, last tool and
     whether it waits on a result, last message, request and token counts,
     transcript path.
4. **Register it** in `src/agents_tree/providers/__init__.py` (`PROVIDERS`). The
   key is what `--provider` takes.
5. **herdr:** `herdr.target_for()` maps the agent herdr reports for a pane
   (`herdr pane get <id>` → `agent`, `agent_session`) to a provider key. If the
   CLI's herdr name differs from the key, map it there.
6. **Stay fast.** The live view reloads every 2 s on a thread. Read transcripts
   incrementally (the Claude provider keeps a byte offset per file) and keep only
   what the last refresh used.
7. **Tests:** a fake directory tree like `tests/conftest.py`'s `FakeClaude`, with
   records shaped like the real ones from step 1. Cover running, finished,
   failed, nested, a session that is not running, and malformed lines.
8. **Check it for real** with the demo in `docs/demo/`, and update README.md.

## Conventions

- Commits: Conventional Commits, one-line subject, imperative, lower case:
  `feat: …`, `fix: …`, `docs: …`, `ci: …`, `test: …`, `refactor: …`, `chore: …`.
- Code, comments and docs in English.
- Never put real session data in the repository: no transcripts, ids, titles or
  paths from someone's actual work, in tests, docs or screenshots. Use the demo.

## Releasing

1. Bump the version in `pyproject.toml`, `src/agents_tree/__init__.py` and
   `herdr-plugin.toml` (a test checks they match), then `uv lock`.
2. Commit, push, wait for CI.
3. Push a tag `vX.Y.Z`. `.github/workflows/publish.yaml` checks the tag against
   the version, tests, builds, and waits for a maintainer to approve the `pypi`
   environment before publishing.
