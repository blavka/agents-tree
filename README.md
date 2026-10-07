# agents-tree

See what your coding agents are doing. `agents-tree` shows every running Claude
Code or Grok Build session and the subagents it spawned as a tree, with each
agent's model and effort, how full its context window is, how long it has been
running, and its status.

In a terminal, `agents-tree .`:

![agents-tree run in a demo project: a session and its five subagents, one nested](https://raw.githubusercontent.com/blavka/agents-tree/main/docs/cli.png)

As a live overlay in [herdr](https://herdr.dev), one key away (`prefix+a`):

![The same tree as a live overlay in herdr](https://raw.githubusercontent.com/blavka/agents-tree/main/docs/tree.png)

It reads the transcripts Claude Code and Grok Build keep on disk and never talks
to a session, so nothing it prints ends up in an agent's context. It has no
dependencies beyond Python 3.10.

Claude Code and Grok Build are supported today; Codex is planned.

## Install

```bash
uv tool install agents-tree      # or: pipx install agents-tree
```

From a checkout: `uv tool install .`, or run it in place with
`PYTHONPATH=src python3 -m agents_tree`.

## Use

```bash
agents-tree                 # every running session on this machine
agents-tree .               # sessions running in this directory (or its newest session)
agents-tree ~/src/api       # same, for another directory
agents-tree 1a2b3c4d        # one session, by id or a unique prefix (an ambiguous one lists the matches)
agents-tree -r              # only running subagents and their parents
agents-tree -w              # live view like top, refreshed every 2 s (-n 5 for 5 s)
agents-tree --provider grok # Grok Build sessions under ~/.grok (or $GROK_HOME)
```

In the live view:

| Key | Does |
| --- | --- |
| `↑` `↓` / `j` `k` / wheel | select an agent |
| `Enter` / `→` / click | open its detail: the prompt it was given, its tool calls (and the one it is waiting on), its last message, request and token counts, transcript path |
| `Esc` / `←` / `q` | back to the tree (in the tree: clear the selection, then quit) |
| `PgUp` `PgDn` `Home` `End` | scroll the detail, or jump through the tree |
| `r` | toggle running-only |
| `Ctrl+C` | quit |

The live view takes the mouse for clicks and the wheel; most terminals still
select text with Shift held, or start it with `--no-mouse`. When the tree is
taller than the terminal, the oldest finished agents are folded into a
"… N older finished agents hidden" line.

`Enter` on a row opens that agent's detail:

![Detail of a running subagent: its prompt, tool calls and the command it waits on](https://raw.githubusercontent.com/blavka/agents-tree/main/docs/detail.png)

| Column | Meaning |
| --- | --- |
| label | `agent-type: description` as the spawning agent gave it; `main` for the session itself |
| model (effort) | model of the agent's latest request, and its effort level (models without effort show none) |
| context | tokens in the latest request (input + cache + output), and that as a share of the context window |
| elapsed | first to last transcript entry; up to now while the agent runs |
| status | subagents: `running` (active in the last 2 minutes, or waiting on a tool call), `completed` / `failed` (as Claude Code reported), `done` (finished, no outcome recorded), `idle` (a teammate waiting for its next message), `stale` (no sign of life and no completion marker); `main`: the session's own state, or `not running` |

### Caveats

- **Context window.** Claude Code does not record it, so it is taken from the
  model family: 1M for Claude 5 models (Opus, Sonnet, Fable), 200k otherwise.
  Grok Build stores it in `summary.json` when set; otherwise 2M for grok-4 /
  grok-code, 128k for other models. Override with `--window 200000` or
  `AGENTS_TREE_WINDOW`.
- **Transcript formats are internal** to each agent CLI and may change between
  releases. Unknown lines are skipped, but a field that moves makes its column
  show `-` until agents-tree is updated.
- **Elapsed** spans wall-clock time, so a session resumed over several days, or a
  subagent woken up again by a message, shows that whole span.
- **Running sessions (Claude)** come from `claude agents --json`; without
  `claude` on `PATH`, agents-tree falls back to `~/.claude/sessions/`
  (interactive sessions only).
- **Running sessions (Grok)** are inferred from recent `updates.jsonl` activity
  under `~/.grok/sessions` (or `$GROK_HOME/sessions`), or from a subagent whose
  `meta.json` still says `running`.

## herdr plugin

[herdr](https://herdr.dev) shows your agents, not their subagents. The plugin in
this repository opens the live tree for the focused agent pane over that pane;
closing it gives the layout back:

```bash
herdr integration install claude                 # herdr learns each pane's session id
herdr plugin install blavka/agents-tree
herdr plugin action invoke agents-tree.setup-keys
```

| Key | Action |
| --- | --- |
| `prefix+a` | live tree of the focused agent's session; `q` or the same key closes it |
| `prefix+shift+a` | live tree of every running session |

On a pane without an agent, `prefix+a` shows every running session. The plugin
runs the copy of agents-tree bundled in the repository, so it needs only
`python3` 3.10+ (or `AGENTS_TREE_PYTHON` pointing at one). `setup-keys` writes
one marked block into your herdr `config.toml` (backed up first) and leaves keys
you already use alone; `agents-tree.remove-keys` takes it out again. The
`open-split` and `open-tab` actions show the same tree in a split or a tab.

For development, link a checkout instead: `herdr plugin link /path/to/agents-tree`.

## Development

```bash
uv sync
uv run pytest
uv run pylint src
uv run pylint tests --disable=redefined-outer-name,protected-access,unused-argument
uv run pyright
```

CI (`.github/workflows/test.yaml`) runs the same on every push and pull request:
lint and types on Python 3.12, then the tests, a test run and a package build on
3.10 to 3.14.

To release, bump `version` in `pyproject.toml`, `src/agents_tree/__init__.py` and
`herdr-plugin.toml`, then push a matching tag (`git tag v0.2.0 && git push origin
v0.2.0`). `.github/workflows/publish.yaml` checks the tag against the version,
tests, builds and publishes to PyPI through trusted publishing.

[AGENTS.md](AGENTS.md) is the guide for working on the code, including adding a
provider; [docs/demo/](docs/demo/README.md) shows how to check agents-tree
against a real agent run in a throwaway project.

Providers live in `src/agents_tree/providers/`: each turns one agent CLI's files
into the provider-neutral `Session`/`Agent` model in `model.py`, which
`render.py` draws.

## License

MIT
