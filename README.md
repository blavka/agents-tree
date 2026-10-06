# agents-tree

See what your coding agents are doing. `agents-tree` shows every running Claude
Code session and the subagents it spawned as a tree, with each agent's model and
effort, how full its context window is, how long it has been running, and its
status.

```
Split protocol modules  [75d94683]  /home/me/src/core  background
main                                            opus-5-5 (xhigh)        742k  74 % of 1M     42h00m  working
├─ Explore: Study dskit modules and services    opus-5-5 (high)          53k   5 % of 1M      2m46s  completed
└─ qp-builder: Builder: query list parsing fix  opus-5-5 (xhigh)        505k  50 % of 1M     20h16m  running

Review flaky test  [b443f8f0]  /home/me/src/api  interactive
main                                            opus-5-5 (high)         228k  23 % of 1M     41m19s  busy
└─ claude-code-guide: Look up hook events       haiku-4-5                61k  31 % of 200k    2m34s  completed
```

It reads the transcripts Claude Code keeps on disk and never talks to a session,
so nothing it prints ends up in an agent's context. It has no dependencies
beyond Python 3.10.

Claude Code is supported today; Codex and Grok are planned.

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
agents-tree b443f8f0        # one session, by id or a unique prefix (an ambiguous one lists the matches)
agents-tree -r              # only running subagents and their parents
agents-tree -w              # live view like top, refreshed every 2 s (-n 5 for 5 s)
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

| Column | Meaning |
| --- | --- |
| label | `agent-type: description` as the spawning agent gave it; `main` for the session itself |
| model (effort) | model of the agent's latest request, and its effort level (models without effort show none) |
| context | tokens in the latest request (input + cache + output), and that as a share of the context window |
| elapsed | first to last transcript entry; up to now while the agent runs |
| status | subagents: `running` (active in the last 2 minutes, or waiting on a tool call), `completed` / `failed` (as Claude Code reported), `done` (finished, no outcome recorded), `idle` (a teammate waiting for its next message), `stale` (no sign of life and no completion marker); `main`: the session's own state, or `not running` |

### Caveats

- **Context window is inferred.** Claude Code does not record it, so it is taken
  from the model family: 1M for Claude 5 models (Opus, Sonnet, Fable), 200k
  otherwise. Override with `--window 200000` or `AGENTS_TREE_WINDOW`.
- **Transcript format is internal to Claude Code** and may change between
  releases. Unknown lines are skipped, but a field that moves makes its column
  show `-` until agents-tree is updated.
- **Elapsed** spans wall-clock time, so a session resumed over several days, or a
  subagent woken up again by a message, shows that whole span.
- Running sessions come from `claude agents --json`; without `claude` on `PATH`,
  agents-tree falls back to `~/.claude/sessions/` (interactive sessions only).

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
```

Providers live in `src/agents_tree/providers/`: each turns one agent CLI's files
into the provider-neutral `Session`/`Agent` model in `model.py`, which
`render.py` draws.

## License

MIT
