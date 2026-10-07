# Demo: check agents-tree against a real run

Unit tests use fake transcripts. This is how to see agents-tree on a real agent
run without touching anyone's real work: a throwaway project, a real agent CLI
in it, a task that fans out into subagents, and agents-tree pointed at that
directory only. It is also how the screenshots in the README were made.

It costs real model usage (a few minutes of five subagents), so it is a manual
check, not part of CI.

## 1. Create the project

```bash
docs/demo/setup.sh            # creates /tmp/agents-tree-demo
```

"Acme Bookshop": a catalog, a cart, an inventory, three planted bugs, one test.
Its own git repository with a made-up author.

## 2. Run an agent in it

Use a separate terminal, or a separate herdr session so your own panes stay out
of the picture:

```bash
cd /tmp/agents-tree-demo && herdr --session demo     # optional
claude --permission-mode acceptEdits                  # or: codex --approve-for-me, grok, ...
```

Give the session a readable name if the CLI can (Claude Code: `/rename Acme
Bookshop`). Let it run shell commands without asking (it runs `uvx pytest`), or
answer its prompts, or the subagents sit "waiting for its result".

## 3. Give it the task

Paste the task from [task.md](task.md). With Claude Code, another session can
also deliver it: find the session with `ListAgents` and send the task with
`SendMessage`.

## 4. Watch it with agents-tree

```bash
agents-tree -w /tmp/agents-tree-demo                      # Claude Code
agents-tree -w --provider <name> /tmp/agents-tree-demo     # another provider
agents-tree -w --provider codex /tmp/agents-tree-demo      # Codex
```

In herdr, `prefix+a` on the agent's pane opens the same view. What to check
while the subagents run and after they finish:

- five subagents under `main`, and the nested one under the review agent;
- the models and efforts the task asked for;
- `running` while they work, then `completed` / `done`; the background one too;
- context tokens and % plausible, elapsed counting up while running;
- `Enter` on a row: the prompt it was given, its tool calls, the command it waits
  on, its last message;
- `r` hides finished agents; a terminal shorter than the tree scrolls and keeps
  the selection visible.

## 5. Screenshots (optional)

Only of the demo: nothing else may be on screen. Crop away the window title bar
(it shows the user and host name) and empty space, e.g. with
`ffmpeg -i in.png -vf "crop=W:H:X:Y" docs/tree.png`. Check the result before
committing it. The README links images by their raw GitHub URL so that PyPI
shows them too.

## 6. Clean up

```bash
herdr session stop demo && herdr session delete demo       # if you used one
rm -rf /tmp/agents-tree-demo
rm -rf ~/.claude/projects/-tmp-agents-tree-demo            # Claude Code's transcripts
```

Other CLIs keep their own session files for the run (Codex, for one, under
`~/.codex/sessions/`). Remove only the files whose `session_meta.cwd` is the
demo directory, or archive the demo session with Codex; do not remove the whole
sessions directory.
