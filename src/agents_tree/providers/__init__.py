"""Agent CLIs agents-tree can read. Claude Code and Grok Build today; Codex later."""

from agents_tree.providers import claude, grok

PROVIDERS = {"claude": claude, "grok": grok}
ALL = "all"  # every provider at once: what --provider and the herdr plugin default to
