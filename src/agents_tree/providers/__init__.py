"""Agent CLIs agents-tree can read."""

from agents_tree.providers import antigravity, claude, codex, grok

PROVIDERS = {"agy": antigravity, "claude": claude, "codex": codex, "grok": grok}
ALL = "all"  # every provider at once: what --provider and the herdr plugin default to
