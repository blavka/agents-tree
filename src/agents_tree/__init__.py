"""Show coding-agent sessions and their subagents as a live tree."""

__version__ = "0.1.0"


def main() -> int:
    from agents_tree.cli import main as cli_main

    return cli_main()
