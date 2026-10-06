"""herdr plugin glue: resolve the focused agent pane's session and open the live tree.

`python3 -m agents_tree.herdr open <scope> [placement]`  (plugin action)
    Opens the tree pane over the focused pane, or closes it when it is the
    focused pane already, so one key toggles it.
`python3 -m agents_tree.herdr pane`                       (plugin pane command)
    Runs the live tree for the target the action resolved.
`python3 -m agents_tree.herdr keys setup|remove`          (plugin actions)
    Adds or removes a marked block of key bindings in herdr's config.toml.

scope "focused": the focused pane's Claude Code session (by the session id
herdr's Claude integration reports, else the pane's directory); every running
session when the pane holds no Claude agent. scope "all": every running session.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

TARGET_ENV = "AGENTS_TREE_TARGET"
PLUGIN_ID = "agents-tree"
PANE_TITLE = "agents-tree"
PLACEMENTS = ("overlay", "split", "tab", "zoomed")


def _herdr() -> str:
    return os.environ.get("HERDR_BIN_PATH") or "herdr"


def _run(*args: str) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run([_herdr(), *args], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None


def _herdr_json(*args: str) -> dict:
    try:
        out = subprocess.run([_herdr(), *args], capture_output=True, text=True,
                             timeout=10).stdout
        data = json.loads(out)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def focused_pane_id() -> str | None:
    """The pane focused when the plugin was invoked (HERDR_PANE_ID is our own pane)."""
    try:
        ctx = json.loads(os.environ.get("HERDR_PLUGIN_CONTEXT_JSON") or "{}")
    except json.JSONDecodeError:
        return None
    pane = ctx.get("focused_pane_id") if isinstance(ctx, dict) else None
    return pane if isinstance(pane, str) and pane else None


def pane_info(pane_id: str) -> dict:
    result = _herdr_json("pane", "get", pane_id).get("result") or {}
    pane = result.get("pane", result) if isinstance(result, dict) else {}
    return pane if isinstance(pane, dict) else {}


def target_for(pane: dict) -> str:
    """agents-tree target for a pane: a session id, a directory, or "" for all sessions."""
    session = pane.get("agent_session")
    if not isinstance(session, dict):
        session = {}
    agent = session.get("agent") or pane.get("agent")
    if agent != "claude":
        return ""
    if session.get("kind") == "id" and session.get("value"):
        return str(session["value"])
    return str(pane.get("foreground_cwd") or pane.get("cwd") or "")


def is_our_pane(pane: dict) -> bool:
    titles = {pane.get("label"), pane.get("title"), pane.get("terminal_title_stripped")}
    return PANE_TITLE in titles and not pane.get("agent")


def open_pane(scope: str, placement: str) -> int:
    plugin = os.environ.get("HERDR_PLUGIN_ID") or PLUGIN_ID
    focused = focused_pane_id()
    pane = pane_info(focused) if focused else {}
    if focused and is_our_pane(pane):
        result = _run("plugin", "pane", "close", focused)
    else:
        target = target_for(pane) if scope == "focused" else ""
        result = _run("plugin", "pane", "open", "--plugin", plugin, "--entrypoint", "tree",
                      "--placement", placement, "--env", f"{TARGET_ENV}={target}")
    if result is None:
        print("agents-tree: could not run herdr", file=sys.stderr)
        return 1
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode


def run_pane() -> int:
    """The pane's terminal is the only place an error is seen: hold it until enter."""
    import traceback

    from agents_tree.cli import main

    target = os.environ.get(TARGET_ENV, "")
    try:
        return main(["-w", *(["--", target] if target else [])])
    except KeyboardInterrupt:
        return 0
    except BaseException as e:  # noqa: BLE001 - anything, including SystemExit from argparse
        if isinstance(e, SystemExit) and not e.code:
            return 0
        traceback.print_exc()
        try:
            input("\n  press enter to close ")
        except (EOFError, KeyboardInterrupt):
            pass
        return 1


# --- key bindings -----------------------------------------------------------------

KEYS = [("prefix+a", "open", "agents-tree: live tree of the focused agent"),
        ("prefix+shift+a", "open-all", "agents-tree: live tree of every running session")]
ALTERNATES = [("open-split", "agents-tree: focused agent, in a split"),
              ("open-tab", "agents-tree: focused agent, in a tab")]
BEGIN = "# >>> agents-tree keys (managed by the agents-tree plugin: setup-keys / remove-keys)"
END = "# <<< agents-tree keys"


def config_path() -> Path:
    if "HERDR_CONFIG_PATH" in os.environ:
        return Path(os.environ["HERDR_CONFIG_PATH"])
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / "herdr" / "config.toml"


def strip_block(text: str) -> str:
    """The config without our block. Raises ValueError when a marker lacks its pair,
    rather than guess where the block ends and take the user's settings with it."""
    lines = text.splitlines()
    begins = [n for n, line in enumerate(lines) if line == BEGIN]
    ends = [n for n, line in enumerate(lines) if line == END]
    if not begins and not ends:
        return text.rstrip("\n")
    if len(begins) != 1 or len(ends) != 1 or ends[0] < begins[0]:
        raise ValueError(f"the agents-tree block markers ({BEGIN!r} ... {END!r}) are "
                         "incomplete; fix or remove them by hand")
    return "\n".join(lines[:begins[0]] + lines[ends[0] + 1:]).strip("\n")


def key_block(plugin: str, keys: list[tuple[str, str, str]]) -> str:
    lines = [BEGIN]
    for key, action, desc in keys:
        lines += ["[[keys.command]]", f'key = "{key}"', 'type = "plugin_action"',
                  f'command = "{plugin}.{action}"', f'description = "{desc}"', ""]
    lines.append("# Same tree in a split or a tab instead of over the pane: point a key here.")
    for action, desc in ALTERNATES:
        lines += ["# [[keys.command]]", '# key = "prefix+..."', '# type = "plugin_action"',
                  f'# command = "{plugin}.{action}"', f'# description = "{desc}"', ""]
    lines.append(END)
    return "\n".join(lines)


def _config_ok() -> bool:
    result = _run("config", "check")
    return result is not None and result.returncode == 0


def _notify(title: str, body: str) -> None:
    print(body)
    _run("notification", "show", title, "--body", body, "--sound", "none")


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.agents-tree-tmp")
    tmp.write_text(text)
    if path.exists():
        shutil.copymode(path, tmp)
    os.replace(tmp, path)


def _is_bound(config: str, key: str) -> bool:
    pattern = rf"""^\s*key\s*=\s*["']{re.escape(key)}["']"""
    return re.search(pattern, config, re.M | re.I) is not None


def _rewrite_config(path: Path, original: str, new: str) -> str | None:
    """Back up, write, check; restore on failure. Returns an error, or None."""
    if path.exists():
        shutil.copy2(path, path.with_name(path.name + ".agents-tree-backup"))
    _write_atomic(path, new)
    if not _config_ok():
        _write_atomic(path, original)
        return "the changed config failed `herdr config check`; restored it"
    _run("server", "reload-config")
    return None


def setup_keys() -> int:
    plugin = os.environ.get("HERDR_PLUGIN_ID") or PLUGIN_ID
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    original = path.read_text() if path.exists() else ""
    if original and not _config_ok():
        _notify("agents-tree", f"{path} does not pass `herdr config check`; nothing changed.")
        return 1
    try:
        rest = strip_block(original)
    except ValueError as e:
        _notify("agents-tree", f"{path}: {e}; nothing changed.")
        return 1
    taken = [k for k, _, _ in KEYS if _is_bound(rest, k)]
    free = [k for k in KEYS if k[0] not in taken]
    if not free:
        _notify("agents-tree", f"{', '.join(taken)} already bound in {path}; nothing changed.")
        return 1
    error = _rewrite_config(path, original,
                            (rest + "\n\n" if rest else "") + key_block(plugin, free) + "\n")
    if error:
        _notify("agents-tree", f"{path}: {error}.")
        return 1
    msg = f"Bound {', '.join(k for k, _, _ in free)} in {path}."
    if taken:
        msg += f" Left {', '.join(taken)} alone: already bound."
    _notify("agents-tree", msg)
    return 0


def remove_keys() -> int:
    path = config_path()
    original = path.read_text() if path.exists() else ""
    if BEGIN not in original and END not in original:
        _notify("agents-tree", f"No agents-tree keys in {path}.")
        return 0
    try:
        rest = strip_block(original)
    except ValueError as e:
        _notify("agents-tree", f"{path}: {e}; nothing changed.")
        return 1
    error = _rewrite_config(path, original, rest + "\n" if rest else "")
    if error:
        _notify("agents-tree", f"{path}: {error}.")
        return 1
    _notify("agents-tree", f"Removed the agents-tree keys from {path}.")
    return 0


def run(argv: list[str]) -> int:
    match argv:
        case ["open", ("focused" | "all") as scope]:
            return open_pane(scope, "overlay")
        case ["open", ("focused" | "all") as scope, placement] if placement in PLACEMENTS:
            return open_pane(scope, placement)
        case ["pane"]:
            return run_pane()
        case ["keys", "setup"]:
            return setup_keys()
        case ["keys", "remove"]:
            return remove_keys()
    print(f"usage: python3 -m agents_tree.herdr open focused|all [{'|'.join(PLACEMENTS)}]"
          " | pane | keys setup|remove", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
