"""herdr plugin glue: resolve the focused agent pane's session and open the live tree.

`python3 -m agents_tree.herdr open <scope> [placement]`  (plugin action)
    Opens the tree pane over the focused pane, or closes it when it is the
    focused pane already, so one key toggles it.
`python3 -m agents_tree.herdr pane`                       (plugin pane command)
    Runs the live tree for the target the action resolved.
`python3 -m agents_tree.herdr keys setup|remove`          (plugin actions)
    Adds or removes a marked block of key bindings in herdr's config.toml.

scope "focused": the focused pane's session (by the session id herdr's agent
integration reports, else the pane's directory) when agents-tree reads that
agent; every running session otherwise. scope "all": every running session.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
import traceback
from collections.abc import Callable
from pathlib import Path

from agents_tree import cli
from agents_tree.providers import ALL, PROVIDERS

TARGET_ENV = "AGENTS_TREE_TARGET"
PROVIDER_ENV = "AGENTS_TREE_PROVIDER"
PLUGIN_ID = "agents-tree"
DEFAULT_PROVIDER = ALL
PLACEMENTS = ("overlay", "split", "tab", "zoomed")


def _herdr() -> str:
    return os.environ.get("HERDR_BIN_PATH") or "herdr"


def _plugin_id() -> str:
    return os.environ.get("HERDR_PLUGIN_ID") or PLUGIN_ID


def _run(*args: str) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run([_herdr(), *args], capture_output=True, text=True, timeout=10,
                              check=False)
    except (OSError, subprocess.SubprocessError):
        return None


def _json_dict(text: str | None) -> dict:
    try:
        data = json.loads(text or "{}")
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def focused_pane_id() -> str | None:
    """The pane focused when the plugin was invoked (HERDR_PANE_ID is our own pane)."""
    pane = _json_dict(os.environ.get("HERDR_PLUGIN_CONTEXT_JSON")).get("focused_pane_id")
    return pane if isinstance(pane, str) and pane else None


def pane_info(pane_id: str) -> dict:
    result = _run("pane", "get", pane_id)
    info = _json_dict(result.stdout if result else None).get("result") or {}
    pane = info.get("pane", info) if isinstance(info, dict) else {}
    return pane if isinstance(pane, dict) else {}


def target_for(pane: dict) -> tuple[str, str]:
    """(provider, target) for a pane: its session id or directory when agents-tree
    reads its agent; every running session of every provider otherwise."""
    session = pane.get("agent_session")
    if not isinstance(session, dict):
        session = {}
    agent = session.get("agent") or pane.get("agent")
    if agent not in PROVIDERS:
        return DEFAULT_PROVIDER, ""
    if session.get("kind") == "id" and session.get("value"):
        return str(agent), str(session["value"])
    return str(agent), str(pane.get("foreground_cwd") or pane.get("cwd") or "")


# --- our own panes ------------------------------------------------------------------
# A tree pane records itself (its pane id and pid) while it runs, so the toggle key
# recognises it by identity rather than by a title another pane could share.

def _marks_dir() -> Path:
    """A directory only this user can write: herdr's state dir for the plugin, else
    the user's runtime or cache directory, never a shared one like /tmp."""
    base = (os.environ.get("HERDR_PLUGIN_STATE_DIR") or os.environ.get("XDG_RUNTIME_DIR")
            or Path.home() / ".cache")
    return Path(base) / "agents-tree-panes"


def _private_marks_dir() -> Path:
    """The marks directory, created 0700; refuses one that is a symlink or not ours."""
    d = _marks_dir()
    d.mkdir(mode=0o700, parents=True, exist_ok=True)
    st = os.lstat(d)
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid():
        raise PermissionError(f"{d} is not a private directory of this user")
    return d


def _mark_path(pane_id: str) -> Path:
    return _marks_dir() / re.sub(r"[^A-Za-z0-9_.-]", "_", pane_id)


def is_our_pane(pane_id: str) -> bool:
    try:
        pid = int(_mark_path(pane_id).read_text(encoding="utf-8"))
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def open_pane(scope: str, placement: str) -> int:
    focused = focused_pane_id()
    if focused and is_our_pane(focused):
        result = _run("plugin", "pane", "close", focused)
    else:
        provider, target = (target_for(pane_info(focused)) if focused and scope == "focused"
                            else (DEFAULT_PROVIDER, ""))
        result = _run("plugin", "pane", "open", "--plugin", _plugin_id(), "--entrypoint", "tree",
                      "--placement", placement, "--env", f"{PROVIDER_ENV}={provider}",
                      "--env", f"{TARGET_ENV}={target}")
    if result is None:
        print("agents-tree: could not run herdr", file=sys.stderr)
        return 1
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode


def run_pane() -> int:
    """The pane's terminal is the only place an error is seen: hold it until enter."""
    mark = _mark_path(os.environ["HERDR_PANE_ID"]) if os.environ.get("HERDR_PANE_ID") else None
    if mark:
        try:
            _private_marks_dir()
            fd = os.open(mark, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(str(os.getpid()))
        except OSError:
            mark = None  # the tree still runs; the toggle key just opens a second one
    target = os.environ.get(TARGET_ENV, "")
    args = ["-w", "--provider", os.environ.get(PROVIDER_ENV) or DEFAULT_PROVIDER]
    try:
        return cli.main([*args, *(["--", target] if target else [])])
    except KeyboardInterrupt:
        return 0
    except SystemExit as e:  # argparse rejecting the target
        if not e.code:
            return 0
        _hold_error()
        return 1
    except Exception:  # pylint: disable=broad-exception-caught  # shown, not lost
        _hold_error()
        return 1
    finally:
        if mark:
            mark.unlink(missing_ok=True)


def _hold_error() -> None:
    traceback.print_exc()
    try:
        input("\n  press enter to close ")
    except (EOFError, KeyboardInterrupt):
        pass


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


def _notify(body: str) -> None:
    print(body)
    _run("notification", "show", "agents-tree", "--body", body, "--sound", "none")


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.agents-tree-tmp")
    tmp.write_text(text, encoding="utf-8")
    if path.exists():
        shutil.copymode(path, tmp)
    os.replace(tmp, path)


def _is_bound(config: str, key: str) -> bool:
    pattern = rf"""^\s*key\s*=\s*["']{re.escape(key)}["']"""
    return re.search(pattern, config, re.M | re.I) is not None


def _edit_config(build: Callable[[str], str]) -> int:
    """Rewrite the herdr config through build(config_without_our_block), which
    returns the new text or raises ValueError with the reason to change nothing.
    Backs up, checks the result with `herdr config check`, restores on failure."""
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    original = path.read_text(encoding="utf-8") if path.exists() else ""
    try:
        if original and not _config_ok():
            raise ValueError("it does not pass `herdr config check`")
        new = build(strip_block(original))
    except ValueError as e:
        _notify(f"{path}: {e}; nothing changed.")
        return 1
    if path.exists():
        shutil.copy2(path, path.with_name(path.name + ".agents-tree-backup"))
    _write_atomic(path, new)
    if not _config_ok():
        _write_atomic(path, original)
        _notify(f"{path}: the changed config failed `herdr config check`; restored it.")
        return 1
    _run("server", "reload-config")
    return 0


def setup_keys() -> int:
    taken: list[str] = []
    free: list[tuple[str, str, str]] = []

    def build(rest: str) -> str:
        taken.extend(k for k, _, _ in KEYS if _is_bound(rest, k))
        free.extend(k for k in KEYS if k[0] not in taken)
        if not free:
            raise ValueError(f"{', '.join(taken)} already bound")
        return (rest + "\n\n" if rest else "") + key_block(_plugin_id(), free) + "\n"

    if _edit_config(build):
        return 1
    msg = f"Bound {', '.join(k for k, _, _ in free)} in {config_path()}."
    if taken:
        msg += f" Left {', '.join(taken)} alone: already bound."
    _notify(msg)
    return 0


def remove_keys() -> int:
    def build(rest: str) -> str:
        return rest + "\n" if rest else ""

    path = config_path()
    original = path.read_text(encoding="utf-8") if path.exists() else ""
    if BEGIN not in original and END not in original:
        _notify(f"No agents-tree keys in {config_path()}.")
        return 0
    if _edit_config(build):
        return 1
    _notify(f"Removed the agents-tree keys from {config_path()}.")
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
