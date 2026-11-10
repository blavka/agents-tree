"""Disk snapshots of a folded transcript, so a new process reads only the tail.

The providers already keep a byte offset in memory and parse only what was
appended. That offset dies with the process, so a one-shot run (and the first
paint of the live view) was re-reading whole transcripts. A snapshot is the
folded transcript plus how far into the file it covers.

Snapshots live in ``$AGENTS_TREE_CACHE`` (default ``~/.cache/agents-tree/transcripts``,
or under ``$XDG_CACHE_HOME``). They hold prompt excerpts. The directory is
created mode 0700 and a snapshot is refused unless it unpickles as the
transcript type the caller asked for. A damaged snapshot is deleted; the
transcript is parsed again and the next save rebuilds it. Nothing in here is
worth crashing over. Set ``AGENTS_TREE_CACHE`` to ``off`` to disable the cache.
"""

# Every boundary here catches Exception on purpose: a damaged cache is discarded.
# pylint: disable=broad-exception-caught

from __future__ import annotations

import hashlib
import io
import os
import pickle
import re
import stat
import struct
from pathlib import Path
from typing import Any, BinaryIO, Callable

_MAGIC = b"ATCACHE1"
_HEADER = struct.Struct("<QQI")  # inode, offset, path length; follows the magic
_MAX_PATH = 4096
_NAME = re.compile(r"^[0-9a-f]{64}$")
_OFF = {"0", "off", "none", "false"}
_MAX_DEPTH = 8

# What a snapshot is allowed to reconstruct. Anything else (os.system, a class
# added later) is a cache miss and the transcript is parsed again.
_ALLOWED = {
    ("agents_tree.providers.claude", "Transcript"),
    ("agents_tree.providers.grok", "Transcript"),
    ("agents_tree.providers.codex", "Transcript"),
    ("agents_tree.providers.codex", "Activity"),
    ("agents_tree.providers.antigravity", "Transcript"),
    ("agents_tree.providers.antigravity", "SubagentSpec"),
    ("builtins", "set"),
    ("builtins", "dict"),
    ("builtins", "list"),
    ("builtins", "tuple"),
    ("builtins", "frozenset"),
}


class _RootMemo:
    """The approved cache directory for the current env. Tests reset both fields."""

    key: str | None = None
    path: Path | None = None


class _Unpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in _ALLOWED:
            raise pickle.UnpicklingError(f"{module}.{name}")
        return super().find_class(module, name)


def _env_key() -> str:
    return f"{os.environ.get('AGENTS_TREE_CACHE')}\0{os.environ.get('XDG_CACHE_HOME')}"


def _configured() -> Path | None:
    raw = os.environ.get("AGENTS_TREE_CACHE")
    if raw is None:
        base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
        return Path(base) / "agents-tree" / "transcripts"
    if raw.strip().lower() in _OFF:
        return None
    return Path(raw)


def _open_root() -> Path | None:
    """The snapshot directory, created 0700, or None when it cannot be kept private."""
    path = _configured()
    if path is None:
        return None
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(path, 0o700)
        info = os.lstat(path)
    except OSError:
        return None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        return None
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        return None
    return path


def root() -> Path | None:
    """Approved snapshot directory, or None when the cache is off or unsafe."""
    key = _env_key()
    if _RootMemo.key == key:
        return _RootMemo.path
    try:
        opened = _open_root()
        if opened is not None:
            _drop_gone(opened)
    except Exception:
        opened = None  # a broken cache directory disables the cache for this process
    _RootMemo.key, _RootMemo.path = key, opened
    return opened


def _identity(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _cache_file(directory: Path, ident: str) -> Path:
    return directory / hashlib.sha256(ident.encode("utf-8", "surrogateescape")).hexdigest()


def _read_header(handle: BinaryIO) -> tuple[int, int, str] | None:
    try:
        head = handle.read(len(_MAGIC) + _HEADER.size)
        if len(head) != len(_MAGIC) + _HEADER.size or head[:len(_MAGIC)] != _MAGIC:
            return None
        inode, offset, path_len = _HEADER.unpack_from(head, len(_MAGIC))
        if path_len > _MAX_PATH:
            return None
        raw = handle.read(path_len)
        if len(raw) != path_len:
            return None
        return inode, offset, raw.decode("utf-8", "surrogateescape")
    except Exception:
        return None


def _with_file(cache_file: Path, read: Callable[[BinaryIO], Any]) -> Any:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(cache_file, flags)
    except OSError:
        return None
    try:
        with os.fdopen(descriptor, "rb") as handle:
            return read(handle)
    except Exception:
        return None


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def _source_missing(text: str) -> bool:
    """True when the transcript path is gone or the header path itself is garbage."""
    try:
        return not Path(text).exists()
    except Exception:
        return True


def _drop_gone(directory: Path) -> None:
    """Delete snapshots whose transcript is gone, and snapshots that cannot be read."""
    try:
        entries = list(directory.iterdir())
    except OSError:
        return
    for entry in entries:
        if not _NAME.match(entry.name):
            continue
        try:
            header = _with_file(entry, _read_header)
            if not isinstance(header, tuple) or _source_missing(header[2]):
                _unlink(entry)
        except Exception:
            _unlink(entry)


def _scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float))


def _value_ok(value: Any, seen: set[int], depth: int) -> bool:
    """A nested value from a folded transcript: scalars, containers, or a dataclass."""
    if depth > _MAX_DEPTH:
        return False
    if _scalar(value):
        return True
    if isinstance(value, tuple):
        return all(_value_ok(item, seen, depth + 1) for item in value)
    if isinstance(value, list):
        return all(_value_ok(item, seen, depth + 1) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, (str, int)) and _value_ok(item, seen, depth + 1)
                   for key, item in value.items())
    return _record_ok(value, seen, depth)


def _record_ok(obj: Any, seen: set[int], depth: int) -> bool:
    """True when a dataclass still has the sets, dicts and lists the parsers expect."""
    fields = getattr(type(obj), "__dataclass_fields__", None)
    if not isinstance(fields, dict) or depth > _MAX_DEPTH:
        return False
    marker = id(obj)
    if marker in seen:
        return True
    seen.add(marker)
    for name, info in fields.items():
        try:
            value = getattr(obj, name)
        except Exception:
            return False
        factory = info.default_factory
        if factory is set:
            ok = isinstance(value, set) and all(
                _value_ok(item, seen, depth + 1) for item in value)
        elif factory is dict:
            ok = (isinstance(value, dict) and all(
                isinstance(key, (str, int)) and _value_ok(item, seen, depth + 1)
                for key, item in value.items()))
        elif factory is list:
            ok = isinstance(value, list) and all(
                _value_ok(item, seen, depth + 1) for item in value)
        else:
            ok = _scalar(value)
        if not ok:
            return False
    return True


def _discard(path: Path) -> None:
    """Best-effort delete of one snapshot. Used when reading it raised."""
    directory = _RootMemo.path
    if directory is None:
        return
    try:
        _unlink(_cache_file(directory, _identity(path)))
    except Exception:
        pass


def load(path: Path, inode: int, size: int, kind: type[Any]) -> tuple[int, Any] | None:
    """(offset, transcript) when a snapshot still covers this file, else None.

    A snapshot that cannot be read, or that unpickles into the wrong shape, is
    deleted. The caller parses the transcript and the next save writes a new one.
    """
    try:
        return _load(path, inode, size, kind)
    except Exception:
        _discard(path)
        return None


def _load(path: Path, inode: int, size: int, kind: type[Any]) -> tuple[int, Any] | None:
    directory = root()
    if directory is None:
        return None
    ident = _identity(path)
    cache_file = _cache_file(directory, ident)

    def read(handle: BinaryIO) -> tuple[int, Any] | None:
        header = _read_header(handle)
        if header is None or header[0] != inode or header[1] > size or header[2] != ident:
            return None
        try:
            obj = _Unpickler(io.BytesIO(handle.read())).load()
        except Exception:
            return None
        if not isinstance(obj, kind) or not _record_ok(obj, set(), 0):
            return None
        return header[1], obj

    loaded = _with_file(cache_file, read)
    if loaded is None and cache_file.is_file():
        # Drop a snapshot we could not use. An inode mismatch belongs to another
        # version of this path; the next save replaces it, so leave it alone.
        header = _with_file(cache_file, _read_header)
        if header is None or (isinstance(header, tuple) and header[0] == inode):
            _unlink(cache_file)
    return loaded


def store(path: Path, inode: int, offset: int, transcript: object) -> None:
    """Write a snapshot. A failure leaves the old file or none; the next run parses."""
    try:
        _store(path, inode, offset, transcript)
    except Exception:
        pass


def _store(path: Path, inode: int, offset: int, transcript: object) -> None:
    directory = root()
    if directory is None or offset < 0:
        return
    ident = _identity(path).encode("utf-8", "surrogateescape")
    if len(ident) > _MAX_PATH:
        return
    payload = _MAGIC + _HEADER.pack(inode, offset, len(ident)) + ident
    payload += pickle.dumps(transcript, protocol=4)
    dest = _cache_file(directory, ident.decode("utf-8", "surrogateescape"))
    tmp = directory / f".{dest.name}.{os.getpid()}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(tmp, flags, 0o600)
    except OSError:
        return
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            os.fchmod(handle.fileno(), 0o600)
        os.replace(tmp, dest)
    except OSError:
        _unlink(tmp)


def resolve(path: Path, st: os.stat_result, hit: Any, kind: type[Any],
            wrap: Callable[[int, int, Any], Any]) -> Any:
    """The memory entry when it still covers the file, else a snapshot, else empty.

    ``hit`` is the in-memory entry (inode, offset, transcript) or None. ``wrap``
    builds a new one of that same type. Any failure falls back to an empty
    transcript so the file is parsed from the start.
    """
    try:
        if hit is not None and hit.inode == st.st_ino and st.st_size >= hit.offset:
            return hit
        loaded = load(path, st.st_ino, st.st_size, kind)
        if loaded is None:
            return wrap(st.st_ino, 0, kind())
        return wrap(st.st_ino, loaded[0], loaded[1])
    except Exception:
        return wrap(st.st_ino, 0, kind())


def save(path: Path, inode: int, before: int, offset: int, transcript: object) -> None:
    """Persist a transcript when this read advanced the offset."""
    try:
        if offset != before:
            store(path, inode, offset, transcript)
    except Exception:
        pass
