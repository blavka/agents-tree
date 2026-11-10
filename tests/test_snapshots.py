"""Disk snapshots of folded transcripts."""

import os

import pytest

from conftest import assistant

from agents_tree import snapshots
from agents_tree.providers import antigravity, claude, codex, grok


def test_cache_directory_is_private(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text("{}\n")
    snapshots.store(path, path.stat().st_ino, path.stat().st_size, claude.Transcript())
    directory = snapshots.root()
    assert directory is not None
    assert directory.stat().st_mode & 0o077 == 0


def test_off_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTS_TREE_CACHE", "off")
    monkeypatch.setattr(snapshots._RootMemo, "key", None)
    monkeypatch.setattr(snapshots._RootMemo, "path", None)
    path = tmp_path / "t.jsonl"
    path.write_text("{}\n")
    snapshots.store(path, path.stat().st_ino, 1, claude.Transcript())
    assert snapshots.root() is None
    assert snapshots.load(path, path.stat().st_ino, path.stat().st_size, claude.Transcript) is None


def test_a_snapshot_for_another_inode_is_ignored(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text("{}\n")
    st = path.stat()
    kept = claude.Transcript()
    kept.context = 50
    snapshots.store(path, st.st_ino + 1, st.st_size, kept)
    assert snapshots.load(path, st.st_ino, st.st_size, claude.Transcript) is None


def test_a_malicious_snapshot_is_not_unpickled(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text("{}\n")
    marker = tmp_path / "pwned"

    class Bad:
        def __reduce__(self):
            return (os.system, (f"touch {marker}",))

    snapshots.store(path, path.stat().st_ino, 1, Bad())
    assert snapshots.load(path, path.stat().st_ino, path.stat().st_size, claude.Transcript) is None
    assert not marker.exists()


def test_nested_objects_roundtrip(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text("{}\n")
    st = path.stat()
    codex_t = codex.Transcript()
    codex_t.activities["a"] = codex.Activity("a", "agent")
    snapshots.store(path, st.st_ino, st.st_size, codex_t)
    loaded = snapshots.load(path, st.st_ino, st.st_size, codex.Transcript)
    assert loaded is not None
    assert loaded[1].activities["a"].path == "agent"

    agy = antigravity.Transcript()
    agy.subagents.append(antigravity.SubagentSpec("c1", role="review"))
    snapshots.store(path, st.st_ino, st.st_size, agy)
    loaded = snapshots.load(path, st.st_ino, st.st_size, antigravity.Transcript)
    assert loaded is not None
    assert loaded[1].subagents[0].role == "review"


def test_a_damaged_snapshot_is_rebuilt_from_the_transcript(fake):
    path = fake.session("s1", "/w/app", [assistant(0, context=1000)])
    assert claude.load(path).main.context_tokens == 1000
    directory = snapshots.root()
    assert directory is not None
    for entry in directory.iterdir():
        entry.write_bytes(b"this is not a snapshot")

    claude._cache.clear()
    assert claude.load(path).main.context_tokens == 1000

    claude._cache.clear()
    st = path.stat()
    loaded = snapshots.load(path, st.st_ino, st.st_size, claude.Transcript)
    assert loaded is not None
    assert loaded[1].context == 1000


def test_a_snapshot_of_the_wrong_shape_is_not_used(fake):
    path = fake.session("s1", "/w/app", [assistant(0, context=1000)])
    claude.load(path)
    broken = claude.Transcript()
    broken.tool_uses = ["not-a-set"]  # type: ignore[assignment]
    st = path.stat()
    snapshots.store(path, st.st_ino, st.st_size, broken)
    claude._cache.clear()

    assert claude.load(path).main.context_tokens == 1000
    claude._cache.clear()
    rebuilt = snapshots.load(path, path.stat().st_ino, path.stat().st_size, claude.Transcript)
    assert rebuilt is not None
    assert rebuilt[1].context == 1000
    assert isinstance(rebuilt[1].tool_uses, set)


def test_an_unexpected_read_error_discards_the_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "t.jsonl"
    path.write_text("{}\n")
    st = path.stat()
    snapshots.store(path, st.st_ino, st.st_size, claude.Transcript())

    def boom(self):
        raise IndexError("damaged")

    monkeypatch.setattr(snapshots._Unpickler, "load", boom)
    assert snapshots.load(path, st.st_ino, st.st_size, claude.Transcript) is None
    directory = snapshots.root()
    assert directory is not None
    assert not any(entry.is_file() for entry in directory.iterdir())


def test_a_header_with_a_broken_path_is_removed(tmp_path):
    directory = snapshots.root()
    assert directory is not None
    raw = b"a\0b"
    name = "ab" * 32
    (directory / name).write_bytes(snapshots._MAGIC + snapshots._HEADER.pack(1, 0, len(raw)) + raw)
    snapshots._drop_gone(directory)
    assert not (directory / name).exists()


def test_a_snapshot_of_a_deleted_transcript_is_removed(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text("{}\n")
    snapshots.store(path, path.stat().st_ino, path.stat().st_size, claude.Transcript())
    directory = snapshots.root()
    assert directory is not None
    assert any(directory.iterdir())
    path.unlink()
    snapshots._drop_gone(directory)
    assert not list(directory.iterdir())


@pytest.mark.parametrize("mod, fn", [
    (claude, "read_transcript"),
    (grok, "read_updates"),
    (codex, "read_transcript"),
    (antigravity, "read_transcript"),
])
def test_a_restart_does_not_reparse_and_an_append_reads_one_line(tmp_path, monkeypatch, mod, fn):
    path = tmp_path / "t.jsonl"
    path.write_text('{"x": 1}\n', encoding="utf-8")
    read = getattr(mod, fn)
    read(path)

    calls = {"n": 0}
    real = mod._read_line

    def spy(transcript, line):
        calls["n"] += 1
        real(transcript, line)

    monkeypatch.setattr(mod, "_read_line", spy)
    mod._cache.clear()
    read(path)
    assert calls["n"] == 0

    path.write_text(path.read_text(encoding="utf-8") + '{"x": 2}\n', encoding="utf-8")
    mod._cache.clear()
    read(path)
    assert calls["n"] == 1
