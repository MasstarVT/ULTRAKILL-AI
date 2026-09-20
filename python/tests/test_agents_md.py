"""AGENTS.md must always be the current CLAUDE.md. No game:  python tests/test_agents_md.py"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import sync_agents_md as sync  # noqa: E402


def test_agents_md_is_the_current_claude_md():
    """The one way this file stays updated: an edit to CLAUDE.md without a re-sync turns the suite red."""
    wanted = sync.render(sync.SOURCE.read_text(encoding="utf-8"))
    assert sync.current(sync.TARGET) == wanted, \
        "AGENTS.md is out of date: run `python scripts/sync_agents_md.py` and commit it with CLAUDE.md"


def test_the_note_sits_under_the_title_and_the_body_is_untouched():
    text = sync.render("# Title\n\nFirst line.\n\n## Section\n- a rule\n")
    lines = text.split("\n")
    assert lines[0] == "# Title" and lines[2].startswith("> **Generated from `CLAUDE.md`")
    assert text.endswith("First line.\n\n## Section\n- a rule\n")


def test_rendering_is_stable_and_ignores_windows_line_endings():
    unix = "# T\n\nbody\n"
    assert sync.render(unix) == sync.render(unix.replace("\n", "\r\n"))
    assert sync.render(unix) == sync.render(unix), "same input, same output: the check must not flap"


def test_a_file_with_no_heading_still_gets_the_note_first():
    assert sync.render("just text\n").startswith("\n> **Generated from `CLAUDE.md`") or \
        sync.render("just text\n").lstrip("\n").startswith("> **Generated from `CLAUDE.md`")


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
