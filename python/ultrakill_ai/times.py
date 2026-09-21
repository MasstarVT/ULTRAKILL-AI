"""The AI's level-time leaderboard: `times.md` at the repo root.

`record` is pure (markdown in, markdown out), so it is tested without touching the real file, and
`record_file` applies it in place. It follows the rules in the file's closing HTML comment: the
generation history gets every recorded run, newest first, with the time change against the newest
earlier run on the same level; the leaderboard keeps one row per level, sorted by campaign order and
replaced by `beats_record` -- HARDEST DIFFICULTY FIRST, then fastest. Level cells use the short form
(`0-1`) and times are `mm:ss.mmm`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from ultrakill_ai.campaign import CAMPAIGN_LEVELS

DIFFICULTY_NAMES = ("Harmless", "Lenient", "Standard", "Violent", "Brutal")
# Every difficulty the game will let itself run. `PrefsManager`'s own validator rejects a stored difficulty
# above 4 -- it returns 4 for anything out of 0..4, on the READ path as well as the write path -- so Brutal is
# the hardest difficulty reachable in this build, and the mod clamps its override to the same -1..4 range
# (`EpisodeController.cs`). The engine does carry a difficulty 5 ("ULTRAKILL MUST DIE", `GameDifficulty.UKMD`)
# with real behaviour behind it, but nothing in the shipped game can select or store it: see
# docs/project-log.md, 2026-09-20.
HARDEST_DIFFICULTY = len(DIFFICULTY_NAMES) - 1  # 4, Brutal
UNKNOWN_DIFFICULTY = -1  # a row or a run that never said which difficulty it played: ranks below every real one
LEADERBOARD_HEADING = "## Leaderboard"
HISTORY_HEADING = "## Generation history"
EMPTY = "—"  # em dash: an empty cell; a row whose first cell is one is a placeholder
MIN_OFFICIAL_SECONDS = 1.0  # an official time at or under this never happened: see `valid_official_seconds`
_TIME = re.compile(r"^(\d+):(\d{1,2}(?:\.\d+)?)$")
_LEVEL_ORDER = {scene.removeprefix("Level "): i for i, scene in enumerate(CAMPAIGN_LEVELS)}


@dataclass
class TimeEntry:
    """One evaluated run, as it goes into both tables."""

    level: str  # scene name, e.g. "Level 0-1"
    seconds: float  # official level time: StatsManager.seconds when the real FinalPit stopped the timer
    rank: str  # "D".."S" or "P" ("" when the level reported no rank thresholds)
    generation: str  # run name and step count, e.g. "campaign_ppo@1.25M"
    difficulty: int  # 0 Harmless .. 4 Brutal
    date: str  # YYYY-MM-DD
    kills: int
    deaths: int
    notes: str = ""


def actions_note(deterministic: bool) -> str:
    """How a recorded run chose its actions, for the notes cell. THE wording, so every writer agrees.

    A row has to say this: the two modes are not the same policy. The action heads are deliberately held near
    7 nats of entropy, so argmax is a policy nobody trained -- measured 2026-09-20 on the promoted 0-1
    specialist, sampling completed 19 of 20 fresh runs and argmax 0 of 5. `post_times.py`'s own rows say
    "training episode (sampled actions)" for the same reason.
    """
    return "deterministic (argmax) actions" if deterministic else "sampled actions"


def valid_official_seconds(seconds) -> float | None:
    """THE predicate for "did the game really report an official time?": the time, or None.

    Every reader of an official level time shares this one test rather than comparing against 0 itself
    (2026-09-19). On 2026-09-19 a fresh-start completion on `spec_0-2_speed` was graded from a frame that
    arrived after the game's level stats had already reset -- a 4,120-decision episode with `seconds` 0.0 and
    `restarts` 3 -- and because that 0.0 was accepted everywhere a time is accepted it became the run's
    `best_time`, its `best_runs` file, and a `00:00.000` leaderboard row that nothing real could ever beat.

    Missing, non-numeric, non-finite and anything at or under `MIN_OFFICIAL_SECONDS` read as MISSING. The
    threshold is safe by a wide margin: the fastest human individual-level record in the whole first act is
    6.6 s (docs/il-records.md), so no real completion can land under a second.
    """
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value <= MIN_OFFICIAL_SECONDS:
        return None
    return value


def difficulty_rank(difficulty) -> int:
    """How hard a difficulty is, for comparison: the number itself, or `UNKNOWN_DIFFICULTY` when there is none.

    A run or a row that never recorded its difficulty ranks BELOW every real one, so a completion that does
    know what it played on replaces it. That is the safe direction: the alternative would let an unlabelled
    row sit above a Brutal one forever.
    """
    try:
        value = int(difficulty)
    except (TypeError, ValueError):
        return UNKNOWN_DIFFICULTY
    return value if 0 <= value else UNKNOWN_DIFFICULTY


def difficulty_name(difficulty) -> str:
    """4 -> "Brutal". Anything outside 0..4 keeps its own text, so an unexpected value is never mislabelled."""
    rank = difficulty_rank(difficulty)
    if 0 <= rank < len(DIFFICULTY_NAMES):
        return DIFFICULTY_NAMES[rank]
    return EMPTY if difficulty is None else str(difficulty)


def parse_difficulty(text) -> int:
    """The inverse of `difficulty_name`: "Brutal" -> 4, "4" -> 4, a dash or anything else -> UNKNOWN_DIFFICULTY."""
    if text is None:
        return UNKNOWN_DIFFICULTY
    cell = str(text).strip()
    for i, name in enumerate(DIFFICULTY_NAMES):
        if cell.casefold() == name.casefold():
            return i
    return difficulty_rank(cell)


def beats_record(new_difficulty, new_seconds, held_difficulty, held_seconds) -> bool:
    """THE leaderboard rule, shared by `record` and `post_times`, so the two can never disagree.

    A level's row is the best time on the HARDEST difficulty that has any valid completion for that level
    (the lead, 2026-09-20, switching training to Brutal):

    - a completion on a HARDER difficulty always replaces the row, EVEN WHEN IT IS SLOWER -- the rows are the
      claim "this is what the AI can do", and a Violent time is not a claim about Brutal;
    - within one difficulty the faster time wins, as it always did;
    - an EASIER difficulty never replaces a harder row, however fast it is.

    A held row whose own time is not one the game could have reported (`valid_official_seconds`) counts as no
    row at all, and a new entry without a usable time is never a record.
    """
    if valid_official_seconds(new_seconds) is None:
        return False
    if valid_official_seconds(held_seconds) is None:
        return True
    new_rank, held_rank = difficulty_rank(new_difficulty), difficulty_rank(held_difficulty)
    if new_rank != held_rank:
        return new_rank > held_rank
    return _ms(new_seconds) < _ms(held_seconds)


def format_time(seconds: float) -> str:
    """83.25 -> "01:23.250", in whole milliseconds."""
    ms = _ms(max(0.0, seconds))
    return f"{ms // 60000:02d}:{ms // 1000 % 60:02d}.{ms % 1000:03d}"


def parse_time(text: str) -> float | None:
    """The inverse of format_time: "01:23.250" -> 83.25, and None for anything else (such as the placeholder dash)."""
    m = _TIME.match(text.strip())
    return int(m.group(1)) * 60 + float(m.group(2)) if m else None


def format_delta(seconds: float) -> str:
    """A signed time change: -1.25 -> "-1.250s", 0.5 -> "+0.500s"."""
    return f"{seconds:+.3f}s"


def short_level(scene: str) -> str:
    """The level cell times.md uses: "Level 0-1" -> "0-1"."""
    return scene.removeprefix("Level ")


def record(markdown: str, entry: TimeEntry) -> str:
    """Returns `markdown` with `entry` added to both tables. Everything outside the table rows is kept as it is.

    Refuses an entry whose time the game never reported (`valid_official_seconds`): a row this file cannot
    hold is better than a row no real run can ever replace. A HELD row whose own time is invalid counts as no
    row at all, so one that slipped in before this guard existed is replaced by the next real time.
    """
    if valid_official_seconds(entry.seconds) is None:
        raise ValueError(f"refusing to record an official time of {entry.seconds!r} for {entry.level}: "
                         "the game never reported one (see times.valid_official_seconds)")
    lines = markdown.splitlines()
    level = short_level(entry.level)
    time_text = format_time(entry.seconds)
    rank = entry.rank or EMPTY
    difficulty = difficulty_name(entry.difficulty)

    # Leaderboard: one row per level, replaced by `beats_record` -- hardest difficulty first, then fastest.
    start, end = _table(lines, LEADERBOARD_HEADING)
    rows = _data_rows(lines[start:end])
    row = [level, time_text, rank, entry.generation, difficulty, entry.date, entry.notes]
    held = next((i for i, cells in enumerate(rows) if cells[0] == level), None)
    if held is None:
        rows.append(row)
    elif beats_record(entry.difficulty, entry.seconds,
                      parse_difficulty(rows[held][4] if len(rows[held]) > 4 else None),
                      parse_time(rows[held][1])):
        rows[held] = row
    rows.sort(key=lambda cells: _LEVEL_ORDER.get(cells[0], len(_LEVEL_ORDER)))
    lines[start:end] = [_format_row(cells) for cells in rows]

    # Generation history: every run, newest first, compared with the newest earlier run on the same level.
    start, end = _table(lines, HISTORY_HEADING)
    rows = _data_rows(lines[start:end])
    previous = next((valid_official_seconds(parse_time(cells[2]))
                     for cells in rows if len(cells) > 2 and cells[1] == level), None)
    delta = EMPTY if previous is None else format_delta((_ms(entry.seconds) - _ms(previous)) / 1000)
    rows.insert(0, [entry.generation, level, time_text, rank, str(entry.kills), str(entry.deaths), delta, entry.date, entry.notes])
    lines[start:end] = [_format_row(cells) for cells in rows]
    return "\n".join(lines) + ("\n" if markdown.endswith("\n") else "")


def record_file(path, entry: TimeEntry) -> None:
    """Adds `entry` to the times.md at `path`, in place."""
    path = Path(path)
    path.write_text(record(path.read_text(encoding="utf-8"), entry), encoding="utf-8")


def _ms(seconds: float) -> int:
    return round(seconds * 1000)


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _format_row(cells: list[str]) -> str:
    return "| " + " | ".join(cell.replace("|", "/") for cell in cells) + " |"


def _data_rows(lines: list[str]) -> list[list[str]]:
    """Table rows as cell lists, placeholder rows dropped."""
    rows = [_cells(line) for line in lines]
    return [cells for cells in rows if cells[0] != EMPTY]


def _table(lines: list[str], heading: str) -> tuple[int, int]:
    """[start, end) line range of the data rows (below the header and separator) of the table under `heading`."""
    if heading not in lines:
        raise ValueError(f"times.md has no {heading!r} section")
    i = lines.index(heading) + 1
    while i < len(lines) and not lines[i].startswith(("|", "#")):
        i += 1
    if i + 1 >= len(lines) or not lines[i].startswith("|"):
        raise ValueError(f"times.md has no table under {heading!r}")
    end = i + 2
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    return i + 2, end
