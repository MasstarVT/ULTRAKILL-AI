"""check_run.py's pure helpers. No game:  python tests/test_check_run.py"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_run  # noqa: E402


def test_parse_pids_keeps_only_whole_numbers():
    assert check_run.parse_pids("14240\r\n31000\r\n") == [14240, 31000]
    assert check_run.parse_pids("") == []
    assert check_run.parse_pids("garbage\n 12 \nx7") == [12]


def test_driver_query_is_read_only_and_names_the_driver():
    """The watch may run this unattended: it must only READ process metadata (no Stop-Process, no kill)."""
    q = check_run.DRIVER_QUERY
    assert "Get-CimInstance" in q and "campaign_driver" in q and "ProcessId" in q
    for forbidden in ("Stop-Process", "taskkill", "Remove-Item", "Invoke-"):
        assert forbidden not in q


def test_tail_rows_reads_only_the_tail_and_drops_the_torn_first_line():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "episodes.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for i in range(200):
                fh.write(json.dumps({"i": i, "pad": "x" * 100}) + "\n")
        rows = check_run.tail_rows(path, max_bytes=2_000)
        assert 0 < len(rows) < 200
        assert rows[-1]["i"] == 199
        assert [r["i"] for r in rows] == list(range(rows[0]["i"], 200)), "contiguous, nothing torn"
        assert check_run.tail_rows(path / "missing") == []


def test_age_of_last_stamp_reads_the_newest_stamped_line():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "driver.log"
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - 120))
        path.write_text("junk line\n%s something happened\nunstamped trailer\n" % stamp, encoding="utf-8")
        age = check_run.age_of_last_stamp(path)
        assert age is not None and 100 < age < 200
        assert check_run.age_of_last_stamp(path / "missing") is None


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} tests passed")
