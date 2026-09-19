"""What the specialist driver is doing, in one screen. Read-only: it opens files, never a port.

    python scripts/specialists_status.py
    python scripts/specialists_status.py --json        # the same thing, machine readable

Prints the current stage (level, run, steps into it, how far off the stage rule it is), the live fresh
completion rate and window, the MEDIAN official time over the last 50 fresh episodes (what a speed stage
promotes on) beside the best one ever recorded (reported only), the hold line and what it is waiting on, and
the table of specialists already promoted to `models/specialists/`. Safe to run beside the driver and beside a
trainer: it reads `driver_state.json`, `status.json`, `best.json` and the promoted sidecars, and nothing else.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from campaign_driver import (  # noqa: E402
    COMPLETE, DRIVER_RUN, SPECIALIST_DIR, SPEED, DriverState, load_plan, read_sample, specialist_path,
    stage_run_name, stage_verdict)
from ultrakill_ai.times import format_time  # noqa: E402


def fmt(value, digits=2, dash="-"):
    return dash if value is None else "%.*f" % (digits, value)


def collect(cwd: Path, plan_path: str, runs_dir: str, models_dir: str) -> dict:
    """Everything the report prints, as plain data, so `--json` and the text share one source."""
    plan = load_plan(cwd / plan_path)
    state = DriverState.load(cwd / runs_dir / DRIVER_RUN / "driver_state.json")
    state.reconcile(plan)  # read-only: the same (level, kind) match the driver does on its own start
    out: dict = {"levels": len(plan.order), "stages": len(plan.stages), "rule": plan.rule.__dict__,
                 "speed_rule": plan.rule_for(SPEED).__dict__, "history": state.history, "current": None}
    # THE HOLD LINE (spec §8a). `hold_before` is the level the ladder will not pass, and `held_by` is what it is
    # still waiting on -- rebuilt here from the same state and plan the driver reads, so this report cannot
    # disagree with the driver about whether the line is up.
    hold = plan.hold_index()
    out["hold_before"] = plan.hold_before
    out["held_by"] = ([{"level": s.level, "kind": s.kind, "rounds": state.rounds(s.key),
                        "status": state.stage_status(s.key),
                        "max_rounds": plan.rule_for(s.kind).max_rounds}
                       for s in plan.stages[:hold]
                       if state.stage_status(s.key) not in ("done", "skipped")] if hold is not None else [])
    out["target_scale"] = plan.target_scale
    stage = state.current
    if stage is not None:
        rule = plan.rule_for(stage.kind)
        sample = read_sample(cwd / runs_dir / stage_run_name(stage.level, stage.kind) / "status.json",
                             cwd / models_dir / stage_run_name(stage.level, stage.kind) / "best.json")
        # A speed stage's target: whatever the stage already holds, else whatever the live run has reported.
        target = stage.target_seconds if stage.target_seconds else sample.target_seconds
        verdict, reached = stage_verdict(sample, stage.start_steps, stage.target_reached_at, rule,
                                         kind=stage.kind, target_seconds=target)
        out["current"] = {
            "level": stage.level, "kind": stage.kind, "run": stage.run, "index": stage.index, "init": stage.init,
            "start_steps": stage.start_steps, "timesteps": sample.timesteps,
            "stage_steps": (sample.timesteps - stage.start_steps) if sample.timesteps is not None else None,
            "fresh_completion_rate": sample.fresh_rate, "fresh_window": sample.fresh_window,
            # `median_time` is what the speed rule promotes on; `best_time` is the run's lifetime minimum and
            # is reported only. Printing both is the point: on the live runs they differ by about 2x.
            "best_time": sample.best_time, "median_time": sample.median_time,
            "best_at": sample.best_at, "target_seconds": target,
            "s_rank_seconds": stage.s_rank_seconds or sample.s_rank_seconds, "round": stage.round,
            "target_rate": rule.target_rate, "max_steps_per_stage": rule.max_steps_per_stage,
            "settle_steps": rule.settle_steps, "min_fresh_window": rule.min_fresh_window,
            "target_reached_at": reached, "verdict": verdict,
        }
    promoted = []
    for level in plan.order:
        sidecar = specialist_path(cwd / models_dir, level).with_suffix(".json")
        if sidecar.exists():
            try:
                promoted.append(json.loads(sidecar.read_text(encoding="utf-8")))
            except ValueError:
                continue
    out["promoted"] = promoted
    done = state.finished_stages()
    out["remaining"] = ["%s (%s)" % (s.level, s.kind) if s.kind != COMPLETE else s.level
                        for s in plan.stages if s.key not in done]
    return out


def render(data: dict) -> str:
    lines = []
    rule = data["rule"]
    current = data["current"]
    if current is None:
        lines.append("no stage running (%d of %d stages left in the plan)"
                     % (len(data["remaining"]), data.get("stages", data["levels"])))
    else:
        lines.append("STAGE %d/%d  %s  [%s, round %d]  run %s  [%s]"
                     % (current["index"] + 1, data.get("stages", data["levels"]), current["level"],
                        current.get("kind", "complete"), current.get("round", 1), current["run"],
                        current["verdict"]))
        lines.append("  steps into the stage: %s of %s (total %s)"
                     % ("{:,.0f}".format(current["stage_steps"] or 0),
                        "{:,}".format(current.get("max_steps_per_stage", rule["max_steps_per_stage"])),
                        "{:,.0f}".format(current["timesteps"] or 0)))
        lines.append("  fresh completion rate: %s over %d fresh episodes (target %s, needs %d)"
                     % (fmt(current["fresh_completion_rate"], 3), current["fresh_window"],
                        fmt(current.get("target_rate", rule["target_rate"]), 2),
                        current.get("min_fresh_window", rule["min_fresh_window"])))
        best, median = current["best_time"], current.get("median_time")
        lines.append("  median official time: %s   best ever: %s   best.zip last moved at %s"
                     % (format_time(median) if median is not None else "-",
                        format_time(best) if best is not None else "-",
                        "{:,.0f}".format(current["best_at"]) if current["best_at"] is not None else "-"))
        if current.get("kind") == SPEED:
            target, s_rank = current.get("target_seconds"), current.get("s_rank_seconds")
            lines.append("  SPEED stage: it promotes only once the MEDIAN time is at or under %s%s"
                         % (format_time(target) if target else
                            "%.2f x the level's own S-rank time" % data.get("target_scale", 0.75),
                            (" (S is %s)" % format_time(s_rank)) if s_rank else
                            ("" if target else " (not read from the run yet)")))
            lines.append("    (the median, not the best: the best is a run-lifetime minimum one lucky load "
                         "sets for good)")
        settle = current.get("settle_steps", rule["settle_steps"])
        if current["target_reached_at"] is None:
            lines.append("  target not reached yet: the stage ends on the rule above, or at the %s-step cap"
                         % "{:,}".format(current.get("max_steps_per_stage", rule["max_steps_per_stage"])))
        else:
            hold_from = max(current["target_reached_at"], current["best_at"] or 0.0)
            left = settle - ((current["timesteps"] or 0) - hold_from)
            lines.append("  target reached at %s steps; %s settle steps left (a new best restarts this)"
                         % ("{:,.0f}".format(current["target_reached_at"]), "{:,.0f}".format(max(0.0, left))))
        lines.append("  resumed from %s" % current["init"])

    held = data.get("held_by") or []
    if data.get("hold_before"):
        lines.append("")
        if held:
            lines.append("holding before %s: waiting on %s"
                         % (data["hold_before"],
                            ", ".join("%s (%s, round %d, %s)"
                                      % (h["level"], h["kind"], h["rounds"], h["status"] or "not started")
                                      for h in held)))
            lines.append("  no stage at or after %s starts until every one of those is done; they are trained "
                         "in round robin, fewest rounds first" % data["hold_before"])
            caps = [h for h in held if h.get("max_rounds", 0) > 0 and h["rounds"] >= h["max_rounds"]]
            if caps:
                lines.append("  OUT OF ROUNDS (the driver stops rather than looping): %s -- retune "
                             "speed.target_scale or speed.targets in the plan"
                             % ", ".join("%s (%s), %d of %d"
                                         % (h["level"], h["kind"], h["rounds"], h["max_rounds"])
                                         for h in caps))
        else:
            lines.append("hold line before %s is OPEN: every stage in front of it is done"
                         % data["hold_before"])

    lines.append("")
    lines.append("promoted specialists (%s/):" % SPECIALIST_DIR)
    if not data["promoted"]:
        lines.append("  none yet")
    for row in data["promoted"]:
        median, target = row.get("median_time"), row.get("target_seconds")
        lines.append("  %-12s %-9s r%-2d %-10s rate %-6s median %-10s target %-10s %s steps  <- %s"
                     % (row.get("level"), row.get("mode", COMPLETE), row.get("round", 1), row.get("status"),
                        fmt(row.get("fresh_completion_rate"), 3),
                        format_time(median) if median is not None else "-",
                        format_time(target) if target else "-",
                        "{:,.0f}".format(row.get("stage_steps") or 0),
                        row.get("source_checkpoint")))
    remaining = data["remaining"]
    lines.append("")
    lines.append("remaining (%d): %s" % (len(remaining), ", ".join(remaining[:8]) + (" ..." if len(remaining) > 8 else "")))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", default="configs/specialists.yaml")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    data = collect(Path.cwd(), a.plan, a.runs_dir, a.models_dir)
    print(json.dumps(data, indent=2, default=str) if a.json else render(data))


if __name__ == "__main__":
    main()
