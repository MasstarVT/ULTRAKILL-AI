# Ladder patience, the fallback target, and the banished-exit guard

Status: approved design, Python only (mod v0.7.1 unchanged; the DLL cannot be rebuilt this session).
Follows `2026-09-16-campaign-gates-unwedge-design.md` (the gate ladder) and
`2026-09-17-multi-level-and-skull-gates-design.md` (the curriculum and the skull carry).

## 1. The two bugs

**A, collapsed ladder.** `GateProgress` treats `campaign.gates[].hops` as a monotone ladder: `best_hops` only
falls, the target is the nearest active gate one rung below it, and `gate` pays per new lower `hops`. But `hops`
is a shortest-path LOWER BOUND in a room graph that multi-room doors over-connect. On `Level 0-3` the gate
nearest spawn (`0,13,330`, hops 2) is "reached" from the pit below — measured at dy −4.1 to −6.0 m and dh
6.0-8.0 m, airborne, in all six probe episodes — so `best_hops` locks at 2 and the target becomes `-16,73,315`
(hops 1), 66 m straight up through a ceiling, for the rest of the load. The walkable route is
2 → 3 → 4 → 5 → 6 → 3 → 2 → 1 → 0, which is not monotone, so every forward leg pays zero and observation slots
448-455 point at the wrong door. Measured collapsed on 0-3, 1-1, 1-2, 2-3, 4-3, 8-1 (four are in the live
11-level curriculum). 0-1 is monotone (`spawn_h == max_hops == 9`) and works; its behaviour must not change.

**B, banished exit.** `CheckPoint.Start` (`decompiled/CheckPoint.cs:132`) and `CheckPoint.ResetRoom`
(`:681`) clone each room and move the ORIGINAL by `transform.position.x + 10000f`. The mod's frozen `FinalPit`
reference follows the banished original, so on `Level 0-2` `campaign.exit.pos` jumps
(−199.0, −86.1, 277.0) → (9801.0, −86.1, 277.0) exactly when checkpoint `-55,-11,277` activates — which is when
the ladder hands the target to the exit. `ResetRoom` runs again per respawn, so the offset is `k * 10000`,
k ≥ 1. The live clone and the real `FinalPit` trigger stay at the original coordinates. The proper fix is
mod-side; this is the Python guard.

## 2. Algorithm (all of it lives in `GateProgress`, `campaign.py`)

New state, and exactly when each is cleared:

| field | scope | cleared by |
|---|---|---|
| `best_hops`, `paid_hops`, `reached`, `hops_reached` | level load | `new_level_load` (today) |
| `parked: set[str]` gate keys | level load | `new_level_load` |
| `park_best: dict[str, float]` the closest the player has been at any park of that key | level load | `new_level_load` only — **kept across an un-park**, and lowered (never raised) by a re-park |
| `park_count: dict[str, int]` parks per key, which doubles its un-park bar | level load | `new_level_load` |
| `paid_fallback: set[str]` keys that paid a fallback `gate` | level load | `new_level_load` |
| `parks: int` | env lifetime | never (the env diffs it per episode) |
| `best_dist`, `target` | episode | `reset_episode` (today, lead ruling R1) |
| `_clock_key`, `_clock`, `_clock_suspended`, `_clock_best` | episode, **and restarted by `mark_paid`** | `reset_episode`, `mark_paid` |
| `_fallback`, `_fallback_key` | episode | `reset_episode` |

`reset_episode` keeps `parked` / `park_best` / `park_count` / `paid_fallback` and restarts the clock: a park is a
fact about this level load's geometry, while the clock measures the current attempt. `mark_paid` (a respawn, or
any reset) absorbs `paid_fallback |= reached`, exactly parallel to `paid_hops = best_hops`, so a respawn that
lands on the current fallback target cannot pay for it — **and restarts the clock**, because a respawn is a new
attempt at the same target over ground whose `gate_approach` is already spent (§3 deviation 4).

```
update(campaign, pos, fought):                      # fought = kills or style rose this step
    prev = target; prev_fallback = _fallback; prev_gate = park_key_of(prev)

    # A4 fallback payment, judged BEFORE retarget so `reached` still describes the previous step
    if prev_fallback and pos and prev_gate not in paid_fallback and prev_gate != EXIT:
        g = gate(prev_gate); h = g.get("hops") if g else None       # guarded: altar_only doors carry hops null
        if h is not None and h >= best_hops and is_reached(g, pos): # best_hops None -> the ladder's own rule pays
            paid_fallback.add(prev_gate)
            if h not in hops_reached: pay 1                         # once per NEW RUNG, not once per door

    # A3 un-parking: passive, every step, never on a timer
    for key in parked:
        bar = unpark_m * 2 ** (park_count[key] - 1)                 # 2, 4, 8, 16 m ...
        if dist3(pos, gate(key).pos) < park_best[key] - bar:
            parked.discard(key)                                     # park_best is KEPT as the next baseline

    paid, approach = <today's update: retarget, ladder instalments, new-best approach>

    # A1 patience. The clock has its OWN baseline; `approach` is the reward, and the two answer
    # different questions across a respawn (§3 deviation 4).
    if key_of(target) != _clock_key:
        _clock_key, _clock, _clock_suspended = key_of(target), 0, 0
        _clock_best = dist3(pos, target.pos)
    if pos is None or park_key in (None, EXIT):   return            # the exit is never parked
    if target.subgoal or gate(park_key).needs_item: return          # never park a carry (§3 deviation 5)
    d = dist3(pos, target.pos)
    if d < _clock_best - min_gain:  _clock_best, _clock, _clock_suspended = d, 0, 0; return
    if arena_enemies_alive > 0 or fought: _clock_suspended += 1     # §3 deviation 1, now BOUNDED
    else:                                 _clock += 1
    forced = _clock + _clock_suspended >= 2 * patience_steps
    if (_clock < patience_steps and not forced) or park_key in parked: return
    _clock = _clock_suspended = 0                                   # park or not, the next is a full window away
    if not _fallback and not better_elsewhere(pos, target): return  # ladder picks only (§3 deviation 2)
    parked.add(park_key)
    park_best[park_key] = min(dist3(pos, gate(park_key).pos), _clock_best, park_best.get(park_key, inf))
    park_count[park_key] += 1; parks += 1; retarget(); _clock_key, _clock_best = ...
    return paid, approach

choose_target(campaign, pos):                        # A2
    rung = <today's candidate set: all active if best_hops is None; the exit at best_hops 0 or when no
            lower rung exists; else the gates at max(hops < best_hops)>
    free = [g in rung if g.key not in parked]
    if free: _fallback = False; return nearest(free, pos)          # today's choice, exactly
    cands = [g active, g.hops is not None, g.key not in reached, g.key not in parked]
    if cands:                                                       # the A2 fallback
        _fallback = True
        keep the sticky `_fallback_key` unless another candidate is more than hysteresis_m nearer
        return it
    _fallback = False; return exit or target                        # A2's tail, now reachable
```

`park_key_of(t)` is `t.gate_key or t.key`, so a park applies to the GATE, not to an instance-keyed sub-goal
(which `CheckPoint.ResetRoom` re-keys anyway) — though a carry leg is now never parked at all. A gate that never
becomes the target never pays, and the `gate` income of a level load is bounded by the **ladder depth** (the
number of distinct `hops` values) under both rules, so it cannot be raised by touring doors and cannot be
re-earned by re-walking.

## 3. Five deviations from the brief, each forced by measurement

1. **The patience clock does not tick while `arena_enemies_alive > 0`, or on a step that scored a kill or style —
   but the suspension is BOUNDED at `2 * patience_steps` of clock plus suspended decisions.**
   Replaying the 32,022 recorded `Level 0-1` decisions (`scratchpad/diag/run_steps.jsonl`,
   `campaign_ppo_ground/latest.zip`) through today's `GateProgress` with 0-1's literal gate array, a bare 20 s
   patience would park 20 target runs. Four of them ran with `arena_enemies_alive > 0` on 96-100% of their steps
   (the projectile arena in front of the hops 2 door, up to 2543 decisions): the level was holding the correct
   gate shut while the agent did the one thing that opens it. Approach progress is impossible there by
   construction. `env._campaign_progress` already treats a kill or a style event as progress for the stuck
   clock; this reuses that test. On 0-3 the suspension cannot hide the bug: the lock on `-16,73,315` runs
   2052-2423 decisions with `arena_enemies_alive == 0` on **100%** of them, and every episode has a kill-free
   tail of 1006-2230 decisions.
   **The bound is the correction the review forced.** `CampaignObserver.ArenaEnemiesAlive` counts every enemy
   whose `ActivateNextWave` has not run ANYWHERE on the level — it is scoped neither to the target gate's arena
   nor to the player's room — so a wave the agent walks away from and never clears froze the clock for the rest
   of the level load. Reproduced: 20,000 decisions under an unreachable door, zero parks, and
   `targets_parked == 0`, which on the dashboard is indistinguishable from "monotone level, mechanism correctly
   inert". Three of the four collapsed levels in the live curriculum (1-1, 1-2, 2-3) have arenas and there is no
   probe data for any of them, so the rule's safety there was asserted, not measured. Suspended decisions are
   now counted separately and the park is forced at `_clock + _clock_suspended >= 2 * patience_steps` — 600
   decisions at the shipped settings, inside the 675 of `stuck_seconds` 45, so the episode remains the outer
   bound. Below the bound the measured behaviour above is unchanged (`test_a6_3` still passes).
2. **A LADDER pick is parked only when some unreached, unparked, active gate is strictly nearer than it; a
   FALLBACK pick is parked unconditionally.** For a ladder pick, parking is then a switch and never a loss:
   without the test the recorded 0-1 run parks 0-13 gates per episode and its target sequence changes in 6 of 7
   episodes; with it, **6 of 7 episodes are byte-identical** in target sequence, `gate` instalments and
   `gate_approach`.
   For a fallback pick the test is vacuous and destructive, which the review proved: `_pick` chooses the
   fallback as the nearest unreached, unparked, active gate and `_nearer_unreached` filters that identical set,
   so nothing can ever be strictly nearer. The mechanism protected exactly one hand-over and then switched
   itself off — reproduced as 4,000 consecutive decisions aimed through a ceiling with `parked == ['rung']` and
   `targets_parked == 1`, i.e. Bug A moved one door over while the telemetry showed the fix as having fired.
   Relaxing it for fallbacks costs 0-1 nothing, measured: **all 46 of 0-1's window expiries are on ladder
   targets and none on a fallback target**, so the same single park happens either way. It does make A2's tail
   ("then the exit when none remain") reachable, which is why `_exhausted` exists.
   **The seventh 0-1 episode, exactly.** Episode 5 parks `40,11,624` once and hands over to `66,21,640` — the
   same gate the ladder was going to pick next, in the same order — **78 decisions early** (3183/1092 becomes
   3105/1170). `gate` instalments are identical at 7, the reached set and `best_hops` are identical, and
   `gate_approach` goes 367.316 → **377.680 m, +10.363 m**, which at `gate_approach` 0.15 is +1.55 reward on an
   episode earning ~55 from that term. An earlier draft of this document called that "identical payments" and
   the hand-over "24 decisions"; both were wrong, `gate_approach` IS a payment, and `test_a6_4` now asserts the
   per-episode delta rather than omitting the field.
3. **The fallback target is sticky with a 10 m hysteresis** rather than re-picking the nearest every step. Pure
   "nearest" flaps between `0,53,330` and `0,13,362` 23-59 times per 0-3 episode, which is noise in observation
   slots 448-455 and in look mode 2. Measured switches per episode: nearest 23-59, hysteresis 10 m 9-20,
   hysteresis 25 m 5-10, full stickiness 3-9. Gate payments on a synthetic walk of 0-3's real route: nearest 8,
   10 m 8, 25 m 8, full stickiness 5. 10 m is the smallest margin that halves the flapping without costing a
   payment, and it keeps A2's "nearest" semantics (a rival must be 10 m nearer to steal the target).
4. **The patience clock runs on its own baseline (`_clock_best`), not on `gate_approach`.** A1 defines the clock
   as "no new best approach", and the brief also carries ruling R1, which makes `best_dist` EPISODE scoped so a
   death respawn inside an episode does not re-earn ground already paid for. The two rules contradict each
   other: after a death the agent must re-walk that ground, the re-walk pays nothing **by design**, and a clock
   reading the reward therefore ran out on a target the agent was walking straight at. Reproduced: a 389 m
   corridor walked flawlessly at 1 m per decision after a respawn, `route` PARKED at decision 299 with 101 m
   still to go, `park_best` set from the stale pre-death minimum of 12.0 m, observation slots 448-455 pointed at
   a side room for the rest of the approach, and touching that side room paying an undeserved instalment.
   `_clock_best` is seeded when the target changes and reset by `mark_paid`, so it measures the current attempt;
   the reward's semantics under R1 are untouched. Within one uninterrupted attempt the two agree by
   construction, which is why A6.1's golden pin and `test_a6_2` are unaffected.
5. **A fetch or carry leg, and any gate still reporting `needs_item`, is never parked.** This one is a safety
   rule. `env._protect_carry` drops the punch button only while `gates.target` is the ALTAR sub-goal of a held
   item, deliberately keyed on one source so the button can never be taken away and left that way. Parking the
   gate deletes the leg, `_subgoal` never rebuilds it, and the protection switches off with the skull still in
   the player's hands — and the live policy presses punch on ~35% of decisions, which `Punch.ActiveStart` turns
   into a throw. Reproduced end to end through the env: stall a red-skull carry past the window and `punch`
   survives into the command. On 1-1 the gate the skull opens is the only route forward, so the level load is
   lost. A skull-locked door is held shut by its altar rather than by geometry, which is the same argument
   deviation 1 makes for an arena; the difference is the cost of being wrong.

**A5, the reach test, is left alone, as A5 itself directs.** The tightening that would fix 0-3 at the source is a
3-D bound on the cylinder (`dist3 <= reach_m` as well as the horizontal/vertical bounds): it rejects all six of
0-3's corner reaches (8.49-9.27 m). But replayed over the 32,022 recorded 0-1 decisions it is NOT inert — ep 5
never reaches `146,31,640` at all and reaches `40,11,624` 934 decisions late — so A5's precondition
("provably leaves 0-1's recorded episodes identical") fails and the change is not adopted. It would not have
fixed Bug A anyway: on 0-3 it only delays the bad reach by 66-1069 decisions.

## 4. Exit guard (B1)

`ExitGuard` in `campaign.py`, one per env, frozen per level load.

```
BANISH_X = 10000.0     # CheckPoint.Start:132 / ResetRoom:681, read from decompiled/CheckPoint.cs
AXIS_EPS = 1.0         # how far y/z may differ and still read as "the same pit, moved along x"
MAX_SHIFT_M = 100.0    # any other jump this big is not a FinalPit moving

new_level_load():  frozen = None; banished = False
apply(campaign) -> rejected:                      # rewrites campaign["exit"]["pos"] IN PLACE
    no campaign / exit / pos      -> False
    frozen is None                -> frozen = pos; False          # first report of this load wins
    d = pos - frozen
    |d| ~ 0                       -> False
    |dy|,|dz| <= EPS and dx ~ +k*BANISH_X, k>=1  -> reject        # the banish, including a repeat ResetRoom
    |dy|,|dz| <= EPS and dx ~ -k*BANISH_X, k>=1  -> accept, re-freeze   # recovery: the banish only ever ADDS,
                                                                  # so this is the true pit coming back
    |d| > MAX_SHIFT_M             -> reject
    otherwise                     -> accept, re-freeze
reject: campaign["exit"]["pos"] = frozen; banished = True
```

Rewriting the block in place is what makes one guard cover all three consumers with no signature change:
`GateProgress._exit` (targeting), `spaces.campaign_block` slots 0-4 and `env._exit_dist_min`. `env._guard_exit`
calls it on every campaign observation the env adopts (`_campaign_reset`, `_step`, `_respawn`,
`_adopt_fresh_load`) and `new_level_load()` beside `GateProgress.new_level_load`.

## 5. Config keys (all defaulting to today's behaviour where that is possible)

| key | default | meaning |
|---|---|---|
| `gate_target_patience_s` | 20.0 | game seconds without a >`gate_min_gain_m` new best before the target is parked. **0 disables parking entirely and restores today's algorithm exactly.** Converted with `fixed_fps / frameskip`; 20 s = 300 decisions at 30/2, well under `stuck_seconds` 45 |
| `gate_unpark_m` | 2.0 | metres nearer than `park_best` the player must passively get to un-park, **doubling per park of that key** (`GateProgress.UNPARK_ESCALATION`), so the bars are 2, 4, 8, 16 m |
| `gate_fallback_hysteresis_m` | 10.0 | metres a rival must beat the sticky fallback target by |
| `exit_max_shift_m` | 100.0 | metres the exit may legitimately move inside one level load |

`EnvConfig.from_dict` drops unknown keys, so every existing `env_config.yaml` and every checkpoint loads
unchanged. Verified against the live run's own file (`models/campaign_gates/env_config.yaml`, 51 keys, none of
the four new ones): it loads, takes all four defaults, and gives 479 inputs and the same 12-component action
space. No observation width and no action-space change.

**`gate_target_patience_s: 0.0` is the per-config opt-out**, and it restores the pre-spec tracker exactly (that
is what A6.1's golden pin replays). The default stays at the brief's 20.0 rather than being switched off and
opted into per level: the mechanism has to be ON for the four collapsed levels already in the live 11-level
curriculum and for any level not yet surveyed, and what it costs the one level known to work is a +1.55 reward
difference on one of seven recorded episodes with an identical target order and identical instalments
(deviation 2). 0-1's inertness is therefore MEASURED, not structural — see the appendix, finding 6.

New numeric per-episode info, both in `CAMPAIGN_INFO_KEYS`: `targets_parked` (parks during this episode) and
`exit_banished` (1 if the guard rejected any report during it). They flow to `status.json` (`mean_100`),
`episodes.jsonl`, the dashboard's campaign panel and `poll_status.py`'s CSV. `poll_status.py` keeps an existing
header, so the live run's `metrics_log.csv` must be moved aside for the two new columns to appear.

## 6. Proof obligations and the tests that discharge them

`tests/test_ladder_replay.py`, against trimmed fixtures in `tests/fixtures/` (positions rounded to 0.1 m, plus
`arena_enemies_alive`, `kills`, `style`; the golden files were generated from the PRE-change implementation).

- **A6.1 (no other behaviour moved).** `gate_target_patience_s = 0` replayed over the whole recorded 0-1 episode
  fixture reproduces `tests/fixtures/ladder_golden_0-1.json` — the target key per step, the `gate` instalments
  and the total `gate_approach` — which was produced by the old implementation. Pins the ladder, the reach test,
  the approach accounting and the exit target byte for byte.
- **A6.2 (inert where the condition holds).** A synthesised walk along 0-1's eleven literal gate positions
  (spec §3.6) in ladder order, reaching every target inside the patience window: patience ON produces exactly
  the same target sequence, `gate` payments and `gate_approach` as patience OFF.
- **A6.3 (the arena pin).** Replaying the recorded 0-1 episode with patience ON, no gate is parked while
  `arena_enemies_alive > 0`, and the whole episode's target sequence and payments are identical to patience OFF.
- **A6.4 / A6.5 (the recorded run, in full).** `test_a6_4` asserts the per-episode `gate_approach` delta
  explicitly — `{0..4: 0.0, 5: +10.363, 6: 0.0}` — plus identical `gate` instalments, identical `reached`,
  identical `best_hops` and an identical target ORDER in every episode. `test_a6_5` pins episode 5's run
  lengths to the decision (`[134, 3034, 42, 181, 3183, 1092, 743, 592]` → `[..., 3105, 1170, ...]`), so the one
  park's entire effect is a 78-decision boundary shift and nothing can move it unseen.
- **A7.1 (0-3, sampled policy).** Replaying the probe's own two episodes, the old tracker targets `-16,73,315`
  for 2052 and 2351 of 2501 decisions; the new one holds it for 1006 and 841 (both at most half), targets a gate
  on the walkable route, pays no less, and records a strictly closer baseline at every park of it.
  `test_a7_1b` pins the park log exactly: parks at (516, 40.87), (1159, 35.98), (1925, 26.13) with un-parks at
  (854, 38.16), (1624, 29.16), (2168, 17.41) — bars of 2, 4 and 8 m against a monotone 24 m closing — and
  episode 1's single park at (991, 29.2) that never comes back.
- **A7.2 (0-3, scripted).** The scripted fixture walks through the hops 2 door in 77 decisions and then mills
  inside a 20 m box for 2,120 more. The old tracker spends all 2,120 pointing through the ceiling; the new one
  holds `-16,73,315` for at most one window, never targets it again, and spends the rest cycling candidates on
  the walkable route, none held longer than a window — each carrying a 2 m un-park bar, so the first genuine
  step toward one brings it straight back. A synthetic walk of the full route pays `gate` on the forward legs:
  **7 instalments against the old 3**, `gate_approach` 469.5 m against 316.5, ending on the exit at `best_hops` 0.
- **B1.** `tests/test_campaign_env.py` against `FakeLevel`: a +10,000 X jump mid-load is ignored for the target,
  for observation slots 0-4 and for `exit_dist_min`; `info["exit_banished"]` reads 1; a repeat (+20,000) is also
  ignored; a genuinely different exit after a new level load is accepted; the negative-multiple recovery works.

Unit tests for each rule (patience, park, un-park, fallback choice and payment, `mark_paid` absorption, the
exit-guard arithmetic) live in `tests/test_campaign.py`.

## 7. Appendix: the eight review findings

Every one was reproduced against this worktree's code before anything was changed; the scripts are in the
session scratchpad under `verify/`. Seven are confirmed and fixed. One is confirmed in substance with one of its
claims rebutted, and one proposed fix is rejected with a reproduction of its own failure.

| # | verdict | fix, and the test that pins it |
|---|---|---|
| 1 | **confirmed** | A fallback target could never be parked: `_pick` and `_nearer_unreached` filter the identical candidate set, so nothing can be strictly nearer than the nearest. Reproduced: 6,000 static decisions, target `high` (30 m through a ceiling), `parked == ['rung']`, `parks == 1` — the wedge one door over, reported as fired. `_tick_patience` now skips the nearness precondition for `_fallback` picks (§3 deviation 2). `test_a_fallback_target_that_never_gets_closer_is_parked_too`, and `test_when_every_gate_is_parked_the_target_becomes_the_exit` for A2's tail, which this makes reachable. |
| 2 | **confirmed** | The instalment was per gate KEY, which does not deliver A4's own "no incentive to tour doors": every door becomes the fallback target in turn. Reproduced on eight side doors sharing one rung — 6 instalments where the ladder depth was 2 (90 points against `level_complete` 100), and 780 available on 8-1's 52 phase-1 gates. Bounded by `hops in hops_reached` instead, so the ceiling is the ladder depth under both rules. Every forward leg of 0-3's route is a new rung and still pays (`test_a7_3`: 7 against 3). `test_a_second_door_on_a_rung_already_reached_pays_nothing`, `test_touring_many_doors_on_one_rung_cannot_out_earn_the_ladder`. |
| 3 | **confirmed; its proposed fix partly rejected** | See below. |
| 4 | **confirmed** | The clock read `gate_approach`, which R1 makes episode scoped, so a post-respawn re-walk paid nothing and ran the clock out on the right door: reproduced, parked at decision 299 of a flawless 389 m approach with 101 m to go, `park_best` taken from the stale pre-death 12.0 m. Fixed with `_clock_best` + a restart in `mark_paid` (§3 deviation 4). `test_a_death_respawn_does_not_park_the_door_the_agent_is_walking_at`, `test_a_respawn_restarts_the_patience_clock`. |
| 5 | **confirmed** | Parking a skull-locked gate silently disabled the punch carry-protection. Reproduced at the unit level (target flips off the altar sub-goal, `_protect_carry` stops removing the button) and end to end through the env. A carry leg and any `needs_item` gate are now never parked (§3 deviation 5). `test_a_stalled_skull_leg_is_never_parked_at_all`, `test_a_gate_still_wanting_a_skull_is_never_parked`, `test_a_carry_that_stalls_past_the_patience_window_keeps_its_punch_protection`. |
| 6 | **confirmed in substance; one claim rebutted; proposed fix not adopted** | See below. |
| 7 | **confirmed** | The arena/`fought` suspension was unbounded and unscoped: 20,000 decisions under an unreachable door, zero parks, `targets_parked == 0`. Bounded at `2 * patience_steps` (§3 deviation 1). `test_an_arena_that_never_dies_still_parks_the_gate_eventually`, `test_an_endless_fight_under_the_door_still_parks_it`. |
| 8 | **confirmed** | `_pay_fallback`'s unguarded `int(gate["hops"])` raises `TypeError` on `hops: null` and `KeyError` when the key is absent — verified by calling it directly — and `env.step` catches only `BridgeRecovered`, so it would take all twelve games. Guarded like every other read in the class. `test_a_fallback_gate_reporting_no_hops_does_not_raise`. |

**Finding 3, in detail.** The mechanism is confirmed and fixed, and half the proposed fix is unsound.

*Confirmed and fixed.* A flat 2 m bar re-read from wherever the player stood at the next park let the wedge
re-form: on the probe, park 1 recorded 40.9 m, park 2 was taken at 77.5 m after the agent wandered off, and
walking back into the pit then cleared the bar for free. `park_best` is now the closest the player has been at
any park of that key this level load — kept across an un-park, lowered only by a re-park — and the bar doubles
per park. Episode 0's cycling falls from three free un-parks to three that cost 2, 4 and 8 m against a monotone
40.9 → 17.4 m closing; a fourth would need the agent within 2 m of the door. Episode 1 parks once and it never
comes back. `test_wandering_away_and_stalling_again_cannot_buy_a_cheap_un_park`,
`test_the_un_park_bar_doubles_each_time_a_door_is_parked_again`, `test_a7_1b`.

*Rejected, with a reproduction.* Part (a) of the proposal — "keep updating `park_best[key] = min(park_best[key],
dist)` while the gate is parked" — cannot work. A baseline that follows the player's running minimum every step
is beaten by at most one decision's travel, so no gradual approach can ever clear a bar wider than that:
walking 1 m per decision straight at a door with a 2 m bar gives `d < (d + 1) - 2`, false at every distance.
Implemented as written, it made the park permanent and failed `test_a7_4` — the agent climbed 34 m to stand on
the door and it stayed parked — which breaks A3 outright. The escalating bar delivers the intended convergence
without it.

*The residual.* Episode 0 still spends 40.2% of its decisions on that door and ends with it un-parked. That is
not slack: 380 of those decisions are the first lock, before any park, because A1's clock is reset by every
genuine new best and the sampled policy does stumble closer to the door on its way round the pit; the rest are
three windows each opened by the agent getting strictly closer than it had ever been, which is what A3 says an
un-park is. The suggested assertion `held < 0.15 * len(steps)` is not reachable under A1 and A3 as specified,
so `test_a7_1` asserts `held(on) <= 0.5 * held(off)` plus the strictly-decreasing park baselines, and
`test_a7_1b` pins every park and un-park to the decision.

**Finding 6, in detail.** The overstatements are confirmed and corrected; the central claim about
`_nearer_unreached` is measured and wrong; the proposed fix is not adopted.

*Confirmed and corrected.* The patience window does expire on 0-1 — **46 times across the 32,022 recorded
decisions** (ep 5: 17, ep 6: 18, ep 1: 7), 45 held only by `_nearer_unreached` and one becoming a real park.
The earlier draft's "identical payments" for ep 5 was wrong, because `gate_approach` is a payment at 0.15/m and
it moves by +10.363 m; and the fallback target `66,21,640` is live for **78** decisions, not 24. `test_a6_4`
now asserts the per-episode `gate_approach` delta instead of omitting the field, and `test_a6_5` pins the run
lengths. §3 deviation 2 carries the corrected numbers.

*Rebutted.* "Relax `_nearer_unreached` as Finding 1 requires and 45 more parks fire on 0-1" does not follow and
does not happen. Finding 1's fix relaxes the precondition for FALLBACK picks only, and measured over the same
fixture **all 46 expiries are on ladder targets and none on a fallback target**: 0-1 has exactly one park with
the relaxation and one without it. The full suite, 0-1 replays included, was run with the relaxation in place.

*Not adopted.* Defaulting `gate_target_patience_s` to 0.0 and opting in per level would turn the mechanism off
on the four collapsed levels already in the live curriculum and on any level not yet surveyed, and there is no
per-level config mechanism to opt them back in with. The measured price of leaving it on for 0-1 is +1.55
reward on one of seven recorded episodes, with the same target order, the same instalments, the same reached set
and the same `best_hops`. The opt-out exists per config (`gate_target_patience_s: 0.0`) and is documented in §5;
what this appendix retracts is the claim that 0-1's inertness is structural. It is measured, and §6's tests are
what measure it.
