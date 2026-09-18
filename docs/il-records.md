# Human individual-level (IL) records - ULTRAKILL

Speed targets for the AI, pulled from the speedrun.com REST API (`https://www.speedrun.com/api/v1`).
Research artifact: nothing in the training pipeline reads this file. The machine-readable copy is
`python/configs/il_records.yaml`.

- **Fetched:** 2026-09-18
- **Game:** ULTRAKILL, speedrun.com id `369p3p81` (<https://www.speedrun.com/ultrakill>)
- **Timing method:** in-game time (IGT) for every row. The game's board default is `ingame` and the IL
  rules say to `leave "Time" blank and fill in IGT (In-Game Time) only, including the 3 decimal digits`,
  so IL runs carry no real-time value at all - every `realtime_t` in this pull was 0. The full-game record
  below is the one place both times exist.
- **Revamp split:** every row is filtered to **Post-Revamp**, the board's own default. Pre-Revamp times are
  on older level geometry and are not comparable to our build; leaving the filter off pulls them in and they
  win (0-1 Pre-Revamp is 2.585 s against 4.915 s Post-Revamp).
- **Exit:** on the nine levels with more than one exit (0-2, 1-1, 2-3, 3-1, 4-2, 5-1, 6-2, 7-3, 8-3) rows are filtered
  to **Main Exit**, which is the exit our `FinalPit` observation grades.

## Difficulty: the boards do not split by it

**There is no difficulty variable on any ULTRAKILL leaderboard.** The full game category rules say, verbatim,
`Any difficulty is allowed, but switching difficulties mid run is not allowed`, and the three IL category
rulesets (Any%, P Rank, No Monsters) do not mention difficulty at all. So every time below was set on
whatever difficulty the runner chose, and the API does not record which one. Difficulty in ULTRAKILL changes
enemy health, damage and aggression but not level geometry, so an Any% route that skips fighting is close to
difficulty-independent while anything that kills is not.

**Read these as a floor, not as a like-for-like target.** Our pilot runs Violent with all gear unlocked; these
records are the fastest any human has moved through the geometry under unrestricted difficulty. The relevant
in-repo reference point stays the human 0-1 playthrough in `CLAUDE.md`: 146.58 s with 59 kills, i.e. a
full clear, against 19.798 s for the inbounds IL record and 4.915 s for the unrestricted one.

(The prose `Game Rules` page at <https://www.speedrun.com/ultrakill/rules> is Cloudflare-protected and
returned HTTP 403, so only the per-category rules the API exposes are quoted here.)

## Full-game record (context)

| Category | Difficulty | IGT | RTA | Runner | Date | Link |
| --- | --- | --- | --- | --- | --- | --- |
| Fraud Full Game - Any%, Post-Revamp | unrestricted | **11:06.259** (666.259 s) | 17:48.099 | dentess | 2026-05-22 | <https://www.speedrun.com/ultrakill/runs/z1063qwm> |
| Fraud Full Game - Inbounds, Post-Revamp | unrestricted | 23:20.439 (1400.439 s) | - | dentess | 2026-04-25 | <https://www.speedrun.com/ultrakill/runs/y6w253qz> |
| Fraud Full Game - P Rank, Post-Revamp | unrestricted | 42:22.213 (2542.213 s) | - | Dabonzack | 2026-07-06 | <https://www.speedrun.com/ultrakill/runs/yjv43r3y> |

The ~11 minute figure is the Any% one: **666.259 s IGT** over 0-1 to 8-4. Its RTA is 1068.099 s, so roughly
6.7 minutes of that run is loads and cutscenes that IGT does not count - worth remembering, because our
`level_seconds` comes from the same in-game timer.

## Per-level records, Any% (fastest known route)

Any route allowed, any rank, any difficulty. On most levels this is an out-of-bounds or major-skip route -
see the next table and the note below.

| Level | Title | Category | Difficulty | IGT record | Runner | Date | Link |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0-1 | Into The Fire | IL Any% (Post-Revamp) | unrestricted | 4.915 | D31taa | 2026-07-23 | <https://www.speedrun.com/ultrakill/runs/z5433ngm> |
| 0-2 | The Meatgrinder | IL Any% (Post-Revamp) | unrestricted | 5.280 | zaalk | 2026-05-22 | <https://www.speedrun.com/ultrakill/runs/y9gprvrm> |
| 0-3 | Double Down | IL Any% (Post-Revamp) | unrestricted | 6.624 | Sowsher | 2026-09-18 | <https://www.speedrun.com/ultrakill/runs/mrpvlx2z> |
| 0-4 | A One-Machine Army | IL Any% (Post-Revamp) | unrestricted | 5.802 | Darknn | 2026-05-26 | <https://www.speedrun.com/ultrakill/runs/ywo2xxpm> |
| 0-5 | Cerberus | IL Any% (Post-Revamp) | unrestricted | 12.595 | Spirtek | 2026-04-19 | <https://www.speedrun.com/ultrakill/runs/me5625qm> |
| 1-1 | Heart Of The Sunrise | IL Any% (Post-Revamp) | unrestricted | 9.837 | xotiic_ | 2026-06-29 | <https://www.speedrun.com/ultrakill/runs/y8434enz> |
| 1-2 | The Burning World | IL Any% (Post-Revamp) | unrestricted | 9.797 | DevilHunter | 2026-08-27 | <https://www.speedrun.com/ultrakill/runs/me51ql0m> |
| 1-3 | Halls Of Sacred Remains | IL Any% (Post-Revamp) | unrestricted | 10.841 | Sowsher | 2026-07-25 | <https://www.speedrun.com/ultrakill/runs/yjv6ov7y> |
| 1-4 | Clair De Lune | IL Any% (Post-Revamp) | unrestricted | 8.377 | CATPERFECT | 2026-07-10 | <https://www.speedrun.com/ultrakill/runs/yjv3j7gy> |
| 2-1 | Bridgeburner | IL Any% (Post-Revamp) | unrestricted | 7.623 | nirichy | 2026-08-16 | <https://www.speedrun.com/ultrakill/runs/yw7j9l2z> |
| 2-2 | Death At 20,000 Volts | IL Any% (Post-Revamp) | unrestricted | 10.153 | vivyaann | 2026-04-26 | <https://www.speedrun.com/ultrakill/runs/yvepqkom> |
| 2-3 | Sheer Heart Attack | IL Any% (Post-Revamp) | unrestricted | 9.130 | vivyaann | 2025-10-27 | <https://www.speedrun.com/ultrakill/runs/yw5dqe3y> |
| 2-4 | Court Of The Corpse King | IL Any% (Post-Revamp) | unrestricted | 21.439 | vivyaann | 2026-08-30 | <https://www.speedrun.com/ultrakill/runs/yjv8od7y> |
| 3-1 | Belly Of The Beast | IL Any% (Post-Revamp) | unrestricted | 12.977 | DevilHunter | 2026-04-20 | <https://www.speedrun.com/ultrakill/runs/y4vgr1qy> |
| 3-2 | In The Flesh | IL Any% (Post-Revamp) | unrestricted | 7.710 | I_am_Awkward | 2026-08-07 | <https://www.speedrun.com/ultrakill/runs/y84eq3xz> |
| 4-1 | Slaves To Power | IL Any% (Post-Revamp) | unrestricted | 5.337 | _Goots_ | 2026-06-01 | <https://www.speedrun.com/ultrakill/runs/zg19jpnz> |
| 4-2 | God Damn The Sun | IL Any% (Post-Revamp) | unrestricted | 21.187 | geshem8 | 2026-01-21 | <https://www.speedrun.com/ultrakill/runs/y400373z> |
| 4-3 | A Shot In The Dark | IL Any% (Post-Revamp) | unrestricted | 19.270 | geshem8 | 2026-05-06 | <https://www.speedrun.com/ultrakill/runs/med0rl2y> |
| 4-4 | Clair De Soleil | IL Any% (Post-Revamp) | unrestricted | 28.597 | FishyBandit | 2026-02-19 | <https://www.speedrun.com/ultrakill/runs/y8l3xxdm> |
| 5-1 | In The Wake Of Poseidon | IL Any% (Post-Revamp) | unrestricted | 17.495 | themeowingdragon | 2026-08-24 | <https://www.speedrun.com/ultrakill/runs/y44x8rqy> |
| 5-2 | Waves Of The Starless Sea | IL Any% (Post-Revamp) | unrestricted | 30.194 | geshem8 | 2026-09-12 | <https://www.speedrun.com/ultrakill/runs/z04xq6oy> |
| 5-3 | Ship Of Fools | IL Any% (Post-Revamp) | unrestricted | 9.617 | Spirtek | 2026-06-22 | <https://www.speedrun.com/ultrakill/runs/z04904jy> |
| 5-4 | Leviathan | IL Any% (Post-Revamp) | unrestricted | 11.237 | wexter | 2026-06-21 | <https://www.speedrun.com/ultrakill/runs/zpv72pvz> |
| 6-1 | Cry For The Weeper | IL Any% (Post-Revamp) | unrestricted | 18.617 | themeowingdragon | 2026-06-05 | <https://www.speedrun.com/ultrakill/runs/m388rnwy> |
| 6-2 | Aesthetics Of Hate | IL Any% (Post-Revamp) | unrestricted | 23.252 | Darknn | 2026-07-02 | <https://www.speedrun.com/ultrakill/runs/y445q62y> |
| 7-1 | Garden Of Forking Paths | IL Any% (Post-Revamp) | unrestricted | 25.524 | tootaroni | 2026-04-09 | <https://www.speedrun.com/ultrakill/runs/m7j6830m> |
| 7-2 | Light Up The Night | IL Any% (Post-Revamp) | unrestricted | 6.822 | dentess | 2026-08-16 | <https://www.speedrun.com/ultrakill/runs/me538e9m> |
| 7-3 | No Sound, No Memory | IL Any% (Post-Revamp) | unrestricted | 24.762 | vivyaann | 2026-06-02 | <https://www.speedrun.com/ultrakill/runs/yve5k54m> |
| 7-4 | ...Like Antennas To Heaven | IL Any% (Post-Revamp) | unrestricted | 20.443 | themeowingdragon | 2026-04-18 | <https://www.speedrun.com/ultrakill/runs/mrwrpndy> |
| 8-1 | Hurtbreak Wonderland | IL Any% (Post-Revamp) | unrestricted | 4.042 | vivyaann | 2026-07-02 | <https://www.speedrun.com/ultrakill/runs/yw78wonz> |
| 8-2 | Through The Mirror | IL Any% (Post-Revamp) | unrestricted | 10.824 | vivyaann | 2026-03-20 | <https://www.speedrun.com/ultrakill/runs/med6r19y> |
| 8-3 | Disintegration Loop | IL Any% (Post-Revamp) | unrestricted | 21.660 | zaalk | 2026-09-11 | <https://www.speedrun.com/ultrakill/runs/zpepw9ny> |
| 8-4 | Final Flight | IL Any% (Post-Revamp) | unrestricted | 30.320 | winterrr | 2026-09-14 | <https://www.speedrun.com/ultrakill/runs/y4465l2y> |

## Per-level records, Any% + Inbounds (intended route)

Same category, filtered to the board's `Inbounds` subcategory tag. This is the closest thing the boards
offer to "no major assists": the route stays inside the level. These are the more meaningful targets for us,
since the agent navigates with a NavMesh path hint and ground-keyed exploration and has no way to reach an
out-of-bounds route.

| Level | Title | Category | Difficulty | IGT record | Runner | Date | Link |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0-1 | Into The Fire | IL Any% + Inbounds (Post-Revamp) | unrestricted | 19.798 | geshem8 | 2026-05-07 | <https://www.speedrun.com/ultrakill/runs/zqn8rdrz> |
| 0-2 | The Meatgrinder | IL Any% + Inbounds (Post-Revamp) | unrestricted | 15.863 | OskariTheHeimfanker | 2026-07-06 | <https://www.speedrun.com/ultrakill/runs/mrp883dz> |
| 0-3 | Double Down | IL Any% + Inbounds (Post-Revamp) | unrestricted | 6.624 | Sowsher | 2026-09-18 | <https://www.speedrun.com/ultrakill/runs/mrpvlx2z> |
| 0-4 | A One-Machine Army | IL Any% + Inbounds (Post-Revamp) | unrestricted | 5.802 | Darknn | 2026-05-26 | <https://www.speedrun.com/ultrakill/runs/ywo2xxpm> |
| 0-5 | Cerberus | IL Any% + Inbounds (Post-Revamp) | unrestricted | 16.279 | bowsette_ | 2026-06-27 | <https://www.speedrun.com/ultrakill/runs/yvk40vom> |
| 1-1 | Heart Of The Sunrise | IL Any% + Inbounds (Post-Revamp) | unrestricted | 26.424 | OskariTheHeimfanker | 2026-07-20 | <https://www.speedrun.com/ultrakill/runs/y9q0kq2z> |
| 1-2 | The Burning World | IL Any% + Inbounds (Post-Revamp) | unrestricted | 14.085 | OskariTheHeimfanker | 2026-07-20 | <https://www.speedrun.com/ultrakill/runs/z54jk4dm> |
| 1-3 | Halls Of Sacred Remains | IL Any% + Inbounds (Post-Revamp) | unrestricted | 13.210 | F1restorm | 2026-05-30 | <https://www.speedrun.com/ultrakill/runs/z02k994m> |
| 1-4 | Clair De Lune | IL Any% + Inbounds (Post-Revamp) | unrestricted | 8.377 | CATPERFECT | 2026-07-10 | <https://www.speedrun.com/ultrakill/runs/yjv3j7gy> |
| 2-1 | Bridgeburner | IL Any% + Inbounds (Post-Revamp) | unrestricted | 7.623 | nirichy | 2026-08-16 | <https://www.speedrun.com/ultrakill/runs/yw7j9l2z> |
| 2-2 | Death At 20,000 Volts | IL Any% + Inbounds (Post-Revamp) | unrestricted | 13.032 | vivyaann | 2025-04-29 | <https://www.speedrun.com/ultrakill/runs/zpoq6gxm> |
| 2-3 | Sheer Heart Attack | IL Any% + Inbounds (Post-Revamp) | unrestricted | 23.725 | Darknn | 2026-06-08 | <https://www.speedrun.com/ultrakill/runs/ydppk7wy> |
| 2-4 | Court Of The Corpse King | IL Any% + Inbounds (Post-Revamp) | unrestricted | 35.604 | sinrotoGT | 2026-08-29 | <https://www.speedrun.com/ultrakill/runs/z14qxe7z> |
| 3-1 | Belly Of The Beast | IL Any% + Inbounds (Post-Revamp) | unrestricted | 32.800 | Melifaro | 2025-12-25 | <https://www.speedrun.com/ultrakill/runs/zp5p198y> |
| 3-2 | In The Flesh | IL Any% + Inbounds (Post-Revamp) | unrestricted | 8.959 | I_am_Awkward | 2026-08-01 | <https://www.speedrun.com/ultrakill/runs/zxrg4dgz> |
| 4-1 | Slaves To Power | IL Any% + Inbounds (Post-Revamp) | unrestricted | 5.337 | _Goots_ | 2026-06-01 | <https://www.speedrun.com/ultrakill/runs/zg19jpnz> |
| 4-2 | God Damn The Sun | IL Any% + Inbounds (Post-Revamp) | unrestricted | 21.187 | geshem8 | 2026-01-21 | <https://www.speedrun.com/ultrakill/runs/y400373z> |
| 4-3 | A Shot In The Dark | IL Any% + Inbounds (Post-Revamp) | unrestricted | 45.165 | dentess | 2026-01-08 | <https://www.speedrun.com/ultrakill/runs/y2rw2xjm> |
| 4-4 | Clair De Soleil | IL Any% + Inbounds (Post-Revamp) | unrestricted | 29.967 | Kolta | 2026-02-04 | <https://www.speedrun.com/ultrakill/runs/y684qg6z> |
| 5-1 | In The Wake Of Poseidon | IL Any% + Inbounds (Post-Revamp) | unrestricted | 1:50.608 | MichalAndru | 2025-11-07 | <https://www.speedrun.com/ultrakill/runs/z0kr3r8m> |
| 5-2 | Waves Of The Starless Sea | IL Any% + Inbounds (Post-Revamp) | unrestricted | 34.673 | MichalAndru | 2026-09-10 | <https://www.speedrun.com/ultrakill/runs/z04xov9y> |
| 5-3 | Ship Of Fools | IL Any% + Inbounds (Post-Revamp) | unrestricted | 1:12.280 | dentess | 2026-01-09 | <https://www.speedrun.com/ultrakill/runs/y40lvdnz> |
| 5-4 | Leviathan | IL Any% + Inbounds (Post-Revamp) | unrestricted | 11.237 | wexter | 2026-06-21 | <https://www.speedrun.com/ultrakill/runs/zpv72pvz> |
| 6-1 | Cry For The Weeper | IL Any% + Inbounds (Post-Revamp) | unrestricted | 42.698 | F1restorm | 2026-02-07 | <https://www.speedrun.com/ultrakill/runs/y40dw3dz> |
| 6-2 | Aesthetics Of Hate | IL Any% + Inbounds (Post-Revamp) | unrestricted | 23.252 | Darknn | 2026-07-02 | <https://www.speedrun.com/ultrakill/runs/y445q62y> |
| 7-1 | Garden Of Forking Paths | IL Any% + Inbounds (Post-Revamp) | unrestricted | 1:03.763 | Kolta | 2025-12-31 | <https://www.speedrun.com/ultrakill/runs/ydon530z> |
| 7-2 | Light Up The Night | IL Any% + Inbounds (Post-Revamp) | unrestricted | 55.329 | Kolta | 2025-05-30 | <https://www.speedrun.com/ultrakill/runs/yv729rey> |
| 7-3 | No Sound, No Memory | IL Any% + Inbounds (Post-Revamp) | unrestricted | 2:25.246 | Erig189 | 2026-01-05 | <https://www.speedrun.com/ultrakill/runs/z0kq5kom> |
| 7-4 | ...Like Antennas To Heaven | IL Any% + Inbounds (Post-Revamp) | unrestricted | 30.576 | vivyaann | 2026-04-05 | <https://www.speedrun.com/ultrakill/runs/mrw6r3dy> |
| 8-1 | Hurtbreak Wonderland | IL Any% + Inbounds (Post-Revamp) | unrestricted | 4.042 | vivyaann | 2026-07-02 | <https://www.speedrun.com/ultrakill/runs/yw78wonz> |
| 8-2 | Through The Mirror | IL Any% + Inbounds (Post-Revamp) | unrestricted | 28.799 | Kolta | 2026-03-22 | <https://www.speedrun.com/ultrakill/runs/m38nkj6y> |
| 8-3 | Disintegration Loop | IL Any% + Inbounds (Post-Revamp) | unrestricted | 2:50.419 | dentess | 2026-03-20 | <https://www.speedrun.com/ultrakill/runs/z5n070dz> |
| 8-4 | Final Flight | IL Any% + Inbounds (Post-Revamp) | unrestricted | 30.320 | winterrr | 2026-09-14 | <https://www.speedrun.com/ultrakill/runs/y4465l2y> |

## Which records rely on skips or going out of bounds

This is read off the board's own subcategory tag, not guessed from videos. Each Any% run carries an
`Extra Subcategory` value; when that value is `Inbounds` the record is the inbounds record, and when it is
`None` the runner did not claim the inbounds tag and a separate, slower inbounds record exists.
**No run comment in this pull names a specific trick**, so no technique is attributed to anyone below - only
the tag and the gap it implies.

Ten levels where the Any% record *is* the inbounds record (the intended route is the fastest known route):

> 0-3, 0-4, 1-4, 2-1, 4-1, 4-2, 5-4, 6-2, 8-1, 8-4

The other 23 levels have an Any% record that is not inbounds-tagged. Sorted by how much the unrestricted
route saves, which is a rough proxy for how much of the level it skips:

| Level | Any% | Inbounds | Saved | Ratio |
| --- | --- | --- | --- | --- |
| 8-3 | 21.660 | 2:50.419 | 148.759 s | 7.87x |
| 7-3 | 24.762 | 2:25.246 | 120.484 s | 5.87x |
| 5-1 | 17.495 | 1:50.608 | 93.113 s | 6.32x |
| 5-3 | 9.617 | 1:12.280 | 62.663 s | 7.52x |
| 7-2 | 6.822 | 55.329 | 48.507 s | 8.11x |
| 7-1 | 25.524 | 1:03.763 | 38.239 s | 2.5x |
| 4-3 | 19.270 | 45.165 | 25.895 s | 2.34x |
| 6-1 | 18.617 | 42.698 | 24.081 s | 2.29x |
| 3-1 | 12.977 | 32.800 | 19.823 s | 2.53x |
| 8-2 | 10.824 | 28.799 | 17.975 s | 2.66x |
| 1-1 | 9.837 | 26.424 | 16.587 s | 2.69x |
| 0-1 | 4.915 | 19.798 | 14.883 s | 4.03x |
| 2-3 | 9.130 | 23.725 | 14.595 s | 2.6x |
| 2-4 | 21.439 | 35.604 | 14.165 s | 1.66x |
| 0-2 | 5.280 | 15.863 | 10.583 s | 3x |
| 7-4 | 20.443 | 30.576 | 10.133 s | 1.5x |
| 5-2 | 30.194 | 34.673 | 4.479 s | 1.15x |
| 1-2 | 9.797 | 14.085 | 4.288 s | 1.44x |
| 0-5 | 12.595 | 16.279 | 3.684 s | 1.29x |
| 2-2 | 10.153 | 13.032 | 2.879 s | 1.28x |
| 1-3 | 10.841 | 13.210 | 2.369 s | 1.22x |
| 4-4 | 28.597 | 29.967 | 1.37 s | 1.05x |
| 3-2 | 7.710 | 8.959 | 1.249 s | 1.16x |

The extremes are worth flagging: 8-3 (21.660 s against 170.419 s inbounds, 7.9x), 7-3 (5.9x), 7-2 (8.1x),
5-3 (7.5x) and 5-1 (6.3x) are levels where the unrestricted record bypasses nearly the whole level. Treating
those Any% numbers as an AI target would be setting a goal that the intended route cannot reach.

## How this was pulled

One request per level per board against
`/leaderboards/369p3p81/level/<level>/5dw44w5k?top=1&embed=players&var-6njpq55n=1dko893l`, plus
`&var-0nwykpd8=14o4m8wq` for the inbounds table and `&var-<exit>=<main>` on the nine multi-exit levels.
66 IL requests and 3 full-game requests, no rate limiting hit. Every board returned `timing: ingame` and a
non-empty top run, so there are no missing levels or categories to report.

9-1 and 9-2 are in `CAMPAIGN_LEVELS` but have no scene bundle in this game build and no leaderboard on
speedrun.com (the board's last Fraud level is 8-4), so they are absent here by design.

