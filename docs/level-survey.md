# ULTRAKILL campaign: offline level-data survey of all main levels

**What this is.** A read-only parse of the shipped scene files for every main campaign level, run entirely
offline (no game launched, no bridge connection, nothing in the repo touched) while the `campaign_ppo_ground`
run was training. It answers, per level, what the agent will need **beyond what works on 0-1**.

**Where the data comes from.** `C:\Program Files (x86)\Steam\steamapps\common\ULTRAKILL\ULTRAKILL_Data\
StreamingAssets\aa\StandaloneWindows64\campaign_scenes_level<N>.bundle`, unpacked with the earlier
investigator's pure-Python UnityFS/SerializedFile reader (`unityfs.py`, `sfile.py`, `scene.py`, copied into
this folder unchanged) and analysed by `survey.py`, `detail.py`, `dump_graph.py`, `dump_anc.py`, `route.py`,
`rules*.py`, `gen_tables.py`. Class and field names were read from `F:\Github\ULTRAKILL-AI\decompiled`
(summarised, never copied). Raw per-level output: `level_survey.json` (all levels) and `raw/<lvl>*.json`.

**Gate/exit rules reproduced exactly.** `survey.py` implements `CampaignObserver.ScanGates` /
`ComputeHops` / `GoalRoom` / `ChooseExit` and spec §3.1-§3.2 literally: a door is a gate iff
`activatedRooms` holds ≥2 distinct GameObjects; room nodes are keyed by
`CampaignPatches.Key(room.transform.position)` (whole metres, `Mathf.RoundToInt`, matched by Python's
round-half-to-even); the goal room is the first room node on the chosen `FinalPit`'s own transform chain;
gate hops = min room hops; gate keys are the closed world position with `#2`-style collision suffixes.

**Verification against the live mod (do trust the numbers below).**

| check | live mod output | this parser |
|---|---|---|
| `Level 0-1` gate keys + hops (11 gates, hops 0..9) | `int_gates_Level_0-1_fresh.json` | **identical, in the same order**, including `40,1,408` at hops 9 and the two-gate hops-2 tier |
| `Level 0-1` exit | `[202.0, -17.1, 354.0]` | `[202.0, -17.1, 354.0]`, target `Level 0-2` |
| `Level 1-1` gate keys + hops (13 gates, hops 0..5) | `int_gates_Level_1-1_fresh.json` | **identical** |
| `Level 1-1` exit | `[81.0, -76.1, 91.0]` | same, target `Level 1-2` |
| `Level 0-1` checkpoints | 6 | 6 |

**Scope gap, please read.** `Level 9-1` and `Level 9-2` **are not in this game build**. The addressables
folder ships 43 scene bundles and none of them is `level9-*`; `GetMissionName.cs` knows missions 34/35, so
the code is newer than the content. **33 of the 35 `CAMPAIGN_LEVELS` can be trained on this install**, and
`eval.py --level "Level 9-1"` will fail at load. Everything below covers those 33.

**Parsed vs inferred.** Every number in Tables 1-4 is *parsed* from the scene file. The only *inferred*
columns are flagged inline: "hop ladder walkable" (derived from parsed room-sharing, rule stated below),
"boss in the exit room" (parsed: a `BossHealthBar` whose scene root equals the exit pit's scene root), and
the tier assignment in §6. The **player's spawn position is NOT determinable offline** - every level's
`Player` object sits at a fixed prefab staging offset inside `FirstRoom` ((0,105,253) on 31 of 33 levels,
(0,105,-47) on 0-2 and 2-4), so no "distance from spawn to the first gate" column is reported.

---

## 1-4. The tables

### Table 1 - route signal (`gates`), all parsed from the scene files with the mod's own rules

| Level | Doors | Gates | with `hops` | max hops | tiers >1 gate | widest fork (m) | rooms reached / nodes | `gates_ordered` | hop ladder walkable | ladder (m) | goal room |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-1 | 33 | 11 | 11 | 9 | 1 | 66.7 | 12/12 | **yes** | yes | 500.4 | 13 - Malicious Face Arena |
| 0-2 | 28 | 17 | 17 | 7 | 5 | 117.6 | 14/14 | **yes** | yes | 246.0 | 9 - Crushers Arena |
| 0-3 | 20 | 12 | 12 | 6 | 3 | 131.0 | 12/12 | **yes** | yes | 253.3 | 11 - Boss Arena - Floor 2 |
| 0-4 | 20 | 7 | 7 | 5 | 1 | 55.4 | 13/13 | **yes** | yes | 239.7 | 8 - Tube Hallway |
| 0-5 | 7 | 3 | 0 | - | 0 | 0.0 | 0/4 | **NO** | n/a | 0 | *none found* |
| 1-1 | 24 | 13 | 13 | 5 | 4 | 132.0 | 14/14 | **yes** | yes | 328.2 | 13 - End Room |
| 1-2 | 18 | 9 | 9 | 4 | 3 | 43.7 | 9/9 | **yes** | yes | 146.9 | 8 - Final Arena |
| 1-3 | 23 | 13 | 0 | - | 0 | 0.0 | 0/12 | **NO** | n/a | 0 | *none found* |
| 1-4 | 16 | 7 | 0 | - | 0 | 0.0 | 0/8 | **NO** | n/a | 0 | *none found* |
| 2-1 | 12 | 4 | 4 | 3 | 0 | 0.0 | 5/5 | **yes** | yes | 553.8 | 6 - Last Hallway |
| 2-2 | 24 | 3 | 3 | 2 | 0 | 0.0 | 5/5 | **yes** | yes | 168.2 | 5 - Second District |
| 2-3 | 21 | 9 | 9 | 2 | 2 | 104.0 | 9/9 | **yes** | yes | 110.9 | 5 - Final Arena |
| 2-4 | 21 | 0 | 0 | - | 0 | 0.0 | 0/0 | **NO** | n/a | 0 | *none found* |
| 3-1 | 22 | 9 | 9 | 7 | 1 | 121.7 | 11/11 | **yes** | yes | 576.6 | 10 - Structure |
| 3-2 | 7 | 4 | 4 | 3 | 0 | 0.0 | 5/5 | **yes** | yes | 647.1 | 4 - Heart Chamber |
| 4-1 | 14 | 9 | 9 | 8 | 0 | 0.0 | 11/11 | **yes** | yes | 693.8 | FinalDoorOpener |
| 4-2 | 25 | 3 | 0 | - | 0 | 0.0 | 0/4 | **NO** | n/a | 0 | *none found* |
| 4-3 | 30 | 5 | 3 | 2 | 0 | 0.0 | 4/7 | **yes** | yes | 138.4 | 7 - Generator Room |
| 4-4 | 21 | 5 | 0 | - | 0 | 0.0 | 0/5 | **NO** | n/a | 0 | *none found* |
| 5-1 | 34 | 3 | 3 | 2 | 0 | 0.0 | 4/4 | **yes** | yes | 300.2 | 4 - Water Processing Chamber |
| 5-2 | 32 | 0 | 0 | - | 0 | 0.0 | 0/0 | **NO** | n/a | 0 | *none found* |
| 5-3 | 47 | 20 | 20 | 13 | 5 | 190.4 | 26/26 | **yes** | yes | 796.7 | 1 - Hallway |
| 5-4 | 6 | 0 | 0 | - | 0 | 0.0 | 0/0 | **NO** | n/a | 0 | *none found* |
| 6-1 | 41 | 12 | 6 | 2 | 1 | 347.6 | 6/13 | **yes** | yes | 272.0 | 14 - Hall of Sacreligious Remains |
| 6-2 | 17 | 0 | 0 | - | 0 | 0.0 | 0/0 | **NO** | n/a | 0 | *none found* |
| 7-1 | 48 | 0 | 0 | - | 0 | 0.0 | 0/0 | **NO** | n/a | 0 | *none found* |
| 7-2 | 83 | 4 | 1 | 0 | 0 | 0.0 | 2/6 | **yes** | n/a (1 tier) | 0 | 15 - Forgotten Archive |
| 7-3 | 46 | 0 | 0 | - | 0 | 0.0 | 0/0 | **NO** | n/a | 0 | *none found* |
| 7-4 | 25 | 1 | 0 | - | 0 | 0.0 | 0/2 | **NO** | n/a | 0 | *none found* |
| 8-1 | 62 | 52 | 28 | 4 | 5 | 601.0 | 36/85 | **yes** | yes | 485.9 | 2 - TV Room / Hotel Hallway |
| 8-2 | 75 | 25 | 24 | 8 | 6 | 1644.2 | 39/41 | **yes** | yes | 836.1 | 19 - Mirror Reaper Arena |
| 8-3 | 85 | 32 | 1 | 0 | 0 | 0.0 | 2/72 | **yes** | n/a (1 tier) | 0 | Pit |
| 8-4 | 26 | 5 | 0 | - | 0 | 0.0 | 0/15 | **NO** | n/a | 0 | *none found* |
| 9-1 | - | - | - | - | - | - | - | - | - | - | *bundle not shipped* |
| 9-2 | - | - | - | - | - | - | - | - | - | - | *bundle not shipped* |

### Table 2 - locks, keys and other progress stoppers (parsed counts)

| Level | altars (`ItemPlaceZone`, real) | altar item types | altars wired to a **gate** door (hops) | carryables (`ItemIdentifier`) | `ItemTrigger` | `DoorLock` | locked `Door`s | `ArenaStatus` | `Breakable` | `Glass` |
|---|---|---|---|---|---|---|---|---|---|---|
| 0-1 | 0 | - | - | - | 0 | 0 | 0 | 0 | 36 | 31 |
| 0-2 | 2 | {'SkullBlue': 2} | - | {'SkullBlue': 2} | 0 | 0 | 2 | 0 | 9 | 13 |
| 0-3 | 0 | - | - | - | 0 | 0 | 0 | 0 | 7 | 22 |
| 0-4 | 0 | - | - | {'CustomKey1': 1} | 8 | 0 | 0 | 0 | 20 | 13 |
| 0-5 | 0 | - | - | - | 0 | 0 | 0 | 0 | 2 | 0 |
| 1-1 | 6 | {'SkullBlue': 4, 'SkullRed': 2} | [('20,-10,381', 2), ('16,20,427', 3)] | {'SkullRed': 2, 'SkullBlue': 3} | 0 | 0 | 0 | 3 | 7 | 2 |
| 1-2 | 5 | {'SkullRed': 2, 'SkullBlue': 3} | [('0,-15,380', 2), ('-26,30,457', 4)] | {'SkullRed': 2, 'SkullBlue': 3} | 0 | 0 | 0 | 1 | 28 | 0 |
| 1-3 | 4 | {'SkullBlue': 2, 'SkullRed': 2} | - | {'SkullBlue': 2, 'SkullRed': 2, 'Soap': 1} | 0 | 2 | 0 | 0 | 23 | 4 |
| 1-4 | 6 | {'SkullBlue': 6} | - | {'Readable': 1, 'SkullBlue': 7} | 0 | 3 | 0 | 0 | 14 | 0 |
| 2-1 | 0 | - | - | - | 0 | 0 | 0 | 0 | 10 | 6 |
| 2-2 | 0 | - | - | {'Readable': 2} | 0 | 6 | 0 | 0 | 23 | 0 |
| 2-3 | 6 | {'SkullRed': 4, 'SkullBlue': 2} | [('-67,8,375', 0), ('0,-2,421', 1), ('47,-9,385', 1)] | {'SkullRed': 3, 'SkullBlue': 3} | 0 | 0 | 0 | 1 | 9 | 10 |
| 2-4 | 0 | - | - | {'SkullRed': 3, 'SkullBlue': 2} | 0 | 0 | 0 | 0 | 3 | 2 |
| 3-1 | 0 | - | - | - | 0 | 0 | 0 | 0 | 4 | 0 |
| 3-2 | 0 | - | - | - | 0 | 0 | 0 | 0 | 1 | 0 |
| 4-1 | 0 | - | - | - | 0 | 0 | 0 | 0 | 3 | 0 |
| 4-2 | 0 | - | - | {'CustomKey1': 1, 'Readable': 1, 'SkullBlue': 2, 'SkullRed': 2} | 1 | 0 | 0 | 1 | 13 | 68 |
| 4-3 | 0 | - | - | {'Torch': 1, 'SkullBlue': 2, 'Soap': 1, 'Readable': 1} | 0 | 0 | 0 | 0 | 16 | 0 |
| 4-4 | 2 | {'SkullBlue': 2} | [('108,648,425', None)] | {'SkullBlue': 3} | 0 | 0 | 0 | 0 | 14 | 0 |
| 5-1 | 3 | {'SkullBlue': 3} | - | {'SkullBlue': 9} | 0 | 3 | 0 | 0 | 6 | 0 |
| 5-2 | 2 | {'SkullBlue': 1, 'SkullRed': 1} | - | {'Readable': 1, 'CustomKey1': 1, 'SkullBlue': 3, 'SkullRed': 2} | 0 | 0 | 0 | 0 | 6 | 24 |
| 5-3 | 4 | {'SkullRed': 2, 'SkullBlue': 2} | - | {'SkullRed': 5, 'SkullBlue': 5} | 0 | 4 | 0 | 0 | 521 | 15 |
| 5-4 | 0 | - | - | - | 0 | 0 | 0 | 0 | 6 | 0 |
| 6-1 | 2 | {'SkullRed': 2} | - | {'Readable': 1, 'CustomKey1': 11, 'SkullRed': 3} | 22 | 0 | 0 | 0 | 79 | 0 |
| 6-2 | 0 | - | - | {'CustomKey1': 3} | 6 | 0 | 0 | 0 | 44 | 0 |
| 7-1 | 4 | {'SkullBlue': 3, 'SkullRed': 1} | - | {'CustomKey1': 2, 'Readable': 2, 'SkullRed': 3, 'SkullBlue': 7} | 1 | 0 | 0 | 0 | 19 | 0 |
| 7-2 | 1 | {'SkullRed': 1} | - | {'CustomKey3': 1, 'CustomKey1': 1, 'CustomKey2': 1, 'Readable': 2, 'SkullRed': 3} | 3 | 0 | 0 | 0 | 25 | 13 |
| 7-3 | 0 | - | - | {'Torch': 1} | 0 | 0 | 7 | 0 | 5 | 0 |
| 7-4 | 0 | - | - | {'Readable': 1} | 0 | 0 | 0 | 1 | 12 | 6 |
| 8-1 | 2 | {'SkullBlue': 1, 'SkullRed': 1} | - | {'SkullBlue': 2, 'SkullRed': 2, 'CustomKey1': 1, 'Soap': 1} | 2 | 0 | 0 | 0 | 166 | 44 |
| 8-2 | 4 | {'SkullBlue': 2, 'SkullRed': 2} | [('16,-10,431', 5), ('285,-10,431', 6)] | {'Readable': 4, 'CustomKey1': 22, 'SkullBlue': 2, 'SkullRed': 2} | 81 | 0 | 0 | 0 | 777 | 45 |
| 8-3 | 4 | {'SkullBlue': 2, 'SkullRed': 2} | - | {'Readable': 2, 'CustomKey1': 4, 'SkullBlue': 2, 'SkullRed': 2} | 22 | 0 | 0 | 0 | 537 | 7 |
| 8-4 | 2 | {'SkullBlue': 1, 'SkullRed': 1} | - | {'SkullBlue': 2, 'CustomKey1': 5, 'SkullRed': 2} | 6 | 0 | 0 | 0 | 50 | 0 |
| 9-1 | - | - | - | - | - | - | - | - | - | - |
| 9-2 | - | - | - | - | - | - | - | - | - | - |

### Table 3 - arenas, bosses and the end of the level (parsed)

| Level | `ActivateArena` | of which lock doors | `ActivateNextWave` | `ActivateNextWaveHP` | enemy objects in arena+wave lists | `EnemyIdentifier` in scene | boss bars | boss in the exit room | end |
|---|---|---|---|---|---|---|---|---|---|
| 0-1 | 16 | 3 | 7 | 0 | 106 | 108 | MALICIOUS FACE | MALICIOUS FACE | FinalPit -> Level 0-2 |
| 0-2 | 15 | 8 | 12 | 0 | 76 | 125 | SWORDSMACHINE | - | FinalPit -> Level 0-3 |
| 0-3 | 18 | 2 | 3 | 0 | 67 | 60 | SWORDSMACHINE | SWORDSMACHINE | FinalPit -> Level 0-4 |
| 0-4 | 5 | 4 | 7 | 0 | 53 | 60 | - | - | FinalPit -> Level 0-5 |
| 0-5 | 1 | 1 | 2 | 1 | 0 | 2 | CERBERUS, GUARDIAN OF HELL | - | FinalPit -> Level 1-1 |
| 1-1 | 14 | 2 | 4 | 0 | 98 | 100 | - | - | FinalPit -> Level 1-2 |
| 1-2 | 13 | 4 | 6 | 0 | 109 | 104 | CANCEROUS RODENT, VERY CANCEROUS RODENT | - | FinalPit -> Level 1-3 |
| 1-3 | 21 | 7 | 20 | 0 | 139 | 141 | HIDEOUS MASS, SWORDSMACHINE "AGONY", SWORDSMACHINE "TUNDRA" | HIDEOUS MASS | FinalPit -> Level 1-4 |
| 1-4 | 1 | 1 | 2 | 0 | 2 | 3 | - | - | FinalPit -> Level 2-1 |
| 2-1 | 17 | 0 | 2 | 0 | 80 | 72 | - | - | FinalPit -> Level 2-2 |
| 2-2 | 22 | 0 | 9 | 0 | 86 | 80 | - | - | FinalPit -> Level 2-3 |
| 2-3 | 8 | 4 | 13 | 1 | 74 | 66 | MINDFLAYER | - | FinalPit -> Level 2-4 |
| 2-4 | 0 | 0 | 0 | 0 | 0 | 2 | THE CORPSE OF KING MINOS | THE CORPSE OF KING MINOS | FinalPit -> Level 3-1 |
| 3-1 | 14 | 4 | 13 | 1 | 104 | 95 | - | - | FinalPit -> Level 3-2 |
| 3-2 | 1 | 0 | 0 | 0 | 2 | 1 | GABRIEL, JUDGE OF HELL | GABRIEL, JUDGE OF HELL | FinalPit -> Intermission1 |
| 4-1 | 14 | 5 | 13 | 0 | 117 | 108 | - | - | FinalPit -> Level 4-2 |
| 4-2 | 10 | 2 | 15 | 0 | 107 | 114 | SISYPHEAN INSURRECTIONIST | SISYPHEAN INSURRECTIONIST | FinalPit -> Level 4-3 |
| 4-3 | 9 | 5 | 12 | 0 | 96 | 76 | MYSTERIOUS DRUID KNIGHT (& OWL) | - | FinalPit -> Level 4-4 |
| 4-4 | 4 | 2 | 1 | 1 | 7 | 5 | - | - | FinalPit -> Level 5-1 |
| 5-1 | 11 | 6 | 12 | 0 | 75 | 69 | - | - | FinalPit -> Level 5-2 |
| 5-2 | 9 | 3 | 12 | 0 | 44 | 44 | FERRYMAN | FERRYMAN | FinalPit -> Level 5-3 |
| 5-3 | 18 | 5 | 18 | 0 | 132 | 128 | - | - | FinalPit -> Level 5-4 |
| 5-4 | 0 | 0 | 0 | 0 | 0 | 1 | LEVIATHAN | LEVIATHAN | FinalPit -> Level 6-1 |
| 6-1 | 13 | 5 | 14 | 1 | 131 | 110 | HIDEOUS MASS, INSURRECTIONIST "ANGRY", INSURRECTIONIST "RUDE" | HIDEOUS MASS | FinalPit -> Level 6-2 |
| 6-2 | 0 | 0 | 0 | 0 | 0 | 1 | GABRIEL, THE APOSTATE OF HATE | GABRIEL, THE APOSTATE OF HATE | FinalPit -> Intermission2 |
| 7-1 | 6 | 6 | 20 | 0 | 39 | 61 | <s>MINOTAUR, BIG JOHNINATOR | <s>MINOTAUR | FinalPit -> Level 7-2 |
| 7-2 | 10 | 4 | 26 | 4 | 60 | 67 | GUTTERMAN | - | FinalPit -> Level 7-3 |
| 7-3 | 4 | 2 | 8 | 0 | 35 | 77 | - | - | FinalPit -> Level 7-4 |
| 7-4 | 5 | 0 | 6 | 1 | 33 | 49 | 1000-THR "EARTHMOVER", 1000-THR DEFENCE SYSTEM | - | FinalPit -> Level 8-1 |
| 8-1 | 11 | 4 | 19 | 1 | 87 | 93 | - | - | FinalPit -> Level 8-2 |
| 8-2 | 16 | 7 | 29 | 2 | 125 | 110 | MIRROR REAPER | MIRROR REAPER | FinalPit -> Level 8-3 |
| 8-3 | 25 | 11 | 53 | 4 | 178 | 201 | FERRYMAN "AGONIS", FERRYMAN "RUDRAKSHA", POWER "CHAUAKIAH", POWER "LEHAHIAH", POWER "MANADEL" | - | FinalPit -> Level 8-4 |
| 8-4 | 1 | 0 | 3 | 0 | 6 | 7 | - | - | FinalPit -> EarlyAccessEnd |
| 9-1 | - | - | - | - | - | - | - | - | - |
| 9-2 | - | - | - | - | - | - | - | - | - |

### Table 4 - start, checkpoints, exit detection, movement mechanics (parsed)

| Level | `CheckPoint` | pit room checkpoint-owned | `FinalPit` after filter | exit unique | `GearCheckEnabler` alt start | `JumpPad` | `MovingPlatform` | `Elevator` | `TramControl` | `Water`/`LimboWater` | `NavMeshLink` | `RandomizeWind` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-1 | 6 | yes | 1 | yes | yes (revalt, rev0) | 0 | 0 | 0 | 0 | 0/0 | 0 | 0 |
| 0-2 | 3 | yes | 1 | yes | no | 0 | 5 | 0 | 0 | 9/0 | 0 | 0 |
| 0-3 | 4 | yes | 1 | yes | yes (sho0) | 0 | 7 | 0 | 0 | 0/0 | 0 | 0 |
| 0-4 | 2 | yes | 1 | yes | no | 0 | 15 | 0 | 0 | 0/0 | 2 | 0 |
| 0-5 | 1 | no | 1 | yes | no | 0 | 0 | 0 | 0 | 3/0 | 0 | 0 |
| 1-1 | 4 | yes | 1 | yes | no | 0 | 0 | 0 | 0 | 0/1 | 0 | 0 |
| 1-2 | 3 | no | 1 | yes | yes (nai0) | 0 | 0 | 0 | 0 | 1/1 | 0 | 0 |
| 1-3 | 8 | no | 1 | yes | no | 0 | 0 | 0 | 0 | 2/1 | 0 | 0 |
| 1-4 | 2 | yes | 1 | yes | no | 0 | 0 | 0 | 0 | 0/1 | 0 | 0 |
| 2-1 | 2 | yes | 1 | yes | no | 18 | 1 | 0 | 0 | 4/0 | 0 | 0 |
| 2-2 | 4 | no | 1 | yes | no | 12 | 0 | 0 | 0 | 2/0 | 0 | 0 |
| 2-3 | 5 | yes | 1 | yes | no | 6 | 0 | 0 | 0 | 6/0 | 0 | 0 |
| 2-4 | 2 | yes | 2 | **no** | no | 0 | 0 | 0 | 1 | 1/0 | 0 | 0 |
| 3-1 | 5 | yes | 2 | **no** | no | 2 | 0 | 0 | 0 | 8/0 | 0 | 0 |
| 3-2 | 1 | yes | 2 | **no** | no | 0 | 9 | 0 | 0 | 1/0 | 0 | 0 |
| 4-1 | 5 | no | 1 | yes | no | 0 | 0 | 0 | 0 | 2/0 | 0 | 0 |
| 4-2 | 4 | no | 1 | yes | no | 1 | 0 | 0 | 0 | 1/0 | 0 | 0 |
| 4-3 | 3 | yes | 1 | yes | no | 0 | 0 | 0 | 0 | 1/0 | 0 | 0 |
| 4-4 | 3 | no | 1 | yes | no | 4 | 0 | 0 | 0 | 3/0 | 8 | 0 |
| 5-1 | 7 | yes | 1 | yes | no | 6 | 0 | 0 | 0 | 17/0 | 2 | 0 |
| 5-2 | 4 | no | 1 | yes | no | 2 | 0 | 0 | 0 | 1/0 | 0 | 0 |
| 5-3 | 6 | no | 1 | yes | no | 1 | 0 | 0 | 0 | 9/0 | 0 | 0 |
| 5-4 | 1 | yes | 1 | yes | no | 1 | 1 | 0 | 0 | 4/0 | 0 | 0 |
| 6-1 | 5 | no | 1 | yes | no | 4 | 11 | 0 | 0 | 9/0 | 0 | 0 |
| 6-2 | 1 | yes | 3 | **no** | no | 3 | 4 | 0 | 0 | 12/0 | 0 | 0 |
| 7-1 | 5 | yes | 1 | yes | no | 0 | 4 | 0 | 8 | 4/0 | 0 | 0 |
| 7-2 | 6 | no | 1 | yes | no | 0 | 8 | 0 | 13 | 5/0 | 0 | 0 |
| 7-3 | 4 | no | 1 | yes | no | 0 | 0 | 0 | 0 | 7/0 | 0 | 0 |
| 7-4 | 7 | no | 1 | yes | no | 15 | 1 | 0 | 0 | 2/0 | 0 | 1 |
| 8-1 | 7 | no | 1 | yes | no | 2 | 9 | 0 | 0 | 11/0 | 0 | 0 |
| 8-2 | 8 | no | 1 | yes | no | 0 | 0 | 4 | 0 | 4/0 | 0 | 0 |
| 8-3 | 13 | no | 1 | yes | no | 1 | 14 | 2 | 0 | 6/0 | 0 | 0 |
| 8-4 | 1 | no | 1 | yes | no | 4 | 2 | 1 | 0 | 0/0 | 0 | 0 |
| 9-1 | - | - | - | - | - | - | - | - | - | - | - | - |
| 9-2 | - | - | - | - | - | - | - | - | - | - | - | - |
---

## 5. What the tables say

### 5.1 The gates signal is a Prelude/Act-I-shaped mechanism, not a campaign-wide one

| | levels |
|---|---|
| `gates_ordered` true **and** every gate has a `hops` value | 0-1, 0-2, 0-3, 0-4, 1-1, 1-2, 2-1, 2-2, 2-3, 3-1, 3-2, 4-1, 5-1, 5-3 (14) |
| `gates_ordered` true but **part** of the level is outside the route | 4-3 (3/5 gates, 4/7 rooms), 6-1 (6/12, 6/13), 8-1 (28/52, 36/85), 8-2 (24/25, 39/41) (4) |
| `gates_ordered` true but the route is **one gate** - effectively dead | 7-2 (1/4 gates, max hops 0), 8-3 (1/32 gates, 2/72 rooms) (2) |
| `gates_ordered` **false** - no goal room at all | 0-5, 1-3, 1-4, 4-2, 4-4, 7-4, 8-4 (7) |
| `gates_ordered` false and there are **no gate candidates whatsoever** | 2-4, 5-2, 5-4, 6-2, 7-1, 7-3 (6) |

So today's signal is fully usable on **14 of 33**, partly usable on 4, and gives nothing on **15**.

> **Trap for Python:** `gates_ordered == true` does **not** mean the route is usable. On 8-3 one gate out of
> 32 carries `hops: 0` and the other 31 are `null`, so `GateProgress` would lock onto a door 892 m away from
> the start of the level and never retarget. Add a usability check before trusting the ladder, e.g.
> `gates_with_hops / gates >= 0.5` (this survey's numbers: 1.00 on the 14 good levels, 0.60/0.50/0.54/0.96
> on the four partial ones, 0.25 and 0.03 on 7-2 and 8-3).

### 5.2 The hop ladder, where it exists, is walkable - no backtracking

Test (inferred from parsed data): for every hop level *h*, is there a gate at tier *h* and a gate at tier
*h-1* that **share a room key**? If yes, you leave gate *h* into a room and cross that same room to gate
*h-1* - one physical step forward. **Every ordered level passes with zero breaks** (column "hop ladder
walkable" in Table 1). Following "walk to the lowest-hops reachable gate" is therefore a valid route on all
of them; the agent never has to go back up the ladder.

Two caveats that are about geometry, not ordering:

- **Forks are large.** "Widest fork" is the distance between two gates sharing one hop value. It is 66.7 m on
  0-1, but 117-132 m on 0-2/0-3/1-1/3-1, 190 m on 5-3, **601 m on 8-1 and 1644 m on 8-2**. Picking the wrong
  member of a tier on 8-2 is a 1.6 km mistake, and §5 of the gates spec picks "nearest gate in the tier",
  which on those levels is a coin flip made once per episode.
- **Single steps can be long.** Largest tier-to-tier step: 339 m (2-1), 344 m (3-2), 451 m (8-2), 275 m (8-1),
  231 m (6-1) versus 81.6 m on 0-1. Total ladder length runs 111 m (2-3) to 836 m (8-2).

### 5.3 Why the 13 unordered/degenerate levels fail, and why there is no cheap rule fix

The exit's room is the 4th/5th ancestor of the `FinalPit` and is always named `<N> - <Room>` under
`<N> Stuff` under `FinalRoom` under `Pit`. It fails to become a room node because **the thing that switches
it on is not a two-room door**. Parsed, per level (`raw/<lvl>.graph.json`, `pit_chain[].refs`):

| what activates the exit room | levels |
|---|---|
| a **one-room** `Door` (`activatedRooms` length 1, so it is not a gate candidate) | 1-3, 1-4, 6-2, 7-1, 7-3, 7-4 |
| `ActivateNextWave.toActivate` only | 0-5, 4-2, 5-2, 8-3 |
| only a `CheckPoint.rooms` entry | 2-4, 5-4 |
| nothing at all - only the `Pit` object itself is referenced | 4-4, 8-4 |

Three candidate rule changes were implemented and measured offline (`rules.py`, `rules2.py`, `rules3.py`).
**All three were rejected; do not spend time re-trying them:**

- **Rule B - use `activatedRooms ∪ deactivatedRooms` for adjacency.** Fixes nothing (0 levels gained) and
  *destroys* the ladder everywhere: `deactivatedRooms` names far-away rooms, which adds shortcut edges and
  collapses 0-1 from 9 hops to 4, 5-3 from 13 to 5, 4-1 from 8 to 3.
- **Rule D - add each one-room door as an edge between the room it opens and the room node it physically
  sits inside (nearest room-node ancestor).** Gains **0** unordered levels; it only makes the object literally
  named `Pit` the goal room on 24 levels and shifts every hop by +1. (Exactly the failure §3.1 rule 2 of the
  spec predicted.)
- **Rule E - rule D plus `ActivateNextWave` as a gate (`toActivate` rooms joined to the wave trigger's own
  room).** Gains exactly **one** level (7-4: 0 → 6 gates with hops). The other wave triggers sit in rooms that
  are not room nodes either, so the edge has nothing to attach to.

**Conclusion:** the door graph genuinely does not reach the end of those levels. The second route source has
to be something else. The cheapest one already in the observation is the **checkpoint chain** - 2-4 and 5-4's
exit rooms are referenced by `CheckPoint.rooms` and nothing else, and the unordered levels average 3.3
checkpoints (1-3 has 8, 7-4 has 7, and the degenerate 8-3 has 13). A `checkpoint hops` ladder built the same
way the gate ladder is built is the natural fallback, and needs no new mod field. The counter-evidence to be
honest about: 0-5, 5-4, 6-2 and 8-4 have exactly **one** checkpoint, so on those the fallback degenerates too
and only exploration plus the straight-line exit vector is left - which is today's behaviour.

### 5.4 Exit detection: one real ambiguity

`ChooseExit`'s filter (drop `fakeEnd`/`secondPit`/`rankless`/templates/empty target/`*-S`) leaves **exactly
one** candidate on **29 of 33** levels. The four exceptions:

- **2-4** (2 candidates) and **3-2** (2): duplicate pits at the same place with the same target. Harmless.
- **3-1** (2): the real pit (`Level 3-2`) plus a **Prime Sanctum pit targeting `Level P-1`**. The successor
  rule resolves it correctly - this is the rule earning its keep.
- **6-2 (3 candidates): genuinely ambiguous and a live bug risk.** The real exit targets `Intermission2`,
  which does not parse as `Level a-b`, so it is *not* a successor and ranks 2+1=3; the `Level P-2` Prime
  Sanctum pit ranks 2+1=3 as well. The tie falls through to `FindObjectsOfType` order, i.e. the mod may pick
  the Prime Sanctum pit as 6-2's exit. **Fix suggestion:** also treat a target that starts with
  `"Intermission"` as a successor, or explicitly drop targets matching `^Level P-\d`. 3-2 targets
  `Intermission1` too but its two candidates are duplicates, so it is only latent there. 8-4 targets
  `EarlyAccessEnd` with a single candidate, so it is fine.

Also noted: on **4-1** the goal room resolves to a GameObject named `FinalDoorOpener`. That is not a bug - a
real two-room door lists it, and it sits at the same rounded key as the pit's `FinalRoom` ancestor and is
listed alongside `10 - End Room` by the same door - but the name in any log will look wrong.

### 5.5 Locks and keys

Mechanism classes found in the level data (names from `decompiled/`):

- **`ItemPlaceZone`** - the altar. `acceptedItemType` (`ItemType`: `SkullBlue`/`SkullRed`/`SkullGreen`/
  `Torch`/`Soap`/`Readable`/`CustomKey1..3`), `doors: Door[]` (calls `Door.Open(enemy:false, skull:true)` on
  success), `reverseDoors: Door[]` (`Close()`), `arenaStatuses: ArenaStatus[]` (increments a counter other
  triggers wait on), `activateOnSuccess/deactivateOnSuccess: GameObject[]`.
- **`ItemIdentifier`** - the carryable. `itemType`, `pickedUp`, `ipz` (the altar it is in), `infiniteSource`.
- **`ItemTrigger`** - the *other* item lock: a trigger volume with `targetType` firing an `UltrakillEvent`.
  Heavily used late: 6-1 has 22, 8-3 22, 8-2 **81**, 0-4 8, 6-2 6, 8-4 6.
- **`DoorLock`** (a door that needs unlocking; 1-3 2, 1-4 3, 2-2 6, 5-1 3, 5-3 4) and `Door.locked`
  (0-2 has 2 locked doors, 7-3 has 7).
- **`ArenaStatus`** - counter gate; 1-1 3, 1-2/2-3/4-2/7-4 1 each.
- **`ActivateNextWaveHP`** - the boss-health phase trigger: 0-5, 2-3, 3-1, 4-4, 6-1, 7-4, 8-1 (1 each),
  8-2 (2), 7-2 and 8-3 (4 each).
- **`Breakable` / `Glass` / `GlassBreaker`** - present everywhere; 5-3 has 521 breakables, 8-2 777, 8-3 537,
  8-1 166. Mostly decorative, but `GlassBreaker` (0-4, 2 of them) is a progress mechanism.
- **Not progress gates, despite the names:** `LimboSwitch`/`LimboSwitchLock` (persistent secret switches,
  `GameProgressSaver.GetLimboSwitch`), `StatueActivator` (turns on a decorative `StatueFake` and destroys
  itself in `Start`), `ScriptActivator` (pistons and light pillars), `AbruptLevelChanger` (the pause menu's
  restart/quit buttons - it is 3/6/9 on every level).

**17 of 33 levels have at least one functional altar** (an `ItemPlaceZone` with a real item type *and* at
least one door/arena-status wired to it): 0-2, 1-1, 1-2, 1-3, 1-4, 2-3, 4-4, 5-1, 5-2, 5-3, 6-1, 7-1, 7-2,
8-1, 8-2, 8-3, 8-4. On **5** of those the altar drives a door that **is a gate**, so the lock is provably on
the route: 1-1 (hops 2, 3), 1-2 (hops 2, 4), **2-3 (hops 0 - the door into the exit room)**, 4-4, 8-2 (hops
5, 6). On the other 12 the altar drives a one-room door, so the criticality of the skull cannot be settled
offline and needs one live check per level.

### 5.6 Arenas, bosses, level end

- **Every one of the 33 levels ends at a `FinalPit`.** There is no boss-death-only or cutscene-only ending in
  the data. `FinalDoor`/`FinalDoorOpener` (2-6 per level) is not an end condition: `FinalDoorOpener.Awake`
  calls `fd.Open()` on its parent `FinalDoor` and, when `startTimer` is set, starts the level timer - it is the
  opening-door helper, the same one that fires at the start of every level.
- **A boss stands in the exit room on 12 levels** (parsed: a `BossHealthBar` whose scene root is the exit
  pit's scene root): 0-1 Malicious Face, 0-3 Swordsmachine, 1-3 Hideous Mass, 2-4 The Corpse of King Minos,
  3-2 Gabriel Judge of Hell, 4-2 Sisyphean Insurrectionist, 5-2 Ferryman, 5-4 Leviathan, 6-1 Hideous Mass,
  6-2 Gabriel the Apostate of Hate, 7-1 `<s>`Minotaur, 8-2 Mirror Reaper. (0-5 Cerberus and 7-4 Earthmover
  sit in a different scene root from their pit but are still the last fight.)
- **Arena size.** `ActivateArena` count 0-25 per level; enemy GameObjects listed by arenas+waves 0-178;
  `EnemyIdentifier` components in the scene 1-201. The extremes: **8-3** (25 arenas, 53 waves, 201 enemies,
  13 checkpoints) and **2-2** (22 arenas, 86 enemy objects, 0 locking arenas). Boss-only levels have
  essentially no arena content: 2-4, 5-4, 6-2 all have 0 arenas and 1-2 `EnemyIdentifier`s.
- 0-1's 16 arenas / 108 enemies is **mid-range**, not small: 1-3 (141), 5-3 (128), 0-2 (125), 8-2 (110),
  6-1 (110) are larger. Combat throughput is not the thing that scales worst; route structure is.

### 5.7 Start, movement, checkpoints

- **Alternate start.** `GearCheckEnabler` reads `GameProgressSaver.CheckGear(gear)`, which
  `CampaignPatches` already forces to 1 whenever `unlock_all_gear` is on. Only three levels carry one:
  **0-1** (`rev0`, `checkForFullIntro`, activates 4 objects and deactivates 3 - this is the whole
  "`1Alt - Short Starting Room` instead of `FirstRoom`" swap, and it is unique to 0-1), **0-3** (`sho0`,
  deactivates 1) and **1-2** (`nai0`, deactivates 1) - those two just remove a weapon pickup, no room swap.
  So the 0-1 start quirk does **not** generalise; every other level starts in its authored first room.
- **Movement mechanics visible in the data**, by component (`JumpPad`, `MovingPlatform`, `Elevator`,
  `TramControl`, `Water`/`WaterObject`/`LimboWater`, `NavMeshLink`, `RandomizeWind`, `SpiderBodyTrigger`):
  - **Jump pads**: 2-1 (18), 7-4 (15), 2-2 (12), 2-3/5-1 (6), 4-4 (4), 6-1 (4), 8-4 (4).
  - **Moving platforms**: 0-4 (15), 8-3 (14), 6-1 (11), 3-2 (9), 8-1 (9), 7-2 (8), 0-3 (7), 0-2 (5).
  - **Trams (`TramControl`)**: 7-2 (13), 7-1 (8), 2-4 (1) - a ride the agent must board and stay on.
  - **Elevators**: 8-2 (4), 8-3 (2), 8-4 (1).
  - **Water**: 5-1 (17), 6-2 (12), 8-1 (11), 5-3/6-1/3-1 (8-9), plus `LimboWater` on 1-1..1-4.
    5-4 is a swimming boss level.
  - **Wind**: only 7-4 (`RandomizeWind`).
  - **Baked NavMesh links** (gaps the mesh itself bridges): 4-4 (8), 0-4 (2), 5-1 (2). Everywhere else the
    NavMesh path hint stops at every jump, which is why `path.status` is `partial` so often.
  - No component in any level encodes "dash/slam storage is required"; that is a movement-tech question the
    data cannot answer. The closest data signal is jump-pad and moving-platform density above.
- **Checkpoints**: 1 (0-5, 3-2, 5-4, 6-2, 8-4) to 13 (8-3); median 4. **The exit pit's room is
  checkpoint-owned on 16 of 33** (Table 4). That matters because §3.1 rule 7 exists precisely for that case -
  a respawn destroys and re-instantiates the room holding the pit, so the hops map must survive the scan. It
  is live on 0-1, 0-2, 0-3, 0-4, 1-1, 1-4, 2-1, 2-3, 2-4, 3-1, 3-2, 4-3, 5-1, 5-4, 6-2, 7-1.

---

## 6. Tier table

One tier per level - the capability that must land **first** for the level to be finishable at all. Secondary
needs are listed after the blocking one.

| Level | Tier | The one blocking mechanism | also needs |
|---|---|---|---|
| 0-1 | **A** | none - the pilot level | - |
| 0-3 | **A** | none (12/12 gates, 12/12 rooms) | mid-level Swordsmachine, 7 moving platforms |
| 0-4 | **A** | none (7/7 gates) | 15 moving platforms, 2 `NavMeshLink`, `GlassBreaker`, 8 `ItemTrigger` |
| 2-1 | **A** | none (4/4 gates) | 18 jump pads, 339 m ladder step |
| 2-2 | **A** | none (3/3 gates) | 12 jump pads, 6 `DoorLock`, 22 arenas |
| 3-1 | **A** | none (9/9 gates, 7 hops) | 14 arenas, 104 enemy objects |
| 4-1 | **A** | none (9/9 gates, 8 hops) | longest clean ladder outside 5-3 (694 m) |
| 0-2 | **B** | 2 blue-skull `ItemPlaceZone`s (criticality unverified) | 8 locking arenas, 5 moving platforms, 44 `DeathZone` |
| 1-1 | **B** | 6 skull altars, **4 on gate doors** (hops 2, 3) | `ArenaStatus` x3, `BigDoor` x28 |
| 1-2 | **B** | 5 skull altars, 3 on gate doors (hops 2, 4) | alt start (`nai0`) |
| 2-3 | **B** | 6 skull altars; **one opens the hops-0 gate into the exit room** | Mindflayer, `ActivateNextWaveHP` |
| 5-1 | **B** | 3 blue-skull altars; route is only 3 gates / 4 rooms | 17 water volumes, 7 checkpoints, 6 locking arenas |
| 5-3 | **B** | 4 skull altars | longest route in the game (20 gates, 13 hops, 797 m), 521 breakables |
| 8-2 | **B** | 4 altars, 2 on gate doors (hops 5, 6) | Mirror Reaper, **1644 m fork**, 81 `ItemTrigger`, 4 elevators |
| 0-5 | **C** | Cerberus kill opens the exit; 0/3 gates ordered, 1 checkpoint | `ActivateNextWaveHP` |
| 2-4 | **C** | Minos corpse; **0 gate candidates**, exit room only `CheckPoint`-referenced | 1 tram, exit not unique (harmless) |
| 3-2 | **C** | Gabriel; route is fine (4/4 gates) but the level is the fight | 9 moving platforms, 1 checkpoint |
| 4-2 | **C** | Sisyphean Insurrectionist; no goal room (wave-activated exit room) | 68 `Glass`, 13 `SpiderBodyTrigger` |
| 5-2 | **C** | Ferryman + the ship; **0 gate candidates** | 6 altars, 24 `Glass` |
| 6-2 | **C** | Gabriel 2; 0 gates, **exit choice ambiguous vs the P-2 pit** | 12 water, 6 `ItemTrigger` |
| 7-4 | **C** | Earthmover; no goal room today (rule E would give 6 gates) | 15 jump pads, the only `RandomizeWind` |
| 1-3 | **D** | no goal room (exit room behind a one-room door) | 4 skull altars, 21 arenas, 8 checkpoints, 141 enemies |
| 1-4 | **D** | no goal room; V2 fight | 6 blue-skull altars, 3 `DoorLock` |
| 4-3 | **D** | partial route (3/5 gates, 4/7 rooms) | Druid Knight, 5 locking arenas |
| 4-4 | **D** | no goal room; exit room referenced by nothing but the pit | 2 blue-skull altars, 8 `NavMeshLink`, 4 jump pads |
| 6-1 | **D** | partial route (6/12 gates, 6/13 rooms) | 2 red-skull altars, 22 `ItemTrigger`, 11 moving platforms |
| 7-3 | **D** | **0 gate candidates** despite 46 doors | 7 locked doors, Torch item |
| 8-1 | **D** | route covers 36/85 rooms; 601 m fork | 2 altars, 166 breakables, 85 room nodes |
| 8-4 | **D** | no goal room; 5 gates all `null`; 1 checkpoint for the whole level | 2 altars, 5 `CustomKey1` |
| 5-4 | **E** | Leviathan in water: 0 gates, 0 arenas, 1 checkpoint, swimming | - |
| 7-1 | **E** | 8 trams: **0 gate candidates** across 48 doors | 4 skull altars, 6 locking arenas |
| 7-2 | **E** | 13 trams; route degenerate (1/4 gates, max hops 0) | 4 `ActivateNextWaveHP`, Gutterman |
| 8-3 | **E** | giant scale: 72 room nodes, route dead (1/32 gates), 13 checkpoints, 201 enemies | 4 altars, 22 `ItemTrigger`, 5 boss bars |
| 9-1 | **-** | **not shipped in this build** | - |
| 9-2 | **-** | **not shipped in this build** | - |

**Counts:** A 7, B 7, C 7, D 8, E 4 = 33 analysable (+2 missing).

---

## 7. Recommended order of sub-projects

1. **Finish the 0-1 pilot, then take the rest of Tier A - in Prelude order first (0-3, 0-4).** Zero new code:
   same gates signal, same arena loop, ladders of the same shape (12/12 and 7/7 gates, walkable chains, no
   key items). 0-3 and 1-2 also exercise `GearCheckEnabler` under `unlock_all_gear`, which is cheap insurance
   that the 0-1 alternate-start path is not a special case. 2-1, 2-2, 3-1, 4-1 are the same capability but
   need the Act-I arsenal, so they slot in after the Prelude is clean.
2. **Tier B: skull carry.** 7 levels, and 5 of them (1-1, 1-2, 2-3, 4-4, 8-2) have an altar wired to a door
   that already carries a `hops` value, so the sub-goals inherit the ladder for free (§8). 1-1 is the next
   level in mission order, so this is the highest return per unit of work and should come straight after
   Tier A. It also unblocks part of Tier D (1-3, 1-4, 6-1, 8-1, 8-4 all have altars on top of their route
   problem).
3. **Tier C: boss end condition.** 7 levels. No new *route* work: 3-2 already has a clean 4-gate ladder, and
   on the rest the route signal is absent anyway, so what is needed is (a) an episode long enough for a boss
   (`ActivateNextWaveHP` phases mean the fight is multi-stage on 7 levels), (b) reward credit for
   `arena_enemies_alive` falling rather than for a new cell, and (c) the 6-2 `ChooseExit` fix in §5.4, which
   is a 2-line mod change and should be done immediately regardless of tier order.
4. **Tier D: a second route source.** 8 levels and the hardest planning item, because §5.3 shows no door-graph
   rule fixes it. Recommended shape: build the same BFS ladder over **checkpoints** (already in the obs, and
   on 2-4/5-4 the exit room is referenced by nothing else), publish `checkpoint hops` next to `gates`, and let
   `GateProgress` fall back to it when `gates_with_hops / gates < 0.5`. That single fallback also rescues the
   two degenerate levels in Tier E (7-2, 8-3).
5. **Tier E: special mechanics last.** Trams (7-1, 7-2) need "board and stay on a moving platform", 5-4 needs
   swimming, 8-3 needs everything at once at 3x the scale. None of them is worth attempting before 2-4 works.

**Two things worth doing out of order, now, because they are cheap:** the 6-2 exit tie-break (§5.4), and the
`gates_with_hops / gates` usability guard in Python (§5.1) - without it 8-3 and 7-2 will silently train
against a one-gate route.

---

## 8. Cheapest level-data-only extension of the gates signal for skull keys

No demos, no recorded routes - the order comes from the lock graph the level already carries.

1. **`campaign.altars[]`** (new): one entry per live `ItemPlaceZone` (`FindObjectsOfType(true)`, existing
   `IsTemplate` filter, rescanned in `Scan()` like `doors`):
   `{"key", "pos", "item": "SkullRed"|…, "filled": bool, "doors": [{"key","pos"}], "reverse_doors": [...]}`.
   `item` is `acceptedItemType` (skip `ItemType.None` - 27 of the 131 zones across the 33 levels are decorative);
   `filled` is `GetComponentInChildren<ItemIdentifier>()?.itemType == acceptedItemType`; the door keys use the
   **same** `ClosedPosition()` + `CampaignPatches.Key()` the gates already use, so they line up by string.
2. **`campaign.items[]`** (new): one entry per live `ItemIdentifier` with `itemType != None`:
   `{"key", "pos", "item", "held": pickedUp, "placed": ipz != null}`. `pos` must be read **per step** (a
   carried skull moves); everything else is scan-time.
3. **`gates[].needs_item`** (new, derived, nullable): the `item` of any **unfilled** altar whose `doors`
   contains this gate's key. This is the only field `GateProgress` has to read to know a gate is skull-locked.
4. **Python (`GateProgress._choose_target`), the whole change:** when the chosen target gate has
   `needs_item = T` and no held item of type `T`, the target becomes the nearest `items[]` entry with
   `item == T and not held and not placed`; once one is held, the target becomes the nearest unfilled
   `altars[]` entry with `item == T`; once it is filled, the target reverts to the gate. Two extra rungs
   **below** that gate's existing `hops` - the ordering is inherited, nothing new is computed.
5. **Rewards:** two `MilestoneTracker` entries paid once per level load, `item_pickup` and `item_placed`,
   keyed exactly like `checkpoint`/`arena_clear`. Reuse `gate_approach` unchanged: `best_dist` is already
   per-target-key, so the skull and the altar each pay their approach once and a skull-altar shuttle is not
   farmable.
6. **Observation:** none of this needs new width - the sub-goal reuses the gate-target slots (448-455), with
   the existing "target changed" semantics.
7. **Why it is enough:** measured, altars drive a door that is already a gate on 1-1 (hops 2, 3), 1-2 (2, 4),
   2-3 (0, 1, 1), 4-4 and 8-2 (5, 6). On the other 12 skull levels the altar drives a one-room door, which is
   exactly why `altars[].doors[].pos` must be published even when the door is not in `gates` - Python can
   still steer to it.
8. **Also cover `ItemTrigger`** in `items[]`'s consumers eventually: it is the same lock with a
   `UnityEvent` instead of doors (8-2 has 81, 6-1 and 8-3 have 22). It can be deferred - no Tier A or Tier B
   level depends on one - but 6-1 and 8-2 will need it.
9. **Cost estimate:** ~120 lines in `CampaignObserver` (two scan lists, one derived field), ~60 lines in
   `campaign.py`, two reward weights, and the fake-level tests in `test_campaign_env.py` extended with a
   skull room. No change to observation width, so **no checkpoint is invalidated**.
10. **What it does not do:** it does not fix the 15 levels with no usable door route (§5.3). Skull carry and
    route repair are independent; do skull carry first because it is bounded and it unblocks 1-1.
