# Times

Leaderboard of the AI's level completion times across training generations.
Times are in-game level time (`mm:ss.mmm`), not wall-clock training time.

## Leaderboard

Best time per level. A generation only takes a spot by beating the current record.

| Level | Time | Rank | Generation | Difficulty | Date | Notes |
|-------|------|------|------------|------------|------|-------|
| 0-1 | 01:57.464 | A | spec_0-1_speed@25.58M | Violent | 2026-09-19 | training episode (sampled actions), fresh start |
| 0-2 | 01:58.742 | S | spec_0-2_speed@19.78M | Violent | 2026-09-19 | training episode (sampled actions), fresh start |
| 0-3 | 04:23.904 | B | campaign_gates@13.39M | Violent | 2026-09-18 | training episode (sampled actions), fresh start |

## Generation history

Best run from each generation, newest first. Keep every generation here, even ones that didn't set a record.

| Generation | Level | Time | Rank | Kills | Deaths | Δ vs previous | Date | Notes |
|------------|-------|------|------|-------|--------|---------------|------|-------|
| spec_0-2_speed@19.78M | 0-2 | 01:58.742 | S | 53 | 1 | -1.872s | 2026-09-19 | training episode (sampled actions), fresh start |
| spec_0-2_speed@19.39M | 0-2 | 02:00.614 | S | 53 | 0 | -7.125s | 2026-09-19 | training episode (sampled actions), fresh start |
| spec_0-2_speed@19.23M | 0-2 | 02:07.739 | A | 53 | 2 | -0.188s | 2026-09-19 | training episode (sampled actions), fresh start |
| spec_0-1_speed@25.58M | 0-1 | 01:57.464 | A | 40 | 0 | -16.601s | 2026-09-19 | training episode (sampled actions), fresh start |
| spec_0-1_speed@25.18M | 0-1 | 02:14.065 | A | 40 | 0 | -4.867s | 2026-09-19 | training episode (sampled actions), fresh start |
| spec_0-1_speed@24.70M | 0-1 | 02:18.932 | A | 49 | 1 | -3.369s | 2026-09-19 | training episode (sampled actions), fresh start |
| spec_0-1_speed@23.14M | 0-1 | 02:22.301 | A | 46 | 0 | -14.683s | 2026-09-19 | training episode (sampled actions), fresh start |
| spec_0-1_speed@19.47M | 0-1 | 02:36.984 | B | 39 | 0 | -26.644s | 2026-09-19 | training episode (sampled actions), fresh start |
| campaign_gates@13.39M | 0-3 | 04:23.904 | B | 6 | 3 | -132.156s | 2026-09-18 | training episode (sampled actions), fresh start |
| campaign_gates@10.18M | 0-2 | 02:07.927 | S | 53 | 0 | -35.284s | 2026-09-17 | training episode (sampled actions), fresh start |
| campaign_gates@11.70M | 0-3 | 06:36.060 | D | 7 | 5 | — | 2026-09-17 | training episode (sampled actions), fresh start |
| campaign_gates@9.79M | 0-2 | 02:43.211 | A | 53 | 5 | -52.549s | 2026-09-17 | training episode (sampled actions), fresh start |
| campaign_gates@9.67M | 0-2 | 03:35.760 | A | 53 | 0 | -188.146s | 2026-09-17 | training episode (sampled actions), fresh start |
| campaign_gates@8.24M | 0-2 | 06:43.906 | B | 51 | 2 | — | 2026-09-17 | training episode (sampled actions), fresh start |
| campaign_gates@6.85M | 0-1 | 03:03.628 | A | 61 | 0 | -76.656s | 2026-09-17 | training episode (sampled actions), fresh start |
| campaign_gates@6.44M | 0-1 | 04:20.284 | B | 63 | 1 | -180.430s | 2026-09-17 | training episode (sampled actions), fresh start |
| campaign_gates@5.57M | 0-1 | 07:20.714 | B | 64 | 1 | — | 2026-09-17 | training episode (sampled actions), fresh start |

<!--
How to add an entry:
- Generation history: add a row at the top of the table for the new generation's best run.
- Leaderboard: if that run beats the level's record, replace the level's row (one row per level, sorted by level order).
- Rank is the in-game style rank (D, C, B, A, S, P). Δ vs previous is the time change from the previous generation on the same level, e.g. -1.250s.
-->
