# ULTRAKILL AI

## Workflow rules

- **Always push to GitHub** after completing a change: commit, then `git push origin main` (remote: https://github.com/MasstarVT/ULTRAKILL-AI).
- **Always update this CLAUDE.md** as part of every change so it reflects the current state of the project (structure, setup, commands, conventions, decisions).

## Project status

- Repository initialized. `times.md` leaderboard template exists but has no entries yet (no generations trained).

## Structure

- `README.md` — project title
- `CLAUDE.md` — guidance for Claude Code sessions (keep current)
- `times.md` — leaderboard of the AI's level times: best time per level, plus a per-generation history table. Update it whenever a training generation finishes (instructions in an HTML comment at the bottom of the file)
