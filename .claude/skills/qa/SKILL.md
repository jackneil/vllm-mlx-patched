---
name: qa
description: "Browser-based QA testing — targeted single-component check (/qa) or parallel multi-aspect review (/ux). (repo)"
---
Two commands are available — read the appropriate one and follow it:
- `.claude/commands/qa.md` — Quick, focused QA pass (single agent). Best for targeted fixes or single-feature verification.
- `.claude/commands/ux.md` — Thorough parallel UX review (multiple agents). Best when changes touch layout, navigation, or multiple components.

Both are read-only detection tools — they return a detailed issue list but do NOT fix code.

Decision guide:
- Changed button styling or a single component? → `/qa`
- Changed layout, interactions, AND multiple pages? → `/ux`

**Note for this repo:** vllm-mlx-patched is a Python backend with no UI. These commands rarely apply here — if the user invokes them, confirm whether they meant the arena UI (`hank-llm-arena`) or the secure-llm Tauri app (`hank-secure-llm`) instead.
