---
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
tags:
  - socratic
  - education
  - science
  - grade-10
  - chat
---

# v7 Socratic Grade 10 science chats (`socratic_v7_final_v3`)

JSONL of `messages` (system / user / assistant) for QLoRA SFT of a Socratic Grade 10 science tutor.

- File: `socratic_v7_final_v3.jsonl` (~1429 chats).
- Train 4B adapters to **Susu11/v7_4b_qwen** (do not overwrite `Susu11/v9socratic4b`).
- Train 7B adapters to **Susu11/v7_qwen7b** (do not overwrite `Susu11/qwen2.5-7b-socratic-tutor`).

Code: [Sushey01/Socratic-Model-Fine-Tune](https://github.com/Sushey01/Socratic-Model-Fine-Tune). Upload: `python upload_v7_socratic_data.py`.
