# v7 dataset audit and expansion

Scope: `dataset7b/socratic_v7_train.jsonl` (7,774), `dataset7b/socratic_v7_val.jsonl` (863), `dataset7b/socratic_dataset_v7_annotated.jsonl` (8,637 = train+val). **28 topics / 54 concepts.** Do not edit training scripts in this track.

After each numbered item: **counts + examples**, then wait. No delete/rewrite/downsample until approved.

## Why the model rambles / dumps answers / goes off-topic

v7 is wide but shallow: 7,774 chats, ~54 concepts, each ~140–160 clones of the same tutor stem. Full-sequence SFT also trained on student tokens (fixed on the training track).

- ~540 `CLARIFY` turns after “explain / tell me” dump a definition with no `?` (breaks the system prompt).
- 100% of train chats end with a no-question VERIFY; ~35% echo “is exactly right.”
- Stock openers (~26% of assistant turns) + dense `SWITCH_TOPIC` / generic “I’m lost, explain” openers teach off-topic jumps.

## Audit

1. **Answer dumps** — list only until approved.
2. **Downsample** — by **science stem** or annotated **concept**, not generic first lines. If >20 chats, propose keep 15–20 with the most varied student first-answers (word count ≥5).
3. **Coverage** vs system-prompt topics; list where **new unique stems** are needed.
4. **Uniqueness** — flag stems with &lt;60% unique first-answers (wc ≥5). List only.
5. **Length** — flag chat-templated sequences &gt;700 tokens (Qwen2.5-7B tokenizer).
6. **Holdout** — after a list of ~30–40 stems/topics, move all matching train rows to `dataset7b/holdout_eval.jsonl`.

When mutations are approved: rewrite or drop definition dumps; strip VERIFY answer-echo; keep only a thin slice of topic-switch and generic openers.

**Grouping caveat:** 316 “unique questions” includes generic openers (`I'm lost, can you explain?` ×90). Downsampling those keys mixes many concepts.

## Expansion (after cleanup, not more clones)

- **8–15 distinct student questions per concept** (~400–800 unique stems), not more paraphrases of the same 54 scripts.
- New chats: Socratic tutor turns (`?` except a short close); never dump the answer when asked to explain.
- Write first into thin coverage areas (D3), not Chemical Reactions (already 639 annotated).
- Hold out 30–40 of the **new** stems.

## Reports

Written under `plan/reports/` as each audit item runs.

- D1 (listed, not removed): [reports/d1-answer-dumps.md](reports/d1-answer-dumps.md) — 378 train dumps; full JSONL alongside.
