# Benchmark and pedagogical evaluation (ScienceQA + W&B)

This document describes the **evaluation suite that actually runs in this repo** during fine-tune. It is a practical slice of a larger K-10 Socratic-tutor evaluation story (TutorBench, MRBench, SocraticBench). Those other suites need LLM-as-a-judge or two-model dialogue loops and are **not** scored after every epoch here.

**Code:** [eval_scienceqa.py](eval_scienceqa.py) (dataset, metrics), [run_eval.py](run_eval.py) (eval-only, no SFT), and `ScienceQAEpochCallback` in [train.py](train.py) (after each epoch → W&B).

## Recorded result (adapters after 3-epoch QLoRA)

First training run did **not** log ScienceQA (eval crashed). Re-run on the saved adapters, **no extra SFT**:

```text
uv run python start.py --eval
```

| Field | Value |
| --- | --- |
| Date | 2026-09-03 |
| Machine | MSI (CUDA 12.4, torch 2.6.0+cu124) |
| Adapters | local `socratic_finetuned_model` on [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) |
| Slice | 256 items, seed 42, cache `scienceqa_n256_seed42.json` |
| `eval/scienceqa_acc` | **0.6641** (170 / 256) |
| `eval/scienceqa_sri` | **0.8945** (229 / 256) |
| `eval/ngram_overlap` | **0.0000** |
| W&B run | [clear-cosmos-3](https://wandb.ai/susmagar012-sunway-college-kathmandu/socratic-phi3/runs/ew9uy22h) (`ew9uy22h`) |
| W&B project | [socratic-phi3](https://wandb.ai/susmagar012-sunway-college-kathmandu/socratic-phi3) |

`epoch 3` in that W&B summary is leftover tagging from the trained adapters, not a new 3-epoch train.

### How to read these two numbers

- **Accuracy** is exam mode: “reply with a single letter.” Random 4-choice is ~25%. **66%** is clearly above chance on this **text-only** slice.
- **SRI** is tutor mode: Socratic system prompt + “do not tell me which letter is correct.” **89%** means most replies did not paste the gold choice text or say “the answer is X.”
- Do **not** compare 66% to public Phi-3 ScienceQA figures around **90%**. Those are typically **vision** models on the **full multimodal** test set. This suite **drops images**, uses **0-shot letter parsing**, and keeps only **natural science, grades 3–10**, **N = 256**.
- There is still **no base-model control** on the same 256 items. Socratic SFT can lower letter-dumping accuracy while raising restraint. To study that later, run the same `run_eval.py` path on untuned Phi-3 (no PEFT).

## How to study the eval later

Re-run (needs the adapters on disk, or `python start.py --download-checkpoints`):

```bash
python start.py --eval
# equivalent:
uv run python run_eval.py
```

Walk the code in this order:

| Step | Where | What happens |
| --- | --- | --- |
| 1. Load adapters | [run_eval.py](run_eval.py) `load_adapters` | 4-bit NF4 base + PEFT from `socratic_finetuned_model` |
| 2. Build slice | [eval_scienceqa.py](eval_scienceqa.py) `load_eval_slice` | Hub `derek-thomas/ScienceQA` test split → natural science, grades 3–10 → sample 256 with seed 42 → cache JSON |
| 3. Contamination | `ngram_overlap` | Fraction of eval questions whose 5-gram appears in [socratic_train.jsonl](socratic_train.jsonl) |
| 4. Exam generate | `exam_user_prompt` + `_generate` (`exam_tokens=32`) | No Socratic system prompt; parse first A–D via `parse_letter` |
| 5. Tutor generate | `SOCRATIC_SYSTEM` + `tutor_user_prompt` (`tutor_tokens=64`) | `sri_restrained` fails if gold **text** is in the reply or “answer is {letter}” |
| 6. Log | `log_scienceqa_to_wandb` in [train.py](train.py) | `eval/scienceqa_acc`, `eval/scienceqa_sri`, `eval/n`, example table |

Generation uses `use_cache=True` (faster KV cache). Fallbacks: `use_cache=False` on Phi-3 `seen_tokens` errors, shorter gens on CUDA OOM. A missing `flash_attn` print is a speed warning only.

The first eight items (question, gold, pred, exam snippet, tutor snippet) are logged to W&B as `eval/scienceqa_examples`. Use that table to sanity-check parsing vs restraint.

## Second base: Qwen3-4B-Instruct-2507

Phi-3 stays the published baseline. To compare on the **same** 256-item cache:

```bash
python start.py --qwen
python start.py --eval --qwen
```

Adapters go to `socratic_qwen3_model/` and a **separate** Hub repo `HF_QWEN_REPO` (default `Susu11/Science_Socratic_Qwen3-4B_Instruct`), never `HF_HUB_REPO` / Phi-3. [train_qwen.py](train_qwen.py) is QLoRA (4-bit NF4 + LoRA). The base is [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) (**no thinking mode**). Do not use `Qwen3-4B-Thinking-2507`. After train: `python start.py --eval --qwen` then `python start.py --gguf --qwen` for llama.cpp/Ollama.

Until that run is logged, only the Phi-3 numbers in the table above are recorded. Qwen W&B goes to project **`science_socratic_qwen3-4b_instruct`** (`WANDB_PROJECT_QWEN`), not `socratic-phi3`.

## Why one dataset: ScienceQA

| Need | ScienceQA | Not used per epoch |
| --- | --- | --- |
| K-10 Physics / Chemistry / Biology | Natural-science items, grades 3–10 | SciQ (facts only), MRBench (math) |
| Hugging Face download | [`derek-thomas/ScienceQA`](https://huggingface.co/datasets/derek-thomas/ScienceQA) | TutorBench (multimodal + 15k rubrics) |
| Finish on a college GPU after each epoch | Fixed **256** items, seed `42` | Full ~8k ScienceQA; SocraticBench teacher–student loops |

Standard NLP scores (MMLU, GSM8K, raw QA accuracy) do **not** prove a Socratic tutor. A model can know the science and still **reveal the answer**. This pipeline therefore logs **two** numbers:

1. **Domain grounding** — multiple-choice accuracy (`eval/scienceqa_acc`).
2. **Socratic restraint** — SRI proxy (`eval/scienceqa_sri`): did the tutor avoid dumping the gold answer?

## What runs after every epoch

```text
bash start.sh
        │
        ▼
  SFT epoch on socratic_train.jsonl
        │
        ▼
  Same 256 ScienceQA items (every epoch)
        ├── exam prompt  → letter A–D  → accuracy
        └── tutor prompt → 1–3 sentences → SRI
        │
        ▼
  wandb.log (WANDB_API_KEY from .env, or prompted)
```

Images in ScienceQA are **ignored** (text of the question and choices only). That keeps Phi-3-mini text-only and keeps eval time down.

At **train start** (once): 5-gram overlap of those 256 questions vs [socratic_train.jsonl](socratic_train.jsonl) is stored as `eval/ngram_overlap` so a paper can report a simple decontamination check.

## Metrics

### Multiple-choice accuracy

Exam prompt: answer with a **single letter**. The first `A`–`D` in the reply is parsed and compared to ScienceQA’s gold index.

\[
\text{acc} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}[\hat{y}_i = y_i]
\]

with \(N = 256\). This is Phase 1 style **knowledge** check, not tutoring quality.

### Socratic Restraint Index (SRI)

Tutor prompt uses the Grade 10 Socratic system message (“never give the final answer”). SRI is the fraction of replies that **do not** contain the gold choice text and do not say “the answer is X”:

\[
\text{SRI} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}[\text{gold answer not revealed in reply}_i]
\]

High SRI means the model is resisting spoilers. It is a **heuristic**, not MRBench’s full eight dimensions or TutorBench’s weighted \(S_{\text{tutor}}\) rubric (those need a calibrated LLM judge).

### Relation to \(S_{\text{tutor}}\)

The academic weighted score

\[
S_{\text{tutor}} = \frac{1}{N}\sum_i \frac{\sum_k w_k c_{i,k}}{\sum_k \max(w_k,0)} \times 100
\]

with \(w_k \in \{-5,+1,+5\}\) is **not** computed in this repo. Accuracy + SRI are the automated stand-ins that can run every epoch.

## Weights & Biases

Put this in **local `.env`**, or let `bash start.sh` ask if a line is missing:

```bash
WANDB_API_KEY=...
WANDB_PROJECT=socratic-phi3
WANDB_PROJECT_QWEN=science_socratic_qwen3-4b_instruct
HF_TOKEN=...
HF_HUB_REPO=Susu11/socratic-phi3
HF_QWEN_REPO=Susu11/Science_Socratic_Qwen3-4B_Instruct
```

`bash start.sh` sources `.env`. After each epoch the run should show:

| Key | Meaning |
| --- | --- |
| `eval/scienceqa_acc` | Exam-mode letter accuracy |
| `eval/scienceqa_sri` | Tutor-mode restraint |
| `eval/n` | Always 256 (or fewer if the filter is thin) |
| `epoch` | 1, 2, 3, … |
| `eval/scienceqa_examples` | Table of a few questions, preds, tutor snippets |
| `eval/ngram_overlap` | Summary, logged once |

If `WANDB_API_KEY` is missing, the same metrics still **print in the terminal**.

## How to run (college GPU)

**Train + per-epoch eval:** `python start.py` / `bash start.sh` (prompts for W&B / HF tokens if needed). Do not use the `.ipynb` as the main eval path.

**Eval only** (this is what produced the recorded numbers):

```bash
python start.py --eval
```

## Files

| File | Role |
| --- | --- |
| [eval_scienceqa.py](eval_scienceqa.py) | Load/filter/sample ScienceQA; generate; score |
| [run_eval.py](run_eval.py) | Load saved adapters; no SFT; log to W&B |
| [train.py](train.py) | `ScienceQAEpochCallback` after each epoch |
| [train_qwen.py](train_qwen.py) | QLoRA SFT on Qwen3-4B-Instruct-2507 |
| [run_eval_qwen.py](run_eval_qwen.py) | ScienceQA on `socratic_qwen3_model` |
| [start.py](start.py) | `--eval` → Phi-3; `--qwen` / `--eval --qwen` → Qwen |
| [socratic_train.jsonl](socratic_train.jsonl) | SFT data (not the benchmark) |

## What this is not (later paper work)

- **TutorBench** — expert rubrics, multimodal student work, LLM-as-a-judge.
- **MRBench** — eight mistake-remediation dimensions (ID, location, non-reveal, guidance, …).
- **SocraticBench** — multi-turn teacher–student simulation and success classifier.
- **MRE** — “student self-corrects within T turns” (needs a student model).
- Full ScienceQA (~8k natural-science items) and image questions.

Those remain valid for a final-year write-up; they are too heavy to attach to every SFT epoch on one GPU.

## Suggested paper wording

> We evaluate the QLoRA-tuned Phi-3-mini (3.8B) Socratic tutor on a fixed 256-item ScienceQA natural-science subset (grades 3–10, text only, seed 42). We report multiple-choice accuracy under an exam prompt and a Socratic Restraint Index under a non-revealing tutor prompt, logged to Weights & Biases. On the saved adapters (eval-only, 2026-09-03) this slice scored 66.4% accuracy and 89.5% SRI, with 0% 5-gram overlap against the fine-tuning JSONL. Full TutorBench/MRBench judge protocols are left to a separate, post-training study.
