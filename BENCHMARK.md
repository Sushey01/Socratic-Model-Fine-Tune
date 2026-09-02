# Benchmark and pedagogical evaluation (ScienceQA + W&B)

This document describes the **evaluation suite that actually runs in this repo** during fine-tune. It is a practical slice of a larger K-10 Socratic-tutor evaluation story (TutorBench, MRBench, SocraticBench). Those other suites need LLM-as-a-judge or two-model dialogue loops and are **not** scored after every epoch here.

**Code:** [eval_scienceqa.py](eval_scienceqa.py) (dataset, metrics) and `ScienceQAEpochCallback` in [train.py](train.py) (after each epoch → W&B).

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
HF_TOKEN=...
HF_HUB_REPO=Susu11/socratic-phi3
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

**One command:** `bash start.sh` (prompts for W&B / HF tokens if needed). Do not run the `.ipynb` for this eval.

## Files

| File | Role |
| --- | --- |
| [eval_scienceqa.py](eval_scienceqa.py) | Load/filter/sample ScienceQA; generate; score |
| [train.py](train.py) | `ScienceQAEpochCallback` after each epoch |
| [start.sh](start.sh) | Loads `.env` (`WANDB_PROJECT` defaults to `socratic-phi3`) |
| [socratic_train.jsonl](socratic_train.jsonl) | SFT data (not the benchmark) |

## What this is not (later paper work)

- **TutorBench** — expert rubrics, multimodal student work, LLM-as-a-judge.
- **MRBench** — eight mistake-remediation dimensions (ID, location, non-reveal, guidance, …).
- **SocraticBench** — multi-turn teacher–student simulation and success classifier.
- **MRE** — “student self-corrects within T turns” (needs a student model).
- Full ScienceQA (~8k natural-science items) and image questions.

Those remain valid for a final-year write-up; they are too heavy to attach to every SFT epoch on one GPU.

## Suggested paper wording

> We evaluate the LoRA-tuned Phi-3 Socratic tutor after each epoch on a fixed 256-item ScienceQA natural-science subset (grades 3–10). We report multiple-choice accuracy under an exam prompt and a Socratic Restraint Index under a non-revealing tutor prompt, logged to Weights & Biases. Five-gram overlap with the fine-tuning JSONL is reported as a contamination check. Full TutorBench/MRBench judge protocols are left to a separate, post-training study.
