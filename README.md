# Socratic model fine-tune

Fine-tune Phi-3 as a Grade 10 Socratic science tutor using [socratic_train.jsonl](socratic_train.jsonl).

**GitHub** = code. **Hugging Face dataset** [Susu11/socraticfinetune](https://huggingface.co/datasets/Susu11/socraticfinetune) = JSONL. **Hugging Face model** (`HF_HUB_REPO`) = LoRA adapters.

## Clone and run (training PC)

Preferred command (**Python**, no Hugging Face CLI — that installer fails in Git Bash with `OS: unknown`):

```bash
python start.py
```

or, if `uv` is installed:

```bash
uv run python start.py
```

`bash start.sh` only forwards to `start.py`.

### `.env` is used automatically

If `.env` already has `WANDB_API_KEY` and `HF_TOKEN` (no `#` in front of those lines), **`start.py` will not ask you to paste keys**. You should see `Using WANDB_API_KEY from .env` and `Using HF_TOKEN from .env`. It only prompts when a key is missing or commented out.

Each line must be `NAME=value` (quotes optional):

```bash
WANDB_API_KEY=...
HF_TOKEN=hf_...
HF_HUB_REPO=Susu11/socratic-phi3
HF_DATASET_REPO=Susu11/socraticfinetune
WANDB_PROJECT=socratic-phi3
```

Do not put a raw token on its own line. Do not `source .env` in Git Bash.

**Windows:** `uv` picked Python **3.14** (CPU-only torch). This repo pins **3.12**. After `git pull`:

```powershell
cd $HOME\Socratic-Model-Fine-Tune
nvidia-smi
uv run python start.py
```

Confirm `torch.cuda.is_available()` becomes True. If `nvidia-smi` works but CUDA is still False, train in **WSL2** or on a Linux GPU box (`bitsandbytes` is unreliable on native Windows).

**One run:** SFT → ScienceQA after each epoch → W&B → push LoRA (Python Hub API, not `hf`).

```bash
python start.py --fresh
python start.py --download-checkpoints
python start.py --gguf
```

## Hugging Face

Keep a local `.env` (gitignored). Do **not** commit it. On the college PC, create `.env` with:

| Variable | Example | What |
| --- | --- | --- |
| `HF_DATASET_REPO` | `Susu11/socraticfinetune` | JSONL via Python Hub API |
| `HF_HUB_REPO` | `Susu11/socratic-phi3` | Adapters + **latest** checkpoint after train |
| `HF_TOKEN` | `hf_...` **without a `#` in front** | Write token; `start.py` reads `.env` |
| `WANDB_API_KEY` | from wandb.ai | Logs ScienceQA after every epoch |
| `WANDB_PROJECT` | `socratic-phi3` | W&B project name (optional) |

`HF_TOKEN=...` must be an active line. A leading `#` means “comment” and the script cannot see it.

**Do not put the Phi-3 model or LoRA files in this GitHub folder.** `train.py` downloads [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) automatically. Checkpoints stay in `socratic_finetuned_model/` (gitignored) and on Hugging Face.

After train, the **model** repo contains:

| Layer | What | Use |
| --- | --- | --- |
| Deploy | PEFT adapters + tokenizer at **repo root** | Load LoRA on [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) |
| Resume | **Only the latest** `checkpoint-N` (e.g. step 500 if you save every 100) | Continue training; you do not need checkpoint-100 if 500 exists |
| GGUF | `gguf/*.gguf` **later** | llama.cpp / Ollama after `bash start.sh --gguf` |

Linked cards:

- Notebook / MLX 4-bit: [Oscilla/Phi-3.5-mini-instruct-mlx-4Bit](https://huggingface.co/Oscilla/Phi-3.5-mini-instruct-mlx-4Bit)
- CUDA train base: [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)

The Hub repo is created **private**. Locally, snapshots still write every 100 steps and keep the last 3 folders; only the newest checkpoint is uploaded.

## Checkpoints

`checkpoint-500` is the model at step 500. Inference uses those adapters (or the final root save). Resume starts at step 501. A crash at 450 resumes from 400 (last completed save).

```bash
git pull && bash start.sh --download-checkpoints
```

## GGUF after fine-tune

Do not merge/quantize inside the default train command (easy OOM). When LoRA is done: `bash start.sh --gguf` prints merge → llama.cpp convert/quantize → upload to the same `HF_HUB_REPO`.

`socratic_train_data.jsonl` is an older instruction-format file and is not used by `train.py`.

## ScienceQA benchmark (W&B)

See **[BENCHMARK.md](BENCHMARK.md)** for dataset choice, metrics (accuracy + SRI), and W&B keys.

After **each training epoch**, `train.py` scores a fixed **256-item** slice of [ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA) (natural science, grades 3–10) and logs `eval/scienceqa_acc`, `eval/scienceqa_sri`, and `eval/ngram_overlap` when `WANDB_API_KEY` is in `.env`.
