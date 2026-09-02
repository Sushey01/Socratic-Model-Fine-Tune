# Socratic model fine-tune

Fine-tune Phi-3 as a Grade 10 Socratic science tutor using [socratic_train.jsonl](socratic_train.jsonl).

**GitHub** = code and `start.sh`. **Hugging Face dataset** [Susu11/socraticfinetune](https://huggingface.co/datasets/Susu11/socraticfinetune) = JSONL. **Hugging Face model** (`HF_HUB_REPO`) = LoRA adapters, latest checkpoint, GGUF later.

## Clone and run (other PC)

```bash
git clone https://github.com/Sushey01/Socratic-Model-Fine-Tune.git && cd Socratic-Model-Fine-Tune && bash start.sh
```

Put Hub settings in **local `.env` only**. `bash start.sh` sources that file and passes `HF_TOKEN` into `hf upload` / `train.py`. You do **not** run `hf auth login` for this project (that uses a different stored token and caused the “not found / invalid token” errors). Never commit `.env` or paste the token into chat.

```bash
bash start.sh --fresh                 # ignore old checkpoints
bash start.sh --download-checkpoints  # pull Hub adapters + latest checkpoint, then continue
bash start.sh --gguf                  # after fine-tune: GGUF steps (not part of train)
```

## Hugging Face

Keep a local `.env` (gitignored). Do **not** commit it. On the college PC, create `.env` with:

| Variable | Example | What |
| --- | --- | --- |
| `HF_DATASET_REPO` | `Susu11/socraticfinetune` | JSONL via `hf upload … --repo-type=dataset` |
| `HF_HUB_REPO` | `Susu11/socratic-phi3` | Adapters + **latest** checkpoint after train |
| `HF_TOKEN` | `hf_...` **without a `#` in front** | Write token; `start.sh` reads this automatically |

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
