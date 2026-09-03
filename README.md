# Socratic model fine-tune

Fine-tune Phi-3 as a Grade 10 Socratic science tutor using [socratic_train.jsonl](socratic_train.jsonl).

**GitHub** = code. **Hugging Face dataset** [Susu11/socraticfinetune](https://huggingface.co/datasets/Susu11/socraticfinetune) = JSONL. **Hugging Face model** (`HF_HUB_REPO`) = LoRA adapters.

## Clone and run (training PC)

Preferred command (**Python**, no Hugging Face CLI — that installer fails in Git Bash with `OS: unknown`):

```bash
python start.py
# or
uv run python start.py
# Windows:
powershell -ExecutionPolicy Bypass -File .\start.ps1
# Linux/WSL:
bash start.sh
```

### `.env` is used automatically

`.env` is **gitignored** and is **not** on GitHub. `git pull` never restores it. Keep a copy in a password manager. Repo template (no secrets): [`.env.example`](.env.example). Do not replace `.env` with the example.

If `.env` already has `WANDB_API_KEY` and `HF_TOKEN` (no `#` in front of those lines), **`start.py` will not ask you to paste keys**. You should see `Using WANDB_API_KEY from .env` and `Using HF_TOKEN from .env`. It only prompts when a key is missing or commented out.

Each line must be `NAME=value` (quotes optional):

```bash
WANDB_API_KEY=...
HF_TOKEN=hf_...
HF_HUB_REPO=Susu11/socratic-phi3
HF_QWEN_REPO=Susu11/Science_Socratic_Qwen3-4B_Instruct
HF_DATASET_REPO=Susu11/socraticfinetune
WANDB_PROJECT=socratic-phi3
```

Do not put a raw token on its own line. Do not `source .env` in Git Bash.

**Windows (MSI / college PC):** one command installs `uv` and Python 3.12 if they are missing, then trains:

```powershell
cd $HOME\Socratic-Model-Fine-Tune
git pull
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

`uv sync` installs CUDA **cu124** torch on Windows/Linux (PyPI torch is CPU-only on Windows). If Application Control blocks `.venv\Scripts\python.exe` (error **4551**), `start.ps1` retries with uv’s managed Python. A WDAC allow-list cannot be automated; IT must allow the folder, or use **WSL2**.

**Colab:** use the existing [socratic_model_fine_tune_3.8b.ipynb](socratic_model_fine_tune_3.8b.ipynb). Mount Drive first; checkpoints go to `MyDrive/socratic_finetuned_model` so they survive a runtime restart. Do not pass `train_sampling_strategy` into `SFTConfig`.

**Kaggle (GPU T4, internet on):** [kaggle_socratic_finetune.ipynb](kaggle_socratic_finetune.ipynb). Add secrets `WANDB_API_KEY` and `HF_TOKEN`. Do not install `wandb[sandbox]` or run `wandb login`.

```bash
python start.py --fresh
python start.py --download-checkpoints
python start.py --eval
python start.py --gguf
python start.py --qwen          # Qwen3-4B-Instruct QLoRA (does not overwrite Phi-3)
python start.py --eval --qwen   # ScienceQA on socratic_qwen3_model
python start.py --gguf --qwen   # Qwen GGUF → HF_QWEN_REPO (not Phi-3)
```

## Hugging Face

Keep a local `.env` (gitignored). Do **not** commit it. On the college PC, create `.env` with:

| Variable | Example | What |
| --- | --- | --- |
| `HF_DATASET_REPO` | `Susu11/socraticfinetune` | JSONL via Python Hub API |
| `HF_HUB_REPO` | `Susu11/socratic-phi3` | **Final** Phi-3 LoRA adapters after train (not step checkpoints) |
| `HF_QWEN_REPO` | `Susu11/Science_Socratic_Qwen3-4B_Instruct` | **Final** Qwen3 Instruct LoRA adapters (`python start.py --qwen`) |
| `HF_TOKEN` | `hf_...` **without a `#` in front** | Write token; `start.py` reads `.env` |
| `WANDB_API_KEY` | from wandb.ai | Logs ScienceQA after every epoch |
| `WANDB_PROJECT` | `socratic-phi3` | W&B project name (optional) |

`HF_TOKEN=...` must be an active line. A leading `#` means “comment” and the script cannot see it.

**Do not put the Phi-3 model or LoRA files in this GitHub folder.** `train.py` downloads [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) automatically. Step checkpoints stay in `socratic_finetuned_model/` on the training PC (gitignored). Hugging Face only receives the **finished** adapters.

After train, the **model** repo contains:

| Layer | What | Use |
| --- | --- | --- |
| Deploy | PEFT adapters + tokenizer at **repo root** | Load LoRA on [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) |
| GGUF | `gguf/socratic-phi3-q8_0.gguf` | llama.cpp / Ollama after `python start.py --gguf` (uploaded 2026-09-03) |

Linked cards:

- Notebook / MLX 4-bit: [Oscilla/Phi-3.5-mini-instruct-mlx-4Bit](https://huggingface.co/Oscilla/Phi-3.5-mini-instruct-mlx-4Bit)
- CUDA train base: [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)

The Hub repo is created **private**. Local snapshots still write every 100 steps and keep the last 3 folders **on that PC only**.

## Checkpoints

`checkpoint-500` is the model at step 500 **on the training disk**. Inference of the published model uses the final root save on Hugging Face. Resume on the same PC starts at the next step after the last local save. A crash at 450 resumes from 400.

To copy the **final** adapters to another machine:

```bash
git pull && bash start.sh --download-checkpoints
```

## GGUF after fine-tune

Do **not** retrain to get GGUF. On the PC that has `socratic_finetuned_model` (or after `--download-checkpoints`):

```bash
python start.py --eval    # ScienceQA acc/sri to W&B (no SFT)
python start.py --gguf    # merge adapters, convert GGUF, upload to HF_HUB_REPO/gguf/
```

Merge needs a lot of RAM. `--eval` prefers KV cache (`use_cache=True`) and falls back if Phi-3 `seen_tokens` or CUDA OOM hits.

`socratic_train_data.jsonl` is an older instruction-format file and is not used by `train.py`.

## Qwen3-4B-Instruct (separate Hub repo + deploy)

**Yes — Qwen must use a different Hugging Face model repo.** Phi-3 stays at `HF_HUB_REPO` (`Susu11/socratic-phi3`). Qwen adapters, model card, and GGUF go to `HF_QWEN_REPO` (`Susu11/Science_Socratic_Qwen3-4B_Instruct`). The same `HF_TOKEN` can write to both; the **repo ids must not be mixed**.

| Artifact | Env | Default Hub |
| --- | --- | --- |
| Dataset JSONL | `HF_DATASET_REPO` | `Susu11/socraticfinetune` |
| Phi-3 PEFT + GGUF | `HF_HUB_REPO` | `Susu11/socratic-phi3` |
| Qwen3 Instruct PEFT + GGUF | `HF_QWEN_REPO` | `Susu11/Science_Socratic_Qwen3-4B_Instruct` |

Add `HF_QWEN_REPO=Susu11/Science_Socratic_Qwen3-4B_Instruct` to `.env` (created private on first push). Instruct-2507 is **non-thinking**. Same JSONL and ScienceQA slice as Phi-3.

```bash
python start.py --qwen              # QLoRA → socratic_qwen3_model → HF_QWEN_REPO
python start.py --eval --qwen       # ScienceQA acc/sri
python start.py --gguf --qwen       # merge → GGUF → HF_QWEN_REPO/gguf/
uv run python infer_qwen.py         # GPU smoke test from local adapters
uv run python infer_qwen.py --hub   # load adapters from HF_QWEN_REPO
```

Code: [train_qwen.py](train_qwen.py) (professional Hub card on push), [run_eval_qwen.py](run_eval_qwen.py), [export_gguf_qwen.py](export_gguf_qwen.py), [infer_qwen.py](infer_qwen.py). Needs `transformers>=4.51`. Merge uses `merged_model_qwen/` so Phi-3 `merged_model/` is untouched.

**Deploy:** PEFT on GPU via `infer_qwen.py` or the snippet on the Hub card; local/edge via the Qwen GGUF in llama.cpp or Ollama after `--gguf --qwen`.

## ScienceQA benchmark (W&B)

See **[BENCHMARK.md](BENCHMARK.md)** to study the eval: metrics, code map, and the recorded adapter scores.

After **each training epoch**, `train.py` scores a fixed **256-item** slice of [ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA) (natural science, grades 3–10) and logs `eval/scienceqa_acc`, `eval/scienceqa_sri`, and `eval/ngram_overlap` when `WANDB_API_KEY` is in `.env`.

Eval-only (no SFT), after adapters exist:

```bash
python start.py --eval
```

**Recorded (2026-09-03, no extra train):** acc **0.6641**, SRI **0.8945**, n **256**, 5-gram overlap **0**. W&B: [clear-cosmos-3](https://wandb.ai/susmagar012-sunway-college-kathmandu/socratic-phi3/runs/ew9uy22h). Do not compare acc to public ~90% ScienceQA numbers (those are multimodal / different protocol). Full interpretation is in [BENCHMARK.md](BENCHMARK.md).
