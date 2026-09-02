# Socratic model fine-tune

Fine-tune `microsoft/Phi-3-mini-4k-instruct` as a Grade 10 Socratic science tutor using [socratic_train.jsonl](socratic_train.jsonl).

- **GitHub** = code + dataset. `git pull` on either machine to get script/data changes.
- **Hugging Face Hub** = LoRA adapters **and checkpoints**. Download those at home to continue training or to run the model.

Do not commit tokens. Log in on each PC yourself.

## One command (college GPU)

Install [uv](https://docs.astral.sh/uv/) once, clone this repo, then:

```bash
huggingface-cli login
export HF_HUB_REPO=YOUR_HF_USER/socratic-phi3-lora
uv run python train.py
```

The first run creates `.venv` and installs packages, then trains. Later runs of the same command **resume from the latest `checkpoint-*`** under `./socratic_finetuned_model`.

Needs an NVIDIA GPU (`nvidia-smi` should work). If `uv` installs a CPU-only PyTorch, install a CUDA wheel in the project env, for example:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Checkpoints (continue later)

Training writes:

- `socratic_finetuned_model/checkpoint-100`, `checkpoint-200`, … (last 3 kept by default)
- Final adapters in `socratic_finetuned_model/`

Resume on the **same machine**:

```bash
uv run python train.py
```

Start over:

```bash
uv run python train.py --no-resume
```

Continue on **another machine**: upload happens automatically when `HF_HUB_REPO` is set. At home:

```bash
git pull
hf download YOUR_HF_USER/socratic-phi3-lora --local-dir ./socratic_finetuned_model
uv run python train.py
```

That download includes checkpoints, so training continues from the last saved step.

## Home: code vs weights

```bash
git pull
hf download YOUR_HF_USER/socratic-phi3-lora --local-dir ./socratic_finetuned_model
```

`socratic_train_data.jsonl` is an older instruction-format file and is not used by `train.py`.
