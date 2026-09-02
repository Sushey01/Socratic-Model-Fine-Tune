# Socratic model fine-tune

Fine-tune Phi-3 as a Grade 10 Socratic science tutor using [socratic_train.jsonl](socratic_train.jsonl).

## Clone and run (other PC)

One paste in the terminal:

```bash
git clone https://github.com/Sushey01/Socratic-Model-Fine-Tune.git && cd Socratic-Model-Fine-Tune && bash start.sh
```

`start.sh` installs [uv](https://docs.astral.sh/uv/) if needed, installs packages, trains, **resumes checkpoints**, and uploads adapters if you give a Hugging Face repo/token (first run asks once and saves `.env`; after that only `bash start.sh`).

Needs an NVIDIA GPU (`nvidia-smi`). Do not paste tokens into GitHub or this README.

Optional:

```bash
bash start.sh --fresh                 # ignore old checkpoints
bash start.sh --download-checkpoints  # pull HF_HUB_REPO then continue training
```

## Hugging Face models (follow these on GitHub)

| What | Hub page |
| --- | --- |
| Starting model (notebook / local MLX 4-bit) | [Oscilla/Phi-3.5-mini-instruct-mlx-4Bit](https://huggingface.co/Oscilla/Phi-3.5-mini-instruct-mlx-4Bit) |
| Model `train.py` loads on a college **NVIDIA GPU** | [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) |
| Your LoRA adapters + checkpoints after training | `HF_HUB_REPO` in `.env` (see [.env.example](.env.example)) |

The Oscilla card is a 4-bit **MLX** build (typical on Apple Silicon). College CUDA training uses the Microsoft Phi-3 checkpoint with bitsandbytes 4-bit LoRA, then you push adapters to **your** Hub repo so home can download them.

- **GitHub** = code + dataset + `start.sh`
- **Hugging Face Hub** = base models above, plus your LoRA adapters **and checkpoints**

## Checkpoints

Training writes `socratic_finetuned_model/checkpoint-*` (last 3 kept) and final adapters in that folder. Running `bash start.sh` again continues from the latest checkpoint.

At home, after a college run that uploaded to the Hub:

```bash
git pull && bash start.sh --download-checkpoints
```

`socratic_train_data.jsonl` is an older instruction-format file and is not used by `train.py`.
