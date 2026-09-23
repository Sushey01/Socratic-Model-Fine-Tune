#!/usr/bin/env python3
"""QLoRA SFT of Qwen3-4B-Instruct-2507 as a Grade 10 Socratic science tutor.

Does not touch Phi-3 adapters (`socratic_finetuned_model` / HF_HUB_REPO).
Output: ./socratic_qwen3_v7_model  Hub: HF_QWEN_REPO (default Susu11/v7_4b_qwen).
Data: dataset7b/socratic_v7_final_v3.jsonl (Hub dataset Susu11/v7_socratic_data).

This is the instruct / non-thinking checkpoint. Do not point --model at
Qwen3-4B-Thinking-2507 (hidden CoT would spoil Socratic restraint).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer

from sft_dataset import apply_qwen_chat_template, prepare_qwen_sft_dataset
from train import (
    HeartbeatCallback,
    ScienceQAEpochCallback,
    init_wandb,
    load_jsonl,
    load_runtime_env,
    make_sft_config,
    prompt_line,
    prompt_secret,
    push_output,
    resolve_checkpoint,
    wandb_enabled,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "dataset7b" / "socratic_v7_final_v3.jsonl"
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_OUTPUT = ROOT / "socratic_qwen3_v7_model"
DEFAULT_HUB = "Susu11/v7_4b_qwen"
LEGACY_HUBS = (
    "Susu11/Science_Socratic_Qwen3-4B_Instruct",
    "Susu11/v9socratic4b",
)
DEFAULT_WANDB_PROJECT = "science_socratic_qwen3-4b_instruct"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def from_pretrained_qwen(model_id: str, **kwargs):
    kwargs.setdefault("attn_implementation", "eager")
    kwargs.setdefault("trust_remote_code", True)
    return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)


def apply_qwen_wandb_project() -> str:
    """Use WANDB_PROJECT_QWEN so Qwen runs are not mixed into the Phi-3 W&B project."""
    project = (os.environ.get("WANDB_PROJECT_QWEN") or "").strip() or DEFAULT_WANDB_PROJECT
    os.environ["WANDB_PROJECT_QWEN"] = project
    os.environ["WANDB_PROJECT"] = project
    # Do not resume a run id that was started under socratic-phi3.
    os.environ.pop("WANDB_RUN_ID", None)
    print(f"W&B project (Qwen): {project}", flush=True)
    return project


def bind_v9_hub() -> str:
    """v7_final_v3 run pushes to Susu11/v7_4b_qwen; leave v9 Hub as a baseline."""
    current = (os.environ.get("HF_QWEN_REPO") or "").strip()
    if not current or current in LEGACY_HUBS:
        os.environ["HF_QWEN_REPO"] = DEFAULT_HUB
        if current in LEGACY_HUBS:
            print(
                f"HF_QWEN_REPO was {current}; this run pushes to {DEFAULT_HUB}.",
                flush=True,
            )
    return (os.environ.get("HF_QWEN_REPO") or DEFAULT_HUB).strip()


def apply_sft_chat_template(tokenizer, messages: list) -> str:
    """Instruct-2507 is non-thinking; pass enable_thinking=False if the template still accepts it."""
    return apply_qwen_chat_template(
        tokenizer, messages, add_generation_prompt=False, enable_thinking=False
    )


def write_qwen_model_card(output_dir: Path, repo_id: str, base_model: str) -> None:
    card = f"""---
library_name: peft
base_model: {base_model}
license: apache-2.0
pipeline_tag: text-generation
tags:
  - qwen3
  - qlora
  - sft
  - socratic
  - education
  - science
---

# {repo_id}

QLoRA adapters for a **Grade 10 Socratic science tutor** on [{base_model}](https://huggingface.co/{base_model}).

This Hub repo is **Qwen-only**. Phi-3 adapters live in a separate model repo (`HF_HUB_REPO`). GitHub [Sushey01/Socratic-Model-Fine-Tune](https://github.com/Sushey01/Socratic-Model-Fine-Tune) holds code and JSONL; this repo holds **weights**.

## What is uploaded

| Path | Contents |
| --- | --- |
| Repo root | Final PEFT adapters + tokenizer after SFT |
| `gguf/` | Optional Q8_0/F16 GGUF after `python start.py --gguf --qwen` |

Trainer `checkpoint-*` folders stay on the training PC (`socratic_qwen3_v9_model/`) and are **not** uploaded.

## Base model

- Instruct / **non-thinking** checkpoint only (no `<think>` blocks).
- Do not load these adapters on `Qwen3-4B-Thinking-2507`.

## Train (QLoRA)

4-bit NF4 + LoRA (`r=8`, `alpha=16`) via `python start.py --qwen` → [train_qwen.py](https://github.com/Sushey01/Socratic-Model-Fine-Tune/blob/main/train_qwen.py). Data: [Susu11/socraticfinetune](https://huggingface.co/datasets/Susu11/socraticfinetune).

## Deploy (Python / GPU)

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "{base_model}"
adapter = "{repo_id}"
tok = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(model, adapter)
messages = [
    {{"role": "system", "content": "You are a Socratic Science Tutor for a Grade 10 student. Never give the final answer directly."}},
    {{"role": "user", "content": "Why does ice float?"}},
]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
print(tok.decode(out[0, inputs.input_ids.shape[-1]:], skip_special_tokens=True))
```

Local helper: `uv run python infer_qwen.py` after adapters exist.

## Deploy (llama.cpp / Ollama)

After merge + convert: `python start.py --gguf --qwen`. Then point llama.cpp or Ollama at `gguf/socratic-qwen3-q8_0.gguf` on this repo.

## Eval

Same ScienceQA 256-item slice as Phi-3: `python start.py --eval --qwen`. Compare `eval/scienceqa_acc` and `eval/scienceqa_sri` in W&B project `science_socratic_qwen3-4b_instruct` (`WANDB_PROJECT_QWEN`).
"""
    (output_dir / "README.md").write_text(card, encoding="utf-8")



def ensure_qwen_secrets() -> None:
    load_runtime_env()
    prompt_secret(
        "WANDB_API_KEY",
        "Weights & Biases API key (wandb.ai/authorize, input hidden): ",
    )
    prompt_secret(
        "HF_TOKEN",
        "Hugging Face write token (huggingface.co/settings/tokens, input hidden): ",
    )
    if not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HF_TOKEN", "")
    prompt_line(
        "HF_QWEN_REPO",
        "Hugging Face repo for Qwen adapters (not the Phi-3 repo)",
        default=os.environ.get("HF_QWEN_REPO") or DEFAULT_HUB,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Socratic Qwen3-4B-Instruct QLoRA fine-tune")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=os.environ.get("BASE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=10)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--push-to-hub",
        default=os.environ.get("HF_QWEN_REPO", ""),
        help="HF repo id (or set HF_QWEN_REPO). Never use HF_HUB_REPO / Phi-3.",
    )
    return parser.parse_args()


def main() -> None:
    load_runtime_env()
    bind_v9_hub()
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["BASE_MODEL"] = args.model
    os.environ.setdefault("WANDB_RUN_NAME", "qwen3-4b-instruct-sft")
    apply_qwen_wandb_project()
    ensure_qwen_secrets()
    if not args.data.is_file():
        raise SystemExit(
            f"Missing {args.data}. Need dataset7b/socratic_v7_final_v3.jsonl (git pull on the GPU PC)."
        )
    if not args.push_to_hub:
        args.push_to_hub = os.environ.get("HF_QWEN_REPO", "") or DEFAULT_HUB
    print(f"v7 Hub push: {args.push_to_hub} | adapters: {args.output_dir} | data: {args.data}", flush=True)

    if "Thinking" in args.model:
        raise SystemExit(
            f"{args.model} is a thinking checkpoint. Use {DEFAULT_MODEL} for this Socratic tutor."
        )

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA GPU not found. This training job needs an NVIDIA GPU. "
            "On the college PC, check `nvidia-smi` and install a CUDA build of PyTorch."
        )

    dataset = load_jsonl(args.data)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset, sft_extra, trainer_extra = prepare_qwen_sft_dataset(
        dataset, tokenizer, enable_thinking=False
    )

    use_bnb = True
    compute = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute,
        bnb_4bit_use_double_quant=True,
    )
    try:
        model = from_pretrained_qwen(args.model, quantization_config=bnb, device_map={"": 0})
    except Exception as exc:
        print(f"4-bit bitsandbytes load failed ({exc}); retrying fp16 without 4-bit.")
        use_bnb = False
        try:
            model = from_pretrained_qwen(args.model, dtype=compute, device_map={"": 0})
        except TypeError:
            model = from_pretrained_qwen(args.model, torch_dtype=compute, device_map={"": 0})

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=LORA_TARGETS,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_arguments = make_sft_config(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        optim="paged_adamw_8bit" if use_bnb else "adamw_torch",
        logging_steps=10,
        learning_rate=2e-4,
        fp16=not use_bf16,
        bf16=use_bf16,
        max_grad_norm=0.3,
        max_steps=-1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="wandb" if wandb_enabled() else "none",
        logging_first_step=True,
        max_length=args.max_length,
        packing=False,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_safetensors=True,
        **sft_extra,
    )

    print("Connecting W&B, then building trainer.", flush=True)

    if not init_wandb():
        training_arguments.report_to = ["none"]

    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "peft_config": lora_config,
        "args": training_arguments,
        "callbacks": [HeartbeatCallback(), ScienceQAEpochCallback(tokenizer, args.data)],
        **trainer_extra,
    }
    print("Building SFTTrainer...", flush=True)
    try:
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = SFTTrainer(tokenizer=tokenizer, **trainer_kwargs)

    resume_from = resolve_checkpoint(args.output_dir, resume=not args.no_resume)
    print("Calling trainer.train() — watch [train] heartbeat lines and GPU use.", flush=True)
    trainer.train(resume_from_checkpoint=resume_from)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved Qwen adapters and tokenizer to {args.output_dir}")
    last = get_last_checkpoint(str(args.output_dir))
    if last:
        print(f"Latest resume checkpoint: {last}")

    repo_id = (args.push_to_hub or os.environ.get("HF_QWEN_REPO") or "").strip()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    if repo_id and not token:
        token = prompt_secret(
            "HF_TOKEN",
            "Hugging Face write token required to push the model (input hidden): ",
        )
    if repo_id and token:
        os.environ["HF_TOKEN"] = token
        push_output(args.output_dir, repo_id, args.model, write_card=write_qwen_model_card)
    elif repo_id:
        print("Skipping Hub upload: no HF_TOKEN after prompt.")
    else:
        print("Skipping Hub upload (set HF_QWEN_REPO or --push-to-hub).")

    if wandb_enabled():
        import wandb

        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
