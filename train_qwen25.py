#!/usr/bin/env python3
"""QLoRA SFT of Qwen2.5-7B-Instruct as a Grade 10 Socratic science tutor.

Does not touch Phi-3 or Qwen3-4B adapters. Train on dataset7b/socratic_v7_final_v3.jsonl
(already messages; no convert_v4). Output: ./socratic_qwen25_v7_model
Hub model: HF_QWEN25_REPO (Susu11/v7_qwen7b). JSONL lives on
HF_V7_DATASET_REPO (Susu11/v7_socratic_data), not the model repo.

College GPU: QLoRA only (4-bit NF4 + LoRA). Do not use the base GGUF for this SFT.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig
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
from train_qwen import LORA_TARGETS, from_pretrained_qwen

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "dataset7b" / "socratic_v7_final_v3.jsonl"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_OUTPUT = ROOT / "socratic_qwen25_v7_model"
DEFAULT_HUB = "Susu11/v7_qwen7b"
DEFAULT_DATASET_REPO = "Susu11/v7_socratic_data"
DEFAULT_WANDB_PROJECT = "science_socratic_qwen25-7b_instruct"


def apply_qwen25_wandb_project() -> str:
    project = (os.environ.get("WANDB_PROJECT_QWEN25") or "").strip() or DEFAULT_WANDB_PROJECT
    os.environ["WANDB_PROJECT_QWEN25"] = project
    os.environ["WANDB_PROJECT"] = project
    os.environ.pop("WANDB_RUN_ID", None)
    print(f"W&B project (Qwen2.5-7B): {project}", flush=True)
    return project


def apply_sft_chat_template(tokenizer, messages: list) -> str:
    return apply_qwen_chat_template(
        tokenizer, messages, add_generation_prompt=False, enable_thinking=False
    )


def write_qwen25_model_card(output_dir: Path, repo_id: str, base_model: str) -> None:
    card = f"""---
library_name: peft
base_model: {base_model}
license: apache-2.0
pipeline_tag: text-generation
tags:
  - qwen2.5
  - qlora
  - sft
  - socratic
  - education
  - science
---

# {repo_id}

QLoRA adapters for a **Grade 10 Socratic science tutor** on [{base_model}](https://huggingface.co/{base_model}).

This Hub repo is **Qwen2.5-7B adapters only**. Do not mix with `HF_HUB_REPO` (Phi-3) or `HF_QWEN_REPO` (Qwen3-4B). JSONL is on [{DEFAULT_DATASET_REPO}](https://huggingface.co/datasets/{DEFAULT_DATASET_REPO}), not here.

## Train

4-bit NF4 + LoRA via `python start.py --qwen25`. Data: `dataset7b/socratic_v7_final_v3.jsonl`. A **base GGUF** of Qwen2.5-7B is **not** used for SFT; convert a new GGUF after merge.

## Deploy

`python start.py --gguf --qwen25` → `gguf/socratic-qwen25-7b-q8_0.gguf` on this repo.

Eval: `python start.py --eval --qwen25` → W&B `WANDB_PROJECT_QWEN25`.
"""
    (output_dir / "README.md").write_text(card, encoding="utf-8")


def ensure_qwen25_secrets() -> None:
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
        "HF_QWEN25_REPO",
        "Hugging Face repo for Qwen2.5-7B adapters (not Phi-3 / not Qwen3-4B)",
        default=os.environ.get("HF_QWEN25_REPO") or DEFAULT_HUB,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Socratic Qwen2.5-7B-Instruct QLoRA fine-tune")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=os.environ.get("BASE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--push-to-hub",
        default=os.environ.get("HF_QWEN25_REPO", ""),
        help="HF repo id (or set HF_QWEN25_REPO).",
    )
    return parser.parse_args()


def main() -> None:
    load_runtime_env()
    cur = (os.environ.get("HF_QWEN25_REPO") or "").strip()
    if not cur or cur == "Susu11/qwen2.5-7b-socratic-tutor":
        os.environ["HF_QWEN25_REPO"] = DEFAULT_HUB
        if cur == "Susu11/qwen2.5-7b-socratic-tutor":
            print(f"HF_QWEN25_REPO was {cur}; this run pushes to {DEFAULT_HUB}.", flush=True)
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["BASE_MODEL"] = args.model
    os.environ.setdefault("WANDB_RUN_NAME", "qwen25-7b-instruct-sft")
    apply_qwen25_wandb_project()
    ensure_qwen25_secrets()
    if not args.push_to_hub:
        args.push_to_hub = os.environ.get("HF_QWEN25_REPO", "") or DEFAULT_HUB

    if not args.data.is_file():
        raise SystemExit(
            f"Missing {args.data}. Need dataset7b/socratic_v7_final_v3.jsonl "
            "(git pull on the GPU PC)."
        )

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA GPU not found. 7B QLoRA needs an NVIDIA GPU. Check nvidia-smi."
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
        print("7B fp16 LoRA often OOMs on college GPUs. Prefer fixing bitsandbytes.", flush=True)
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

    resume_from = resolve_checkpoint(
        args.output_dir, resume=not args.no_resume, num_train_epochs=args.epochs
    )
    print("Calling trainer.train() — QLoRA 7B, batch=1. Watch VRAM.", flush=True)
    trainer.train(resume_from_checkpoint=resume_from)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved Qwen2.5-7B adapters to {args.output_dir}")
    last = get_last_checkpoint(str(args.output_dir))
    if last:
        print(f"Latest resume checkpoint: {last}")

    repo_id = (args.push_to_hub or os.environ.get("HF_QWEN25_REPO") or "").strip()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    if repo_id and not token:
        token = prompt_secret(
            "HF_TOKEN",
            "Hugging Face write token required to push the model (input hidden): ",
        )
    if repo_id and token:
        os.environ["HF_TOKEN"] = token
        push_output(args.output_dir, repo_id, args.model, write_card=write_qwen25_model_card)
    elif repo_id:
        print("Skipping Hub upload: no HF_TOKEN after prompt.")
    else:
        print("Skipping Hub upload (set HF_QWEN25_REPO or --push-to-hub).")

    if wandb_enabled():
        import wandb

        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
