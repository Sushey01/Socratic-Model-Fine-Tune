#!/usr/bin/env python3
"""QLoRA SFT of Qwen3-4B-Instruct-2507 as a Grade 10 Socratic science tutor.

Does not touch Phi-3 adapters (`socratic_finetuned_model` / HF_HUB_REPO).
Output: ./socratic_qwen3_model  Hub: HF_QWEN_REPO (default Susu11/socratic-qwen3).

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

from train import (
    DEFAULT_DATA,
    HeartbeatCallback,
    ScienceQAEpochCallback,
    init_wandb,
    load_jsonl,
    make_sft_config,
    prompt_line,
    prompt_secret,
    push_output,
    resolve_checkpoint,
    wandb_enabled,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_OUTPUT = ROOT / "socratic_qwen3_model"
DEFAULT_HUB = "Susu11/socratic-qwen3"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def from_pretrained_qwen(model_id: str, **kwargs):
    kwargs.setdefault("attn_implementation", "eager")
    kwargs.setdefault("trust_remote_code", True)
    return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)


def apply_sft_chat_template(tokenizer, messages: list) -> str:
    """Instruct-2507 is non-thinking; pass enable_thinking=False if the template still accepts it."""
    kw = dict(tokenize=False, add_generation_prompt=False)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kw)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kw)


def ensure_qwen_secrets() -> None:
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
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--push-to-hub",
        default=os.environ.get("HF_QWEN_REPO", ""),
        help="HF repo id (or set HF_QWEN_REPO). Never use HF_HUB_REPO / Phi-3.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["BASE_MODEL"] = args.model
    os.environ.setdefault("WANDB_RUN_NAME", "qwen3-4b-instruct-sft")
    ensure_qwen_secrets()
    if not args.push_to_hub:
        args.push_to_hub = os.environ.get("HF_QWEN_REPO", "") or DEFAULT_HUB

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
        lr_scheduler_type="constant",
        report_to="wandb" if wandb_enabled() else "none",
        logging_first_step=True,
        max_length=args.max_length,
        packing=False,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_safetensors=True,
        dataset_text_field="text",
    )

    def to_text(example):
        messages = example["messages"]
        if not isinstance(messages, list):
            raise ValueError("Expected 'messages' to be a list")
        return {"text": apply_sft_chat_template(tokenizer, messages)}

    dataset = dataset.map(to_text, remove_columns=[c for c in dataset.column_names if c != "text"])
    print("Dataset mapped. Connecting W&B, then building trainer.", flush=True)

    if not init_wandb():
        training_arguments.report_to = ["none"]

    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "peft_config": lora_config,
        "args": training_arguments,
        "callbacks": [HeartbeatCallback(), ScienceQAEpochCallback(tokenizer, args.data)],
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
        push_output(args.output_dir, repo_id, args.model)
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
