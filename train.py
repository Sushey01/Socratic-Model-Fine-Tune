#!/usr/bin/env python3
"""Fine-tune Phi-3-mini on the Socratic Grade 10 chat dataset.

Checkpoints are written under OUTPUT_DIR (default ./socratic_finetuned_model).
Re-run the same command to resume from the latest checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "socratic_train.jsonl"
DEFAULT_OUTPUT = ROOT / "socratic_finetuned_model"
DEFAULT_MODEL = "microsoft/Phi-3-mini-4k-instruct"


def load_jsonl(path: Path) -> Dataset:
    rows = []
    errors = []
    with path.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append(f"line {i}: {exc}")
    if errors:
        print("JSONL parse errors:", file=sys.stderr)
        for err in errors:
            print(f"  {err}", file=sys.stderr)
        raise SystemExit(1)
    if not rows:
        raise SystemExit(f"No examples found in {path}")
    if "messages" not in rows[0]:
        raise SystemExit(f"{path} must contain a 'messages' field (use socratic_train.jsonl)")
    print(f"Loaded {len(rows)} examples from {path}")
    return Dataset.from_list(rows)


def resolve_checkpoint(output_dir: Path, resume: bool) -> str | None:
    if not resume or not output_dir.is_dir():
        return None
    last = get_last_checkpoint(str(output_dir))
    if last:
        print(f"Resuming from checkpoint: {last}")
    else:
        print("No checkpoint found; starting a new run.")
    return last


def write_model_card(output_dir: Path, repo_id: str, base_model: str) -> None:
    last = get_last_checkpoint(str(output_dir))
    last_name = Path(last).name if last else "none"
    card = f"""---
library_name: peft
base_model: {base_model}
tags:
  - lora
  - sft
  - socratic
  - education
---

# {repo_id}

Grade 10 Socratic science tutor LoRA. **GitHub** holds code and data; **this Hub repo** holds weights.

## What is in this repo

1. **Deploy (repo root):** PEFT adapters + tokenizer. Load with `PeftModel.from_pretrained("{repo_id}")` on top of `{base_model}`.
2. **Resume:** only the **latest** Trainer folder (`{last_name}`). Continue with `bash start.sh --download-checkpoints` then `bash start.sh`.
3. **GGUF (later):** after fine-tune, `bash start.sh --gguf` (merge + llama.cpp quant). Not uploaded during training.

Older `checkpoint-*` folders are not kept on the Hub. Local training still snapshots every 100 steps and keeps the last 3 on disk.

## Linked models

- CUDA train base: [{base_model}](https://huggingface.co/{base_model})
- Notebook / MLX 4-bit reference: [Oscilla/Phi-3.5-mini-instruct-mlx-4Bit](https://huggingface.co/Oscilla/Phi-3.5-mini-instruct-mlx-4Bit)
"""
    (output_dir / "README.md").write_text(card, encoding="utf-8")


def _hub_checkpoint_names(api, repo_id: str) -> list[str]:
    names: set[str] = set()
    try:
        for item in api.list_repo_tree(repo_id, repo_type="model", recursive=False):
            path = getattr(item, "path", None) or str(item)
            if path.startswith("checkpoint-"):
                names.add(path.split("/")[0])
    except Exception as exc:
        print(f"Could not list Hub checkpoints ({exc}); skip prune.")
    return sorted(names)


def push_output(output_dir: Path, repo_id: str, base_model: str) -> None:
    from huggingface_hub import HfApi, login

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)
    api = HfApi()
    api.create_repo(repo_id, exist_ok=True, repo_type="model", private=True)

    write_model_card(output_dir, repo_id, base_model)
    last = get_last_checkpoint(str(output_dir))
    last_name = Path(last).name if last else None

    print(f"Uploading adapters + tokenizer (repo root) to {repo_id}")
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=[
            "checkpoint-*",
            "checkpoint-*/**",
            "runs/**",
            "tmp_trainer/**",
            "*.tmp",
        ],
    )

    if last:
        print(f"Uploading latest checkpoint only: {last_name}")
        api.upload_folder(
            folder_path=last,
            path_in_repo=last_name,
            repo_id=repo_id,
            repo_type="model",
            ignore_patterns=["rng_state*", "*.tmp"],
        )
        for name in _hub_checkpoint_names(api, repo_id):
            if name != last_name:
                print(f"Removing old Hub checkpoint {name}")
                api.delete_folder(repo_id=repo_id, path_in_repo=name, repo_type="model")

    print(f"Pushed to https://huggingface.co/{repo_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Socratic Phi-3 LoRA fine-tune")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=os.environ.get("BASE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing checkpoints and start over",
    )
    parser.add_argument(
        "--push-to-hub",
        default=os.environ.get("HF_HUB_REPO", ""),
        help="HF repo id (or set HF_HUB_REPO). Uploads adapters plus the latest checkpoint only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA GPU not found. This training job needs an NVIDIA GPU. "
            "On the college PC, check `nvidia-smi` and install a CUDA build of PyTorch."
        )

    dataset = load_jsonl(args.data)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
        if "type" not in config.rope_scaling:
            config.rope_scaling = None

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        quantization_config=quantization_config,
        device_map={"": 0},
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    training_arguments = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        optim="paged_adamw_8bit",
        logging_steps=10,
        learning_rate=2e-4,
        fp16=not use_bf16,
        bf16=use_bf16,
        max_grad_norm=0.3,
        max_steps=-1,
        lr_scheduler_type="constant",
        report_to="none",
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
        return {
            "text": tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    dataset = dataset.map(to_text, remove_columns=[c for c in dataset.column_names if c != "text"])

    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "peft_config": lora_config,
        "args": training_arguments,
    }
    try:
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = SFTTrainer(tokenizer=tokenizer, **trainer_kwargs)

    resume_from = resolve_checkpoint(args.output_dir, resume=not args.no_resume)
    trainer.train(resume_from_checkpoint=resume_from)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved adapters and tokenizer to {args.output_dir}")
    last = get_last_checkpoint(str(args.output_dir))
    if last:
        print(f"Latest resume checkpoint: {last}")

    repo_id = (args.push_to_hub or "").strip()
    if repo_id:
        push_output(args.output_dir, repo_id, args.model)
    else:
        print("Skipping Hub upload (set HF_HUB_REPO or --push-to-hub).")


if __name__ == "__main__":
    main()
