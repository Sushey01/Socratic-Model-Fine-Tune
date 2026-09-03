#!/usr/bin/env python3
"""ScienceQA eval for Qwen3 QLoRA adapters. Does not load Phi-3."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer, BitsAndBytesConfig

from eval_scienceqa import load_eval_slice, ngram_overlap, run_scienceqa_eval
from train import DEFAULT_DATA, init_wandb, log_scienceqa_to_wandb, wandb_enabled
from train_qwen import DEFAULT_MODEL, DEFAULT_OUTPUT, from_pretrained_qwen

ROOT = Path(__file__).resolve().parent


def load_adapters(adapter_dir: Path, base: str):
    tok = AutoTokenizer.from_pretrained(
        adapter_dir if (adapter_dir / "tokenizer_config.json").is_file() else base,
        trust_remote_code=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    compute = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute,
        bnb_4bit_use_double_quant=True,
    )
    try:
        model = from_pretrained_qwen(base, quantization_config=bnb, device_map={"": 0})
    except Exception as exc:
        print(f"4-bit load failed ({exc}); using fp16.", flush=True)
        try:
            model = from_pretrained_qwen(base, dtype=compute, device_map={"": 0})
        except TypeError:
            model = from_pretrained_qwen(base, torch_dtype=compute, device_map={"": 0})
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return model, tok


def main() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("WANDB_RUN_NAME", "qwen3-4b-instruct-eval")
    if not torch.cuda.is_available():
        raise SystemExit("Need a CUDA GPU for eval.")
    adapter = Path(os.environ.get("ADAPTER_DIR") or DEFAULT_OUTPUT)
    if not adapter.is_dir():
        raise SystemExit(
            f"No Qwen adapters at {adapter}. Train first: python start.py --qwen"
        )
    base = os.environ.get("BASE_MODEL") or DEFAULT_MODEL
    os.environ["BASE_MODEL"] = base
    print(f"Eval adapters={adapter} base={base}", flush=True)
    model, tok = load_adapters(adapter, base)
    items = load_eval_slice()
    overlap = ngram_overlap([x["question"] for x in items], DEFAULT_DATA)
    print(f"ScienceQA n={len(items)} 5-gram overlap={overlap:.4f}", flush=True)
    init_wandb()
    if wandb_enabled():
        import wandb

        if wandb.run is not None:
            wandb.summary["eval/ngram_overlap"] = overlap
            wandb.summary["eval/n"] = len(items)
    print("Running ScienceQA (KV cache on; can take a while)...", flush=True)
    metrics = run_scienceqa_eval(model, tok, items)
    print(
        f"ScienceQA acc={metrics['eval/scienceqa_acc']:.4f} "
        f"sri={metrics['eval/scienceqa_sri']:.4f} n={int(metrics['eval/n'])}",
        flush=True,
    )
    log_scienceqa_to_wandb(metrics, epoch=3.0, step=0)
    if wandb_enabled():
        import wandb

        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
