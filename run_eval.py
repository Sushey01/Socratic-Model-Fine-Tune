#!/usr/bin/env python3
"""ScienceQA eval only — load saved QLoRA adapters, no SFT. Logs to W&B if WANDB_API_KEY is set."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer, BitsAndBytesConfig

from eval_scienceqa import load_eval_slice, ngram_overlap, run_scienceqa_eval
from train import (
    DEFAULT_DATA,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT,
    from_pretrained_phi3,
    init_wandb,
    log_scienceqa_to_wandb,
    wandb_enabled,
)

ROOT = Path(__file__).resolve().parent


def load_adapters(adapter_dir: Path, base: str):
    tok = AutoTokenizer.from_pretrained(
        adapter_dir if (adapter_dir / "tokenizer_config.json").is_file() else base,
        trust_remote_code=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    try:
        model = from_pretrained_phi3(base, quantization_config=bnb, device_map={"": 0})
    except Exception as exc:
        print(f"4-bit load failed ({exc}); using fp16.", flush=True)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        try:
            model = from_pretrained_phi3(base, dtype=dtype, device_map={"": 0})
        except TypeError:
            model = from_pretrained_phi3(base, torch_dtype=dtype, device_map={"": 0})
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return model, tok


def main() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if not torch.cuda.is_available():
        raise SystemExit("Need a CUDA GPU for eval.")
    adapter = Path(os.environ.get("ADAPTER_DIR") or DEFAULT_OUTPUT)
    if not adapter.is_dir():
        raise SystemExit(f"No adapters at {adapter}. Train first or: python start.py --download-checkpoints")
    base = os.environ.get("BASE_MODEL", DEFAULT_MODEL)
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
