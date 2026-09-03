#!/usr/bin/env python3
"""Load Qwen QLoRA adapters and run one Socratic tutor turn (GPU deploy smoke test)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer

from train_qwen import DEFAULT_HUB, DEFAULT_MODEL, DEFAULT_OUTPUT, from_pretrained_qwen

ROOT = Path(__file__).resolve().parent
SYSTEM = (
    "You are a Socratic Science Tutor for a Grade 10 student, covering the full Grade 10 "
    "science curriculum. Never give the final answer directly. Guide the student toward it "
    "with questions. Keep responses to 1-3 sentences."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Socratic Qwen adapter inference")
    parser.add_argument("--adapter", default=os.environ.get("ADAPTER_DIR") or str(DEFAULT_OUTPUT))
    parser.add_argument("--base", default=os.environ.get("BASE_MODEL") or DEFAULT_MODEL)
    parser.add_argument("--hub", action="store_true", help=f"Load adapters from HF_QWEN_REPO ({DEFAULT_HUB})")
    parser.add_argument("--prompt", default="Why does ice float on water?")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    adapter = (os.environ.get("HF_QWEN_REPO") or DEFAULT_HUB) if args.hub else args.adapter
    print(f"base={args.base} adapter={adapter}", flush=True)
    tok = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    try:
        model = from_pretrained_qwen(args.base, dtype=dtype, device_map="auto")
    except TypeError:
        model = from_pretrained_qwen(args.base, torch_dtype=dtype, device_map="auto")
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": args.prompt},
    ]
    try:
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    reply = tok.decode(out[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    print(reply.strip())


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
