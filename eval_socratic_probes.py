#!/usr/bin/env python3
"""Score first-turn Socratic probes: question + no leak phrases + short.

Does not replace ScienceQA. GPU generate is optional (--adapter).
Otherwise pass --replies evals/v7_4b_socratic_replies.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROBES = ROOT / "evals" / "socratic_probes.json"


def sentence_count(text: str) -> int:
    bits = [p for p in re.split(r"[.!?]+", text.strip()) if p.strip()]
    return max(len(bits), 1 if text.strip() else 0)


def score_reply(reply: str, leak_phrases: list[str]) -> dict:
    blob = (reply or "").strip()
    low = blob.lower()
    has_q = "?" in blob
    leaks = [p for p in leak_phrases if p.lower() in low]
    short = sentence_count(blob) <= 3
    ok = bool(blob) and has_q and not leaks and short
    return {
        "has_question": has_q,
        "no_leak": not leaks,
        "leak_hits": leaks,
        "short": short,
        "sentence_count": sentence_count(blob),
        "pass": ok,
        "reply": blob,
    }


def load_probes() -> list[dict]:
    return json.loads(PROBES.read_text(encoding="utf-8"))["probes"]


def generate_replies(adapter: Path, base: str) -> dict[str, str]:
    import os

    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    from train_qwen import from_pretrained_qwen

    system = (
        "You are a Socratic Science Tutor for a Grade 10 student, covering the full Grade 10 "
        "science curriculum. Never give the final answer directly. Guide the student toward it "
        "with questions. Keep responses to 1-3 sentences."
    )
    tok = AutoTokenizer.from_pretrained(
        adapter if (adapter / "tokenizer_config.json").is_file() else base,
        trust_remote_code=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    try:
        model = from_pretrained_qwen(base, dtype=dtype, device_map="auto")
    except TypeError:
        model = from_pretrained_qwen(base, torch_dtype=dtype, device_map="auto")
    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    out: dict[str, str] = {}
    for probe in load_probes():
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": probe["prompt"]},
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
            gen = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        reply = tok.decode(gen[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        out[probe["id"]] = reply.strip()
        print(f"{probe['id']}: {out[probe['id']][:80]}", flush=True)
    _ = os.environ
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replies", type=Path, help="JSON map probe_id -> reply text")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--base", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--out", type=Path, default=ROOT / "evals" / "v7_4b_socratic_probes.json")
    parser.add_argument("--model-id", default="v7_4b")
    args = parser.parse_args()

    replies: dict[str, str] = {}
    if args.replies and args.replies.is_file():
        replies = json.loads(args.replies.read_text(encoding="utf-8"))
    elif args.adapter:
        replies = generate_replies(args.adapter, args.base)
    else:
        known = ROOT / "evals" / "v7_4b_socratic_replies.json"
        if known.is_file():
            replies = json.loads(known.read_text(encoding="utf-8"))

    scored = []
    for probe in load_probes():
        reply = replies.get(probe["id"], "")
        row = {"id": probe["id"], "prompt": probe["prompt"], "scored": bool(reply)}
        if reply:
            row.update(score_reply(reply, probe["leak_phrases"]))
        else:
            row["pass"] = None
        scored.append(row)

    done = [r for r in scored if r.get("scored")]
    n_pass = sum(1 for r in done if r.get("pass"))
    report = {
        "model_id": args.model_id,
        "n_probes": len(scored),
        "n_scored": len(done),
        "n_pass": n_pass,
        "socratic_probe_score": (n_pass / len(done)) if done else None,
        "probes": scored,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {args.out} score={report['socratic_probe_score']} "
        f"({n_pass}/{len(done)} scored of {len(scored)})",
        flush=True,
    )


if __name__ == "__main__":
    main()
