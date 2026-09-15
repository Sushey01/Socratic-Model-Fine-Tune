#!/usr/bin/env python3
"""Convert v4 annotated turns JSONL into SFT chat messages JSONL.

Maps student→user, tutor→assistant. Drops strategy and other metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_SRC = ROOT / "socratic_dataset_v4_annotated.jsonl"
DEFAULT_OUT = ROOT / "socratic_train_v4.jsonl"

SOCRATIC_SYSTEM = (
    "You are a Socratic Science Tutor for a Grade 10 student, covering the full Grade 10 "
    "science curriculum (life processes, control and coordination, reproduction, heredity, "
    "electricity and magnetism, light, the human eye, pressure, gases, waves, chemical "
    "reactions, acids and bases, metals and non-metals, carbon compounds, classification, "
    "motion and force, and more). Never give the final answer directly. Guide the student "
    "toward it with questions, using their previous response to decide your next move. "
    "If the student switches to a different topic, follow their lead. Keep responses to "
    "1-3 sentences."
)

ROLE_MAP = {"student": "user", "tutor": "assistant"}


def convert_row(row: dict) -> dict | None:
    turns = row.get("turns")
    if not isinstance(turns, list) or not turns:
        return None
    messages = [{"role": "system", "content": SOCRATIC_SYSTEM}]
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        role = ROLE_MAP.get(str(turn.get("role") or "").strip().lower())
        content = str(turn.get("content") or "").strip()
        if not role or not content:
            continue
        messages.append({"role": role, "content": content})
    if len(messages) < 3:
        return None
    return {"messages": messages}


def convert(src: Path, dest: Path) -> int:
    n_in = 0
    n_out = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    with src.open(encoding="utf-8") as hin, dest.open("w", encoding="utf-8") as hout:
        for i, line in enumerate(hin, start=1):
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"skip line {i}: {exc}", file=sys.stderr)
                continue
            converted = convert_row(row)
            if converted is None:
                print(f"skip line {i}: no usable turns", file=sys.stderr)
                continue
            hout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"Converted {n_out}/{n_in} conversations → {dest}")
    return n_out


def main() -> None:
    parser = argparse.ArgumentParser(description="v4 turns JSONL → SFT messages JSONL")
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not args.src.is_file():
        raise SystemExit(f"Missing {args.src}")
    n = convert(args.src, args.out)
    if n == 0:
        raise SystemExit("Converted 0 rows.")


if __name__ == "__main__":
    main()
