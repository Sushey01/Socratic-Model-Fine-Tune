#!/usr/bin/env python3
"""Keep v7 chats whose first tutor turn is a short question without a dump/close.

Does not mix v9 JSONL. Writes a filtered JSONL + a JSON report for review.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "dataset7b" / "socratic_v7_final_v3.jsonl"
OUT = ROOT / "dataset7b" / "socratic_v7_final_v3_socratic.jsonl"
REPORT = ROOT / "evals" / "v7_dataset_refine_report.json"

DUMP_USER = re.compile(r"\b(explain|tell me the answer|just tell me|give me the answer)\b", re.I)
CLOSE = re.compile(r"^\s*that'?s it\b", re.I)
LEAKY_FIRST = re.compile(
    r"less dense|the answer is|f\s*=\s*ma|unit of |is defined as|consists of",
    re.I,
)


def sentences(text: str) -> int:
    return len([p for p in re.split(r"[.!?]+", text.strip()) if p.strip()])


def first_assistant(messages: list) -> str:
    for m in messages:
        if m.get("role") == "assistant":
            return str(m.get("content") or "")
    return ""


def last_assistant(messages: list) -> str:
    text = ""
    for m in messages:
        if m.get("role") == "assistant":
            text = str(m.get("content") or "")
    return text


def decide(messages: list) -> str | None:
    """Return drop reason or None to keep."""
    first = first_assistant(messages)
    if not first:
        return "no_assistant"
    if "?" not in first:
        return "first_turn_no_question"
    if sentences(first) > 3 or len(first) > 420:
        return "first_turn_dump"
    if LEAKY_FIRST.search(first):
        return "first_turn_leak_phrase"
    users = [str(m.get("content") or "") for m in messages if m.get("role") == "user"]
    if any(DUMP_USER.search(u) for u in users):
        # keep misconception / I don't know; drop only if assistant then lectures
        for m in messages:
            if m.get("role") != "assistant":
                continue
            body = str(m.get("content") or "")
            if "?" not in body and len(body) > 200:
                return "explain_dump"
    if CLOSE.search(last_assistant(messages)):
        return "close_with_thats_it"
    return None


def main() -> None:
    keep: list[str] = []
    reasons: dict[str, int] = {}
    n = 0
    for line in SRC.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        n += 1
        row = json.loads(line)
        reason = decide(row.get("messages") or [])
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        keep.append(line)
    OUT.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
    report = {
        "src": str(SRC.relative_to(ROOT)),
        "out": str(OUT.relative_to(ROOT)),
        "src_n": n,
        "kept_n": len(keep),
        "dropped_n": n - len(keep),
        "drop_reasons": reasons,
        "policy": [
            "first assistant turn must contain ?",
            "first turn <= 3 sentences and <= 420 chars",
            "first turn must not match leak regex",
            "drop chats whose last assistant turn starts That's it",
            "drop explain-path lectures with no question",
            "do not mix socratic_v9_train.jsonl",
        ],
        "next": "Review OUT then upload a new Hub dataset version; do not overwrite v7_socratic_data until reviewed.",
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
