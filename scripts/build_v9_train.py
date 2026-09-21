#!/usr/bin/env python3
"""Build dataset7b/socratic_v9_train.jsonl + holdout eval.

Cleaned v7 (same as v8) + gold stems (2 paths) + extra train intents (2 paths).
Does not overwrite v7. Last 3 extras per concept are holdout only.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_v8_train import CLEAN, clean_v7, write_jsonl  # noqa: E402
from gold_v8_stems import GOLD_STEMS  # noqa: E402
from v9_chats import two_paths  # noqa: E402
from v9_extra import V9_EXTRA  # noqa: E402
from v9_extra_rest import V9_EXTRA_REST  # noqa: E402

import v9_extra_rest2  # noqa: F401,E402
import v9_extra_rest3  # noqa: F401,E402
import v9_extra_rest4  # noqa: F401,E402
import v9_extra_rest5  # noqa: F401,E402
import v9_extra_rest6  # noqa: F401,E402
import v9_extra_rest7  # noqa: F401,E402
import v9_extra_rest8  # noqa: F401,E402
import v9_extra_rest9  # noqa: F401,E402

V9 = ROOT / "dataset7b" / "socratic_v9_train.jsonl"
HOLDOUT = ROOT / "dataset7b" / "holdout_eval.jsonl"
INTENTS = ROOT / "dataset7b" / "socratic_v9_intents.jsonl"
SEED = 42
STUCK = ("wrong", "idk", "explain")


def extras_by_concept() -> dict[str, list[dict[str, str]]]:
    merged = dict(V9_EXTRA_REST)
    merged.update(V9_EXTRA)
    return merged


def strip_holdout_q(s: dict[str, str]) -> dict[str, str]:
    q = s["q"]
    if q.startswith("HOLDOUT:"):
        q = q[len("HOLDOUT:") :].strip()
    out = dict(s)
    out["q"] = q
    return out


def expand(stems: list[dict[str, str]], start: int = 0) -> list[dict]:
    rows: list[dict] = []
    for i, s in enumerate(stems):
        stuck = STUCK[(start + i) % len(STUCK)]
        rows.extend(two_paths(strip_holdout_q(s), stuck))
    return rows


def main() -> None:
    extras = extras_by_concept()
    missing = sorted(set(GOLD_STEMS) - set(extras))
    extra_only = sorted(set(extras) - set(GOLD_STEMS))
    if missing:
        raise SystemExit(f"Missing v9 extras for {len(missing)} concepts: {missing}")
    if extra_only:
        raise SystemExit(f"Unknown extra concepts: {extra_only}")
    short = [c for c, rows in extras.items() if len(rows) < 10]
    if short:
        raise SystemExit(f"Need 10 extras (last 3 holdout) for: {short}")

    clean, stats = clean_v7()
    write_jsonl(CLEAN, clean)

    train_chats: list[dict] = []
    hold_chats: list[dict] = []
    intent_rows: list[dict] = []

    for concept, gold in GOLD_STEMS.items():
        extra = extras[concept]
        train_extra, hold_extra = extra[:-3], extra[-3:]
        for s in gold:
            intent_rows.append({"concept": concept, "split": "train", "source": "gold", "q": s["q"]})
        for s in train_extra:
            intent_rows.append(
                {"concept": concept, "split": "train", "source": "extra", "q": strip_holdout_q(s)["q"]}
            )
        for s in hold_extra:
            intent_rows.append(
                {"concept": concept, "split": "holdout", "source": "extra", "q": strip_holdout_q(s)["q"]}
            )
        train_chats.extend(expand(gold, 0))
        train_chats.extend(expand(train_extra, 1))
        for i, s in enumerate(hold_extra):
            s2 = strip_holdout_q(s)
            for chat in two_paths(s2, STUCK[i % 3]):
                hold_chats.append({"concept": concept, "q": s2["q"], **chat})

    rng = random.Random(SEED)
    v9 = list(clean) + train_chats
    rng.shuffle(v9)
    write_jsonl(V9, v9)
    write_jsonl(HOLDOUT, hold_chats)
    write_jsonl(INTENTS, intent_rows)

    n_gold_intents = sum(1 for r in intent_rows if r["source"] == "gold")
    n_train_extra = sum(1 for r in intent_rows if r["split"] == "train" and r["source"] == "extra")
    n_hold = sum(1 for r in intent_rows if r["split"] == "holdout")
    print("v7 train in:", stats["train_in"], "dumps dropped:", stats["dropped_dumps"])
    print("clean v7:", stats["clean_out"], "->", CLEAN)
    print("concepts:", len(GOLD_STEMS))
    print("gold intents:", n_gold_intents, "extra train intents:", n_train_extra, "holdout intents:", n_hold)
    print("gold+extra train chats:", len(train_chats), "(2 paths each)")
    print("v9 total:", len(v9), "->", V9)
    print("holdout chats:", len(hold_chats), "->", HOLDOUT)
    print("intent list:", INTENTS)
    print("Train with: --data", V9, "--epochs 2")


if __name__ == "__main__":
    main()
