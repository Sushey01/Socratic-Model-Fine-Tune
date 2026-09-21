#!/usr/bin/env python3
"""Build dataset7b/socratic_v8_train.jsonl: cleaned v7 + gold Socratic chats.

Does not overwrite v7 train/val/annotated.
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from gold_v8_stems import GOLD_STEMS, SYSTEM, make_gold_conversations  # noqa: E402

ANN = ROOT / "dataset7b" / "socratic_dataset_v7_annotated.jsonl"
TRAIN = ROOT / "dataset7b" / "socratic_v7_train.jsonl"
CLEAN = ROOT / "dataset7b" / "socratic_v7_clean_train.jsonl"
V8 = ROOT / "dataset7b" / "socratic_v8_train.jsonl"
KEEP_PER_CONCEPT = 18
SEED = 42

ASK = re.compile(
    r"(explain.{0,40}(more|clear|further|again)|more clearly|"
    r"just tell me|tell me (the answer|what it is|directly)|please just tell|"
    r"what does that mean|could you elaborate|go deeper|break that down|"
    r"i don't want a hint|can't you just tell|tell me directly|"
    r"i'm still confused|could you go deeper|explain that further|"
    r"i'm (a bit )?lost|still not clicking)",
    re.I,
)
NUDGE = re.compile(
    r"(try this angle|here's a nudge|work this one out|think about|"
    r"think of|i think you can get|let's find it together)",
    re.I,
)
ECHO = re.compile(r"\s*.{0,200}?\bis exactly right\.?\s*", re.I)


def is_dump(text: str) -> bool:
    if "?" in text:
        return False
    if NUDGE.search(text) and len(text) < 280:
        return False
    return True


def turns_key(turns: list) -> tuple:
    parts = []
    for t in turns:
        role = t.get("role")
        if role == "student":
            role = "user"
        elif role == "tutor":
            role = "assistant"
        parts.append((role, (t.get("content") or "").strip()))
    return tuple(parts)


def msgs_key(msgs: list) -> tuple:
    return tuple(
        (m.get("role"), (m.get("content") or "").strip())
        for m in msgs
        if m.get("role") != "system"
    )


def row_is_dump(msgs: list) -> bool:
    for j in range(len(msgs) - 1):
        if msgs[j].get("role") != "user":
            continue
        if not ASK.search(msgs[j].get("content") or ""):
            continue
        nxt = msgs[j + 1]
        if nxt.get("role") == "assistant" and is_dump(nxt.get("content") or ""):
            return True
    return False


def strip_verify_echo(text: str) -> str:
    raw = (text or "").strip()
    if "exactly right" not in raw.lower():
        return raw
    cut = ECHO.sub(" ", raw)
    cut = re.sub(r"\s+", " ", cut).strip(" ,.")
    first = re.split(r"(?<=[.!?])\s+", raw, maxsplit=1)[0].strip()
    praise = re.match(
        r"^(Right|That's it|That's correct|Exactly|Well done|Correct|Perfect|Yes, exactly)[^.!?]*[.!]?",
        first,
        re.I,
    )
    if praise:
        s = praise.group(0)
        if not s.endswith((".", "!", "?")):
            s += "."
        return s
    if cut and len(cut) < 80:
        return cut + ("." if not cut.endswith((".", "!", "?")) else "")
    return "That's it — you reasoned that through."


def first_answer(msgs: list) -> str:
    users = [m["content"] for m in msgs if m.get("role") == "user"]
    if len(users) >= 2:
        return users[1].strip().lower()
    return users[0].strip().lower() if users else ""


def downsample(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    if len(rows) <= n:
        return list(rows)
    scored = []
    seen = set()
    for r in rows:
        ans = first_answer(r["messages"])
        words = [w for w in re.findall(r"[a-z0-9']+", ans) if len(w) > 1]
        uniq = " ".join(words) if len(words) >= 5 else ans
        bonus = 0 if uniq in seen else 1
        seen.add(uniq)
        scored.append((bonus, rng.random(), r))
    scored.sort(key=lambda x: (-x[0], x[1]))
    picked = [r for _, _, r in scored[:n]]
    rng.shuffle(picked)
    return picked


def load_ann_index() -> dict[tuple, dict]:
    idx = {}
    with ANN.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            idx[turns_key(r["turns"])] = r
    return idx


def clean_v7() -> tuple[list[dict], dict]:
    rng = random.Random(SEED)
    ann_idx = load_ann_index()
    by_concept: dict[str, list[dict]] = defaultdict(list)
    n_in = n_dump = n_unmatched = 0
    with TRAIN.open(encoding="utf-8") as f:
        for line in f:
            n_in += 1
            row = json.loads(line)
            msgs = row["messages"]
            if row_is_dump(msgs):
                n_dump += 1
                continue
            meta = ann_idx.get(msgs_key(msgs))
            if meta is None:
                n_unmatched += 1
                concept = "_unmatched"
            else:
                concept = meta.get("concept") or "_unknown"
            last = None
            for i in range(len(msgs) - 1, -1, -1):
                if msgs[i].get("role") == "assistant":
                    last = i
                    break
            if last is not None:
                msgs = [dict(m) for m in msgs]
                msgs[last]["content"] = strip_verify_echo(msgs[last]["content"])
            by_concept[concept].append({"messages": msgs, "concept": concept})

    kept: list[dict] = []
    per = {}
    for concept, rows in sorted(by_concept.items()):
        chosen = downsample(rows, KEEP_PER_CONCEPT, rng)
        per[concept] = (len(rows), len(chosen))
        for r in chosen:
            kept.append({"messages": r["messages"]})
    rng.shuffle(kept)
    stats = {
        "train_in": n_in,
        "dropped_dumps": n_dump,
        "unmatched": n_unmatched,
        "clean_out": len(kept),
        "concepts": len(per),
        "per_concept": per,
    }
    return kept, stats


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    clean, stats = clean_v7()
    write_jsonl(CLEAN, clean)
    gold = make_gold_conversations(SYSTEM)
    missing = sorted(set(GOLD_STEMS) - set(stats["per_concept"]) - {"_unmatched", "_unknown"})
    v8 = list(clean) + gold
    random.Random(SEED).shuffle(v8)
    write_jsonl(V8, v8)
    print("v7 train in:", stats["train_in"])
    print("dropped dumps:", stats["dropped_dumps"])
    print("unmatched to annotated:", stats["unmatched"])
    print("clean v7 out:", stats["clean_out"], "->", CLEAN)
    print("gold chats:", len(gold), "stems-file concepts:", len(GOLD_STEMS))
    print("v8 total:", len(v8), "->", V8)
    print("concepts after clean:", stats["concepts"])
    if missing:
        print("gold concepts not in clean map (ok if only gold):", missing[:10], "...")
    thin = [(c, a, b) for c, (a, b) in stats["per_concept"].items() if b < 10]
    if thin:
        print("thin concepts after downsample:", thin[:8])


if __name__ == "__main__":
    main()
