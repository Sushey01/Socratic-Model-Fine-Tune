# v8 train file (for review)

**File:** `dataset7b/socratic_v8_train.jsonl` (2,064 chats)  
**Original v7 train/val/annotated were not overwritten.**

Also written: `dataset7b/socratic_v7_clean_train.jsonl` (972 chats) — cleaned v7 only.

## What’s in v8

| Part | Count |
| --- | --- |
| Cleaned v7 | 972 |
| New gold-style chats | 1,092 |
| **Total** | **2,064** |

Cleaned v7: dropped **378** definition-dumps; kept **18** chats per concept (54 concepts); shortened VERIFY lines that said “is exactly right.”

Gold: **273** unique everyday questions × 4 student variants (on-track, wrong, “I don’t know”, “Explain it more clearly”). All 273 explain-more replies contain a `?`.

Rebuild: `python3 scripts/build_v8_train.py`

Train (after you approve the file):

```text
python train_qwen.py --data dataset7b/socratic_v8_train.jsonl --epochs 2
python train_qwen25.py --data dataset7b/socratic_v8_train.jsonl --epochs 2
```
