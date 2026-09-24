#!/usr/bin/env python3
"""
Socratic Dataset Auditor & Topic Variance Checker
=================================================
A self-contained tool to:
1. Detect exact and near-duplicate user questions.
2. Distinguish between redundant exact duplicates and valuable distinct response variants.
3. Track topic distribution across Grade 10 Physics, Chemistry, Biology, and Environmental Science.
4. Detect "topic loops" (over-represented micro-topics).
5. Highlight missing/under-represented syllabus topics to guide further data generation.
6. Pre-check new incoming data (via file, pipe, or paste) before appending.
"""

import sys
import os
import json
import re
import argparse
from collections import defaultdict, Counter
from difflib import SequenceMatcher

DEFAULT_DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "socratic_dataset.jsonl")

# Grade 10 Science Topic Taxonomy
CURRICULUM_TAXONOMY = {
    "Physics": {
        "Light - Reflection & Refraction": [
            "refraction", "reflection", "mirror", "lens", "focal length", "prism",
            "dispersion", "rainbow", "straw", "bent", "apparent depth", "snell",
            "concave", "convex", "radius of curvature", "optical", "speed of light"
        ],
        "The Human Eye & Colourful World": [
            "myopia", "hypermetropia", "retina", "cornea", "eye", "scattering",
            "sky", "sunset", "sunrise", "tyndall", "twinkling", "presbyopia"
        ],
        "Electricity": [
            "resistance", "resistor", "ohm", "volt", "voltage", "current", "ampere",
            "potential difference", "circuit", "series", "parallel", "resistivity",
            "joule heating", "power", "watt", "charge", "coulomb"
        ],
        "Magnetic Effects of Current": [
            "magnetic", "magnet", "solenoid", "motor", "generator", "induction",
            "electromagnetic", "lenz", "fleming", "compass", "eddy current", "copper pipe"
        ],
        "Mechanics, Forces & Motion": [
            "inertia", "momentum", "newton", "friction", "skateboard", "bus stops",
            "parachute", "drag", "air resistance", "terminal velocity", "ball", "force",
            "action", "reaction", "angular momentum", "skater", "spin", "swivel chair"
        ],
        "Buoyancy & Fluids": [
            "buoyancy", "float", "sink", "ship", "nail", "archimedes", "density",
            "displace"
        ],
        "Work, Energy & Waves": [
            "work", "joule", "energy", "frequency", "hertz", "wave", "sound",
            "light speed", "thunder", "lightning"
        ]
    },
    "Chemistry": {
        "Chemical Reactions & Equations": [
            "chemical reaction", "displacement", "double displacement", "precipitate",
            "combination", "decomposition", "redox", "oxidation", "reduction",
            "exothermic", "endothermic", "chalk", "acid", "reaction rate", "collision",
            "catalyst", "activation energy"
        ],
        "Acids, Bases & Salts": [
            "acid", "base", "ph", "neutralization", "salt", "litmus", "indicator",
            "turmeric", "hydrogen ion", "hydroxide"
        ],
        "Metals & Non-Metals": [
            "metal", "non-metal", "corrosion", "rust", "galvanization", "reactivity series",
            "cation", "anion", "alloy", "brass", "bronze", "steel", "ionic compound",
            "lattice", "conduct electricity"
        ],
        "Carbon & Its Compounds": [
            "carbon", "catenation", "tetravalency", "hydrocarbon", "alkane", "alkene",
            "alkyne", "covalent", "polymer", "monomer", "plastic", "soap", "detergent",
            "isomer"
        ],
        "Periodic Classification & Atomic Structure": [
            "atomic number", "valency", "valence", "octet", "electron", "proton",
            "neutron", "isotope", "shell", "oxygen", "chlorine", "noble gas"
        ],
        "States of Matter & Solutions": [
            "saturated", "solution", "solute", "solvent", "boiling point elevation",
            "condensation", "breath", "winter", "evaporation", "colligative"
        ]
    },
    "Biology": {
        "Life Processes - Nutrition & Digestion": [
            "photosynthesis", "chlorophyll", "chloroplast", "villi", "small intestine",
            "absorption", "digestive", "enzyme"
        ],
        "Life Processes - Respiration": [
            "respiration", "aerobic", "anaerobic", "lactic acid", "cramp", "oxygen debt",
            "sprint", "breathe heavier", "yawn", "alveoli", "gas exchange"
        ],
        "Life Processes - Transportation & Osmosis": [
            "xylem", "phloem", "osmosis", "transpiration", "root hair", "plant roots",
            "celery", "stomata", "guard cell", "turgor"
        ],
        "Control & Coordination": [
            "nervous system", "endocrine", "hormone", "insulin", "pancreas",
            "blood sugar", "glucose", "homeostasis", "reflex arc", "neuron", "synapse"
        ],
        "Reproduction": [
            "vegetative propagation", "asexual", "sexual", "clone", "cutting",
            "budding", "fission", "flower", "pollination"
        ],
        "Heredity & Genetics": [
            "heredity", "gene", "allele", "mendel", "heterozygous", "homozygous",
            "dominant", "recessive", "pea plant", "ribosome", "protein synthesis", "dna"
        ]
    },
    "Environmental Science": {
        "Our Environment & Ecosystems": [
            "ecosystem", "food chain", "food web", "biological magnification", "trophic",
            "biomagnification", "pesticide", "abiotic", "biotic", "decomposer"
        ],
        "Natural Resources & Earth": [
            "earth", "atmosphere", "magnetic pole", "natural resources", "conservation",
            "pollution", "ozone"
        ]
    }
}


def normalize_text(text: str) -> str:
    """Normalize text by lowercasing, stripping punctuation, and collapsing whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def extract_plan_goal(assistant_content: str) -> str:
    """Extract content inside <plan>...</plan>."""
    match = re.search(r"<plan>(.*?)</plan>", assistant_content, re.DOTALL | re.IGNORECASE)
    if match:
        clean = match.group(1).strip()
        clean = re.sub(r"^(goal:|topic:.*?goal:)\s*", "", clean, flags=re.IGNORECASE).strip()
        return clean
    return ""


def classify_topic(user_question: str, plan_text: str, full_dialogue: str) -> tuple[str, str]:
    """Classify dialogue into Major Category and Specific Subtopic."""
    search_corpus = f"{user_question} {plan_text} {full_dialogue}".lower()

    best_cat = "General Science"
    best_subtopic = "Miscellaneous"
    highest_score = 0

    for category, subtopics in CURRICULUM_TAXONOMY.items():
        for subtopic, keywords in subtopics.items():
            score = 0
            for kw in keywords:
                # Direct word boundary or phrase match
                if " " in kw:
                    if kw in search_corpus:
                        score += 3
                else:
                    if re.search(r"\b" + re.escape(kw) + r"\b", search_corpus):
                        score += 1
            if score > highest_score:
                highest_score = score
                best_cat = category
                best_subtopic = subtopic

    return best_cat, best_subtopic


def parse_dialogue_entry(line_str: str, line_no: int):
    """Parse JSON line and extract key fields."""
    try:
        data = json.loads(line_str.strip().rstrip(","))
    except Exception as e:
        return None, f"Line {line_no}: Invalid JSON - {e}"

    messages = data.get("messages", [])
    if not messages:
        return None, f"Line {line_no}: Empty messages list"

    user_q = ""
    first_assistant = ""
    all_assistant = []
    plan_goals = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user" and not user_q:
            user_q = content.strip()
        elif role == "assistant":
            if not first_assistant:
                first_assistant = content.strip()
            all_assistant.append(content.strip())
            goal = extract_plan_goal(content)
            if goal:
                plan_goals.append(goal)

    turn_count = sum(1 for m in messages if m.get("role") == "assistant")

    norm_q = normalize_text(user_q)
    full_text = " ".join([m.get("content", "") for m in messages])
    major_cat, subtopic = classify_topic(user_q, " ".join(plan_goals), full_text)

    return {
        "line_no": line_no,
        "raw": data,
        "user_q": user_q,
        "norm_q": norm_q,
        "first_assistant": first_assistant,
        "all_assistant_text": " ".join(all_assistant),
        "plan_goals": plan_goals,
        "turns": turn_count,
        "major_category": major_cat,
        "subtopic": subtopic,
    }, None


class DatasetAuditor:
    def __init__(self, dataset_path: str = DEFAULT_DATASET):
        self.dataset_path = dataset_path
        self.entries = []
        self.errors = []
        self.load_dataset()

    def load_dataset(self):
        if not os.path.exists(self.dataset_path):
            print(f"Error: Dataset not found at {self.dataset_path}")
            return

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                entry, err = parse_dialogue_entry(line, idx)
                if err:
                    self.errors.append(err)
                else:
                    self.entries.append(entry)

    def find_duplicates(self, fuzzy_threshold: float = 0.85):
        """Find exact and near duplicate user questions."""
        exact_groups = defaultdict(list)
        for e in self.entries:
            exact_groups[e["norm_q"]].append(e)

        exact_dups = {k: v for k, v in exact_groups.items() if len(v) > 1}

        # Near duplicates across distinct questions
        unique_norm_qs = list(exact_groups.keys())
        near_dups = []

        for i in range(len(unique_norm_qs)):
            q1 = unique_norm_qs[i]
            for j in range(i + 1, len(unique_norm_qs)):
                q2 = unique_norm_qs[j]
                # Length check optimization
                if abs(len(q1) - len(q2)) / max(len(q1), len(q2)) > 0.4:
                    continue
                sim = SequenceMatcher(None, q1, q2).ratio()
                if sim >= fuzzy_threshold:
                    near_dups.append({
                        "q1": exact_groups[q1][0]["user_q"],
                        "q2": exact_groups[q2][0]["user_q"],
                        "similarity": round(sim * 100, 1),
                        "entries_q1": exact_groups[q1],
                        "entries_q2": exact_groups[q2],
                    })

        return exact_dups, near_dups

    def analyze_variants(self, entries_list):
        """Analyze if entries with the same question have distinct responses or identical copies."""
        response_hashes = defaultdict(list)
        for e in entries_list:
            norm_resp = normalize_text(e["first_assistant"])
            response_hashes[norm_resp].append(e)

        identical_count = sum(len(v) - 1 for v in response_hashes.values() if len(v) > 1)
        distinct_variants = len(response_hashes)

        return {
            "total_instances": len(entries_list),
            "distinct_variants": distinct_variants,
            "identical_redundant": identical_count,
            "variants_detail": list(response_hashes.values()),
        }

    def topic_breakdown(self):
        cat_counts = Counter(e["major_category"] for e in self.entries)
        subtopic_counts = Counter(e["subtopic"] for e in self.entries)
        return cat_counts, subtopic_counts

    def find_loops(self, threshold: int = 3):
        """Find concepts that appear repeatedly (potential loops)."""
        exact_dups, _ = self.find_duplicates()
        looped_questions = []

        for norm_q, entries in exact_dups.items():
            if len(entries) >= threshold:
                variant_analysis = self.analyze_variants(entries)
                looped_questions.append({
                    "question": entries[0]["user_q"],
                    "count": len(entries),
                    "distinct_responses": variant_analysis["distinct_variants"],
                    "identical_responses": variant_analysis["identical_redundant"],
                    "category": entries[0]["major_category"],
                    "subtopic": entries[0]["subtopic"],
                    "lines": [e["line_no"] for e in entries]
                })

        # Sort by frequency descending
        looped_questions.sort(key=lambda x: x["count"], reverse=True)
        return looped_questions

    def find_missing_or_sparse_topics(self, sparse_threshold: int = 4):
        """Identify Grade 10 topics that are under-represented or missing."""
        cat_counts, subtopic_counts = self.topic_breakdown()
        missing = []
        sparse = []

        for category, subtopics in CURRICULUM_TAXONOMY.items():
            for subtopic in subtopics.keys():
                count = subtopic_counts.get(subtopic, 0)
                if count == 0:
                    missing.append({"category": category, "subtopic": subtopic, "count": 0})
                elif count < sparse_threshold:
                    sparse.append({"category": category, "subtopic": subtopic, "count": count})

        return missing, sparse

    def check_new_entries(self, new_raw_lines: list[str]):
        """Check a list of new JSON strings against the existing dataset."""
        results = []
        seen_in_batch = defaultdict(list)

        for idx, line in enumerate(new_raw_lines, start=1):
            line = line.strip()
            if not line:
                continue
            entry, err = parse_dialogue_entry(line, idx)
            if err:
                results.append({
                    "index": idx,
                    "status": "ERROR",
                    "reason": err,
                    "entry": None
                })
                continue

            user_q = entry["user_q"]
            norm_q = entry["norm_q"]

            # 1. Check exact match in existing dataset
            exact_matches = [e for e in self.entries if e["norm_q"] == norm_q]
            # 2. Check in batch
            batch_dups = seen_in_batch[norm_q]
            seen_in_batch[norm_q].append(idx)

            # 3. Fuzzy matches in existing dataset
            fuzzy_matches = []
            if not exact_matches:
                for existing in self.entries:
                    sim = SequenceMatcher(None, norm_q, existing["norm_q"]).ratio()
                    if sim >= 0.85:
                        fuzzy_matches.append((existing, round(sim * 100, 1)))

            # Assess verdict
            if exact_matches:
                # Check response novelty
                existing_resp_norms = [normalize_text(e["first_assistant"]) for e in exact_matches]
                new_resp_norm = normalize_text(entry["first_assistant"])

                if new_resp_norm in existing_resp_norms:
                    status = "EXACT_REDUNDANT"
                    reason = f"Identical question & identical response as line(s) {[e['line_no'] for e in exact_matches]}."
                else:
                    status = "DISTINCT_VARIANT"
                    reason = f"Duplicate question as line(s) {[e['line_no'] for e in exact_matches]}, BUT offers a distinct pedagogical response/plan."
            elif fuzzy_matches:
                top_match, top_sim = sorted(fuzzy_matches, key=lambda x: x[1], reverse=True)[0]
                status = "NEAR_DUPLICATE"
                reason = f"{top_sim}% similar to line {top_match['line_no']} ('{top_match['user_q']}')."
            else:
                status = "NOVEL"
                reason = "Fresh question not currently in dataset."

            results.append({
                "index": idx,
                "status": status,
                "reason": reason,
                "entry": entry,
                "category": entry["major_category"],
                "subtopic": entry["subtopic"]
            })

        return results

    def print_summary_report(self):
        total = len(self.entries)
        print("=" * 80)
        print(f"📊 SOCRATIC DATASET AUDIT REPORT: {os.path.basename(self.dataset_path)}")
        print("=" * 80)
        print(f"Total Valid Dialogues: {total}")
        if self.errors:
            print(f"⚠️ Syntax / Parse Errors: {len(self.errors)}")
            for err in self.errors[:5]:
                print(f"   - {err}")

        # Duplicates
        exact_dups, near_dups = self.find_duplicates()
        total_dup_entries = sum(len(v) for v in exact_dups.values())
        print(f"\n🔍 Duplicate Analysis:")
        print(f"  • Unique User Questions: {len(self.entries) - total_dup_entries + len(exact_dups)}")
        print(f"  • Repeated Question Sets: {len(exact_dups)} (involving {total_dup_entries} dialogues)")
        print(f"  • Near-Duplicate Pairs (Similarity >= 85%): {len(near_dups)}")

        # Variant breakdown
        identical_redundant = 0
        valid_variants = 0
        for norm_q, entries in exact_dups.items():
            analysis = self.analyze_variants(entries)
            identical_redundant += analysis["identical_redundant"]
            valid_variants += (analysis["distinct_variants"] - 1)

        print(f"    - Identical Duplicate Responses (Redundant copies): {identical_redundant}")
        print(f"    - Distinct Pedagogical Variants (Different plans/dialogues): {valid_variants}")

        # Category distribution
        cat_counts, subtopic_counts = self.topic_breakdown()
        print(f"\n📚 Major Subject Distribution:")
        for cat, cnt in cat_counts.most_common():
            pct = (cnt / total * 100) if total else 0
            bar = "█" * int(pct / 3)
            print(f"  • {cat:<24} : {cnt:>4} ({pct:>5.1f}%) {bar}")

        # Looped micro-topics
        loops = self.find_loops(threshold=3)
        if loops:
            print(f"\n🔁 Looped Micro-Topics (Appeared >= 3 times):")
            for item in loops[:10]:
                print(f"  • [{item['count']}x] \"{item['question']}\"")
                print(f"      Category: {item['category']} -> {item['subtopic']}")
                print(f"      Distinct variants: {item['distinct_responses']} | Identical copies: {item['identical_responses']} | Lines: {item['lines']}")

        # Missing / Sparse Topics
        missing, sparse = self.find_missing_or_sparse_topics(sparse_threshold=3)
        if missing or sparse:
            print(f"\n🎯 Curriculum Coverage Gaps (Syllabus Areas to Target Next):")
            if missing:
                print(f"  🚫 Completely Missing Chapters ({len(missing)}):")
                for m in missing:
                    print(f"     - [{m['category']}] {m['subtopic']}")
            if sparse:
                print(f"  ⚠️ Sparse Coverage (< 3 entries) ({len(sparse)}):")
                for s in sparse[:10]:
                    print(f"     - [{s['category']}] {s['subtopic']} ({s['count']} entries)")

        print("=" * 80)


def run_cli():
    parser = argparse.ArgumentParser(description="Audit and inspect Socratic Science dataset.")
    parser.add_argument("--file", "-f", default=DEFAULT_DATASET, help="Path to socratic_dataset.jsonl")
    parser.add_argument("--check-new", "-c", help="Check a new JSONL file before appending")
    parser.add_argument("--paste", "-p", action="store_true", help="Paste new dialogues into terminal to verify")
    parser.add_argument("--duplicates", "-d", action="store_true", help="Print detailed duplicate breakdown")
    parser.add_argument("--missing", "-m", action="store_true", help="Print all missing curriculum topics")
    parser.add_argument("--clean-export", help="Export cleaned dataset without exact identical copies")

    args = parser.parse_args()

    auditor = DatasetAuditor(args.file)

    if args.paste:
        print("\n📋 PASTE MODE: Paste your JSON / JSONL batch below.")
        print("When done, press Ctrl+D (Linux/macOS) or Ctrl+Z then Enter (Windows) on an empty line:\n")
        raw_input_lines = sys.stdin.readlines()
        results = auditor.check_new_entries(raw_input_lines)
        print("\n" + "=" * 70)
        print("🔍 INCOMING BATCH VERIFICATION RESULTS:")
        print("=" * 70)
        for res in results:
            status = res["status"]
            icon = {
                "NOVEL": "✅ [FRESH TOPIC]",
                "DISTINCT_VARIANT": "💡 [VALID VARIANT]",
                "NEAR_DUPLICATE": "⚠️ [NEAR DUPLICATE]",
                "EXACT_REDUNDANT": "❌ [REDUNDANT COPY]",
                "ERROR": "🚨 [SYNTAX ERROR]"
            }.get(status, status)

            q_text = res["entry"]["user_q"] if res["entry"] else "(invalid JSON)"
            print(f"Item #{res['index']}: {icon}")
            print(f"  Question: \"{q_text}\"")
            print(f"  Verdict:  {res['reason']}")
            if res.get("category"):
                print(f"  Topic:    {res['category']} -> {res['subtopic']}")
            print("-" * 70)
        return

    if args.check_new:
        if not os.path.exists(args.check_new):
            print(f"File not found: {args.check_new}")
            sys.exit(1)
        with open(args.check_new, "r", encoding="utf-8") as f:
            lines = f.readlines()
        results = auditor.check_new_entries(lines)
        print("\n" + "=" * 70)
        print(f"🔍 BATCH VERIFICATION RESULTS: {args.check_new}")
        print("=" * 70)
        for res in results:
            status = res["status"]
            icon = {
                "NOVEL": "✅ [FRESH TOPIC]",
                "DISTINCT_VARIANT": "💡 [VALID VARIANT]",
                "NEAR_DUPLICATE": "⚠️ [NEAR DUPLICATE]",
                "EXACT_REDUNDANT": "❌ [REDUNDANT COPY]",
                "ERROR": "🚨 [SYNTAX ERROR]"
            }.get(status, status)
            q_text = res["entry"]["user_q"] if res["entry"] else "(invalid JSON)"
            print(f"Item #{res['index']}: {icon} | \"{q_text}\"")
            print(f"  -> {res['reason']}")
        return

    if args.clean_export:
        exact_dups, _ = auditor.find_duplicates()
        seen_fingerprints = set()
        kept_entries = []
        dropped_count = 0

        for entry in auditor.entries:
            # Fingerprint based on normalized question + normalized response
            fp = (entry["norm_q"], normalize_text(entry["first_assistant"]))
            if fp in seen_fingerprints:
                dropped_count += 1
            else:
                seen_fingerprints.add(fp)
                kept_entries.append(entry["raw"])

        with open(args.clean_export, "w", encoding="utf-8") as f:
            for item in kept_entries:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"✨ Cleaned dataset written to {args.clean_export}")
        print(f"   Original count: {len(auditor.entries)}")
        print(f"   Dropped redundant identical copies: {dropped_count}")
        print(f"   Retained entries (including distinct pedagogical variants): {len(kept_entries)}")
        return

    if args.duplicates:
        exact_dups, near_dups = auditor.find_duplicates()
        print("\n" + "=" * 80)
        print("🔍 DETAILED DUPLICATE REPORT")
        print("=" * 80)
        for norm_q, entries in exact_dups.items():
            analysis = auditor.analyze_variants(entries)
            print(f"\nQuestion: \"{entries[0]['user_q']}\" (Appeared {len(entries)} times)")
            print(f"  Category: {entries[0]['major_category']} -> {entries[0]['subtopic']}")
            print(f"  Distinct Variants: {analysis['distinct_variants']} | Identical Redundancies: {analysis['identical_redundant']}")
            print(f"  Line numbers: {[e['line_no'] for e in entries]}")
            for v_idx, variant_group in enumerate(analysis["variants_detail"], start=1):
                first_e = variant_group[0]
                plan_g = first_e["plan_goals"][0] if first_e["plan_goals"] else "No plan tag"
                print(f"    [Variant {v_idx}] ({len(variant_group)} copies) Plan Goal: \"{plan_g}\"")
                print(f"       Response Preview: {first_e['first_assistant'][:110]}...")

        if near_dups:
            print("\n" + "-" * 80)
            print("🔍 NEAR-DUPLICATE QUESTION PAIRS (Similarity >= 85%)")
            print("-" * 80)
            for pair in near_dups:
                print(f"  • {pair['similarity']}% similar:")
                print(f"     Q1 (Line {pair['entries_q1'][0]['line_no']}): \"{pair['q1']}\"")
                print(f"     Q2 (Line {pair['entries_q2'][0]['line_no']}): \"{pair['q2']}\"")
        return

    if args.missing:
        missing, sparse = auditor.find_missing_or_sparse_topics(sparse_threshold=4)
        print("\n" + "=" * 80)
        print("🎯 COMPLETE GRADE 10 CURRICULUM GAP ANALYSIS")
        print("=" * 80)
        print("1. Uncovered Topics (0 entries in dataset):")
        for m in missing:
            print(f"   [ ] {m['category']:<22} | {m['subtopic']}")
        print("\n2. Sparse Topics (< 4 entries in dataset):")
        for s in sparse:
            print(f"   [!] {s['category']:<22} | {s['subtopic']:<35} ({s['count']} entries)")
        return

    # Default action: Full summary
    auditor.print_summary_report()


if __name__ == "__main__":
    run_cli()
