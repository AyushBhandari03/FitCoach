"""
Step 2: Merge Data Sources + Clean + Split
Combines two sources into the final training pool:
  1. data/raw_dataset.jsonl    - freshly generated via Groq (Step 1, 01_generate_dataset.py)
  2. data/colab_dataset.jsonl  - old Dataset from various sources

Both use the same {"instruction": ..., "response": ...} format. We merge, dedupe (case-insensitive
on the instruction text), filter out junk/too-short examples, truncate extreme outlier-length
responses, then do a 90/10 train/val split.
"""
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SOURCES = [
    ("groq_generated", DATA_DIR / "raw_dataset.jsonl"),
    ("colab_dataset", DATA_DIR / "colab_dataset.jsonl"),
]
TRAIN_PATH = DATA_DIR / "train.jsonl"
VAL_PATH = DATA_DIR / "val.jsonl"

MIN_INSTRUCTION_LEN = 8
MIN_RESPONSE_LEN = 15
MAX_RESPONSE_LEN = 1200
VAL_FRACTION = 0.1
SEED = 42


def load_source(path: Path):
    examples = []
    if not path.exists():
        return examples
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            instr = obj.get("instruction", "")
            resp = obj.get("response", "")
            if not isinstance(instr, str) or not isinstance(resp, str):
                continue  
            instr, resp = instr.strip(), resp.strip()
            if len(instr) < MIN_INSTRUCTION_LEN or len(resp) < MIN_RESPONSE_LEN:
                continue
            if len(resp) > MAX_RESPONSE_LEN:
                resp = resp[:MAX_RESPONSE_LEN].rsplit(".", 1)[0] + "."
            examples.append({"instruction": instr, "response": resp})
    return examples


def main():
    found_any = False
    seen = set()
    merged = []
    per_source_counts = {}

    for name, path in SOURCES:
        loaded = load_source(path)
        if not loaded and not path.exists():
            print(f"  note: {path} not found, skipping ({name})")
            continue
        found_any = True
        kept = 0
        for ex in loaded:
            key = ex["instruction"].lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(ex)
            kept += 1
        per_source_counts[name] = {"raw": len(loaded), "kept_after_dedupe": kept}

    if not found_any:
        raise SystemExit(
            "No source datasets found. Run 01_generate_dataset.py first, and make sure "
            "data/colab_dataset.jsonl is present."
        )

    print("Source breakdown:")
    for name, stats in per_source_counts.items():
        print(f"  {name}: {stats['raw']} raw -> {stats['kept_after_dedupe']} kept "
              f"(after cross-source dedupe)")

    random.seed(SEED)
    random.shuffle(merged)

    n_val = max(1, int(len(merged) * VAL_FRACTION))
    val_examples = merged[:n_val]
    train_examples = merged[n_val:]

    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")

    with open(VAL_PATH, "w", encoding="utf-8") as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nTotal merged & deduped examples: {len(merged)}")
    print(f"  train: {len(train_examples)} -> {TRAIN_PATH}")
    print(f"  val:   {len(val_examples)} -> {VAL_PATH}")


if __name__ == "__main__":
    main()
