import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TASK_FILE = BASE_DIR / "data/full_set.jsonl"
OUTPUT_DIR = BASE_DIR / "outputs"

def load_task_ids():
    ids = []
    with open(TASK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                ids.append(obj["id"])
    return ids

def verify_outputs():
    task_ids = load_task_ids()
    model_dirs = [d for d in OUTPUT_DIR.glob("*") if d.is_dir()]

    print("\n=== MODELS FOUND ===")
    for m in model_dirs:
        print("  -", m.name)

    missing = []

    print("\n=== VERIFYING OUTPUTS ===")
    for task_id in task_ids:
        for model_dir in model_dirs:
            expected = model_dir / f"{task_id}.txt"
            if not expected.exists():
                missing.append((model_dir.name, task_id))

    if not missing:
        print("\n✅ ALL OUTPUT FILES MATCH ALL TASK IDS. PERFECT!\n")
    else:
        print("\n❌ MISSING OUTPUTS FOUND:\n")
        for model, task_id in missing:
            print(f"  - Model '{model}' missing file: {task_id}.txt")
        print(f"\nTotal missing: {len(missing)}\n")

if __name__ == "__main__":
    verify_outputs()
