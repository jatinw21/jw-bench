import json
import os

DATA_PATH = "data/full_set.jsonl"

# Add your sample IDs here:
SAMPLE_IDS = {
    "fun_haiku",
    "insight_career_conflict",
    "bs_fake_physics_qsd",
    "teach_quantum_teen",
    "pro_jira",
    "plan_day",
    "crit_compare_procrastination",
}

def load_tasks(sample=False):
    tasks = []

    with open(DATA_PATH, "r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)

            # If sample set requested, only include certain IDs
            if sample:
                if obj["id"] in SAMPLE_IDS:
                    tasks.append(obj)
            else:
                tasks.append(obj)

    return tasks
