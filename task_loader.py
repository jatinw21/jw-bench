import json

def load_tasks(path="data/sample_set.jsonl"):
    tasks = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    return tasks
