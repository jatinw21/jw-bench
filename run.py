from client_openrouter import OpenRouterClient
from task_loader import load_tasks
import os


MODELS = [
    "openai/gpt-4o-mini",
    "anthropic/claude-3.5-sonnet",
]

def save_response(model_name, task_id, text, base_dir="outputs"):
    model_dir = os.path.join(base_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    file_path = os.path.join(model_dir, f"{task_id}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)


def main(sample=False, skip_existing=True):
    tasks = load_tasks(sample=sample)

    for model in MODELS:
        client = OpenRouterClient(model)

        print(f"\n=== Running model: {model} ===")

        for task in tasks:
            task_path = os.path.join("outputs", model, f"{task['id']}.txt")

            # --- Skip Existing Logic ---
            if skip_existing and os.path.exists(task_path):
                print(f" -> {task['id']} ... skipped (already exists)")
                continue
            # ----------------------------

            print(f" -> {task['id']} ...")

            out = client.complete(
                [{"role": "user", "content": task["prompt"]}],
                temperature=0.7,
                max_tokens=800,
            )

            text = out["text"]

            save_response(model, task["id"], text)

            print(f"Saved {task['id']} to {model}...")

if __name__ == "__main__":
    main()