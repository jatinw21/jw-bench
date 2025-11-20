from client_openrouter import OpenRouterClient
from task_loader import load_tasks

MODELS = [
    "openai/gpt-4o-mini",
    "anthropic/claude-3.5-sonnet",
]

def main():
    tasks = load_tasks(sample=True)

    for model in MODELS:
        client = OpenRouterClient(model)
        print(f"\n=== Model: {model} ===")

        for task in tasks:
            print(f"\n--- {task['id']} ---")
            out = client.complete([{"role": "user", "content": task["prompt"]}],
                                  temperature=0.7,
                                  max_tokens=600)
            print(out["text"])

if __name__ == "__main__":
    main()
