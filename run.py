import json, re, pandas as pd
from pathlib import Path
from client_openrouter import OpenRouterClient

MODELS = [
    "openrouter/anthropic/claude-3.5-sonnet",
    "openrouter/openai/gpt-4o-mini",
]

def format_messages(q):
    prompt = (
        "You are a helpful assistant. Answer with only the option letter (A, B, C, or D).\n\n"
        f"Question: {q['question']}\nChoices:\n" + "\n".join(q["choices"]) + "\nAnswer:"
    )
    return [{"role": "user", "content": prompt}]

def normalize_letter(text: str) -> str:
    m = re.search(r"\b([ABCD])\b", text.strip(), re.IGNORECASE)
    return m.group(1).upper() if m else "?"

def main():
    data = [json.loads(l) for l in Path("data/simple_mc.jsonl").read_text().splitlines()]
    rows = []
    for model in MODELS:
        client = OpenRouterClient(model)
        for q in data:
            out = client.complete(format_messages(q), temperature=0.0, max_tokens=2)
            pred_letter = normalize_letter(out["text"])
            correct = pred_letter == q["answer"]
            rows.append({
                "id": q["id"], "model": model, "gold": q["answer"],
                "pred": pred_letter, "correct": int(correct),
                "latency_s": round(out["latency_s"], 3),
                "total_tokens": out["usage"].get("total_tokens"),
            })
    df = pd.DataFrame(rows)
    df.to_csv("results.csv", index=False)
    summary = df.groupby("model")["correct"].mean().sort_values(ascending=False)
    print("\nAccuracy by model:")
    print((summary*100).round(1).astype(str) + "%")
    print("\nSaved per-question results → results.csv")

if __name__ == "__main__":
    main()
