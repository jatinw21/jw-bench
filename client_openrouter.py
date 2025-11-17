import os, time, requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

class OpenRouterClient:
    def __init__(self, model: str, api_key: str | None = None, timeout: float = 60):
        self.model = model
        self.key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.key:
            raise RuntimeError("Set OPENROUTER_API_KEY")
        self.timeout = timeout

    def complete(self, messages, **params):
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": params.get("temperature", 0.0),
            "max_tokens": params.get("max_tokens", 64),
        }
        t0 = time.time()
        r = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        j = r.json()
        latency_s = time.time() - t0
        text = j["choices"][0]["message"]["content"]
        usage = j.get("usage", {})  # may include prompt_tokens, completion_tokens, total_tokens
        return {"text": text, "usage": usage, "latency_s": latency_s}
