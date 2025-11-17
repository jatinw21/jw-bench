import os, time, requests
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

class OpenRouterClient:
    def __init__(self, model: str, api_key: str | None = None, timeout: float = 60):
        self.model = model
        self.key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not found. Create a .env file in the repo root "
                "with OPENROUTER_API_KEY=... (see .env.example)."
            )
        self.timeout = timeout

    def complete(self, messages, **params):
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            # These two are optional but recommended by OpenRouter:
            "HTTP-Referer": "https://localhost",  # or your repo URL
            "X-Title": "jw-bench",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": params.get("temperature", 0.0),
            "max_tokens": params.get("max_tokens", 64),
        }
        t0 = time.time()
        r = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=self.timeout)
        # --- Better error surfacing ---
        if r.status_code >= 400:
            try:
                detail = r.json()
            except Exception:
                detail = {"text": r.text[:500]}
            raise requests.HTTPError(
                f"{r.status_code} error from OpenRouter for model='{self.model}': {detail}",
                response=r,
            )
        j = r.json()
        latency_s = time.time() - t0
        text = j["choices"][0]["message"]["content"]
        usage = j.get("usage", {})
        return {"text": text, "usage": usage, "latency_s": latency_s}
