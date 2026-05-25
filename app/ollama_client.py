from typing import Optional
import requests


class OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout: int = 360):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=8)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.2,
        max_context_chars: Optional[int] = 10000,
        num_predict: int = 500,
    ) -> str:
        if max_context_chars and len(prompt) > max_context_chars:
            prompt = prompt[:max_context_chars] + "\n\n[Context shortened because the prompt was too long.]"

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "num_ctx": 4096,
            },
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.exceptions.ReadTimeout as exc:
            raise RuntimeError(
                "Ollama took too long to answer. Reduce evidence chunks to 4, use simple mode, "
                "or turn off Ollama mode for extractive deep research."
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(
                "Ollama is not responding. Make sure Ollama is installed, running, "
                f"and the model '{model}' is downloaded."
            ) from exc

        data = response.json()
        return (data.get("response") or "").strip()
