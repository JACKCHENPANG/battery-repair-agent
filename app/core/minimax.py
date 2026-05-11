"""MiniMax & GLM API client for vision and chat."""
import httpx
import os

GLM_API_KEY = os.getenv("GLM_API_KEY", "ec3ea013c5d24f63a32158c323bbf899.XtRuYm5ekp10JH4c")
GLM_API_BASE = "https://open.bigmodel.cn/api/paas/v4"
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")
MINIMAX_API_BASE = os.getenv("MINIMAX_API_BASE", "https://api.minimaxi.com")


class MiniMaxClient:
    def __init__(self, api_key: str = "", base_url: str = MINIMAX_API_BASE):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    async def analyze_image(self, image_data: str, mime: str = "image/jpeg", prompt: str = "请识别这张 PCB 照片中的所有电子元件。") -> str:
        """Analyze PCB image using GLM-4V-Flash."""
        if image_data.startswith("data:"):
            parts = image_data.split(",", 1)
            if len(parts) == 2:
                mime_part = parts[0].split(";")[0].replace("data:", "")
                if mime_part:
                    mime = mime_part
                image_data = parts[1]

        payload = {
            "model": "glm-4v-flash",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_data}"}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            "max_tokens": 1200
        }
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(
                f"{GLM_API_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {GLM_API_KEY}"},
                json=payload
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    async def chat(self, messages: list, model: str = "MiniMax-M1") -> str:
        """Text chat — uses MiniMax if key is available, otherwise falls back to GLM."""
        if self.api_key:
            async with httpx.AsyncClient(timeout=60.0) as client:
                payload = {"model": model, "messages": messages, "max_tokens": 1200}
                resp = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
        else:
            async with httpx.AsyncClient(timeout=60.0) as client:
                payload = {"model": "glm-4-flash", "messages": messages, "max_tokens": 1200}
                resp = await client.post(
                    f"{GLM_API_BASE}/chat/completions",
                    headers={"Authorization": f"Bearer {GLM_API_KEY}"},
                    json=payload
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]


_client: MiniMaxClient | None = None


def get_client() -> MiniMaxClient:
    global _client
    if _client is None:
        _client = MiniMaxClient(api_key=MINIMAX_API_KEY, base_url=MINIMAX_API_BASE)
    return _client
