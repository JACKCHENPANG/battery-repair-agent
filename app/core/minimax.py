"""MiniMax & GLM API client for vision and chat."""
import httpx
import base64
import os

# GLM API (vision)
GLM_API_KEY = os.getenv("GLM_API_KEY", "ec3ea013c5d24f63a32158c323bbf899.XtRuYm5ekp10JH4c")
GLM_API_BASE = "https://open.bigmodel.cn/api/paas/v4"

# MiniMax API (chat only)
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")
MINIMAX_API_BASE = os.getenv("MINIMAX_API_BASE", "https://api.minimaxi.com")


class MiniMaxClient:
    """MiniMax Chat + GLM Vision."""

    async def analyze_image(self, image_data: str, prompt: str = "请识别这张 PCB 照片中的所有电子元件（电阻、电容、IC、MOSFET、二极管等），并标注它们的大致位置和编号。") -> str:
        """Analyze PCB image using GLM-4V-Flash."""
        async with httpx.AsyncClient(timeout=90.0) as client:
            if image_data.startswith("data:"):
                image_data = image_data.split(",", 1)[1]
            
            payload = {
                "model": "glm-4v-flash",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ],
                "max_tokens": 800
            }
            resp = await client.post(
                f"{GLM_API_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {GLM_API_KEY}"},
                json=payload
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    async def chat(self, messages: list, model: str = "MiniMax-M2.5") -> str:
        """Chat using MiniMax."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {"model": model, "messages": messages, "max_tokens": 600}
            resp = await client.post(
                f"{MINIMAX_API_BASE}/v1/chat/completions",
                headers={"Authorization": f"Bearer {MINIMAX_API_KEY}"},
                json=payload
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]


_client = None

def get_client():
    global _client
    if _client is None:
        _client = MiniMaxClient()
    return _client
