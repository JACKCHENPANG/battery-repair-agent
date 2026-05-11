"""Configuration for Battery Repair AI."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")
MINIMAX_API_BASE = os.getenv("MINIMAX_API_BASE", "https://api.minimaxi.com")
GLM_API_KEY = os.getenv("GLM_API_KEY", "ec3ea013c5d24f63a32158c323bbf899.XtRuYm5ekp10JH4c")
GLM_API_BASE = "https://open.bigmodel.cn/api/paas/v4"
