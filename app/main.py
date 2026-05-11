"""FastAPI entry point for Battery Repair AI."""
import json
import uuid
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import aiofiles

from app.config import HOST, PORT, UPLOAD_DIR
from app.api.websocket import websocket_endpoint

app = FastAPI(title="BatteryRepairAI", version="0.2.0")

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

HISTORY_FILE = Path(__file__).parent.parent / "history.json"


def _load_history() -> list:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_history(records: list) -> None:
    try:
        HISTORY_FILE.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "templates" / "index.html"
    return FileResponse(str(html_path))


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/history")
async def get_history():
    records = _load_history()
    return {"history": list(reversed(records[-100:]))}


@app.post("/api/history")
async def add_history(item: dict):
    if item.get("clear"):
        _save_history([])
        return {"ok": True}
    records = _load_history()
    item["timestamp"] = datetime.now().isoformat()
    records.append(item)
    if len(records) > 100:
        records = records[-100:]
    _save_history(records)
    return {"ok": True}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    upload_path = Path(UPLOAD_DIR) / file.filename
    async with aiofiles.open(upload_path, "wb") as f:
        content = await file.read()
        await f.write(content)
    return {"filename": file.filename, "path": str(upload_path)}


@app.websocket("/ws/{session_id}")
async def ws(websocket: WebSocket, session_id: str):
    await websocket.accept()
    actual_id = session_id if session_id else str(uuid.uuid4())[:8]
    await websocket.send_json({"type": "connected", "session_id": actual_id})
    await websocket_endpoint(websocket, actual_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=True)
