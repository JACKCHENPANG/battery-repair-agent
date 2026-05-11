"""FastAPI entry point for Battery Repair AI."""
import uuid
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import aiofiles

from app.config import HOST, PORT, UPLOAD_DIR
from app.api.websocket import websocket_endpoint

app = FastAPI(title="BatteryRepairAI", version="0.1.0")

# Mount static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# History store
_history = []


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main HTML page."""
    html_path = Path(__file__).parent / "templates" / "index.html"
    return FileResponse(str(html_path))


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/history")
async def get_history():
    """Get repair session history."""
    return {"history": list(reversed(_history[-50:]))}


@app.post("/api/history")
async def add_history(item: dict):
    """Add a history record."""
    item["timestamp"] = datetime.now().isoformat()
    _history.append(item)
    return {"ok": True}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Backup file upload endpoint."""
    upload_path = Path(UPLOAD_DIR) / file.filename
    async with aiofiles.open(upload_path, "wb") as f:
        content = await file.read()
        await f.write(content)
    return {"filename": file.filename, "path": str(upload_path)}


@app.websocket("/ws/{session_id}")
async def ws(websocket: WebSocket, session_id: str):
    """WebSocket endpoint — delegates to websocket module."""
    await websocket.accept()
    actual_id = session_id if session_id else str(uuid.uuid4())[:8]
    await websocket.send_json({"type": "connected", "session_id": actual_id})
    websocket.session_id = actual_id
    await websocket_endpoint(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=True)
