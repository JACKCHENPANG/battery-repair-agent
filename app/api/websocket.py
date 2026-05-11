"""WebSocket endpoint for real-time repair session."""
import json
import uuid
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect
from app.models.schemas import ImageAck, AnalysisResult, ReasoningResult, GuidanceResult, ErrorResult, StatusMessage
from app.services.repair import RepairSession

_sessions: dict[str, RepairSession] = {}


def get_session(session_id: str) -> RepairSession:
    if session_id not in _sessions:
        _sessions[session_id] = RepairSession(session_id)
    return _sessions[session_id]


def parse_client_message(raw_data: str) -> dict:
    try:
        data = json.loads(raw_data)
        msg_type = data.get("type")
        if msg_type == "image":
            return {"parsed": True, "type": "image", "data": data.get("data", ""),
                    "image_id": data.get("image_id", ""), "filename": data.get("filename", ""),
                    "mime": data.get("mime", "image/jpeg")}
        elif msg_type == "symptom":
            return {"parsed": True, "type": "symptom", "symptom": data.get("symptom", "")}
        elif msg_type == "measure":
            return {"parsed": True, "type": "measure", "step": data.get("step", 1), "result": data.get("result", "")}
        elif msg_type == "text":
            return {"parsed": True, "type": "text", "content": data.get("content", "")}
        else:
            return {"parsed": False, "error": f"Unknown message type: {msg_type}"}
    except Exception as e:
        return {"parsed": False, "error": str(e)}


async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """Handle a WebSocket repair session. Caller must already have accepted the socket."""
    session = get_session(session_id)

    try:
        while True:
            raw = await websocket.receive_text()
            parsed = parse_client_message(raw)

            if not parsed.get("parsed"):
                await websocket.send_json(ErrorResult(message=parsed.get("error", "解析失败")).model_dump())
                continue

            msg_type = parsed["type"]

            if msg_type == "image":
                await websocket.send_json(StatusMessage(message="正在分析 PCB 照片，请稍候...").model_dump())
                try:
                    result = await session.analyze_components(parsed["data"], parsed.get("mime", "image/jpeg"))
                    comp_count = len(result["components"])
                    await websocket.send_json(
                        AnalysisResult(
                            components=result["components"],
                            positions=result["positions"],
                            raw_text=result["raw_text"]
                        ).model_dump()
                    )
                    await websocket.send_json(ImageAck(image_id=parsed["image_id"]).model_dump())
                    summary = f"识别完成，共发现 **{comp_count}** 个元件。\n\n请描述您观察到的故障现象（如：充电时发热、无法开机、电量显示异常等）。"
                    await websocket.send_json({"type": "chat", "content": summary})
                except Exception as e:
                    await websocket.send_json(ErrorResult(message=f"图片分析失败: {e}").model_dump())

            elif msg_type == "symptom":
                await websocket.send_json(StatusMessage(message="正在推理故障原因...").model_dump())
                try:
                    causes = await session.reason_causes(parsed["symptom"])
                    await websocket.send_json(ReasoningResult(causes=causes).model_dump())
                except Exception as e:
                    await websocket.send_json(ErrorResult(message=f"推理失败: {e}").model_dump())

            elif msg_type == "measure":
                await websocket.send_json(StatusMessage(message="正在分析测量结果...").model_dump())
                try:
                    guidance = await session.get_guidance(parsed["step"], parsed["result"])
                    await websocket.send_json(
                        GuidanceResult(
                            step=guidance["step"],
                            instruction=guidance["instruction"],
                            measurement_point=guidance.get("measurement_point", ""),
                            expected_value=guidance.get("expected_value", "")
                        ).model_dump()
                    )
                except Exception as e:
                    await websocket.send_json(ErrorResult(message=f"指导生成失败: {e}").model_dump())

            elif msg_type == "text":
                content = parsed.get("content", "")
                if "报告" in content or "report" in content.lower():
                    await websocket.send_json(StatusMessage(message="正在生成维修报告...").model_dump())
                    try:
                        report = await session.generate_report()
                        await websocket.send_json({"type": "report", "content": report})
                    except Exception as e:
                        await websocket.send_json(ErrorResult(message=f"报告生成失败: {e}").model_dump())
                else:
                    await websocket.send_json(StatusMessage(message="请描述故障现象或点击「生成报告」按钮").model_dump())

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
