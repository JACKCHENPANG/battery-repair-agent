"""Repair session service."""
import json
import re
from app.core.minimax import get_client


class RepairSession:
    """Manages a repair session conversation."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.components: list = []
        self.positions: list = []
        self.raw_text: str = ""
        self.symptom: str = ""
        self.causes: list = []
        self.measurement_history: list = []

    async def analyze_components(self, image_data: str, mime: str = "image/jpeg") -> dict:
        """Analyze PCB image and extract components."""
        client = get_client()
        prompt = (
            "你是专业的电池维修技师。请仔细识别这张 PCB/电路板照片中的所有电子元件。\n"
            "特别注意识别：电阻器、电容器、IC芯片、MOSFET/三极管、二极管、LED指示灯、连接器/接口、晶振、保险丝等。\n"
            "请以 JSON 格式返回，不要包含其他文字：\n"
            '{"components": [{"name": "元件名称/型号", "type": "元件类型", "value": "参数值", "notes": "备注"}], '
            '"positions": [{"name": "元件名称", "location": "板面位置", "notes": "备注"}]}'
        )
        raw = await client.analyze_image(image_data, mime, prompt)

        data = _extract_json(raw)
        self.components = data.get("components", [])
        self.positions = data.get("positions", [])
        self.raw_text = raw
        return {"components": self.components, "positions": self.positions, "raw_text": raw}

    async def reason_causes(self, symptom: str) -> list:
        """Reason about possible fault causes."""
        self.symptom = symptom
        client = get_client()

        comp_summary = json.dumps(self.components[:10], ensure_ascii=False) if self.components else "未知"
        prompt = (
            f"你是电池检测设备维修专家。\n"
            f"已识别的 PCB 元件（部分）：{comp_summary}\n"
            f"用户描述故障：{symptom}\n\n"
            "请推理可能的故障原因，返回 JSON 格式，不要包含其他文字：\n"
            '{"causes": [{"cause": "原因描述", "probability": 0.85, "fix": "修复建议"}], '
            '"measurements": [{"point": "测量点", "expected": "正常值"}]}'
        )

        raw = await client.chat([{"role": "user", "content": prompt}])

        data = _extract_json(raw)
        causes = data.get("causes", [])
        if not causes:
            causes = [{"cause": "需要进一步检测才能确定原因", "probability": 0.5, "fix": "建议联系技术支持"}]
        self.causes = causes
        return causes

    async def get_guidance(self, step: int, measurement_result: str) -> dict:
        """Get next guidance based on measurement result."""
        self.measurement_history.append({"step": step, "result": measurement_result})
        client = get_client()

        prompt = (
            f"症状：{self.symptom}\n"
            f"已完成测量：{json.dumps(self.measurement_history, ensure_ascii=False)}\n"
            f"用户刚完成测量：步骤{step}，结果：{measurement_result}\n\n"
            "请提供下一步测量指导，返回 JSON 格式，不要包含其他文字：\n"
            '{"step": 3, "instruction": "下一步操作", "measurement_point": "测量点", "expected_value": "期望值"}'
        )

        raw = await client.chat([{"role": "user", "content": prompt}])
        data = _extract_json(raw)
        if not data:
            data = {
                "step": step + 1,
                "instruction": "根据测量结果判断是否需要进一步检查",
                "measurement_point": "",
                "expected_value": ""
            }
        return data

    async def generate_report(self) -> str:
        """Generate a repair report in Markdown."""
        client = get_client()

        prompt = (
            "请生成一份专业的电池检测设备维修报告（Markdown 格式）。\n\n"
            f"**设备型号**：JCY系列电池检测设备\n"
            f"**故障现象**：{self.symptom or '待描述'}\n"
            f"**已识别元件**：{json.dumps(self.components, ensure_ascii=False)}\n"
            f"**可能原因**：{json.dumps(self.causes, ensure_ascii=False)}\n"
            f"**测量历史**：{json.dumps(self.measurement_history, ensure_ascii=False)}\n\n"
            "报告需包含以下章节：\n"
            "## 1. 设备信息\n## 2. 故障描述\n## 3. 诊断过程\n## 4. 故障原因分析\n## 5. 维修建议\n## 6. 预防措施"
        )

        return await client.chat([{"role": "user", "content": prompt}])


def _extract_json(text: str) -> dict:
    """Extract first JSON object from text, return empty dict on failure."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return {}
