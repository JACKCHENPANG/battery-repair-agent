"""Repair session service."""
import json
import re
from app.core.minimax import get_client


class RepairSession:
    """Manages a repair session conversation."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.components = []
        self.positions = []
        self.raw_text = ""
        self.symptom = ""
        self.causes = []
        self.measurement_history = []
        self.finished = False

    async def analyze_components(self, image_data: str) -> dict:
        """Analyze PCB image and extract components."""
        client = get_client()
        prompt = (
            "你是专业的电池维修技师。请仔细识别这张 PCB/电路板照片中的所有电子元件。\n"
            "特别注意识别：电阻器、电容器、IC芯片、MOSFET/三极管、二极管、LED指示灯、连接器/接口、晶振、保险丝等。\n"
            "请以 JSON 格式返回：\n"
            '{"components": [{"name": "元件名称/型号", "type": "元件类型", "value": "参数值", "notes": "备注"}], '
            '"positions": [{"name": "元件名称", "location": "板面位置", "notes": "备注"}]}'
        )
        raw = await client.analyze_image(image_data, prompt)
        
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            try:
                match = re.search(r'\{[\s\S]*\}', raw)
                if match:
                    data = json.loads(match.group())
                else:
                    data = {"components": [], "positions": []}
            except Exception:
                data = {"components": [], "positions": []}
        
        self.components = data.get("components", [])
        self.positions = data.get("positions", [])
        self.raw_text = raw
        return {"components": self.components, "positions": self.positions, "raw_text": raw}

    async def reason_causes(self, symptom: str) -> list:
        """Reason about possible fault causes."""
        self.symptom = symptom
        client = get_client()
        
        prompt = (
            f"你是电池检测设备维修专家。用户描述故障：{symptom}\n\n"
            "请推理可能原因，返回 JSON 格式：\n"
            '{"causes": [{"cause": "原因描述", "probability": 0.85, "fix": "修复建议"}], '
            '"measurements": [{"point": "测量点", "expected": "正常值"}]}'
        )
        
        raw = await client.chat([{"role": "user", "content": prompt}])
        
        causes = []
        try:
            data = json.loads(raw)
            causes = data.get("causes", [])
        except json.JSONDecodeError:
            try:
                match = re.search(r'\{[\s\S]*\}', raw)
                if match:
                    causes = json.loads(match.group()).get("causes", [])
            except Exception:
                pass
            if not causes:
                causes = [{"cause": "需要进一步检测才能确定原因", "probability": 0.5, "fix": "建议联系技术支持"}]
        
        self.causes = causes
        return causes

    async def get_guidance(self, step: int, measurement_result: str) -> dict:
        """Get next guidance based on measurement."""
        self.measurement_history.append({"step": step, "result": measurement_result})
        client = get_client()
        
        prompt = (
            f"症状：{self.symptom}\n"
            f"已完成测量：{json.dumps(self.measurement_history, ensure_ascii=False)}\n"
            f"用户刚完成测量：步骤{step}，结果：{measurement_result}\n\n"
            "请提供下一步测量指导，返回 JSON 格式：\n"
            '{"step": 3, "instruction": "下一步操作", "measurement_point": "测量点", "expected_value": "期望值"}'
        )
        
        raw = await client.chat([{"role": "user", "content": prompt}])
        
        try:
            data = json.loads(raw)
            return data
        except Exception:
            return {
                "step": step + 1,
                "instruction": "根据测量结果判断是否需要进一步检查",
                "measurement_point": "",
                "expected_value": ""
            }

    async def generate_report(self) -> str:
        """Generate a repair report."""
        client = get_client()
        
        prompt = (
            f"生成维修报告。设备型号：JCY系列电池检测设备\n"
            f"故障现象：{self.symptom}\n"
            f"已识别元件：{json.dumps(self.components, ensure_ascii=False)}\n"
            f"可能原因：{json.dumps(self.causes, ensure_ascii=False)}\n"
            f"测量历史：{json.dumps(self.measurement_history, ensure_ascii=False)}\n\n"
            "请生成完整维修报告，用 Markdown 格式，包含：\n"
            "- 设备信息\n- 故障描述\n- 诊断过程\n- 可能原因\n- 维修建议\n- 预防措施"
        )
        
        raw = await client.chat([{"role": "user", "content": prompt}])
        return raw
