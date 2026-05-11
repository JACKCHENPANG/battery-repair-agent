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
            "特别注意识别以下类型：\n"
            "1. 电阻器 (Resistor) - 标注阻值（如 10Ω, 1kΩ, 10kΩ）和功率\n"
            "2. 电容器 (Capacitor) - 标注容值（如 10uF, 100nF）和耐压\n"
            "3. 电感器 (Inductor)\n"
            "4. IC 芯片 - 标注芯片型号/丝印（非常重要！）\n"
            "5. MOSFET / 三极管 - 标注型号（非常重要！）\n"
            "6. 二极管 / 稳压管\n"
            "7. LED 指示灯\n"
            "8. 连接器 / 接口（USB, JST, XT30/60 等）\n"
            "9. 晶振 (Crystal)\n"
            "10. 保险丝 / 自恢复保险 (Fuse / PTC)\n"
            "\n请以 JSON 格式返回，components 包含识别到的所有元件（含推测型号），positions 描述各元件在板上的大致位置：\n"
            '{"components": [{"name": "元件名称/型号", "type": "元件类型", "value": "参数值（尽量详细）", "notes": "备注或风险提示"}], '
            '"positions": [{"name": "元件名称/型号", "location": "板面位置（如左上角、中央偏右）", "notes": "备注"}]}'
        )
        raw = await client.analyze_image(image_data, prompt)
        
        # Try to extract JSON from response
        try:
            # Try direct JSON parse first
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try to find JSON in text (model may include thinking tags)
            try:
                json_match = re.search(r'\{[\s\S]*\}', raw)
                if json_match:
                    data = json.loads(json_match.group())
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
        
        messages = [{"role": "user", "content": prompt}]
        
        # Use GLM for cleaner JSON output
        raw = await client.analyze_image("", f"推理故障原因：{symptom}")
        
        # Try to find JSON array in response
        causes = []
        try:
            data = json.loads(raw)
            causes = data.get("causes", [])
        except json.JSONDecodeError:
            # Try extracting JSON from thinking text
            try:
                match = re.search(r'\[.*\]', raw, re.DOTALL)
                if match:
                    causes = json.loads(match.group())
            except Exception:
                causes = [{"cause": "请提供更多信息以便诊断", "probability": 0.5, "fix": "建议联系技术支持"}]
        
        self.causes = causes
        return causes

    async def get_guidance(self, step: int, measurement_result: str) -> dict:
        """Get next guidance based on measurement."""
        self.measurement_history.append({"step": step, "result": measurement_result})
        client = get_client()
        
        context = f"症状：{self.symptom}\n已完成测量：{self.measurement_history}"
        prompt = (
            f"基于以下信息，提供下一步测量指导：\n{context}\n"
            f"用户刚完成测量：步骤{step}，结果：{measurement_result}\n"
            "返回 JSON 格式：\n"
            '{"step": 3, "instruction": "下一步操作", "measurement_point": "测量点", "expected_value": "期望值"}'
        )
        
        messages = [{"role": "user", "content": prompt}]
        # For guidance, use text-only (no image)
        raw = await client.chat(messages, model="MiniMax-M2.5")
        
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
            "请生成完整维修报告，包含：设备信息、诊断过程、维修结果、建议。"
        )
        
        messages = [{"role": "user", "content": prompt}]
        raw = await client.chat(messages, model="MiniMax-M2.5")
        return raw
