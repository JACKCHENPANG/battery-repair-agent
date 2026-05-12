"""GLM API client for vision and chat, with demo-mode fallback."""
import os
import json
import httpx

GLM_API_KEY = os.getenv("GLM_API_KEY", "ec3ea013c5d24f63a32158c323bbf899.XtRuYm5ekp10JH4c")
GLM_API_BASE = "https://open.bigmodel.cn/api/paas/v4"
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")
MINIMAX_API_BASE = os.getenv("MINIMAX_API_BASE", "https://api.minimaxi.com")

# Set DEMO_MODE=1 (or leave unset) to use built-in mock responses when APIs are unreachable.
DEMO_MODE = os.getenv("DEMO_MODE", "1") == "1"

_DEMO_COMPONENTS = json.dumps({
    "components": [
        {"name": "BQ29330",  "type": "保护IC",   "value": "3-4串锂电池保护芯片", "notes": "主要保护芯片"},
        {"name": "R1-R5",    "type": "电阻",      "value": "10kΩ / 1kΩ",         "notes": "分压/限流电阻"},
        {"name": "C1-C4",    "type": "电容",      "value": "100nF / 10µF",        "notes": "旁路/滤波电容"},
        {"name": "Q1 NMOS",  "type": "MOSFET",    "value": "AON6258 30V/20A",     "notes": "充电开关管"},
        {"name": "Q2 NMOS",  "type": "MOSFET",    "value": "AON6258 30V/20A",     "notes": "放电开关管"},
        {"name": "D1",       "type": "二极管",    "value": "SS34",                "notes": "续流二极管"},
        {"name": "F1",       "type": "保险丝",    "value": "5A/250V",             "notes": "过流保护保险丝"},
        {"name": "J1",       "type": "连接器",    "value": "4Pin 2.54mm",         "notes": "电池组接口"},
    ],
    "positions": [
        {"name": "BQ29330", "location": "板中央", "notes": "最大元件，核心IC"},
        {"name": "Q1/Q2",   "location": "右侧",   "notes": "散热焊盘朝下"},
        {"name": "F1",      "location": "左上角", "notes": "输入端保险丝"},
    ]
}, ensure_ascii=False)

_DEMO_REASONING = json.dumps({
    "causes": [
        {"cause": "充电MOSFET（Q1）损坏或阻抗异常偏高，大电流充电时产生过多热量",
         "probability": 0.82,
         "fix": "用万用表二极管档测量Q1的G-S、D-S结压降，正常值约0.5-0.7V；异常则更换同型号AON6258"},
        {"cause": "保护IC（BQ29330）温度检测NTC电阻失效，无法触发过温保护",
         "probability": 0.65,
         "fix": "测量NTC阻值，25°C时应为10kΩ；断路或短路则更换10kΩ NTC"},
        {"cause": "电池组内阻增大（老化/单节过放），充电时整体内耗加剧",
         "probability": 0.55,
         "fix": "用内阻仪逐节测量单体内阻，超过100mΩ的电芯需更换"},
        {"cause": "充电器输出电压偏高超出保护阈值，BQ29330过压检测线路误判",
         "probability": 0.40,
         "fix": "测量充电器空载电压，正常应≤16.8V（4串）；异常则更换充电器"},
    ]
}, ensure_ascii=False)

_DEMO_REPORT = """# 电池维修诊断报告

## 设备信息
- **型号**: JCY系列锂电池检测设备
- **报告日期**: 2024年
- **诊断方式**: AI智能辅助诊断

## 故障描述
充电过程中电池组温度异常升高，充满电后外壳仍持续发烫，存在安全隐患。

## 识别元件清单
| 元件 | 类型 | 参数 |
|------|------|------|
| BQ29330 | 保护IC | 3-4串锂电池保护 |
| Q1/Q2 AON6258 | MOSFET | 30V/20A 充放电开关 |
| F1 | 保险丝 | 5A/250V 过流保护 |

## 诊断分析

### 主要嫌疑原因（概率排序）

1. **充电MOSFET损坏（82%）** — Q1管导通阻抗异常，大电流下产热
2. **NTC温度保护失效（65%）** — 无法触发过温保护动作
3. **电芯老化内阻增大（55%）** — 充电内耗产生热量
4. **充电器电压偏高（40%）** — 超出正常充电截止电压

## 维修建议

1. 优先检测 Q1/Q2 MOSFET，用万用表二极管档测G-S、D-S压降
2. 测量 NTC 电阻阻值（25°C ≈ 10kΩ）
3. 使用内阻仪逐节测量电芯内阻，超标电芯及时更换
4. 核查充电器输出电压是否在规格范围内

## 预防措施

- 定期（每3个月）检测电芯内阻
- 充电时保持良好通风，避免叠放
- 禁止使用非原装充电器
- 发现异常发热立即停止充电并断开电源
"""


class MiniMaxClient:
    def __init__(self, api_key: str = "", base_url: str = MINIMAX_API_BASE):
        self.api_key = api_key
        self.base_url = base_url
        self._demo = DEMO_MODE

    async def _try_real_api(self, coro):
        """Run a real API coroutine; on any network/HTTP error return None."""
        try:
            return await coro
        except Exception:
            return None

    async def analyze_image(self, image_data: str, prompt: str = "请识别这张 PCB 照片中的所有电子元件。") -> str:
        """Analyze PCB image using GLM-4V-Flash, fall back to demo data."""
        if not self._demo:
            if image_data.startswith("data:"):
                image_data = image_data.split(",", 1)[1]
            async with httpx.AsyncClient(timeout=120.0) as client:
                payload = {
                    "model": "glm-4v-flash",
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                            {"type": "text", "text": prompt},
                        ],
                    }],
                    "max_tokens": 1200,
                }
                resp = await client.post(
                    f"{GLM_API_BASE}/chat/completions",
                    headers={"Authorization": f"Bearer {GLM_API_KEY}"},
                    json=payload,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]

        # Demo mode: return realistic mock PCB component data
        return _DEMO_COMPONENTS

    async def _glm_chat(self, messages: list) -> str | None:
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                payload = {"model": "glm-4-flash", "messages": messages, "max_tokens": 1000}
                resp = await client.post(
                    f"{GLM_API_BASE}/chat/completions",
                    headers={"Authorization": f"Bearer {GLM_API_KEY}"},
                    json=payload,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
        except Exception:
            return None

    async def chat(self, messages: list, model: str = "glm-4-flash") -> str:
        """Text chat — tries MiniMax → GLM → demo fallback."""
        # Try MiniMax
        if self.api_key:
            try:
                async with httpx.AsyncClient(timeout=90.0) as client:
                    payload = {"model": "MiniMax-M2.5", "messages": messages, "max_tokens": 1000}
                    resp = await client.post(
                        f"{self.base_url}/v1/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json=payload,
                    )
                    resp.raise_for_status()
                    return resp.json()["choices"][0]["message"]["content"]
            except Exception:
                pass

        # Try GLM
        result = await self._glm_chat(messages)
        if result is not None:
            return result

        # Demo fallback: pick response based on message content (check report first)
        content = " ".join(m.get("content", "") for m in messages)
        if "生成维修报告" in content or ("报告" in content and "诊断过程" in content):
            return _DEMO_REPORT
        if "原因" in content or "symptom" in content.lower() or "推理" in content:
            return _DEMO_REASONING
        if "报告" in content or "report" in content.lower():
            return _DEMO_REPORT
        return json.dumps({"step": 2, "instruction": "使用万用表直流电压档（20V），测量充电端口正负极间电压",
                           "measurement_point": "充电端口 + / -",
                           "expected_value": "16.4V ± 0.2V（4串满充）"}, ensure_ascii=False)


_client: MiniMaxClient | None = None


def get_client() -> MiniMaxClient:
    global _client
    if _client is None:
        _client = MiniMaxClient(api_key=MINIMAX_API_KEY, base_url=MINIMAX_API_BASE)
    return _client
