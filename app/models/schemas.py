"""Pydantic models for API messages."""
from pydantic import BaseModel
from typing import Optional


class ImageAck(BaseModel):
    image_id: str
    type: str = "image_ack"


class AnalysisResult(BaseModel):
    components: list
    positions: list
    raw_text: str
    type: str = "analysis"


class ReasoningResult(BaseModel):
    causes: list
    type: str = "reasoning"


class GuidanceResult(BaseModel):
    step: int
    instruction: str
    measurement_point: str = ""
    expected_value: str = ""
    type: str = "guidance"


class ErrorResult(BaseModel):
    message: str
    type: str = "error"


class StatusMessage(BaseModel):
    message: str
    type: str = "status"
