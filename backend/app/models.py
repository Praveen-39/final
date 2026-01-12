from pydantic import BaseModel
from typing import Optional

class PredictRequest(BaseModel):
    audio_base64: str
    sample_rate: int = 16000
    language: str = "en"

class PredictResponse(BaseModel):
    emotion: str
    emotion_confidence: float
    sarcasm: bool
    sarcasm_score: float
    transcript: str
    pitch: Optional[float] = None
    energy: Optional[float] = None
    spectral_centroid: Optional[float] = None
