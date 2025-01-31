from fastapi import requests
from src.core.base_model import BaseModel
import numpy as np


class AudioSealModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.endpoint = self.config.get("endpoint", "http://localhost:5001")

    def embed(
        self, audio: np.ndarray, watermark_data: np.ndarray, sampling_rate: int
    ) -> np.ndarray:
        response = requests.post(
            self.endpoint + "/embed",
            json={
                "audio": audio.tolist(),
                "watermark_data": watermark_data.tolist(),
                "sampling_rate": sampling_rate,
            },
        )
        return np.array(response.json()["watermarked_audio"])

    def detect(self, audio: np.ndarray, sampling_rate: int) -> np.ndarray:
        response = requests.post(
            self.endpoint + "/detect",
            json={
                "audio": audio.tolist(),
                "sampling_rate": sampling_rate
            }
        )
        return np.array(response.json()["watermark"])
