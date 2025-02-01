import requests
from src.core.base_model import BaseModel
import numpy as np


class WavMarkModel(BaseModel):

    def embed(
        self, audio: np.ndarray, watermark_data: np.ndarray, sampling_rate: int
    ) -> np.ndarray:
        response = requests.post(
            self.config["endpoint"] + "/embed",
            json={
                "audio": audio.tolist(),
                "watermark_data": watermark_data.tolist(),
                "sampling_rate": sampling_rate,
            },
        )
        return np.array(response.json()["watermarked_audio"])

    def detect(self, audio: np.ndarray, sampling_rate: int) -> np.ndarray:
        response = requests.post(
            self.config["endpoint"] + "/detect",
            json={
                "audio": audio.tolist(),
                "sampling_rate": sampling_rate
            }
        )
        return np.array(response.json()["watermark"])
