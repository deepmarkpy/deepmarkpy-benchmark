import logging
import os
import numpy as np

from core.base_model import BaseModel

logging = logging.getLogger(__name__)

class PerthModel(BaseModel):
    def __init__(self):
        super().__init__()

        host = "localhost" 
        port = os.getenv("PERTH_PORT", "7010")
        if not port:
             logging.error("PERTH_PORT environment variable not set.")
             raise ValueError("PERTH_PORT must be set for PerthModel")

        self.base_url = f"http://{host}:{port}"
        logging.info(f"PerthModel initialized. Target API: {self.base_url}")

    def embed(
        self, audio: np.ndarray, watermark_data: np.ndarray, sampling_rate: int
    ) -> np.ndarray:
        """Embeds a watermark using the Perth service."""
        payload = {
            "audio": audio.tolist(),
            "watermark_data": watermark_data.tolist(),
            "sampling_rate": sampling_rate,
        }
        # Use the helper method from BaseModel
        response_data = self._make_request(endpoint="/embed", json_data=payload, method="POST")

        if "watermarked_audio" not in response_data:
             logging.error("'/embed' response did not contain 'watermarked_audio' key.")
             raise KeyError("Missing 'watermarked_audio' in response from /embed")
        return np.array(response_data["watermarked_audio"])
    
    
    def detect(self, audio: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Detects a watermark using the Perth service."""
        payload = {"audio": audio.tolist(), "sampling_rate": sampling_rate}
       
        # Use the helper method from BaseModel
        response_data = self._make_request(endpoint="/detect", json_data=payload, method="POST")

        if "watermark" not in response_data:
             logging.error("'/detect' response did not contain 'watermark' key.")
             raise KeyError("Missing 'watermark' in response from /detect")
        # Handle potential None value from the API
        watermark = response_data["watermark"]
        return np.array(watermark) if watermark is not None else None
    