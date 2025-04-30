import logging
import os

import numpy as np

from core.base_model import BaseModel

logger = logging.getLogger(__name__)

class AudioSealModel(BaseModel):
    def __init__(self):
        super().__init__()

        # Determine the APP PORT from environment variables
        port = os.getenv("AUDIOSEAL_PORT", "5001")

        if not port:
            logger.error("AUDIOSEAL_PORT environment variable not set and no default provided.")
            raise ValueError("AUDIOSEAL_PORT must be set")

        self.endpoint = f"http://localhost:{port}"
        logger.info(f"AudioSealModel initialized. Target API: {self.endpoint}")

    def embed(
        self, audio: np.ndarray, watermark_data: np.ndarray, sampling_rate: int
    ) -> np.ndarray:
        """Embeds a watermark into the audio using the AudioSeal service."""
        payload = {
            "audio": audio.tolist(),
            "watermark_data": watermark_data.tolist(),
            "sampling_rate": sampling_rate,
        }
        
        response_data = self._make_request(endpoint="/embed", json_data=payload, method="POST")
        
        if "watermarked_audio" not in response_data:
             logger.error("'/embed' response did not contain 'watermarked_audio' key.")
             raise KeyError("Missing 'watermarked_audio' in response from /embed")
        
        return np.array(response_data["watermarked_audio"])

    def detect(self, audio: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Detects a watermark in the audio using the AudioSeal service."""
        payload = {"audio": audio.tolist(), "sampling_rate": sampling_rate}
        
        response_data = self._make_request(endpoint="/detect", json_data=payload, method="POST")

        if "watermark" not in response_data:
             logger.error("'/detect' response did not contain 'watermark' key.")
             raise KeyError("Missing 'watermark' in response from /detect")
        
        return np.array(response_data["watermark"])
        
 
