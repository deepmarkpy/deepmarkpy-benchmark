from core.base_attack import BaseAttack

import requests
import numpy as np


class SpeechEnhancementAttack(BaseAttack):
    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        sampling_rate = kwargs.get("sampling_rate", None)
        noise_strength = kwargs.get("noise_strength", self.config.get("noise_strength"))
        if sampling_rate is None:
            raise ValueError("'sampling_rate' must be provided in kwargs.")

        response = requests.post(
            self.config["endpoint"] + "/attack",
            json={
                "audio": audio.tolist(),
                "sampling_rate": sampling_rate,
                "noise_strength": noise_strength,
            },
        )
        return np.array(response.json()["audio"])
