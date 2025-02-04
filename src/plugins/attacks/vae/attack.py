import requests
from core.base_attack import BaseAttack

import numpy as np

class VAEAttack(BaseAttack):
    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        sampling_rate = kwargs.get("sampling_rate", None)
        if sampling_rate is None:
            raise ValueError("'sampling_rate' must be provided in kwargs.")

        response = requests.post(
            self.config["endpoint"] + "/attack",
            json={
                "audio": audio.tolist(),
                "sampling_rate": sampling_rate
            },
        )
        return np.array(response.json()["audio"])
