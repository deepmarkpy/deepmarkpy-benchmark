from core.base_attack import BaseAttack

import requests
import numpy as np


class DiffusionAttack(BaseAttack):
    def apply(self, audio, **kwargs):
        sampling_rate = kwargs.get("sampling_rate", None)
        diffusion_steps = kwargs.get(
            "diffusion_steps", self.config.get("diffusion_steps")
        )
        assert diffusion_steps <= 150, "number of steps is too large."
        if sampling_rate is None:
            raise ValueError("'sampling_rate' must be provided in kwargs.")

        response = requests.post(
            self.config["endpoint"] + "/attack",
            json={
                "audio": audio.tolist(),
                "sampling_rate": sampling_rate,
                "diffusion_steps": diffusion_steps,
            },
        )
        return np.array(response.json()["audio"])
