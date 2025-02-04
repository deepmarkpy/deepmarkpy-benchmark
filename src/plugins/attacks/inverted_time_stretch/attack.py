from core.base_attack import BaseAttack
from plugins.attacks.time_stretch.attack import TimeStretchAttack

import numpy as np


class InvertedTimeStretch(BaseAttack):

    def __init__(self):
        super().__init__()
        self.time_stretch = TimeStretchAttack()

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform an inverted time stretch operation.

        Args:
        audio (np.ndarray): Input audio signal.
        **kwargs: Additional parameters for time stretching.
            - sampling_rate (int): Sampling rate of the audio in Hz (required).
            - inverted_stretch_rate (float): Stretching factor (>1 for slower, <1 for faster) (Optional).

        Returns:
        np.ndarray: The audio signal after time stretching and inverting.

        Raises:
            ValueError: If `sampling_rate` is not provided in kwargs.

        """
        sampling_rate = kwargs.get("sampling_rate", None)
        if sampling_rate is None:
            raise ValueError("'sampling_rate' must be provided in kwargs.")
        stretch_rate = kwargs.get(
            "inverted_stretch_rate", self.config.get("inverted_stretch_rate")
        )
        stretched_audio = self.time_stretch.apply(
            audio, sampling_rate=sampling_rate, stretch_rate=stretch_rate
        )
        return self.time_stretch.apply(
            stretched_audio, sampling_rate=sampling_rate, stretch_rate=1 / stretch_rate
        )
