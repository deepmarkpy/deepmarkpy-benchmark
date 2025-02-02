from src.core.base_attack import BaseAttack

import numpy as np
import pyrubberband as pyrb


class TimeStretchAttack(BaseAttack):

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform time stretching on an audio signal using pyrubberband.

        Args:
            audio (np.ndarray): Input audio signal to be stretched.
            **kwargs: Additional parameters for time stretching.
                - sampling_rate (int): Sampling rate of the audio in Hz (required).
                - stretch_rate (float): Stretching factor (>1 for slower, <1 for faster) (optional).

        Returns:
            np.ndarray: The time-stretched audio signal.

        Raises:
            ValueError: If `sampling_rate` is not provided in kwargs.

        Notes:
            - This function uses the pyrubberband library, which provides high-quality
            time-stretching capabilities while maintaining pitch integrity.
            - Ensure that pyrubberband and the Rubber Band Library are installed before use.
            - Stretch rate of `1.0` implies no change in speed.
            - Values greater than `1.0` slow down the audio, while values less than `1.0` speed it up.
        """
        sampling_rate = kwargs.get("sampling_rate", None)

        stretch_rate = kwargs.get("stretch_rate", self.config.get("stretch_rate"))

        if sampling_rate is None:
            raise ValueError(
                "Both 'sampling_rate' and 'stretch_rate' must be provided in kwargs."
            )

        return pyrb.time_stretch(audio, sampling_rate, stretch_rate)
