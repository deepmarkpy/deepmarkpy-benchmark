from core.base_attack import BaseAttack

import numpy as np
import pyrubberband as pyrb


class PitchShiftAttack(BaseAttack):

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform pitch shifting using pyrubberband.

        Args:
            audio (np.ndarray): Input audio signal.
            **kwargs: Additional parameters for pitch shifting.
                - sampling_rate (int): Sampling rate of the audio in Hz (optional).
                - cents (float): Pitch shift in cents (1 cent = 1/100 of a semitone) (optional).

        Returns:
            np.ndarray: The pitch-shifted audio signal.

        Notes:
            - This function uses the pyrubberband library, which provides high-quality
            pitch shifting without altering the speed of the audio.
            - Ensure that pyrubberband and the Rubber Band Library are installed before use.
        """

        sampling_rate = kwargs.get("sampling_rate", self.config.get("cents"))
        cents = kwargs.get("cents", self.config.get("cents"))

        sampling_rate = kwargs.get("sampling_rate", None)
        cents = kwargs.get("cents", None)

        semitones = cents / 100
        return pyrb.pitch_shift(audio, sampling_rate, semitones)
