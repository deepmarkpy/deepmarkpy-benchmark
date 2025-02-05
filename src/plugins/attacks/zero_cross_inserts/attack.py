from core.base_attack import BaseAttack

import numpy as np


class ZeroCrossInsertsAttack(BaseAttack):
    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform Zero-Cross-Inserts on an audio signal.

        Args:
            audio (np.ndarray): Input audio signal.
            **kwargs: Additional parameters for the operation.
                - sampling_rate (int): Sampling rate of the audio in Hz (required).
                - zero_cross_pause_length (int): Number of zeros to insert at each zero-crossing point. Default is 20.
                - zero_cross_min_distance (float): Minimum distance between pauses in seconds. Default is 1.0.

        Returns:
            np.ndarray: Audio signal with inserted pauses at zero-crossing points.

        Raises:
            ValueError: If 'sampling_rate' is not provided in kwargs.
        """

        sampling_rate = kwargs.get("sampling_rate")
        pause_length = kwargs.get(
            "zero_cross_pause_length", self.config.get("zero_cross_pause_length")
        )
        min_distance = kwargs.get(
            "zero_cross_min_distance", self.config.get("zero_cross_pause_length")
        )

        if sampling_rate is None:
            raise ValueError("'sampling_rate' must be provided in kwargs.")

        min_sample_distance = int(min_distance * sampling_rate)

        zero_crossings = np.where(np.diff(np.sign(audio)))[0]

        modified_audio = []
        last_insert_pos = -min_sample_distance

        for i in zero_crossings:
            if i - last_insert_pos >= min_sample_distance:
                modified_audio.extend(audio[last_insert_pos:i])
                modified_audio.extend([0] * pause_length)
                last_insert_pos = i

        modified_audio.extend(audio[last_insert_pos:])

        return np.array(modified_audio, dtype=audio.dtype)
