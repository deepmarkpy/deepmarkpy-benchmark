from core.base_attack import BaseAttack

import numpy as np


class FlipSamplesAttack(BaseAttack):
    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform a "Flip Samples" attack by randomly exchanging the positions
        of selected samples in the audio signal.

        Args:
            audio (np.ndarray): Input audio signal.
            **kwargs: Additional parameters.
                - sampling_rate (int): Sampling rate of the audio in Hz (required).
                - num_flips (int): Number of sample pairs to flip in the specified duration. Default is 20 (Optional).
                - flip_duration (float): Duration (in seconds) over which flips should occur. Default is 0.5 seconds (Optional).

        Returns:
            np.ndarray: Audio signal with flipped sample positions.

        Raises:
            ValueError: If 'sampling_rate' is not provided in kwargs.
        """
        sampling_rate = kwargs.get("sampling_rate", None)
        num_flips = kwargs.get("num_flips", self.config.get("num_flips"))
        duration = kwargs.get("flip_duration", self.config.get("flip_duration"))

        if sampling_rate is None:
            raise ValueError("'sampling_rate' must be provided in kwargs.")

        total_samples = int(duration * sampling_rate)

        total_samples = min(total_samples, len(audio))

        flip_indices = np.random.choice(
            range(total_samples), size=num_flips * 2, replace=False
        )
        flip_indices = flip_indices.reshape(-1, 2)

        modified_audio = audio.copy()
        for idx1, idx2 in flip_indices:
            modified_audio[idx1], modified_audio[idx2] = (
                modified_audio[idx2],
                modified_audio[idx1],
            )

        return modified_audio
