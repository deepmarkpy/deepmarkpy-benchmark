from src.core.base_attack import BaseAttack

import numpy as np


class CutSamplesAttack(BaseAttack):

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform a "Cut Samples" attack by randomly deleting short sequences of samples
        while maintaining inaudibility constraints.

        Args:
            audio (np.ndarray): Input audio signal.
            **kwargs: Additional parameters.
                - sampling_rate (int): Sampling rate of the audio in Hz (required).
                - cut_max_sequence_length (int): Maximum length of each cut sequence. Default is 50 samples (Optional).
                - cut_num_sequences (int): Number of sequences to cut in the specified duration. Default is 20 (Optional).
                - cut_duration (float): Duration (in seconds) over which cuts should occur. Default is 0.5 seconds (Optional).
                - cut_max_value_difference (float): Maximum allowed difference between start and end sample of a cut. Default is 0.1 (Optional).

        Returns:
            np.ndarray: Audio signal with random samples cut.

        Raises:
            ValueError: If 'sampling_rate' is not provided in kwargs.
        """
        sampling_rate = kwargs.get("sampling_rate", None)
        max_sequence_length = kwargs.get(
            "cut_max_sequence_length", self.config.get("cut_max_sequence_length")
        )
        num_sequences = kwargs.get(
            "cut_num_sequences", self.config.get("cut_num_sequences")
        )
        duration = kwargs.get("cut_duration", self.config.get("cut_duration"))
        max_value_difference = kwargs.get(
            "cut_max_value_difference", self.config.get("cut_max_value_difference")
        )

        if sampling_rate is None:
            raise ValueError("'sampling_rate' must be provided in kwargs.")

        total_samples = int(duration * sampling_rate)

        total_samples = min(total_samples, len(audio))

        cut_positions = np.random.choice(
            range(total_samples - max_sequence_length),
            size=num_sequences,
            replace=False,
        )

        cut_positions = np.sort(cut_positions)

        modified_audio = []
        prev_cut_end = 0

        for start_pos in cut_positions:
            sequence_length = np.random.randint(1, max_sequence_length + 1)
            end_pos = start_pos + sequence_length

            if end_pos >= len(audio):
                break

            if abs(audio[start_pos] - audio[end_pos]) > max_value_difference:
                continue

            modified_audio.extend(audio[prev_cut_end:start_pos])

            prev_cut_end = end_pos

        modified_audio.extend(audio[prev_cut_end:])

        return np.array(modified_audio, dtype=audio.dtype)
