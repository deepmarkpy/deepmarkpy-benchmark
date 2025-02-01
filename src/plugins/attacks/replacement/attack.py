import numpy as np
from src.core.base_attack import BaseAttack
from src.plugins.attacks.replacement.replacement_attack import replacement_attack


class ReplacementAttack(BaseAttack):

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform a replacement attack on an audio signal.

        Args:
            audio (np.ndarray): The input audio signal.
            **kwargs: Additional parameters for the replacement attack:
                - sampling_rate (int): The sampling rate of the audio signal in Hz (required).
                - replacement_block_size (int): Size of each block for processing in samples (default: 1024).
                - replacement_overlap_factor (float): Overlap factor between consecutive blocks (default: 0.75).
                Must be in the range [0, 1), where 0 means no overlap and values closer to 1
                indicate higher overlap.
                - replacement_lower_bound (float): The lower bound of the similarity distance for considering a block as a candidate (default: 0).
                - replacement_upper_bound (float): The upper bound of the similarity distance for considering a block as a candidate (default: 10).
                - replacement_k (int): Maximum number of similar blocks to consider (default: 20).
                - replacement_use_masking (bool): Whether to use psychoacoustic masking for distance calculation (default: False).

        Returns:
            np.ndarray: The processed audio signal with the replacement attack applied.

        Raises:
            ValueError: If the `sampling_rate` is not provided in `kwargs`.

        """
        sampling_rate = kwargs.get("sampling_rate", None)
        block_size = kwargs.get(
            "replacement_block_size", self.config.get("replacement_block_size")
        )
        overlap_factor = kwargs.get(
            "replacement_overlap_factor", self.config.get("replacement_overlap_factor")
        )
        lower_bound = kwargs.get(
            "replacement_lower_bound", self.config.get("replacement_lower_bound")
        )
        upper_bound = kwargs.get(
            "replacement_upper_bound", self.config.get("replacement_upper_bound")
        )
        use_masking = kwargs.get(
            "replacement_use_masking", self.config.get("replacement_use_masking")
        )
        if sampling_rate is None:
            raise ValueError("'sampling_rate' must be provided in kwargs.")
        return replacement_attack(
            x=audio,
            sampling_rate=sampling_rate,
            block_size=block_size,
            overlap_factor=overlap_factor,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            use_masking=use_masking,
        )
