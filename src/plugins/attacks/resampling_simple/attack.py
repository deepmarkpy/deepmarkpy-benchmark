import numpy as np

from core.base_attack import BaseAttack

class ResamplingSimpleAttack(BaseAttack):
    """
        Perform a resampling attack on an audio signal, by downsampling it first to 16kHz, and then upsampling it
        to the starting sampling rate. 
        Args:
            audio (np.ndarray): The input audio signal.
            **kwargs: Additional parameters for the resampling attack:
                - sampling_rate (int): Sampling rate of the signal.
        Returns:
            np.ndarray: The processed audio signal with the resampling applied.

        Raises:
            ValueError: If the sampling rate is not provided in `kwargs`.

    """

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:

        sampling_rate = kwargs.get("sampling_rate", None)
        if sampling_rate is None:
            raise ValueError("'sampling_rate' must be provided in kwargs.")
        
        downsample_factor = sampling_rate // 16000
        if downsample_factor > 1:
            # Simple decimation (take every nth sample)
            downsampled_audio = audio[::downsample_factor]
            downsampled_sr = 16000
            print(f"Downsampled to {downsampled_sr} Hz: {len(downsampled_audio)} samples")
                
            # Upsample back to original rate using linear interpolation
            upsampled_audio = np.interp(
                np.arange(len(audio)), 
                np.arange(0, len(audio), downsample_factor),
                downsampled_audio
            )
            print(f"Upsampled back to {sampling_rate} Hz: {len(upsampled_audio)} samples")
            return upsampled_audio
        else:
            print(f"Audio already at or below 16kHz ({sampling_rate} Hz), skipping resampling")
        return audio