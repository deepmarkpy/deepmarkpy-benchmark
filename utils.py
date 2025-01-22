import librosa
import numpy as np

def load_audio(file_path, target_sr=None, mono=True):
    """
    Load an audio file and resample it to the specified sampling rate.

    Args:
        file_path (str): Path to the audio file.
        target_sr (int, optional): Target sampling rate for the audio. If None, the original sampling rate is used.
        mono (bool, optional): If True, the audio is converted to mono. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - y (np.ndarray): The audio signal as a NumPy array.
            - sr (int): The sampling rate of the loaded audio.
    """
    y, sr = librosa.load(file_path, sr=target_sr, mono=mono)
    return y, sr


def snr(signal, noisy_signal):
    """
    Calculate the Signal-to-Noise Ratio (SNR) in decibels (dB) between a clean signal
    and a noisy signal.

    Args:
        signal (np.ndarray): The original clean signal.
        noisy_signal (np.ndarray): The noisy version of the clean signal.

    Returns:
        float: The SNR value in decibels (dB).
    """
    if signal.shape != noisy_signal.shape:
        min_len = np.min([len(signal), len(noisy_signal)])
        signal = signal[:min_len]
        noisy_signal = noisy_signal[:min_len]

    # Calculate the noise by subtracting the clean signal from the noisy signal
    noise = noisy_signal - signal

    # Calculate signal and noise powers
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)

    # Handle the case where noise power is zero
    if noise_power == 0:
        return np.inf

    # Compute SNR in decibels
    snr = 10 * np.log10(signal_power / noise_power)

    return snr