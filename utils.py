import librosa
import numpy as np
import pywt

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

    noise = noisy_signal - signal

    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)

    if noise_power == 0:
        return np.inf

    snr = 10 * np.log10(signal_power / noise_power)

    return snr

def compute_threshold(audio, wavelet):
    """
    Compute the universal threshold for wavelet-based denoising.

    Args:
        audio (np.ndarray): Input audio signal.
        wavelet (str): Wavelet type (e.g., 'db1', 'sym5', etc.) used for decomposition.

    Returns:
        float: The calculated threshold value.

    Notes:
        - This function uses the universal threshold formula:
          Threshold = sigma * sqrt(2 * log(n)),
          where sigma is the noise standard deviation estimated from the detail coefficients,
          and n is the length of the audio signal.
        - The estimation of sigma uses the robust formula:
          sigma = median(|coeffs[-1]|) / 0.6745,
          which is based on the assumption of Gaussian white noise.
        - The universal threshold is particularly effective for denoising signals corrupted by
          additive white Gaussian noise.

    """
    coeffs = pywt.wavedec(audio, wavelet)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(audio)))
    return threshold