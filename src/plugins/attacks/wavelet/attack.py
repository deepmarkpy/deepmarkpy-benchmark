import pywt
import numpy as np

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