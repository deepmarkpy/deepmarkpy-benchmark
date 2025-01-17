import numpy as np
from watermarking_wrapper import WatermarkingWrapper

class Attacks:
    """
    A class to perform various attacks on watermarking models and benchmark their performance.
    """

    def __init__(self, wrapper=None):
        """
        Initialize the Attacks class.

        Args:
            wrapper (WatermarkingWrapper, optional): The watermarking wrapper to use for model operations.
                                                     If None, a new wrapper instance is created.
        """
        self.wrapper = wrapper or WatermarkingWrapper()
        self.attacks = {
            "additive_noise": self.additive_noise_attack
        }

    def benchmark(self, filepaths, model_name, watermark_data=None, attack_types=None, sampling_rate=16000, **kwargs):
        """
        Benchmark the watermarking models against selected attacks.

        Args:
            filepaths (str or list): Path(s) to the audio file(s) to benchmark.
            model_name (str): The model to benchmark ('AudioSeal', 'WavMark', or 'SilentCipher').
            watermark_data (np.ndarray, optional): The binary watermark data to embed. Defaults to random message.
            attack_types (list, optional): A list of attack types to perform. Defaults to all attacks.
            **kwargs: Additional parameters for specific attacks.

        Returns:
            dict: A dictionary containing benchmark results for each file and attack.
        """
        if isinstance(filepaths, str):
            filepaths = [filepaths]

        attack_types = attack_types or list(self.attacks.keys())
        results = {}

        for filepath in filepaths:
            print(f"Processing file: {filepath}")
            results[filepath] = {}

            if watermark_data is None:
                watermark_data = np.random.randint(0, 2, size=40 if model_name=='SilentCipher' else 16, dtype=np.int32)

            watermarked_audio = self.wrapper.embed(model_name, filepath, watermark_data)

            for attack in attack_types:
                if attack not in self.attacks:
                    print(f"Attack '{attack}' not found. Skipping.")
                    continue

                print(f"Applying attack: {attack}")
                
                attacked_audio = self.attacks[attack](watermarked_audio, **kwargs)
                
                detected_message = self.wrapper.detect(model_name, attacked_audio, sampling_rate=sampling_rate)
                
                accuracy = self.compare_watermarks(watermark_data, detected_message)
                results[filepath][attack] = {
                    "accuracy": accuracy,
                }

        return results

    def compare_watermarks(self, original, detected):
        """
        Compare the original and detected watermarks.

        Args:
            original (np.ndarray): The original binary watermark.
            detected (np.ndarray): The detected binary watermark.

        Returns:
            float: The accuracy of the detected watermark (percentage).
        """
        matches = np.sum(original == detected)
        return matches / len(original) * 100

    def additive_noise_attack(self, audio, noise_level=0.01):
        """
        Additive Gaussian noise attack.

        Args:
            audio (np.ndarray): The input audio signal.
            noise_level (float): The standard deviation of the Gaussian noise.

        Returns:
            np.ndarray: The audio signal with noise added.
        """
        noise = np.random.normal(0, noise_level, audio.shape)
        return audio + noise