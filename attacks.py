import numpy as np
from watermarking_wrapper import WatermarkingWrapper
import random
import soundfile as sf

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
            "additive_noise": self.additive_noise_attack,
            "multiple_watermarking": self.multiple_watermarking,
            "collusion_attack": self.collusion_attack
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
                attack_kwargs = {**kwargs}
                attack_kwargs['model_name'] = model_name
                attack_kwargs['watermark_data'] = watermark_data
                attack_kwargs['sampling_rate'] = sampling_rate
                attack_kwargs['filepath'] = filepath
                attacked_audio = self.attacks[attack](watermarked_audio, **attack_kwargs)

                sf.write('test.wav', attacked_audio, samplerate=44100 if model_name=='SilentCipher' else 16000)
                
                detected_message = self.wrapper.detect(model_name, attacked_audio, sampling_rate=sampling_rate)

                print(watermark_data)
                print(detected_message)
                
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

    def additive_noise_attack(self, audio, **kwargs):
        """
        Additive Gaussian noise attack.

        Args:
            audio (np.ndarray): The input audio signal.
            noise_level (float): The standard deviation of the Gaussian noise.

        Returns:
            np.ndarray: The audio signal with noise added.
        """
        noise_level = kwargs.get('noise_level', 0.01)
        noise = np.random.normal(0, noise_level, audio.shape)
        return audio + noise
    
    def multiple_watermarking(self, audio, **kwargs):
        """
        Perform multiple watermarking by embedding a watermark using a specified model
        or a randomly chosen model if allowed.

        Args:
            audio (np.ndarray): The input audio signal to watermark.
            **kwargs: Additional parameters for the watermarking process.
                - model_name (str): The name of the current watermarking model.
                - sampling_rate (int): The sampling rate of the audio signal.
                - mwm_type (str): The type of multiple watermarking to apply.
                                Defaults to 'other'.

        Returns:
            np.ndarray: The watermarked audio signal.

        Behavior:
            - If `mwm_type` is 'other', a different model than `model_name` is chosen randomly
            from the available models in the wrapper.
            - If no other models are available, `None` is used as the model.
            - Otherwise, the specified `model_name` is reused.
        """
        model_name = kwargs.get('model_name', None)
        input_sr = kwargs.get('sampling_rate', None)
        mwm_type = kwargs.get('mwm_type', 'other')

        if mwm_type == 'other':
            filtered_list = [key for key in self.wrapper.models.keys() if key != model_name]
            new_model_name = random.choice(filtered_list) if filtered_list else None
        else:
            new_model_name = model_name
        return self.wrapper.embed(new_model_name, audio, input_sr=input_sr)
    
    def collusion_attack(self, audio, **kwargs):
        """
        Perform a collusion attack by embedding a second watermark into the audio and
        recombining segments of both watermarked audios into one.

        Args:
            audio (np.ndarray): The original watermarked audio.
            **kwargs: Additional parameters for the attack.
                - model_name (str): Name of the watermarking model.
                - sampling_rate (int): Sampling rate of the audio.
                - filepath (str): Path to the original audio file.

        Returns:
            np.ndarray: The recombined audio after the collusion attack.
        """
        model_name = kwargs.get('model_name', None)
        filepath = kwargs.get('filepath', None)

        if model_name is None or filepath is None:
            raise ValueError("Both 'model_name' and 'filepath' must be provided for the collusion attack.")

        second_audio = self.wrapper.embed(model_name, filepath)

        segment_size = kwargs.get('collusion_size', 25)
        num_segments = len(audio) // segment_size
        mixed_audio = np.zeros_like(audio)

        for i in range(num_segments):
            start_idx = i * segment_size
            end_idx = start_idx + segment_size

            if np.random.rand() > 0.5:
                mixed_audio[start_idx:end_idx] = audio[start_idx:end_idx]
            else:
                mixed_audio[start_idx:end_idx] = second_audio[start_idx:end_idx]

        if num_segments * segment_size < len(audio):
            remaining_start = num_segments * segment_size
            if np.random.rand() > 0.5:
                mixed_audio[remaining_start:] = audio[remaining_start:]
            else:
                mixed_audio[remaining_start:] = second_audio[remaining_start:]

        return mixed_audio
