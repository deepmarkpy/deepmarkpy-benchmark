import numpy as np
from utils import load_audio, snr
from watermarking_wrapper import WatermarkingWrapper
import random
import pyrubberband as pyrb
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
            "collusion_attack": self.collusion_attack,
            "pitch_shift": self.pitch_shift,
            "time_stretch": self.time_stretch,
            "inverted_time_stretch": self.inverted_time_stretch,
            "zero_cross_inserts": self.zero_cross_inserts,
            "cut_samples": self.cut_samples
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

            audio, sampling_rate = load_audio(filepath, target_sr=sampling_rate)
            
            watermarked_audio = self.wrapper.embed(model_name, audio, watermark_data, sampling_rate)

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

                sf.write('test.wav', attacked_audio, sampling_rate)
                
                detected_message = self.wrapper.detect(model_name, attacked_audio, sampling_rate=sampling_rate)
                
                accuracy = self.compare_watermarks(watermark_data, detected_message)
                results[filepath][attack] = {
                    "accuracy": accuracy,
                    "snr": "N/A" if np.abs(len(audio)-len(attacked_audio)>1) else snr(audio, attacked_audio)
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
        noise_level = kwargs.get('noise_level', 0.001)
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
                - collusion_size(int): Size of the collusion segment.

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
    
    def pitch_shift(self, audio, **kwargs):
        """
        Perform pitch shifting using pyrubberband.

        Args:
            audio (np.ndarray): Input audio signal.
            **kwargs: Additional parameters for pitch shifting.
                - sampling_rate (int): Sampling rate of the audio in Hz (required).
                - cents (float): Pitch shift in cents (1 cent = 1/100 of a semitone) (required).

        Returns:
            np.ndarray: The pitch-shifted audio signal.

        Raises:
            ValueError: If `sampling_rate` or `cents` is not provided in kwargs.

        Notes:
            - This function uses the pyrubberband library, which provides high-quality
            pitch shifting without altering the speed of the audio.
            - Ensure that pyrubberband and the Rubber Band Library are installed before use.
        """
        sampling_rate = kwargs.get('sampling_rate', None)
        cents = kwargs.get('cents', None)

        if sampling_rate is None or cents is None:
            raise ValueError("Both 'sampling_rate' and 'cents' must be provided in kwargs.")

        semitones = cents / 100
        return pyrb.pitch_shift(audio, sampling_rate, semitones)
    
    def time_stretch(self, audio, **kwargs):
        """
        Perform time stretching on an audio signal using pyrubberband.

        Args:
            audio (np.ndarray): Input audio signal to be stretched.
            **kwargs: Additional parameters for time stretching.
                - sampling_rate (int): Sampling rate of the audio in Hz (required).
                - stretch_rate (float): Stretching factor (>1 for slower, <1 for faster) (required).

        Returns:
            np.ndarray: The time-stretched audio signal.

        Raises:
            ValueError: If `sampling_rate` or `stretch_rate` is not provided in kwargs.

        Notes:
            - This function uses the pyrubberband library, which provides high-quality
            time-stretching capabilities while maintaining pitch integrity.
            - Ensure that pyrubberband and the Rubber Band Library are installed before use.
            - Stretch rate of `1.0` implies no change in speed.
            - Values greater than `1.0` slow down the audio, while values less than `1.0` speed it up.
        """
        sampling_rate = kwargs.get('sampling_rate', None)
        stretch_rate = kwargs.get('stretch_rate', None)

        if sampling_rate is None or stretch_rate is None:
            raise ValueError("Both 'sampling_rate' and 'stretch_rate' must be provided in kwargs.")
        
        return pyrb.time_stretch(audio, sampling_rate, stretch_rate)
    
    def inverted_time_stretch(self, audio, **kwargs):
        """
        Perform an inverted time stretch operation.

        Args:
            audio (np.ndarray): Input audio signal.
            **kwargs: Additional parameters for time stretching.
                - inverted_stretch_rate (float): Stretching factor (>1 for slower, <1 for faster). Required.

        Returns:
            np.ndarray: The audio signal after time stretching and inverting.

        Raises:
            ValueError: If 'inverted_stretch_rate' is not provided in kwargs.
        """
        args = kwargs.copy()
        stretch_rate = args.get('inverted_stretch_rate')
        args.pop('stretch_rate')
        if stretch_rate is None:
            raise ValueError("'inverted_stretch_rate' must be provided in kwargs.")
        audio = self.time_stretch(audio, stretch_rate = stretch_rate, **args)
        return self.time_stretch(audio, stretch_rate = 1 / stretch_rate, **args)

    def zero_cross_inserts(self, audio, **kwargs):
        """
        Perform Zero-Cross-Inserts on an audio signal.

        Args:
            audio (np.ndarray): Input audio signal.
            **kwargs: Additional parameters for the operation.
                - sampling_rate (int): Sampling rate of the audio in Hz (required).
                - pause_length (int): Number of zeros to insert at each zero-crossing point. Default is 20.
                - min_distance (float): Minimum distance between pauses in seconds. Default is 1.0.

        Returns:
            np.ndarray: Audio signal with inserted pauses at zero-crossing points.

        Raises:
            ValueError: If 'sampling_rate' is not provided in kwargs.
        """

        sampling_rate = kwargs.get('sampling_rate')
        pause_length = kwargs.get('pause_length', 20)
        min_distance = kwargs.get('min_distance', 1.0)

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

    import numpy as np

    def cut_samples(self, audio, **kwargs):
        """
        Perform a "Cut Samples" attack by randomly deleting short sequences of samples
        while maintaining inaudibility constraints.

        Args:
            audio (np.ndarray): Input audio signal.
            **kwargs: Additional parameters.
                - sampling_rate (int): Sampling rate of the audio in Hz (required).
                - max_sequence_length (int): Maximum length of each cut sequence. Default is 50 samples.
                - num_sequences (int): Number of sequences to cut in the specified duration. Default is 20.
                - duration (float): Duration (in seconds) over which cuts should occur. Default is 0.5 seconds.
                - max_value_difference (float): Maximum allowed difference between start and end sample of a cut. Default is 0.1.

        Returns:
            np.ndarray: Audio signal with random samples cut.

        Raises:
            ValueError: If 'sampling_rate' is not provided in kwargs.
        """
        sampling_rate = kwargs.get('sampling_rate', None)
        max_sequence_length = kwargs.get('max_sequence_length', 50)
        num_sequences = kwargs.get('num_sequences', 20)
        duration = kwargs.get('duration', 0.5)
        max_value_difference = kwargs.get('max_value_difference', 0.1)

        if sampling_rate is None:
            raise ValueError("'sampling_rate' must be provided in kwargs.")

        total_samples = int(duration * sampling_rate)

        total_samples = min(total_samples, len(audio))

        cut_positions = np.random.choice(
            range(total_samples - max_sequence_length),
            size=num_sequences,
            replace=False
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