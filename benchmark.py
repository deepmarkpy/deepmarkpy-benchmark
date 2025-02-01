import numpy as np
from audio_diffusion_attacks import AudioDDIMAttacker
from utils import compute_threshold, load_audio, resample_audio, snr
from watermarking_wrapper import WatermarkingWrapper
import random
import pyrubberband as pyrb
import torch
from vaewmattacker import VAEWMAttacker
import pywt
from replacement_attack import replacement_attack


class Benchmark:
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
            "same_model_watermarking": self.same_model_watermarking,
            "cross_model_watermarking": self.cross_model_watermarking,
            "collusion_attack": self.collusion_attack,
            "pitch_shift": self.pitch_shift,
            "time_stretch": self.time_stretch,
            "inverted_time_stretch": self.inverted_time_stretch,
            "zero_cross_inserts": self.zero_cross_inserts,
            "cut_samples": self.cut_samples,
            "flip_samples": self.flip_samples,
            "wavelet_denoise": self.wavelet_denoise,
            "replacement_attack": self.replacement_attack,
            "vae_wm_attack": self.vae_wm_attack,
            "ddim_attack": self.ddim_attack
        }

    def run(
        self,
        filepaths,
        model_name,
        watermark_data=None,
        attack_types=None,
        sampling_rate=16000,
        verbose=True,
        **kwargs,
    ):
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
            if verbose:
                print(f"Processing file: {filepath}")
            results[filepath] = {}

            if watermark_data is None:
                watermark_data = np.random.randint(
                    0,
                    2,
                    size=40 if model_name == "SilentCipher" else 16,
                    dtype=np.int32,
                )

            audio, sampling_rate = load_audio(filepath, target_sr=sampling_rate)

            watermarked_audio = self.wrapper.embed(
                model_name, audio, watermark_data, sampling_rate
            )

            for attack in attack_types:
                if attack not in self.attacks:
                    print(f"Attack '{attack}' not found. Skipping.")
                    continue
                if verbose:
                    print(f"Applying attack: {attack}")
                attack_kwargs = {**kwargs}
                attack_kwargs["model_name"] = model_name
                attack_kwargs["watermark_data"] = watermark_data
                attack_kwargs["sampling_rate"] = sampling_rate
                attack_kwargs["filepath"] = filepath
                attacked_audio = self.attacks[attack](
                    watermarked_audio, **attack_kwargs
                )

                detected_message = self.wrapper.detect(
                    model_name, attacked_audio, sampling_rate=sampling_rate
                )

                accuracy = self.compare_watermarks(watermark_data, detected_message)
                results[filepath][attack] = {
                    "accuracy": accuracy,
                    "snr": "N/A"
                    if np.abs(len(audio) - len(attacked_audio) > 1)
                    else snr(audio, attacked_audio),
                }

        return results

    def compute_mean_accuracy(self, results):
        """
        Compute the mean accuracy for each attack across all files.

        Args:
            results (dict): Dictionary where each key is a filepath, and the value is another dictionary
                            containing attack results with accuracy and other metrics.

        Returns:
            dict: A dictionary with attacks as keys and their mean accuracy as values.
        """
        attack_accuracies = {}

        for _, attacks in results.items():
            for attack, metrics in attacks.items():
                if attack not in attack_accuracies:
                    attack_accuracies[attack] = []

                attack_accuracies[attack].append(metrics["accuracy"])

        mean_accuracies = {
            attack: np.mean(accuracies)
            for attack, accuracies in attack_accuracies.items()
        }

        return mean_accuracies

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

    def same_model_watermarking(self, audio, **kwargs):
        """
        Perform multiple watermarking using the same model repeatedly.

        Args:
            audio (np.ndarray): The input audio signal to watermark.
            **kwargs: Additional parameters for the watermarking process.
                - model_name (str): The name of the watermarking model to use.
                - sampling_rate (int): The sampling rate of the audio signal.

        Returns:
            np.ndarray: The watermarked audio signal.
        """
        model_name = kwargs.get("model_name", None)
        input_sr = kwargs.get("sampling_rate", None)

        if model_name is None:
            raise ValueError("A model name must be specified for same_model_watermarking.")

        return self.wrapper.embed(model_name, audio, input_sr=input_sr)


    def cross_model_watermarking(self, audio, **kwargs):
        """
        Perform multiple watermarking by embedding a watermark using a randomly chosen
        model other than the specified one.

        Args:
            audio (np.ndarray): The input audio signal to watermark.
            **kwargs: Additional parameters for the watermarking process.
                - model_name (str): The name of the current watermarking model to avoid.
                - sampling_rate (int): The sampling rate of the audio signal.

        Returns:
            np.ndarray: The watermarked audio signal.

        Raises:
            ValueError: If no other models are available for cross-model watermarking.
        """
        model_name = kwargs.get("model_name", None)
        input_sr = kwargs.get("sampling_rate", None)

        filtered_list = [
            key for key in self.wrapper.models.keys() if key != model_name
        ]

        if not filtered_list:
            raise ValueError("No other models are available for cross_model_watermarking.")

        new_model_name = random.choice(filtered_list)

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
        model_name = kwargs.get("model_name", None)
        filepath = kwargs.get("filepath", None)

        if model_name is None or filepath is None:
            raise ValueError(
                "Both 'model_name' and 'filepath' must be provided for the collusion attack."
            )

        second_audio = self.wrapper.embed(model_name, filepath)

        segment_size = kwargs.get("collusion_size", 25)
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
        sampling_rate = kwargs.get("sampling_rate", None)
        cents = kwargs.get("cents", None)

        if sampling_rate is None or cents is None:
            raise ValueError(
                "Both 'sampling_rate' and 'cents' must be provided in kwargs."
            )

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
        sampling_rate = kwargs.get("sampling_rate", None)
        stretch_rate = kwargs.get("stretch_rate", None)

        if sampling_rate is None or stretch_rate is None:
            raise ValueError(
                "Both 'sampling_rate' and 'stretch_rate' must be provided in kwargs."
            )

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
        stretch_rate = args.get("inverted_stretch_rate")
        args.pop("stretch_rate")
        if stretch_rate is None:
            raise ValueError("'inverted_stretch_rate' must be provided in kwargs.")
        audio = self.time_stretch(audio, stretch_rate=stretch_rate, **args)
        return self.time_stretch(audio, stretch_rate=1 / stretch_rate, **args)

    def zero_cross_inserts(self, audio, **kwargs):
        """
        Perform Zero-Cross-Inserts on an audio signal.

        Args:
            audio (np.ndarray): Input audio signal.
            **kwargs: Additional parameters for the operation.
                - sampling_rate (int): Sampling rate of the audio in Hz (required).
                - zero_cross_pause_length (int): Number of zeros to insert at each zero-crossing point. Default is 20.
                - zero_cross_min_distance (float): Minimum distance between pauses in seconds. Default is 1.0.

        Returns:
            np.ndarray: Audio signal with inserted pauses at zero-crossing points.

        Raises:
            ValueError: If 'sampling_rate' is not provided in kwargs.
        """

        sampling_rate = kwargs.get("sampling_rate")
        pause_length = kwargs.get("zero_cross_pause_length", 20)
        min_distance = kwargs.get("zero_cross_min_distance", 1.0)

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

    def cut_samples(self, audio, **kwargs):
        """
        Perform a "Cut Samples" attack by randomly deleting short sequences of samples
        while maintaining inaudibility constraints.

        Args:
            audio (np.ndarray): Input audio signal.
            **kwargs: Additional parameters.
                - sampling_rate (int): Sampling rate of the audio in Hz (required).
                - cut_max_sequence_length (int): Maximum length of each cut sequence. Default is 50 samples.
                - cut_num_sequences (int): Number of sequences to cut in the specified duration. Default is 20.
                - cut_duration (float): Duration (in seconds) over which cuts should occur. Default is 0.5 seconds.
                - cut_max_value_difference (float): Maximum allowed difference between start and end sample of a cut. Default is 0.1.

        Returns:
            np.ndarray: Audio signal with random samples cut.

        Raises:
            ValueError: If 'sampling_rate' is not provided in kwargs.
        """
        sampling_rate = kwargs.get("sampling_rate", None)
        max_sequence_length = kwargs.get("cut_max_sequence_length", 50)
        num_sequences = kwargs.get("cut_num_sequences", 20)
        duration = kwargs.get("cut_duration", 0.5)
        max_value_difference = kwargs.get("cut_max_value_difference", 0.1)

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

    def flip_samples(self, audio, **kwargs):
        """
        Perform a "Flip Samples" attack by randomly exchanging the positions
        of selected samples in the audio signal.

        Args:
            audio (np.ndarray): Input audio signal.
            **kwargs: Additional parameters.
                - sampling_rate (int): Sampling rate of the audio in Hz (required).
                - num_flips (int): Number of sample pairs to flip in the specified duration. Default is 20.
                - flip_duration (float): Duration (in seconds) over which flips should occur. Default is 0.5 seconds.

        Returns:
            np.ndarray: Audio signal with flipped sample positions.

        Raises:
            ValueError: If 'sampling_rate' is not provided in kwargs.
        """
        sampling_rate = kwargs.get("sampling_rate", None)
        num_flips = kwargs.get("num_flips", 20)
        duration = kwargs.get("flip_duration", 0.50)

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

    def wavelet_denoise(self, audio, **kwargs):
        """
        Perform wavelet-based denoising on an audio signal.

        Args:
            audio (np.ndarray): Input audio signal.
            **kwargs: Additional parameters for the wavelet denoising.
                - wavelet (str): Wavelet type (e.g., 'db1', 'sym5'). Default is 'db1'.
                - wt_mode (str): Thresholding mode ('soft' or 'hard'). Default is 'soft'.

        Returns:
            np.ndarray: The denoised audio signal.
        """
        wavelet = kwargs.get("wavelet", "db1")
        mode = kwargs.get("wt_mode", "soft")

        threshold = compute_threshold(audio, wavelet)

        coeffs = pywt.wavedec(audio, wavelet)
        coeffs_denoised = [pywt.threshold(c, threshold, mode=mode) for c in coeffs]

        denoised_audio = pywt.waverec(coeffs_denoised, wavelet)

        return denoised_audio

    def replacement_attack(self, audio, **kwargs):
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
                - replacement_upper_bound (float): The upper bound of the similarity distance for considering a block as a candidate (default: 0).
                - replacement_k (int): Maximum number of similar blocks to consider (default: 20).
                - replacement_use_masking (bool): Whether to use psychoacoustic masking for distance calculation (default: False).

        Returns:
            np.ndarray: The processed audio signal with the replacement attack applied.

        Raises:
            ValueError: If the `sampling_rate` is not provided in `kwargs`.

        """
        sampling_rate = kwargs.get("sampling_rate", None)
        block_size = kwargs.get("replacement_block_size", 1024)
        overlap_factor = kwargs.get("replacement_overlap_factor", 0.75)
        lower_bound = kwargs.get("replacement_lower_bound", 0)
        upper_bound = kwargs.get("replacement_upper_bound", 10)
        use_masking = kwargs.get("replacement_use_masking", False)
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

    def vae_wm_attack(self, audio, **kwargs):
        """
        Applies a VAE-based watermarking attack on the given audio signal.
        
        Args:
            audio (np.ndarray): The input audio signal.
            **kwargs: Additional parameters.
                - sampling_rate (int): The original sampling rate of the audio (required).

        Returns:
            np.ndarray: The attacked audio signal.

        Raises:
            ValueError: If 'sampling_rate' is not provided in kwargs.
        """
        sampling_rate = kwargs.get("sampling_rate", None)
        if sampling_rate is None:
            raise ValueError("'sampling_rate' must be provided in kwargs.")
        audio = np.squeeze(audio)

        block_size = 2048
        original_length = len(audio)
        new_length = (original_length // block_size) * block_size
        audio = audio[:new_length]

        audio = resample_audio(audio, sampling_rate, target_sr=48000)

        waveform_tensor = torch.from_numpy(audio).float()

        model = 'voice_vctk_b2048_r44100_z22.ts'
        device = "cuda" if torch.cuda.is_available() else "cpu"
        attacker = VAEWMAttacker(model=model, device=device)
        attacked = attacker.attack(waveform_tensor)

        return attacked
    
    def ddim_attack(self, audio, **kwargs):
        attacker = AudioDDIMAttacker()
        sampling_rate = kwargs.get("sampling_rate", None)
        if sampling_rate is None:
            raise ValueError("'sampling_rate' must be provided in kwargs.")
        return attacker.attack(audio, sampling_rate)