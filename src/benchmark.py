import numpy as np
import inspect
from utils.utils import load_audio, snr
from plugin_manager import PluginManager
import json


class Benchmark:
    """
    A class to perform various attacks on watermarking models and benchmark their performance.
    """

    def __init__(self):
        """
        Initialize Benchmark class with PluginManager.
        """
        self.plugin_manager = PluginManager()
        # Now these are dicts of the form { "class_name": {"class": ActualClass, "config": {...}} }
        self.attacks = self.plugin_manager.get_attacks()
        self.models = self.plugin_manager.get_models()

    def show_available_plugins(self):
        """
        Print out all discovered models and attacks, including any __init__ parameters
        and key-value pairs from config.json (defaults).
        """
        print("===== Available Models =====")
        for model_name, model_entry in self.models.items():
            model_cls = model_entry["class"]
            config = model_entry.get("config") or {}

            # Get the __init__ signature to show possible constructor parameters
            signature = inspect.signature(model_cls.__init__)
            params = [p for p in signature.parameters.values() if p.name != "self"]
            # Convert to {param_name: default_value}
            init_params = {
                p.name: (None if p.default is inspect.Parameter.empty else p.default)
                for p in params
            }

            print(f"\nModel: {model_name}")
            print(f"  - Constructor parameters: {init_params}")

            print("  - Arguments defaults:")
            if config:
                for key, val in config.items():
                    print(f"    {key}: {val}")
            else:
                print("    (none found)")

        print("\n===== Available Attacks =====")
        for attack_name, attack_entry in self.attacks.items():
            attack_cls = attack_entry["class"]
            config = attack_entry.get("config") or {}

            # Get the __init__ signature
            signature = inspect.signature(attack_cls.__init__)
            params = [p for p in signature.parameters.values() if p.name != "self"]
            init_params = {
                p.name: (None if p.default is inspect.Parameter.empty else p.default)
                for p in params
            }

            print(f"\nAttack: {attack_name}")
            print(f"  - Constructor parameters: {init_params}")

            print("  - Argument defaults:")
            if config:
                for key, val in config.items():
                    print(f"    {key}: {val}")
            else:
                print("    (none found)")

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
            model_name (str): The model to benchmark (e.g., 'AudioSeal', 'WavMark', 'SilentCipher').
            watermark_data (np.ndarray, optional): The binary watermark data to embed. Defaults to random message.
            attack_types (list, optional): A list of attack types to perform. Defaults to all available attacks.
            sampling_rate (int, optional): Target sampling rate for loading audio. Defaults to 16000.
            verbose (bool, optional): Print verbose info. Defaults to True.
            **kwargs: Additional parameters for specific attacks.

        Returns:
            dict: A dictionary containing benchmark results for each file and attack.
        """
        if isinstance(filepaths, str):
            filepaths = [filepaths]

        # If user doesn't specify attacks, use them all
        attack_types = attack_types or list(self.attacks.keys())
        results = {}

        # Make sure the requested model exists
        if model_name not in self.models:
            raise ValueError(
                f"Model '{model_name}' not found. Available: {list(self.models.keys())}"
            )

        model_cls = self.models[model_name]["class"]
        model_instance = model_cls()

        attack_kwargs = {
            **kwargs,
            "model_name": model_name,
            "watermark_data": watermark_data,
            "sampling_rate": sampling_rate,
        }

        for filepath in filepaths:
            if verbose:
                print(f"\nProcessing file: {filepath}")
            results[filepath] = {}

            # If no user-supplied watermark, pick a random message size
            if watermark_data is None:
                message_size = 40 if model_name == "SilentCipher" else 16
                watermark_data = np.random.randint(
                    0, 2, size=message_size, dtype=np.int32
                )

            # Load audio
            audio, sampling_rate = load_audio(filepath, target_sr=sampling_rate)

            # Embed watermark
            watermarked_audio = model_instance.embed(
                audio=audio, watermark_data=watermark_data, sampling_rate=sampling_rate
            )

            # Apply each attack
            for attack_name in attack_types:
                if attack_name not in self.attacks:
                    print(f"Attack '{attack_name}' not found. Skipping.")
                    continue

                if verbose:
                    print(f"  Applying attack: {attack_name}")

                # Create attack instance from plugin data
                attack_instance = self.attacks[attack_name]["class"]()

                # Construct the kwargs for this attack

                # Apply attack
                attacked_audio = attack_instance.apply(
                    watermarked_audio, **attack_kwargs
                )

                # Detect (extract) watermark from attacked audio
                detected_message = model_instance.detect(attacked_audio, sampling_rate)

                # Compute accuracy
                accuracy = self.compare_watermarks(watermark_data, detected_message)

                # Compute SNR if lengths match, else "N/A"
                if abs(len(audio) - len(attacked_audio)) > 1:
                    snr_val = "N/A"
                else:
                    snr_val = snr(audio, attacked_audio)

                results[filepath][attack_name] = {
                    "accuracy": accuracy,
                    "snr": snr_val,
                }

        return json.dumps(results, indent=2)

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

        for _, attack_dict in results.items():
            for attack_name, metrics in attack_dict.items():
                if attack_name not in attack_accuracies:
                    attack_accuracies[attack_name] = []
                attack_accuracies[attack_name].append(metrics["accuracy"])

        # Compute mean accuracy
        mean_accuracies = {
            attack_name: np.mean(accuracies)
            for attack_name, accuracies in attack_accuracies.items()
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
        return (matches / len(original)) * 100
