import numpy as np

from utils import load_audio
from .plugin_manager import discover_plugins

class Benchmark:
    def __init__(self, plugin_folder="src/plugins"):
        """
        Discover all attacker and watermark model plugins.
        """
        self.attack_plugins, self.watermark_plugins = discover_plugins(plugin_folder)

    def run(
        self,
        audio_files,
        watermark_model_name,
        attacks=None,
        watermark_data=None,
        sampling_rate=16000,
    ):
        """
        Run the watermarking and attacks on the provided audio files.
        """
        if watermark_model_name not in self.watermark_plugins:
            raise ValueError(f"No watermark model found for {watermark_model_name}")

        model = self.watermark_plugins[watermark_model_name]
        if attacks is None:
            # If no attacks specified, run all
            attacks = list(self.attack_plugins.keys())

        # For storing results
        results = {}

        for audio_file in audio_files:
            # Load your audio file
            # or do whatever is needed
            y, sr = load_audio(audio_file, sampling_rate)
            # Possibly create random watermark_data if not provided
            if watermark_data is None:
                # e.g. 16 bits
                watermark_data = np.random.randint(0, 2, size=16, dtype=np.int32)

            # 1) Embed watermark
            watermarked_audio = model.embed(y, sr, watermark_data)

            file_result = {}
            for attack_name in attacks:
                if attack_name not in self.attack_plugins:
                    print(f"Attack plugin '{attack_name}' not found, skipping.")
                    continue
                attacker = self.attack_plugins[attack_name]

                # 2) Attack
                attacked_audio = attacker.attack(watermarked_audio, sr)
                # 3) Detect
                detected_bits = model.detect(attacked_audio, sr)
                # 4) Evaluate accuracy
                accuracy = self._compare_watermarks(watermark_data, detected_bits)
                file_result[attack_name] = {"accuracy": accuracy}

            results[audio_file] = file_result

        return results

    def _compare_watermarks(self, original, detected):
        if detected is None or len(detected) != len(original):
            return 0.0
        matches = np.sum(original == detected)
        return 100.0 * matches / len(original)