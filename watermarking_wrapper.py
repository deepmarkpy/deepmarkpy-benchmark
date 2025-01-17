import torch
import librosa
import numpy as np

from audioseal import AudioSeal
import wavmark
import silentcipher

class WatermarkingWrapper:
    """
    A unified wrapper class to embed and detect watermarks in audio files
    using AudioSeal, WavMark, and SilentCipher models.
    """

    def __init__(self):
        """
        Initialize the watermarking wrapper.
        Preloads AudioSeal (generator & detector), WavMark, and SilentCipher models.
        """
        self.models = {}
        self.build_models()

    def build_models(self):
        """
        Load and initialize all the models.

        Raises:
            RuntimeError: If any model fails to load.
        """
        try:
            print("Loading AudioSeal models...")
            self.models["AudioSeal"] = {
                "generator": AudioSeal.load_generator("audioseal_wm_16bits"),
                "detector": AudioSeal.load_detector("audioseal_detector_16bits")
            }

            print("Loading WavMark model...")
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.models["WavMark"] = wavmark.load_model().to(device)

            print("Loading SilentCipher model...")
            self.models["SilentCipher"] = silentcipher.get_model(model_type='44.1k', device='cuda')

            print("All models loaded successfully!")
        except Exception as e:
            raise RuntimeError(f"Error loading models: {e}")

    def embed(self, model_name, audio_path, watermark_data, output_path=None):
        """
        Embed a watermark into an audio file using the specified model.

        Args:
            model_name (str): The name of the model to use ('AudioSeal', 'WavMark', or 'SilentCipher').
            audio_path (str): Path to the input audio file.
            watermark_data (np.ndarray): The binary watermark data to embed (e.g., a NumPy array of 0 and 1).
            output_path (str, optional): Path to save the watermarked audio file. Currently unused in this implementation.

        Returns:
            np.ndarray: The watermarked audio as a NumPy array.

        Raises:
            ValueError: If the model_name is invalid.
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Choose from {list(self.models.keys())}.")

        sr = 44100 if model_name == 'SilentCipher' else 16000
        y, sr = self.load_audio(audio_path, sr)
        model = self.models[model_name]
        if model_name == "AudioSeal":
            model = model["generator"]
            wav = torch.tensor(y, dtype=torch.float32)
            wav = wav.unsqueeze(0).unsqueeze(0)
            msg = torch.from_numpy(watermark_data).unsqueeze(0)
            watermark = model.get_watermark(wav, message=msg, sample_rate=sr)
            watermarked_audio = wav + watermark
            watermarked_audio = watermarked_audio.detach().numpy()
            return watermarked_audio
        elif model_name == 'WavMark':
            watermarked_audio, _ = wavmark.encode_watermark(model, y, watermark_data, show_progress=False)
            return watermarked_audio
        else:
            watermark_data = np.split(watermark_data, len(watermark_data) // 8)
            watermark_data = [int("".join(map(str, arr)), 2) for arr in watermark_data]
            watermarked_audio, _ = model.encode_wav(y, sr, watermark_data)
            return watermarked_audio

    def detect(self, model_name, watermarked_audio, sampling_rate):
        """
        Detect a watermark from a watermarked audio signal.

        Args:
            model_name (str): The name of the model to use ('AudioSeal', 'WavMark', or 'SilentCipher').
            watermarked_audio (np.ndarray): The watermarked audio signal.
            sampling_rate (int): The sampling rate of the audio signal.

        Returns:
            np.ndarray: The extracted binary watermark as a NumPy array.

        Raises:
            ValueError: If the model_name is invalid.
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Choose from {list(self.models.keys())}.")
        model = self.models[model_name]
        if model_name == "AudioSeal":
            model = model["detector"]
            watermarked_audio = torch.tensor(watermarked_audio, dtype=torch.float32)
            _, message = model.detect_watermark(watermarked_audio, sampling_rate)
            message = message.squeeze().cpu().numpy()
            return message
        elif model_name == 'WavMark':
            message, _ = wavmark.decode_watermark(model, watermarked_audio, show_progress=False)
            return message
        else:
            message = model.decode_wav(watermarked_audio, sampling_rate, phase_shift_decoding=False)
            message = message['messages'][0]
            message = [np.array(list(f"{val:08b}"), dtype=np.int32) for val in message]
            message = np.concatenate(message)
            return message

    def load_audio(self, file_path, target_sr=None, mono=True):
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