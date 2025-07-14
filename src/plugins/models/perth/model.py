import logging
import os
import numpy as np

from core.base_model import BaseModel

logging = logging.getLogger(__name__)

class PerthModel(BaseModel):
    def __init__(self):
        super().__init__()

        host = "localhost" 
        port = os.getenv("PERTH_PORT", "7010")
        if not port:
             logging.error("PERTH_PORT environment variable not set.")
             raise ValueError("PERTH_PORT must be set for PerthModel")

        self.base_url = f"http://{host}:{port}"
        logging.info(f"PerthModel initialized. Target API: {self.base_url}")

    def embed(
        self, audio: np.ndarray, watermark_data: np.ndarray, sampling_rate: int
    ) -> np.ndarray:
        """Embeds a watermark using the Perth service."""
        payload = {
            "audio": audio.tolist(),
            "watermark_data": watermark_data.tolist(),
            "sampling_rate": sampling_rate,
        }
        # Use the helper method from BaseModel
        response_data = self._make_request(endpoint="/embed", json_data=payload, method="POST")

        if "watermarked_audio" not in response_data:
             logging.error("'/embed' response did not contain 'watermarked_audio' key.")
             raise KeyError("Missing 'watermarked_audio' in response from /embed")
        return np.array(response_data["watermarked_audio"])
    
    
    def detect(self, audio: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Detects a watermark using the Perth service."""
        payload = {"audio": audio.tolist(), "sampling_rate": sampling_rate}
       
        # Use the helper method from BaseModel
        response_data = self._make_request(endpoint="/detect", json_data=payload, method="POST")

        if "watermark" not in response_data:
             logging.error("'/detect' response did not contain 'watermark' key.")
             raise KeyError("Missing 'watermark' in response from /detect")
        # Handle potential None value from the API
        watermark = response_data["watermark"]
        return np.array(watermark) if watermark is not None else None
    

if __name__ == "__main__":
    import librosa
    import soundfile as sf
    import requests

    # Create a dummy audio file for testing
    input_audio_path = "oh-yeah-everything-is-fine.wav"
    sampling_rate = 44100
    duration_seconds = 3
    if not os.path.exists(input_audio_path):
        logging.info(f"Creating a dummy audio file: {input_audio_path}")
        dummy_audio = np.random.uniform(-0.5, 0.5, int(sampling_rate * duration_seconds)).astype(np.float32)
        sf.write(input_audio_path, dummy_audio, sampling_rate)
    
    # Load the dummy audio
    try:
        audio_data, sr = librosa.load(input_audio_path, sr=None)
        logging.info(f"Loaded audio from {input_audio_path} with sampling rate {sr} and shape {audio_data.shape}")
    except Exception as e:
        logging.error(f"Could not load audio file {input_audio_path}: {e}")
        exit(1)

    # Initialize the client
    try:
        client = PerthModel()
    except ValueError as e:
        logging.critical(f"Failed to initialize PerthModel client: {e}")
        logging.info("Ensure PERTH_PORT environment variable is set or server is on default port 7010.")
        exit(1)

    # --- Test Embedding ---
   
    watermark_to_embed_dummy = np.random.randint(0, 2, size=client.config.get("watermark_size", 10), dtype=np.int32)
    logging.info(f"Dummy watermark data for embedding (PerthImplicitWatermarker uses implicit): {watermark_to_embed_dummy.tolist()}")

    watermarked_audio_output_path = "watermarked_perth_test_audio.wav"
    try:
        logging.info("Attempting to embed watermark via FastAPI server (PerthImplicitWatermarker)...")
        watermarked_audio = client.embed(audio_data, watermark_to_embed_dummy, sr)
        sf.write(watermarked_audio_output_path, watermarked_audio, sr)
        logging.info(f"Watermark embedded. Saved watermarked audio to: {watermarked_audio_output_path}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to connect to FastAPI server for embedding. Is it running on {client.base_url}? Error: {e}")
        exit(1)
    except Exception as e:
        logging.error(f"An error occurred during embedding: {e}")
        exit(1)

    # --- Test Detection ---
    try:
        logging.info("Attempting to detect watermark via FastAPI server (PerthImplicitWatermarker)...")
        # The detect method of PerthModel (client) will call the /detect endpoint
        # The server's /detect endpoint uses model.get_watermark, which returns a single float (0 or 1)
        extracted_watermark_result = client.detect(watermarked_audio, sr)
        logging.info(f"Successfully detected watermark (confidence/presence): {extracted_watermark_result}")

        # Based on the PerthImplicitWatermarker's get_watermark, this will be a single 0 or 1.
        if extracted_watermark_result == 1:
            logging.info("Watermark detected (presence confirmed).")
            print(1)
        elif extracted_watermark_result == 0:
            logging.info("No watermark detected (presence denied).")
            print(0)
        else:
            logging.warning(f"Unexpected detection result: {extracted_watermark_result}")

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to connect to FastAPI server for detection. Is it running on {client.base_url}? Error: {e}")
    except Exception as e:
        logging.error(f"An error occurred during detection: {e}")
