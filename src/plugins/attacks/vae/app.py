import json
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
from vae import VAE
import torch
import numpy as np
import uvicorn
from vae import VAE
from utils.utils import resample_audio

app = FastAPI()

with open('config.json') as json_file:
    config = json.load(json_file)

model_name = config["model_name"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VAE(model_name, device)

class AttackRequest(BaseModel):
    audio: List[float]
    sampling_rate: int

@app.post("/attack")
async def attack(request: AttackRequest):
    """
        Applies a VAE-based watermarking attack on the given audio signal.
        
        Args:
            audio (np.ndarray): The input audio signal.
            **kwargs: Additional parameters.
                - sampling_rate (int): The original sampling rate of the audio (required).

        Returns:
            np.ndarray: The attacked audio signal.
    """
    sampling_rate = request.sampling_rate
    audio = np.array(request.audio)
    audio = np.squeeze(audio)

    block_size = 2048
    original_length = len(audio)
    new_length = (original_length // block_size) * block_size
    audio = audio[:new_length]

    audio = resample_audio(audio, sampling_rate, target_sr=48000)

    waveform_tensor = torch.from_numpy(audio).float()

    attacked = model.inference(waveform_tensor)

    return {"audio": audio.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config["port"])
