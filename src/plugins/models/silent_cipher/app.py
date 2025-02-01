import json
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import silentcipher
import uvicorn
import torch
import logging

from utils.utils import resample_audio

logger = logging.getLogger(__name__)

app = FastAPI()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = silentcipher.get_model(model_type='44.1k', device=device)

with open('config.json') as json_file:
    config = json.load(json_file)

class EmbedRequest(BaseModel):
    audio: List[float]
    watermark_data: List[int]
    sampling_rate: int

class DetectRequest(BaseModel):
    audio: List[float]
    sampling_rate: int

@app.post("/embed")
async def embed(request: EmbedRequest):
    """Embed a watermark in an audio file."""
    audio = np.array(request.audio)
    watermark_data = np.array(request.watermark_data)
    sampling_rate = request.sampling_rate
    if sampling_rate != config["sampling_rate"]:
        audio = resample_audio(request.audio, sampling_rate, config["sampling_rate"])

    watermark_data = np.split(watermark_data, len(watermark_data) // 8)
    watermark_data = [int("".join(map(str, arr)), 2) for arr in watermark_data]
    watermarked_audio, _ = model.encode_wav(audio, config["sampling_rate"], watermark_data)

    if sampling_rate != config["sampling_rate"]:
        watermarked_audio = resample_audio(watermarked_audio, sampling_rate, config["sampling_rate"])

    return {"watermarked_audio": watermarked_audio.tolist()}


@app.post("/detect")
async def detect(request: DetectRequest):
    """Detect a watermark from an audio file."""
    audio = np.array(request.audio)
    sampling_rate = request.sampling_rate

    message = model.decode_wav(audio, sampling_rate, phase_shift_decoding=True)
    try:
        message = message['messages'][0]
        message = [np.array(list(f"{val:08b}"), dtype=np.int32) for val in message]
        message = np.concatenate(message)
        message = message.toList()
    except:  # noqa: E722
        message = None
    
    return {"watermark": message}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config["port"])