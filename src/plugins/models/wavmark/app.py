import json
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import wavmark
import uvicorn
import torch

from utils.utils import resample_audio

app = FastAPI()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = wavmark.load_model().to(device)

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

    watermarked_audio, _ = wavmark.encode_watermark(model, audio, watermark_data, show_progress=False)  
    
    if sampling_rate != config["sampling_rate"]:
        watermarked_audio = resample_audio(watermarked_audio, config["sampling_rate"], sampling_rate)
    return {"watermarked_audio": watermarked_audio.tolist()}


@app.post("/detect")
async def detect(request: DetectRequest):
    """Detect a watermark from an audio file."""
    audio = np.array(request.audio)
    sampling_rate = request.sampling_rate
    if sampling_rate != config["sampling_rate"]:
        audio = resample_audio(audio, sampling_rate, config["sampling_rate"])
    message, _ = wavmark.decode_watermark(model, audio, show_progress=False)
    return {"watermark": message if message is None else message.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config["port"])