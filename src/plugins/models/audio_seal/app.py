import json
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from audioseal import AudioSeal
import uvicorn
import torch
import logging

from utils.utils import resample_audio

logger = logging.getLogger(__name__)

app = FastAPI()

model = {
    "generator": AudioSeal.load_generator("audioseal_wm_16bits"),
    "detector": AudioSeal.load_detector("audioseal_detector_16bits"),
}

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

    generator = model["generator"]
    wav = torch.tensor(audio, dtype=torch.float32)
    wav = wav.unsqueeze(0).unsqueeze(0)
    msg = torch.from_numpy(watermark_data).unsqueeze(0)
    watermark = generator.get_watermark(
        wav, message=msg, sample_rate=config["sampling_rate"]
    )
    watermarked_audio = wav + watermark
    watermarked_audio = watermarked_audio.detach().numpy()
    watermarked_audio = np.squeeze(watermarked_audio)

    if sampling_rate != config["sampling_rate"]:
        watermarked_audio = resample_audio(watermarked_audio, sampling_rate, config["sampling_rate"])

    return {"watermarked_audio": watermarked_audio.tolist()}


@app.post("/detect")
async def detect(request: DetectRequest):
    """Detect a watermark from an audio file."""
    audio = np.array(request.audio)
    sampling_rate = request.sampling_rate
    detector = model["detector"]
    watermarked_audio = np.expand_dims(audio, axis=[0, 1])
    watermarked_audio = torch.tensor(watermarked_audio, dtype=torch.float32)
    _, message = detector.detect_watermark(watermarked_audio, sampling_rate)
    message = message.squeeze().cpu().numpy()
    return {"watermark": message.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config["port"])