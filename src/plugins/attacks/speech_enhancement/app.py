import logging
import os
import sys
from typing import List

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from speech_brain import SpeechBrain

from utils.utils import load_config

logger = logging.getLogger(__name__)

app = FastAPI()

try:
    config = load_config("config.json")
except (FileNotFoundError, ValueError, IOError) as e:
    logger.critical(f"Failed to load configuration: {e}. Application cannot start.")
    sys.exit(1)

model = SpeechBrain(config["type"])


class AttackRequest(BaseModel):
    audio: List[float]
    sampling_rate: int
    noise_strength: float


@app.post("/attack")
async def attack(request: AttackRequest):
    sampling_rate = request.sampling_rate
    audio = np.array(request.audio)
    noise_strength = request.noise_strength

    audio = model.inference(audio, sampling_rate, noise_strength)

    return {"audio": audio.tolist()}


if __name__ == "__main__":
    # Use the default as a fallback if SPEECH_ENHANCEMENT_PORT is not set in the environment
    app_port = int(os.getenv("SPEECH_ENHANCEMENT_PORT", 10005))
    host = os.environ.get("HOST", "0.0.0.0")

    logger.info(f"Starting server on port {app_port}")
    uvicorn.run(app, host={host}, port={app_port})
