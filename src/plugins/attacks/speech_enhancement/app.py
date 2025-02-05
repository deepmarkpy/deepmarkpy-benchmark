from typing import List
import numpy as np
from pydantic import BaseModel
import torch
from fastapi import FastAPI
import uvicorn
import json

from speech_brain import SpeechBrain

app = FastAPI()

with open("config.json") as json_file:
    config = json.load(json_file)

type = config["type"]

model = SpeechBrain(type)


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
    uvicorn.run(app, host="0.0.0.0", port=config["port"])
