from typing import List
import numpy as np
from pydantic import BaseModel
import torch
from fastapi import FastAPI
import uvicorn
import json
from xcodec import XCodec

app = FastAPI()

with open("config.json") as json_file:
    config = json.load(json_file)

model_name = config["model_name"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = XCodec(model_name, device)


class AttackRequest(BaseModel):
    audio: List[float]
    sampling_rate: int


@app.post("/attack")
async def attack(request: AttackRequest):
    sampling_rate = request.sampling_rate
    audio = np.array(request.audio)

    audio = model.inference(audio, sampling_rate)

    return {"audio": audio.tolist()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config["port"])
