import logging
import os
import sys
from typing import List

import numpy as np
import torch
import uvicorn
import yaml
from fastapi import FastAPI
from pydantic import BaseModel

from utils.utils import load_config, resample_audio

logger = logging.getLogger(__name__)

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.append("TimbreWatermarking/watermarking_model")

def load_model(process_config, model_config, train_config):
    if model_config["structure"]["transformer"]:
        if model_config["structure"]["mel"]:
            from wm_model.mel_modules import Encoder, Decoder
        else:
            from wm_model.modules import Encoder, Decoder
    elif model_config["structure"].get("conv2", False):
        from wm_model.conv2_modules import Encoder, Decoder
    elif model_config["structure"].get("conv2mel", False):
        if not model_config["structure"].get("ab", False):
            from wm_model.conv2_mel_modules import Encoder, Decoder
        else:
            from wm_model.conv2_mel_modules_ab import Encoder, Decoder
    else:
        from wm_model.conv_modules import Encoder, Decoder
    win_dim = process_config["audio"]["win_len"]
    embedding_dim = model_config["dim"]["embedding"]
    nlayers_encoder = model_config["layer"]["nlayers_encoder"]
    nlayers_decoder = model_config["layer"]["nlayers_decoder"]
    attention_heads_encoder = model_config["layer"]["attention_heads_encoder"]
    attention_heads_decoder = model_config["layer"]["attention_heads_decoder"]
    msg_length = train_config["watermark"]["length"]
    if model_config["structure"].get("mel", False) or model_config["structure"].get("conv2", False):
        encoder = Encoder(process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_encoder=nlayers_encoder, attention_heads=attention_heads_encoder).to(device)
        decoder = Decoder(process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_decoder=nlayers_decoder, attention_heads=attention_heads_decoder).to(device)
    else:
        encoder = Encoder(model_config, msg_length, win_dim, embedding_dim, nlayers_encoder=nlayers_encoder, attention_heads=attention_heads_encoder).to(device)
        decoder = Decoder(model_config, msg_length, win_dim, embedding_dim, nlayers_decoder=nlayers_decoder, attention_heads=attention_heads_decoder).to(device)
    path_model = model_config["test"]["model_path"]
    model_name = model_config["test"]["model_name"]
    if model_name:
        model = torch.load(os.path.join(path_model, model_name), map_location=device)
    else:
        index = model_config["test"]["index"]
        model_list = os.listdir(path_model)
        model_list = sorted(model_list, key=lambda x: os.path.getmtime(os.path.join(path_model, x)))
        model_path = os.path.join(path_model, model_list[index])
        model = torch.load(model_path, map_location=device)
    encoder.load_state_dict(model["encoder"])
    decoder.load_state_dict(model["decoder"], strict=False)
    encoder.eval()
    decoder.eval()
    return encoder, decoder, msg_length


process_config = yaml.load(open("TimbreWatermarking/watermarking_model/config/process.yaml", "r"), Loader=yaml.FullLoader)
model_config = yaml.load(open("TimbreWatermarking/watermarking_model/config/model.yaml", "r"), Loader=yaml.FullLoader)
train_config = yaml.load(open("TimbreWatermarking/watermarking_model/config/train.yaml", "r"), Loader=yaml.FullLoader)
embedder, detector, msg_length = load_model(process_config, model_config, train_config)

try:
    config = load_config("config.json")
except (FileNotFoundError, ValueError, IOError) as e:
    logger.critical(f"Failed to load configuration: {e}. Application cannot start.")
    sys.exit(1)

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

    wav = torch.tensor(audio, dtype=torch.float32)
    wav = wav.unsqueeze(0).unsqueeze(0).to(device)
    msg = torch.from_numpy(watermark_data).unsqueeze(0).unsqueeze(0).to(device)
    msg = msg * 2 - 1

    with torch.no_grad():
        watermarked_audio, _ = embedder.test_forward(wav, msg)

    watermarked_audio = watermarked_audio.squeeze().cpu().numpy()
    
    if sampling_rate != config["sampling_rate"]:
        watermarked_audio = resample_audio(watermarked_audio, config["sampling_rate"], sampling_rate)

    return {"watermarked_audio": watermarked_audio.tolist()}


@app.post("/detect")
async def detect(request: DetectRequest):
    """Detect a watermark from an audio file."""
    audio = np.array(request.audio)
    sampling_rate = request.sampling_rate
    if sampling_rate != config["sampling_rate"]:
        audio = resample_audio(request.audio, sampling_rate, config["sampling_rate"])

    wav = torch.tensor(audio, dtype=torch.float32)
    wav = wav.unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        message = detector.test_forward(wav)

    message = (message + 1) / 2
    message = message.squeeze().cpu().numpy()
    return {"watermark": message if message is None else message.tolist()}

if __name__ == "__main__":
    # Use the default as a fallback if TIMBREWM_PORT is not set in the environment
    app_port = int(os.getenv("TIMBREWM_PORT", 59001))
    host = os.environ.get("HOST", "0.0.0.0")

    logger.info(f"Starting server on port {app_port}")
    uvicorn.run(app, host={host}, port={app_port})