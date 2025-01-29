import os
import torch
from tqdm import tqdm
import numpy as np
from huggingface_hub import hf_hub_download

class WMAttacker:
  def attack(self, waveform, out_path):
    raise NotImplementedError

class VAEWMAttacker(WMAttacker):
  def __init__(self, model_path, device='cpu'):
    repo_id = "Intelligent-Instruments-Lab/rave-models"
    if not os.path.exists(model_path):
        print(f"Model '{model_path}' not found. Downloading from Hugging Face...")
        model_path = hf_hub_download(repo_id=repo_id, filename=model_path)

    self.model = torch.jit.load(model_path).eval().to(device)
    self.device = device

  def attack(self, waveform):
    waveform = waveform.to(self.device)
    if waveform.dim() == 1:
      waveform = waveform.unsqueeze(0)
    
    waveform = waveform / waveform.abs().max()

    with torch.no_grad():
      reconstructed = self.model.forward(waveform.unsqueeze(0))
    reconstructed = reconstructed.squeeze().clamp(-1, 1)
    
    return reconstructed.cpu().numpy()