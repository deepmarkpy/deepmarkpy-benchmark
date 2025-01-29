import os
import torch
from huggingface_hub import hf_hub_download
import shutil


class WMAttacker:
    def attack(self, waveform):
        raise NotImplementedError


class VAEWMAttacker(WMAttacker):
    def __init__(self, model="voice_vctk_b2048_r44100_z22.ts", device="cpu"):
        model_path = os.path.join('models', model)
        repo_id = "Intelligent-Instruments-Lab/rave-models"
        local_model_dir = "models"
        os.makedirs(local_model_dir, exist_ok=True)

        # Get just the filename without any path
        model_filename = os.path.basename(model_path)
        local_model_path = os.path.join(local_model_dir, model_filename)

        if not os.path.exists(local_model_path):
            print(
                f"Model '{model_filename}' not found. Downloading from Hugging Face..."
            )
            downloaded_path = hf_hub_download(repo_id=repo_id, filename=model_filename)
            shutil.copy2(downloaded_path, local_model_path)
            model_path = local_model_path
        else:
            model_path = local_model_path

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
