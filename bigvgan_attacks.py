"""
Installation:
git clone https://github.com/NVIDIA/BigVGAN
cd BigVGAN
pip install -r requirements.txt
"""
import sys
import torch
import librosa
import numpy as np

sys.path.append("BigVGAN")
import bigvgan
from meldataset import get_mel_spectrogram

class BigVGANAttacker:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_44khz_128band_512x')
        self.model.remove_weight_norm()
        self.model.eval().to(self.device)

    def attack(self, audio, sample_rate):
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=44100)
        audio = torch.FloatTensor(audio).unsqueeze(0)

        mel = get_mel_spectrogram(audio, self.model.h).to(self.device) 
        
        with torch.inference_mode():
            output = self.model(mel) 

        output = output.squeeze().cpu().numpy() 
        
        output = librosa.resample(output, orig_sr=44100, target_sr=sample_rate)

        return output


if __name__ == "__main__":
    audio, sr = librosa.load("1.wav", sr=None, mono=True) 
    attacker = BigVGANAttacker()
    y = attacker.attack(audio, sr)

    from scipy.io.wavfile import write

    if np.issubdtype(y.dtype, np.floating):
        y = np.int16(y / np.max(np.abs(y)) * 32767)

    write("output_vgan.wav", sr, y)