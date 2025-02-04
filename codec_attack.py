"""
Installation:
pip install xcodec2==0.1.3
"""
import torch
import librosa
import numpy as np
from xcodec2.modeling_xcodec2 import XCodec2Model

class XCodecAttacker:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = XCodec2Model.from_pretrained("HKUST-Audio/xcodec2")
        self.model.eval().to(self.device)

    def attack(self, audio, sample_rate):
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        
        audio = torch.from_numpy(audio).float().unsqueeze(0) 

        with torch.no_grad():
            vq_code = self.model.encode_code(input_waveform=audio)
    
            output = self.model.decode_code(vq_code)[0, 0, :].cpu().numpy()  

        output = librosa.resample(output, orig_sr=16000, target_sr=sample_rate)
        return output


if __name__ == "__main__":
    audio, sr = librosa.load("1.wav", sr=None, mono=True) 
    attacker = XCodecAttacker()
    y = attacker.attack(audio, sr)

    from scipy.io.wavfile import write

    if np.issubdtype(y.dtype, np.floating):
        y = np.int16(y / np.max(np.abs(y)) * 32767)

    write("output_codec.wav", sr, y)