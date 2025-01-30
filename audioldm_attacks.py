"""
To apply these attacks, you must run:
pip3 install git+https://github.com/haoheliu/AudioLDM.git
"""
import torch
import librosa
import numpy as np

from audioldm.utils import default_audioldm_config
from audioldm.audio.stft import TacotronSTFT
from audioldm.audio.tools import normalize_wav, pad_wav, get_mel_from_wav
from audioldm.pipeline import build_model


class AudioLDMAttacker:
    def __init__(self):
        self.model = build_model("models/audioLDM")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        self.model.eval().to(self.device)

    def prepare_input(self, audio, sampling_rate):
        config = default_audioldm_config()

        fn_STFT = TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
        audio = normalize_wav(audio)
        audio = audio[None, ...]
        audio = pad_wav(audio, None)
        
        audio = audio / np.max(np.abs(audio))
        audio = 0.5 * audio
        
        audio = audio[0, ...]
        audio = torch.FloatTensor(audio)

        mel, _, _ = get_mel_from_wav(audio, fn_STFT)

        mel = torch.FloatTensor(mel.T)

        return mel

    def vae_attack(self, audio, sampling_rate, scaling_factor=1e-4):
        
        assert scaling_factor <= 1e-4, "should be at most 1e-4."

        mel = self.prepare_input(audio, sampling_rate)
        mel = mel.unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            latent_dist = self.model.encode_first_stage(mel) 

        latent_sample = latent_dist.mean + scaling_factor * latent_dist.std * torch.randn(latent_dist.mean.shape, device=self.device, dtype=latent_dist.mean.dtype)

        if(torch.max(torch.abs(latent_sample)) > 1e2):
            latent_sample = torch.clip(latent_sample, min=-10, max=10)

        with torch.no_grad():
            mel_decoded = self.model.decode_first_stage(latent_sample[:,:,:-3,:])
            output = self.model.first_stage_model.decode_to_waveform(mel_decoded).squeeze(0)
        
        output = np.float32(output/2**16)
        output = librosa.resample(output, orig_sr=16000, target_sr=sr)
        
        return output

    def vocoder_attack(self, audio, sampling_rate, model=None):

        mel = self.prepare_input(audio, sampling_rate)
        mel = mel.unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model.mel_spectrogram_to_waveform(mel)

        output = np.float32(output/2**16)
        output = librosa.resample(output, orig_sr=16000, target_sr=sr)

        return output

if __name__ == "__main__":
    x, sr = librosa.load("1.wav", sr=None)
    attacker = AudioLDMAttacker()
    y = attacker.vocoder_attack(x, sr) # y = attacker.vae_attack(x, sr, scaling_factor=1e-5)
    
    from scipy.io.wavfile import write

    if np.issubdtype(y.dtype, np.floating):
        y = np.int16(y / np.max(np.abs(y)) * 32767)

    write("output_vocoder.wav", sr, y)