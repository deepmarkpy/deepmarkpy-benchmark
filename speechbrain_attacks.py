import numpy as np
import  librosa
import torch 
from speechbrain.inference.enhancement import SpectralMaskEnhancement, WaveformEnhancement

class SpeechBrainAttacker:
    def __init__(self, type="waveform"):
        assert type=="waveform" or type=="spectral_mask", "type must be either 'waveform' or 'spectral_mask'."

        if type=="waveform":
            self.model = WaveformEnhancement.from_hparams(source="speechbrain/mtl-mimic-voicebank", savedir='models/sepformer-wham16k-waveform-enhancement')
        else:
            self.model = SpectralMaskEnhancement.from_hparams(source="speechbrain/metricgan-plus-voicebank", savedir='models/sepformer-wham16k-spectralmask-enhancement')
        
    def attack(self, audio, sample_rate, noise_strength=0.01):
        assert abs(noise_strength) <= 0.01, "noise_strength should not be greater than 0.01."
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        noisy = audio+noise_strength*np.random.normal(0, 1, size=(len(audio)))
        noisy = np.expand_dims(noisy, axis=[0])
        noisy = torch.FloatTensor(noisy)
        lengths = torch.FloatTensor([1.0])
        with torch.no_grad():
            enhanced = self.model.enhance_batch(noisy, lengths=lengths)
            enhanced = enhanced.squeeze().detach().numpy()

        enhanced = librosa.resample(enhanced, orig_sr=16000, target_sr=sample_rate)
        return enhanced

if __name__ == "__main__":
    x, sr = librosa.load("1.wav", mono=True, sr=None)
    attacker = SpeechBrainAttacker(type="waveform")
    y = attacker.attack(x, sr)

    from scipy.io.wavfile import write

    if np.issubdtype(y.dtype, np.floating):
        y = np.int16(y / np.max(np.abs(y)) * 32767)

    write("output_enhancer.wav", sr, y)