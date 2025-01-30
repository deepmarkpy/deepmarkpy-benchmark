import torch
import librosa
import numpy as np
from diffusers import DiffusionPipeline

class AudioDiffusionAttacker:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
    def attack(self, audio, sampling_rate, model, start_step):
        mel = model.mel
        mel_sample_rate = mel.get_sample_rate()
        slice_size = mel.x_res * mel.hop_length

        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=mel_sample_rate)

        overlap_secs = 2
        overlap_samples = overlap_secs * mel_sample_rate
        stride = min(slice_size - overlap_samples, len(audio))

        generator = torch.Generator(device=self.device)
        seed = generator.seed()

        output = np.array([])
        new_audio_slice = np.array([])
        not_first = 0
        for sample in range(len(audio) // stride):
            generator.manual_seed(seed)
            audio_slice = np.array(audio[sample * stride:sample * stride + slice_size])
            if not_first:
                # Normalize and re-insert generated audio
                audio_slice[:overlap_samples] = new_audio_slice[-overlap_samples:] * np.max(audio_slice[:overlap_samples]) / np.max(new_audio_slice[-overlap_samples:])
            with torch.no_grad():
                slice = model(raw_audio=audio_slice,
                                        start_step=start_step,
                                        generator=generator,
                                        mask_start_secs=overlap_secs * not_first)
                new_audio_slice = slice.audios[0, 0]
                output = np.concatenate([output, new_audio_slice[overlap_samples * not_first:]])
                not_first = 1
        
        output = librosa.resample(output, orig_sr=mel_sample_rate, target_sr=sampling_rate)
        return output 

class AudioDDPMAttacker(AudioDiffusionAttacker):
    def __init__(self):
        super(AudioDDPMAttacker, self).__init__()
        self.model = DiffusionPipeline.from_pretrained("teticio/audio-diffusion-256", savedir="models/teticio")
        
        self.model.to(self.device)

    def attack(self, audio, sampling_rate, steps=150):
        assert steps<=150,  "number of steps is too large."
        return super().attack(audio, sampling_rate, self.model, 1000-steps)

class AudioDDIMAttacker(AudioDiffusionAttacker):
    def __init__(self):
        super(AudioDDIMAttacker, self).__init__()
        self.model = DiffusionPipeline.from_pretrained("teticio/audio-diffusion-ddim-256", savedir="models/teticio")
        self.model.to(self.device)

    def attack(self, audio, sampling_rate, steps=5):
        assert steps <= 5, "number of steps is too large."
        return super().attack(audio, sampling_rate, self.model, 50-steps)


if __name__ == "__main__":
    x, sr = librosa.load("saxophone.wav", mono=True, sr=None)
    attacker = AudioDDIMAttacker()
    y = attacker.attack(x, sr)

    from scipy.io.wavfile import write

    if np.issubdtype(y.dtype, np.floating):
        y = np.int16(y / np.max(np.abs(y)) * 32767)

    write("output_ddim.wav", sr, y)