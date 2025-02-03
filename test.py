from src.plugins.attacks.replacement.attack import ReplacementAttack
from src.plugins.attacks.pitch_shift.attack import PitchShiftAttack
from src.plugins.models.wavmark.model import WavMarkModel
from src.plugins.models.silent_cipher.model import SilentCipherModel
from src.plugins.models.audio_seal.model import AudioSealModel
from src.utils.utils import load_audio
import numpy as np
from src.plugins.attacks.additive_noise.attack import AdditiveNoiseAttack
from src.plugins.attacks.time_stretch.attack import TimeStretchAttack
from src.plugins.attacks.inverted_time_stretch.attack import InvertedTimeStretch
from src.plugins.attacks.zero_cross_inserts.attack import ZeroCrossInsertsAttack
from src.plugins.attacks.cut_samples.attack import CutSamplesAttack
from src.plugins.attacks.flip_samples.attack import FlipSamplesAttack
from src.plugins.attacks.wavelet.attack import WaveletAttack

from src.plugins.attacks.vae.attack import VAEAttack

from src.plugins.attacks.diffusion.attack import DiffusionAttack

model = AudioSealModel()

attack = DiffusionAttack()

audio, sr = load_audio("test.wav")

watermark_data = np.random.randint(
                    0,
                    2,
                    size=16,
                    dtype=np.int32,
                )

watermarked_audio = model.embed(audio, watermark_data, sr)

watermarked_audio = attack.apply(watermarked_audio, sampling_rate=sr)

watermark = model.detect(watermarked_audio, sr)

print(watermark)

