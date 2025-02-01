from src.plugins.models.wavmark.model import WavMarkModel
from src.plugins.models.silent_cipher.model import SilentCipherModel
from src.plugins.models.audio_seal.model import AudioSealModel
from src.utils.utils import load_audio
import numpy as np
from src.plugins.attacks.additive_noise.attack import AdditiveNoiseAttack

model = AudioSealModel()

attack = AdditiveNoiseAttack()

audio, sr = load_audio("test.wav")

watermark_data = np.random.randint(
                    0,
                    2,
                    size=16,
                    dtype=np.int32,
                )

watermarked_audio = model.embed(audio, watermark_data, sr)

watermarked_audio = attack.apply(watermarked_audio)

watermark = model.detect(watermarked_audio, sr)

print(watermark)

print("===============================================")

watermark_data = np.random.randint(
                    0,
                    2,
                    size=40,
                    dtype=np.int32,
                )

model = SilentCipherModel()

watermarked_audio = model.embed(audio, watermark_data, sr)

print(watermarked_audio.shape)

watermarked_audio = attack.apply(audio)

watermark = model.detect(watermarked_audio, sr)

print(watermark)

print("===============================================")

watermark_data = np.random.randint(
                    0,
                    2,
                    size=16,
                    dtype=np.int32,
                )

model = WavMarkModel()

watermarked_audio = model.embed(audio, watermark_data, sr)

watermarked_audio = attack.apply(watermarked_audio)

watermark = model.detect(watermarked_audio, sr)

print(watermark)