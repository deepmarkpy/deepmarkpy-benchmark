from src.plugins.attacks.pitch_shift.attack import PitchShiftAttack
from src.plugins.models.wavmark.model import WavMarkModel
from src.plugins.models.silent_cipher.model import SilentCipherModel
from src.plugins.models.audio_seal.model import AudioSealModel
from src.utils.utils import load_audio
import numpy as np
from src.plugins.attacks.additive_noise.attack import AdditiveNoiseAttack

model = AudioSealModel()

attack = PitchShiftAttack()

audio, sr = load_audio("test.wav")

watermark_data = np.random.randint(
                    0,
                    2,
                    size=16,
                    dtype=np.int32,
                )

watermarked_audio = model.embed(audio, watermark_data, sr)

watermarked_audio = attack.apply(watermarked_audio, cents=5, sampling_rate=sr)

watermark = model.detect(watermarked_audio, sr)

print(watermark)

