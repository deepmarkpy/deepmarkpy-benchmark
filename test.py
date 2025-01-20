from watermarking_wrapper import WatermarkingWrapper
import numpy as np
from attacks import Attacks

# wrapper = WatermarkingWrapper()

# watermark = np.random.randint(0, 2, size=16, dtype=np.int32)

# print(watermark)

# watermarked_audio = wrapper.embed(model_name='AudioSeal', audio_path='harvard.wav', watermark_data=watermark)
# message = wrapper.detect(model_name='AudioSeal', watermarked_audio=watermarked_audio, sampling_rate=16000)

# print(message)

# watermarked_audio = wrapper.embed(model_name='WavMark', audio_path='harvard.wav', watermark_data=watermark)
# message = wrapper.detect(model_name='WavMark', watermarked_audio=watermarked_audio, sampling_rate=16000)

# print(message)

# watermark = np.random.randint(0, 2, size=40, dtype=np.int32)

# print(watermark)

# watermarked_audio = wrapper.embed(model_name='SilentCipher', audio_path='harvard.wav', watermark_data=watermark)
# message = wrapper.detect(model_name='SilentCipher', watermarked_audio=watermarked_audio, sampling_rate=44100)

# print(message)

attacks = Attacks()
results = attacks.benchmark(['harvard.wav'], 'SilentCipher', sampling_rate=441000, mwm_type='other', collusion_size=100)
print(results)