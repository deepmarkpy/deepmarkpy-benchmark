from watermarking_wrapper import WatermarkingWrapper
import numpy as np
from attacks import Attacks
import os
import json

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

wav_files = ['data/'+f for f in os.listdir('data') if f.endswith(".wav")]

wav_files = wav_files[:3]

attacks = Attacks()
results = attacks.benchmark(wav_files, 'WavMark', sampling_rate=16000, mwm_type='other', collusion_size=10, cents=5, stretch_rate=0.9, inverted_stretch_rate=2.0)

with open('result.json', 'w') as fp:
    json.dump(results, fp)