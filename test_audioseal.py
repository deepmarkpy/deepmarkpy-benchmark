
from audioseal import AudioSeal
import librosa
import torch

model = AudioSeal.load_generator("audioseal_wm_16bits")

wav, sr = librosa.load('harvard.wav', sr=16000)

wav = torch.tensor(wav, dtype=torch.float32)

wav = wav.unsqueeze(0).unsqueeze(0)

msg = torch.randint(0, 2, (wav.shape[0], model.msg_processor.nbits), device=wav.device)
watermark = model.get_watermark(wav, message = msg)

watermarked_audio = wav + watermark

detector = AudioSeal.load_detector("audioseal_detector_16bits")

result, message = detector.detect_watermark(watermarked_audio, sr)

print(result)
print(message)

result, message = detector(watermarked_audio, sr)

print(result[:, 1 , :])  

print(message)  