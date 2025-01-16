import numpy as np
import librosa
import torch
import wavmark

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = wavmark.load_model().to(device)

payload = np.random.choice([0, 1], size=16)
print("Payload:", payload)

signal, sample_rate = librosa.load('harvard.wav', sr=16000)

watermarked_signal, _ = wavmark.encode_watermark(model, signal, payload, show_progress=True)

payload_decoded, _ = wavmark.decode_watermark(model, watermarked_signal, show_progress=True)
BER = (payload != payload_decoded).mean() * 100

print("Decode BER:%.1f" % BER)