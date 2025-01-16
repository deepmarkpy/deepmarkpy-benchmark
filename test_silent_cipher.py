import librosa
import silentcipher

model = silentcipher.get_model(
    model_type='44.1k',
    device='cuda'
)
y, sr = librosa.load('harvard.wav', sr=44100)

encoded, sdr = model.encode_wav(y, sr, [123, 234, 111, 222, 11])

result = model.decode_wav(encoded, sr, phase_shift_decoding=False)

print(result['status'])
print(result['messages'][0] == [123, 234, 111, 222, 11])
print(result['confidences'][0])