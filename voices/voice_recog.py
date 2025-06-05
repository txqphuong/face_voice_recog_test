from speechbrain.inference import SpeakerRecognition
import torchaudio
import torch
import os

# Load model
model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="tmp/spkrec"
)

# Function to get embedding from file path
def get_embedding(file_path):
    signal, fs = torchaudio.load(file_path)
    if fs != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
        signal = resampler(signal)
    with torch.no_grad():
        embedding = model.encode_batch(signal).squeeze(0).mean(dim=0)
    return embedding

# Enroll speaker profiles
huy_embedding = get_embedding("/Users/phuongtxq/Desktop/test/voices/whisper/voice_person1.flac")
phuong_embedding = get_embedding("/Users/phuongtxq/Desktop/test/voices/whisper/voice_person_2.flac")

# Load unknown voice
unknown_embedding = get_embedding("/Users/phuongtxq/Desktop/test/voices/whisper/voice_unknown.flac")

# Compute similarities
similarity_phuong = model.similarity(phuong_embedding, unknown_embedding).item()
similarity_huy = model.similarity(huy_embedding, unknown_embedding).item()

# Print similarity scores
print(f"🔊 Similarity to Phuong: {similarity_phuong:.3f}")
print(f"🔊 Similarity to Huy: {similarity_huy:.3f}")

# Decision logic
threshold = 0.5
if similarity_phuong > similarity_huy and similarity_phuong > threshold:
    print("✅ Predicted speaker: Phuong")
elif similarity_huy > threshold:
    print("✅ Predicted speaker: Huy")
