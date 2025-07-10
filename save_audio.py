import torch
import torchaudio
from utils import prepare_data
import os

ds = prepare_data(max_shards=2)
audios = []
sample_rates = []

for i in range(10):                                          
    audio = torch.from_numpy(ds[i]["mp3"]["array"]).float()  # type: ignore[attr-defined] 
    sample_rates.append(ds[i]["mp3"]["sampling_rate"])       # type: ignore[attr-defined]
    audios.append(audio)

# Create output directory
os.makedirs("saved_audio", exist_ok=True)

# Save each audio as .wav file
for i, (audio, sr) in enumerate(zip(audios, sample_rates)):
    # Ensure audio is 2D (channels, samples)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)  # Add channel dimension
    
    filename = f"saved_audio/audio_{i:03d}.wav"
    torchaudio.save(filename, audio, sr)
    print(f"Saved: {filename} (SR: {sr}Hz, Duration: {audio.shape[1]/sr:.2f}s)") 