import os
import torch
import numpy as np

from pathlib import Path
from scipy.io import wavfile
from pyannote.audio import Pipeline


OUTPUT_ROOT = Path("output")

audio_files = list(OUTPUT_ROOT.glob("*/audio_clean.wav"))

if not audio_files:
    raise FileNotFoundError("Không tìm thấy audio.wav trong output")

# lấy audio.wav mới nhất
AUDIO_FILE = max(audio_files, key=lambda p: p.stat().st_mtime)

# kết quả sẽ lưu cùng thư mục với audio.wav đó
RESULT_DIR = AUDIO_FILE.parent

print("Using audio:", AUDIO_FILE)
print("Saving result to:", RESULT_DIR)


# load pyannote

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=os.environ["HF_TOKEN"]
)

pipeline.instantiate({
    "segmentation": {
        "min_duration_off": 0.3
    },
    "clustering": {
        "threshold": 0.6,
        "Fa": 0.07,
        "Fb": 0.8
    }
})

# gpu

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

pipeline.to(device)

print("Device:", device)


# load Wav 

sample_rate, audio = wavfile.read(AUDIO_FILE)

print("Sample rate:", sample_rate)
print("Original shape:", audio.shape)


# int16 -> float32 [-1, 1]
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0
else:
    audio = audio.astype(np.float32)


# Nếu stereo thì chuyển thành mono
if audio.ndim == 2:
    audio = audio.mean(axis=1)


# pyannote cần dạng [channels, samples]
waveform = torch.from_numpy(audio).unsqueeze(0)


print("Waveform shape:", waveform.shape)
print("Running diarization...")


# diarization

audio_input = {
    "waveform": waveform,
    "sample_rate": sample_rate
}


result = pipeline(
    audio_input,
    num_speakers=2
)

diarization = result.speaker_diarization

diarization = result.speaker_diarization


# thư mục lưu kết quả của datk1
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# save TXT

result_file = RESULT_DIR / "result.txt"

with open(result_file, "w", encoding="utf-8") as f:
    
    for turn, speaker in diarization:
        line = f"{turn.start:.2f}s -> {turn.end:.2f}s : {speaker}\n"

        print(line, end="")
        f.write(line)

print("\nSaved result to:", result_file)


# save CSV

import csv

csv_file = RESULT_DIR / "diarization.csv"

with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow([
        "speaker",
        "start",
        "end",
        "duration"
    ])

    for turn, speaker in diarization:
        writer.writerow([
            speaker,
            round(turn.start, 3),
            round(turn.end, 3),
            round(turn.end - turn.start, 3)
        ])

print("Saved CSV:", csv_file)


# save RTTM

rttm_file = RESULT_DIR / "diarization.rttm"

with open(rttm_file, "w", encoding="utf-8") as f:
    diarization.write_rttm(f)

print("Saved RTTM:", rttm_file)