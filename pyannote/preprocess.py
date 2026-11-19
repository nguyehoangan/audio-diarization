import subprocess
from pathlib import Path

OUTPUT_ROOT = Path("output")

# tìm audio.wav mới nhất
audio_files = list(OUTPUT_ROOT.glob("*/audio.wav"))

if not audio_files:
    raise FileNotFoundError("Không tìm thấy audio.wav")

INPUT = max(audio_files, key=lambda p: p.stat().st_mtime)

OUTPUT_DIR = INPUT.parent
OUTPUT = OUTPUT_DIR / "audio_clean.wav"

print("Input :", INPUT)
print("Output:", OUTPUT)

command = [
    "ffmpeg",
    "-y",
    "-i", str(INPUT),

    "-af",
    "highpass=f=90,"
    "equalizer=f=50:t=q:w=1.5:g=-18,"
    "equalizer=f=100:t=q:w=1.8:g=-12,"
    "equalizer=f=150:t=q:w=2:g=-8,"
    "lowpass=f=7000,"
    "acompressor=threshold=-28dB:ratio=5:attack=15:release=280:makeup=6,"
    "alimiter=limit=0.85",

    "-ac", "1",
    "-ar", "16000",
    "-c:a", "pcm_s16le",

    str(OUTPUT)
]

subprocess.run(command, check=True)

print("\nDone:", OUTPUT)