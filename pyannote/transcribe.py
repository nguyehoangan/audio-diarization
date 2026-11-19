from pathlib import Path
from faster_whisper import WhisperModel
from scipy.io import wavfile
import numpy as np

OUTPUT_ROOT = Path("output")

# tìm audio.wav mới nhất
audio_files = list(OUTPUT_ROOT.glob("*/audio.wav"))

if not audio_files:
    raise FileNotFoundError("Không tìm thấy audio.wav")

AUDIO_FILE = max(audio_files, key=lambda p: p.stat().st_mtime)
RESULT_DIR = AUDIO_FILE.parent

print("Using audio:", AUDIO_FILE)

# load audio
sample_rate, audio = wavfile.read(AUDIO_FILE)

if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0
else:
    audio = audio.astype(np.float32)

if audio.ndim == 2:
    audio = audio.mean(axis=1)

# Whisper model
model = WhisperModel(
    "medium",
    device="cuda",
    compute_type="float16"
)

CHUNK_SECONDS = 30
OVERLAP_SECONDS = 2

chunk_samples = CHUNK_SECONDS * sample_rate
overlap_samples = OVERLAP_SECONDS * sample_rate

transcript_file = RESULT_DIR / "transcript.txt"

with open(transcript_file, "w", encoding="utf-8") as f:

    start_sample = 0

    while start_sample < len(audio):

        end_sample = min(
            start_sample + chunk_samples,
            len(audio)
        )

        chunk = audio[start_sample:end_sample]

        chunk_start_time = start_sample / sample_rate

        print(
            f"\nProcessing "
            f"{chunk_start_time:.2f}s -> "
            f"{end_sample / sample_rate:.2f}s"
        )

        segments, info = model.transcribe(
            chunk,
            language="en",
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=400,
                speech_pad_ms=200
            ),
            condition_on_previous_text=False,
            word_timestamps=True
        )

        for segment in segments:

            global_start = (
                chunk_start_time + segment.start
            )

            global_end = (
                chunk_start_time + segment.end
            )

            text = segment.text.strip()

            if not text:
                continue

            line = (
                f"[{global_start:.2f}s -> "
                f"{global_end:.2f}s] "
                f"{text}\n"
            )

            print(line, end="")
            f.write(line)

        if end_sample >= len(audio):
            break

        start_sample = (
            end_sample - overlap_samples
        )

print("\nSaved:", transcript_file)