from pathlib import Path
import csv
import re

OUTPUT_ROOT = Path("output")

# =========================
# FIND LATEST RESULT FOLDER
# =========================

folders = [p for p in OUTPUT_ROOT.iterdir() if p.is_dir()]

if not folders:
    raise FileNotFoundError("Không tìm thấy thư mục nào trong output")

RESULT_DIR = max(folders, key=lambda p: p.stat().st_mtime)

DIARIZATION_FILE = RESULT_DIR / "diarization.csv"
TRANSCRIPT_FILE = RESULT_DIR / "transcript.txt"
MERGED_FILE = RESULT_DIR / "conversation.txt"

print("Using folder:", RESULT_DIR)


# =========================
# LOAD DIARIZATION
# =========================

speaker_segments = []

with open(DIARIZATION_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        speaker_segments.append({
            "speaker": row["speaker"],
            "start": float(row["start"]),
            "end": float(row["end"])
        })


# =========================
# LOAD TRANSCRIPT
# =========================

transcript_segments = []

pattern = re.compile(
    r"\[(\d+(?:\.\d+)?)s\s*->\s*(\d+(?:\.\d+)?)s\]\s*(.*)"
)

with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        match = pattern.match(line)

        if not match:
            continue

        transcript_segments.append({
            "start": float(match.group(1)),
            "end": float(match.group(2)),
            "text": match.group(3).strip()
        })


# =========================
# SPEAKER ASSIGNMENT
# =========================

MAX_NEAREST_DISTANCE = 2.5  # seconds


def get_speaker(transcript_start, transcript_end):

    best_speaker = None
    best_overlap = 0.0

    # 1. Ưu tiên speaker có overlap lớn nhất
    for segment in speaker_segments:

        overlap_start = max(
            transcript_start,
            segment["start"]
        )

        overlap_end = min(
            transcript_end,
            segment["end"]
        )

        overlap = max(
            0.0,
            overlap_end - overlap_start
        )

        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = segment["speaker"]

    if best_speaker is not None:
        return best_speaker

    # 2. Nếu pyannote bỏ sót đoạn speech,
    # tìm speaker gần nhất
    transcript_mid = (
        transcript_start + transcript_end
    ) / 2

    nearest_speaker = None
    nearest_distance = float("inf")

    for segment in speaker_segments:

        # khoảng cách từ transcript tới segment
        if transcript_end < segment["start"]:
            distance = segment["start"] - transcript_end

        elif transcript_start > segment["end"]:
            distance = transcript_start - segment["end"]

        else:
            distance = 0.0

        if distance < nearest_distance:
            nearest_distance = distance
            nearest_speaker = segment["speaker"]

    if nearest_distance <= MAX_NEAREST_DISTANCE:
        return nearest_speaker

    return "UNKNOWN"


# =========================
# MERGE
# =========================

merged = []

for segment in transcript_segments:

    speaker = get_speaker(
        segment["start"],
        segment["end"]
    )

    merged.append({
        "speaker": speaker,
        "start": segment["start"],
        "end": segment["end"],
        "text": segment["text"]
    })


# =========================
# COMBINE SAME SPEAKER
# =========================

combined = []

for item in merged:

    if (
        combined
        and combined[-1]["speaker"] == item["speaker"]
        and item["start"] - combined[-1]["end"] <= 2.0
    ):

        combined[-1]["text"] += " " + item["text"]
        combined[-1]["end"] = item["end"]

    else:
        combined.append(item.copy())


# =========================
# SAVE
# =========================

with open(MERGED_FILE, "w", encoding="utf-8") as f:

    for item in combined:

        line = (
            f"[{item['start']:.2f}s -> "
            f"{item['end']:.2f}s] "
            f"{item['speaker']}: "
            f"{item['text']}\n"
        )

        print(line, end="")
        f.write(line)

print("\nSaved:", MERGED_FILE)