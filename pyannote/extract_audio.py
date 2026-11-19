from pathlib import Path
import subprocess

INPUT_DIR = Path("input")
OUTPUT_ROOT = Path("output")

# Các định dạng hỗ trợ
extensions = {
    ".mp4",
    ".m4a",
    ".wav",
    ".mp3",
    ".mov",
    ".mkv"
}

files = [
    f for f in INPUT_DIR.iterdir()
    if f.is_file() and f.suffix.lower() in extensions
]

if not files:
    raise FileNotFoundError("Không có file audio/video trong thư mục input")

# Lấy file được thêm/sửa gần nhất
INPUT = max(files, key=lambda p: p.stat().st_mtime)

# Tạo tên thư mục output theo tên file
safe_name = (
    INPUT.stem
    .replace(" ", "_")
    .replace("(", "_")
    .replace(")", "")
)

OUTPUT_DIR = OUTPUT_ROOT / safe_name
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUTPUT_DIR / "audio.wav"

print("Input :", INPUT)
print("Output:", OUTPUT)

command = [
    "ffmpeg",
    "-y",
    "-i", str(INPUT),
    "-ac", "1",
    "-ar", "16000",
    "-c:a", "pcm_s16le",
    str(OUTPUT)
]

subprocess.run(command, check=True)

print("\nDone")
print("Saved:", OUTPUT)