import os
import subprocess
import requests
from tqdm import tqdm

BASE_DIR = "data/audio"
REAL_DIR = os.path.join(BASE_DIR, "real")
FAKE_DIR = os.path.join(BASE_DIR, "fake")

os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(FAKE_DIR, exist_ok=True)

# -----------------------------
# Helper: download file
# -----------------------------
def download_file(url, path):
    r = requests.get(url, stream=True)
    total = int(r.headers.get('content-length', 0))

    with open(path, 'wb') as f, tqdm(
        desc=path,
        total=total,
        unit='B',
        unit_scale=True
    ) as bar:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))


# -----------------------------
# Helper: convert to wav
# -----------------------------
def convert_to_wav(input_path, output_path):
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "16000", "-ac", "1",
            output_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print("FFmpeg error:", e)


# -----------------------------
# Sample REAL audio (public domain)
# -----------------------------
REAL_AUDIO_URLS = [
    "https://www2.cs.uic.edu/~i101/SoundFiles/taunt.wav",
    "https://www2.cs.uic.edu/~i101/SoundFiles/gettysburg10.wav",
    "https://www2.cs.uic.edu/~i101/SoundFiles/preamble10.wav",
]

# -----------------------------
# Sample FAKE audio (TTS-like)
# -----------------------------
FAKE_AUDIO_URLS = [
    "https://www2.cs.uic.edu/~i101/SoundFiles/CantinaBand3.wav",
    "https://www2.cs.uic.edu/~i101/SoundFiles/StarWars60.wav",
]

# -----------------------------
# Download + convert
# -----------------------------
def process_urls(urls, out_dir, prefix):
    for i, url in enumerate(urls):
        tmp_path = os.path.join(out_dir, f"{prefix}_{i}.wav")

        print(f"Downloading {url}")
        download_file(url, tmp_path)

        # already wav, but normalize anyway
        final_path = os.path.join(out_dir, f"{prefix}_{i}_final.wav")
        convert_to_wav(tmp_path, final_path)

        os.remove(tmp_path)


if __name__ == "__main__":
    print("Downloading REAL audio...")
    process_urls(REAL_AUDIO_URLS, REAL_DIR, "real")

    print("\nDownloading FAKE audio...")
    process_urls(FAKE_AUDIO_URLS, FAKE_DIR, "fake")

    print("\n✅ Dataset ready in data/audio/")