"""
Download Pre-Trained Weights & Deep Learning Models
Fetches YuNet Face Detector (ONNX) and OpenVINO Retail Age/Gender Model (ONNX).
"""

from pathlib import Path
from urllib.request import urlopen

MODELS = Path(__file__).resolve().parent.parent / "models"
DOWNLOADS = {
    "face_detection_yunet_2023mar.onnx": (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
        "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ),
    "age-gender-recognition-retail-0013.onnx": (
        "https://storage.googleapis.com/ailia-models/age-gender-recognition-retail/"
        "age-gender-recognition-retail-0013.onnx"
    ),
}

def main() -> None:
    MODELS.mkdir(exist_ok=True)
    for filename, url in DOWNLOADS.items():
        destination = MODELS / filename
        if destination.exists() and destination.stat().st_size > 1024:
            print(f"✓ Model already verified: {filename} ({destination.stat().st_size // 1024} KB)")
            continue
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            print(f"Downloading {filename}...")
            with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            if temporary.stat().st_size <= 1024:
                raise RuntimeError(f"Downloaded file is unexpectedly small: {filename}")
            temporary.replace(destination)
            print(f"✓ Successfully downloaded {filename}")
        finally:
            temporary.unlink(missing_ok=True)
    print(f"All models are ready in {MODELS}")

if __name__ == "__main__":
    main()
