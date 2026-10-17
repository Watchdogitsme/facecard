"""
Download Pre-Trained Weights & Deep Learning Models
Fetches:
1. YuNet Face & 5-Point Landmark Detector (ONNX)
2. SFace 128-D Deep Face Recognition Model (ONNX)
3. OpenVINO Retail Age/Gender Recognition Model (ONNX)
"""

from pathlib import Path
from urllib.request import Request, urlopen

MODELS = Path(__file__).resolve().parent.parent / "models"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FaceCardModelDownloader/2.0"

DOWNLOADS = {
    "face_detection_yunet_2023mar.onnx": [
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
        "models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    ],
    "face_recognition_sface_2021dec.onnx": [
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
        "models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    ],
    "age-gender-recognition-retail-0013.onnx": [
        "https://storage.googleapis.com/ailia-models/age-gender-recognition-retail/"
        "age-gender-recognition-retail-0013.onnx",
    ],
}


def download_file(filename: str, urls: list[str]) -> bool:
    destination = MODELS / filename
    if destination.exists() and destination.stat().st_size > 100 * 1024:
        print(f"✓ Model already verified: {filename} ({destination.stat().st_size // 1024} KB)")
        return True

    temporary = destination.with_suffix(destination.suffix + ".part")
    for url in urls:
        try:
            print(f"Downloading {filename} from {url}...")
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=120) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            if temporary.stat().st_size <= 100 * 1024:
                raise RuntimeError(f"Downloaded file is unexpectedly small ({temporary.stat().st_size} B): {filename}")
            temporary.replace(destination)
            print(f"✓ Successfully downloaded {filename} ({destination.stat().st_size // 1024} KB)")
            return True
        except Exception as exc:
            print(f"  [ Retry ] Failed from {url}: {exc}")
            temporary.unlink(missing_ok=True)
    return False


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    for filename, urls in DOWNLOADS.items():
        download_file(filename, urls)
    print(f"All models are ready in {MODELS}")


if __name__ == "__main__":
    main()
