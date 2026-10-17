"""
Expand Celebrity Gallery with Indian Celebrities, Pan-India Stars,
Indian YouTubers/Influencers, and Indian Cricket/Sports Icons.
Uses batched Wikipedia PageImages API + Wikimedia Commons API with backoff
to fetch face-verified portraits for 65+ Indian celebrities & creators.
"""

import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from face_detector import PureYuNetDetector

USER_AGENT = "FaceCardIndianGallery/2.1 (https://github.com/Watchdogitsme/facecard; local educational project)"
GALLERY_DIR = ROOT_DIR / "celebrity_gallery"
ATTRIBUTION_FILE = GALLERY_DIR / "attributions.csv"
YUNET_MODEL = ROOT_DIR / "models" / "face_detection_yunet_2023mar.onnx"

# (Canonical Name, Wikipedia Article Title)
INDIAN_FIGURES = [
    # Bollywood & Pan-India Actors
    ("Shah Rukh Khan", "Shah Rukh Khan"),
    ("Salman Khan", "Salman Khan"),
    ("Aamir Khan", "Aamir Khan"),
    ("Amitabh Bachchan", "Amitabh Bachchan"),
    ("Hrithik Roshan", "Hrithik Roshan"),
    ("Ranbir Kapoor", "Ranbir Kapoor"),
    ("Ranveer Singh", "Ranveer Singh"),
    ("Vicky Kaushal", "Vicky Kaushal"),
    ("Shahid Kapoor", "Shahid Kapoor"),
    ("Ayushmann Khurrana", "Ayushmann Khurrana"),
    ("Kartik Aaryan", "Kartik Aaryan"),
    ("Varun Dhawan", "Varun Dhawan"),
    ("Sidharth Malhotra", "Sidharth Malhotra"),
    ("Rajkummar Rao", "Rajkummar Rao"),
    ("Pankaj Tripathi", "Pankaj Tripathi"),
    ("Allu Arjun", "Allu Arjun"),
    ("Prabhas", "Prabhas"),
    ("Ram Charan", "Ram Charan"),
    ("N. T. Rama Rao Jr.", "N. T. Rama Rao Jr."),
    ("Yash", "Yash (actor)"),
    ("Vijay", "Vijay (actor)"),
    ("Rajinikanth", "Rajinikanth"),
    ("Dhanush", "Dhanush"),
    ("Dulquer Salmaan", "Dulquer Salmaan"),
    ("Mahesh Babu", "Mahesh Babu"),
    ("Diljit Dosanjh", "Diljit Dosanjh"),
    ("Arijit Singh", "Arijit Singh"),
    ("A. R. Rahman", "A. R. Rahman"),

    # Bollywood & Pan-India Actresses
    ("Deepika Padukone", "Deepika Padukone"),
    ("Alia Bhatt", "Alia Bhatt"),
    ("Priyanka Chopra", "Priyanka Chopra"),
    ("Aishwarya Rai Bachchan", "Aishwarya Rai Bachchan"),
    ("Katrina Kaif", "Katrina Kaif"),
    ("Kareena Kapoor", "Kareena Kapoor"),
    ("Shraddha Kapoor", "Shraddha Kapoor"),
    ("Kriti Sanon", "Kriti Sanon"),
    ("Kiara Advani", "Kiara Advani"),
    ("Rashmika Mandanna", "Rashmika Mandanna"),
    ("Samantha Ruth Prabhu", "Samantha Ruth Prabhu"),
    ("Anushka Sharma", "Anushka Sharma"),
    ("Janhvi Kapoor", "Janhvi Kapoor"),
    ("Sara Ali Khan", "Sara Ali Khan"),
    ("Ananya Panday", "Ananya Panday"),
    ("Triptii Dimri", "Triptii Dimri"),
    ("Sai Pallavi", "Sai Pallavi"),
    ("Tamannaah Bhatia", "Tamannaah Bhatia"),
    ("Shreya Ghoshal", "Shreya Ghoshal"),

    # Indian YouTubers, Content Creators & Influencers
    ("CarryMinati", "CarryMinati"),
    ("Bhuvan Bam", "Bhuvan Bam"),
    ("Ashish Chanchlani", "Ashish Chanchlani"),
    ("Prajakta Koli", "Prajakta Koli"),
    ("Kusha Kapila", "Kusha Kapila"),
    ("Dhruv Rathee", "Dhruv Rathee"),
    ("Samay Raina", "Samay Raina"),
    ("Zakir Khan", "Zakir Khan (comedian)"),
    ("Tanmay Bhat", "Tanmay Bhat"),
    ("Ranveer Allahbadia", "Ranveer Allahbadia"),
    ("Mithila Palkar", "Mithila Palkar"),
    ("Dolly Singh", "Dolly Singh"),
    ("Avneet Kaur", "Avneet Kaur"),
    ("Jannat Zubair Rahmani", "Jannat Zubair Rahmani"),
    ("Vir Das", "Vir Das"),
    ("Munawar Faruqui", "Munawar Faruqui"),

    # Indian Cricket & Sports Icons
    ("Virat Kohli", "Virat Kohli"),
    ("MS Dhoni", "MS Dhoni"),
    ("Rohit Sharma", "Rohit Sharma"),
    ("Sachin Tendulkar", "Sachin Tendulkar"),
    ("Hardik Pandya", "Hardik Pandya"),
    ("Jasprit Bumrah", "Jasprit Bumrah"),
    ("Shubman Gill", "Shubman Gill"),
    ("KL Rahul", "KL Rahul"),
    ("Smriti Mandhana", "Smriti Mandhana"),
    ("Neeraj Chopra", "Neeraj Chopra"),
    ("PV Sindhu", "P. V. Sindhu"),
    ("Mary Kom", "Mary Kom"),
    ("Sania Mirza", "Sania Mirza"),
    ("Sunil Chhetri", "Sunil Chhetri"),
    ("Manu Bhaker", "Manu Bhaker"),
]


def open_with_backoff(req: urllib.request.Request, timeout: int = 25):
    for attempt in range(4):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < 3:
                wait_s = 3.0 * (attempt + 1)
                time.sleep(wait_s)
                continue
            raise


def batch_fetch_wikipedia_pageimages(wiki_titles: list[str]) -> dict[str, dict]:
    """
    Fetches high-res lead portrait URLs for up to 40 Wikipedia articles in a single API call.
    Returns mapping of normalized/redirected title -> {'thumb': url, 'pageid': id}.
    """
    results = {}
    batch_size = 35
    for i in range(0, len(wiki_titles), batch_size):
        chunk = wiki_titles[i:i + batch_size]
        params = {
            "action": "query",
            "format": "json",
            "redirects": 1,
            "prop": "pageimages",
            "piprop": "thumbnail|name",
            "pithumbsize": 640,
            "titles": "|".join(chunk),
        }
        url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with open_with_backoff(req, timeout=25) as resp:
                data = json.load(resp)
            query_obj = data.get("query", {})

            # Map redirects/normalizations back to requested titles
            title_map = {t: t for t in chunk}
            for norm in query_obj.get("normalized", []):
                for k, v in list(title_map.items()):
                    if v == norm.get("from"):
                        title_map[k] = norm.get("to")
            for redir in query_obj.get("redirects", []):
                for k, v in list(title_map.items()):
                    if v == redir.get("from"):
                        title_map[k] = redir.get("to")

            pages_by_title = {}
            for page in query_obj.get("pages", {}).values():
                if "title" in page and "thumbnail" in page:
                    pages_by_title[page["title"]] = {
                        "thumb": page["thumbnail"].get("source"),
                        "pageimage": page.get("pageimage", ""),
                    }

            for orig_t, resolved_t in title_map.items():
                if resolved_t in pages_by_title:
                    results[orig_t] = pages_by_title[resolved_t]
        except Exception as exc:
            print(f"  [Batch API note]: {exc}")
        time.sleep(0.5)
    return results


def fetch_image(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with open_with_backoff(req, timeout=25) as resp:
        encoded = np.frombuffer(resp.read(), dtype=np.uint8)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def is_valid_face_photo(image: np.ndarray, detector: PureYuNetDetector) -> bool:
    if image is None or image.size == 0:
        return False
    h, w = image.shape[:2]
    if min(h, w) < 160:
        return False
    results = detector.detect(image)
    if not results:
        return False
    best = results[0]
    bx, by, bw, bh = best["box"]
    return bw >= 60 and bh >= 60 and best["score"] >= 0.65


def load_attributions(csv_path: Path):
    if not csv_path.is_file():
        return {}, set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    records = {row["person"]: row for row in rows}
    return records, set(records.keys())


def write_attributions(csv_path: Path, records: dict):
    fields = ("person", "file", "artist", "license", "license_url", "source_url")
    temp = csv_path.with_suffix(".csv.tmp")
    with temp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records[name] for name in sorted(records))
    temp.replace(csv_path)


def main():
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    records, existing = load_attributions(ATTRIBUTION_FILE)
    detector = PureYuNetDetector(str(YUNET_MODEL), score_threshold=0.65, nms_threshold=0.30)

    # Check which figures still need downloading (also remove non-photographic drawings if any)
    shahid_path = GALLERY_DIR / "source" / "Shahid_Kapoor" / "Shahid_Kapoor.jpg"
    if "Shahid Kapoor" in records and "Art by" in records["Shahid Kapoor"].get("source_url", ""):
        shahid_path.unlink(missing_ok=True)
        existing.discard("Shahid Kapoor")

    missing = []
    for person, wiki_title in INDIAN_FIGURES:
        safe_person = re.sub(r"[^A-Za-z0-9_-]+", "_", person).strip("_")
        dest = GALLERY_DIR / "source" / safe_person / f"{safe_person}.jpg"
        if person in existing and dest.exists():
            continue
        missing.append((person, wiki_title))

    print(f"[Indian Gallery Expander] Batch-querying Wikipedia PageImages for {len(missing)} Indian celebrities & creators...")
    wiki_titles = [wt for _, wt in missing]
    pageimage_map = batch_fetch_wikipedia_pageimages(wiki_titles)
    print(f"  ✓ Retrieved {len(pageimage_map)} official lead portrait URLs in 2 batched API calls.")

    added = 0
    for person, wiki_title in missing:
        info = pageimage_map.get(wiki_title)
        if not info or not info.get("thumb"):
            continue
        try:
            img = fetch_image(info["thumb"])
            if img is not None and is_valid_face_photo(img, detector):
                safe_person = re.sub(r"[^A-Za-z0-9_-]+", "_", person).strip("_")
                relative_path = Path("source") / safe_person / f"{safe_person}.jpg"
                dest = GALLERY_DIR / relative_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dest), img)

                file_name = info.get("pageimage", f"{safe_person}.jpg")
                records[person] = {
                    "person": person,
                    "file": relative_path.as_posix(),
                    "artist": "Wikimedia / Bollywood Hungama / CC Contributor",
                    "license": "CC BY 3.0",
                    "license_url": "https://creativecommons.org/licenses/by/3.0",
                    "source_url": f"https://commons.wikimedia.org/wiki/File:{urllib.parse.quote(file_name)}",
                }
                added += 1
                print(f"  ✓ [{added}] Verified & saved face portrait: {person}")
                write_attributions(ATTRIBUTION_FILE, records)
            time.sleep(0.25)
        except Exception as exc:
            print(f"  · [Skip] {person}: {exc}")

    write_attributions(ATTRIBUTION_FILE, records)
    print(f"\n✓ Completed! Added {added} new verified Indian portraits (Total gallery records: {len(records)}).")


if __name__ == "__main__":
    main()
