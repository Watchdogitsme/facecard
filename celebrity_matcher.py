"""
Deep Learning & Biometric Celebrity + Influencer Matcher Module
Features:
- Hybrid 128-D SFace Deep Face Recognition Embeddings + 12 Anthropometric Facial Proportions
- 5-Point Affine Face Alignment for pose & head-tilt invariance
- Unified 189+ Celebrity, YouTuber, Influencer & Athlete Database (with neutral category tags)
- Learned Metric Calibration from train_models.py
"""

import os
import math
import cv2
import numpy as np

from face_detector import PureSFaceRecognizer, align_face


# Region-neutral metadata catalog (Category, Title, Gender, Approx Age, Color)
CELEBRITY_METADATA = {
    # --- FILM STARS & ACTORS ---
    "Shah Rukh Khan": ("ACTOR", "Cinema Superstar & Icon", "male", "indian", 58, (0, 200, 255)),
    "Salman Khan": ("ACTOR", "Action Cinema Megastar", "male", "indian", 58, (255, 160, 60)),
    "Aamir Khan": ("ACTOR", "Acclaimed Cinema Icon", "male", "indian", 59, (120, 210, 255)),
    "Amitabh Bachchan": ("ACTOR", "Legendary Screen Icon", "male", "indian", 81, (210, 180, 90)),
    "Hrithik Roshan": ("ACTOR", "Action & Dance Superstar", "male", "indian", 50, (80, 220, 255)),
    "Ranbir Kapoor": ("ACTOR", "Award-Winning Film Star", "male", "indian", 41, (255, 140, 90)),
    "Ranveer Singh": ("ACTOR", "Powerhouse Film Superstar", "male", "indian", 39, (100, 240, 180)),
    "Vicky Kaushal": ("ACTOR", "National Award-Winning Actor", "male", "indian", 36, (120, 200, 240)),
    "Shahid Kapoor": ("ACTOR", "Acclaimed Leading Actor", "male", "indian", 43, (240, 180, 100)),
    "Ayushmann Khurrana": ("ACTOR", "Award-Winning Actor & Singer", "male", "indian", 39, (160, 220, 120)),
    "Kartik Aaryan": ("ACTOR", "Leading Film Superstar", "male", "indian", 33, (255, 170, 80)),
    "Varun Dhawan": ("ACTOR", "Action & Comedy Film Star", "male", "indian", 37, (100, 190, 255)),
    "Sidharth Malhotra": ("ACTOR", "Action & Drama Film Star", "male", "indian", 39, (140, 210, 180)),
    "Rajkummar Rao": ("ACTOR", "Critically Acclaimed Actor", "male", "indian", 39, (180, 220, 140)),
    "Pankaj Tripathi": ("ACTOR", "Award-Winning Character Actor", "male", "indian", 48, (200, 170, 100)),
    "Allu Arjun": ("ACTOR", "National Award-Winning Star", "male", "indian", 42, (0, 190, 255)),
    "Prabhas": ("ACTOR", "Action Blockbuster Megastar", "male", "indian", 44, (240, 150, 70)),
    "Ram Charan": ("ACTOR", "Action Cinema Superstar", "male", "indian", 39, (80, 200, 255)),
    "N. T. Rama Rao Jr.": ("ACTOR", "Action & Drama Powerhouse", "male", "indian", 41, (255, 180, 60)),
    "Yash": ("ACTOR", "Blockbuster Cinema Star", "male", "indian", 38, (200, 170, 70)),
    "Vijay": ("ACTOR", "Cinema Megastar & Icon", "male", "indian", 50, (100, 210, 255)),
    "Rajinikanth": ("ACTOR", "Legendary Cinema Superstar", "male", "indian", 73, (255, 215, 0)),
    "Dhanush": ("ACTOR", "Award-Winning Actor & Director", "male", "indian", 41, (140, 200, 220)),
    "Dulquer Salmaan": ("ACTOR", "Acclaimed Film Star", "male", "indian", 38, (180, 220, 255)),
    "Mahesh Babu": ("ACTOR", "Leading Cinema Superstar", "male", "indian", 49, (120, 210, 240)),
    "Diljit Dosanjh": ("POP STAR", "Music & Film Superstar", "male", "indian", 40, (255, 130, 180)),
    "Arijit Singh": ("POP STAR", "Chart-Topping Vocal Icon", "male", "indian", 37, (160, 200, 255)),
    "A. R. Rahman": ("POP STAR", "Oscar & Grammy-Winning Maestro", "male", "indian", 57, (220, 190, 100)),

    # --- ACTRESSES ---
    "Deepika Padukone": ("ACTOR", "Fashion & Cinema Superstar", "female", "indian", 38, (255, 150, 200)),
    "Alia Bhatt": ("ACTOR", "Award-Winning Film Actress", "female", "indian", 31, (255, 170, 210)),
    "Priyanka Chopra": ("ACTOR", "Film Star & Cultural Icon", "female", "indian", 42, (230, 140, 210)),
    "Aishwarya Rai Bachchan": ("ACTOR", "Miss World & Cinema Icon", "female", "indian", 50, (240, 190, 120)),
    "Katrina Kaif": ("ACTOR", "Action & Dance Film Star", "female", "indian", 41, (255, 160, 190)),
    "Kareena Kapoor": ("ACTOR", "Acclaimed Leading Actress", "female", "indian", 43, (255, 140, 175)),
    "Shraddha Kapoor": ("ACTOR", "Leading Film Actress", "female", "indian", 37, (255, 175, 220)),
    "Kriti Sanon": ("ACTOR", "National Award-Winning Actress", "female", "indian", 34, (240, 160, 230)),
    "Kiara Advani": ("ACTOR", "Leading Cinema Star", "female", "indian", 32, (255, 180, 200)),
    "Rashmika Mandanna": ("ACTOR", "Blockbuster Leading Actress", "female", "indian", 28, (255, 150, 180)),
    "Samantha Ruth Prabhu": ("ACTOR", "Action & Drama Film Icon", "female", "indian", 37, (230, 170, 240)),
    "Anushka Sharma": ("ACTOR", "Actress & Film Producer", "female", "indian", 36, (245, 180, 160)),
    "Janhvi Kapoor": ("ACTOR", "Leading Film Actress", "female", "indian", 27, (255, 165, 210)),
    "Sara Ali Khan": ("ACTOR", "Leading Cinema Actress", "female", "indian", 29, (240, 190, 220)),
    "Ananya Panday": ("ACTOR", "Film & Youth Style Icon", "female", "indian", 25, (255, 185, 215)),
    "Triptii Dimri": ("ACTOR", "Acclaimed Breakout Actress", "female", "indian", 30, (235, 160, 220)),
    "Sai Pallavi": ("ACTOR", "Acclaimed Actress & Dancer", "female", "indian", 32, (255, 190, 160)),
    "Tamannaah Bhatia": ("ACTOR", "Leading Cinema Star", "female", "indian", 34, (245, 165, 205)),
    "Shreya Ghoshal": ("POP STAR", "Award-Winning Vocal Icon", "female", "indian", 40, (240, 175, 230)),

    # --- YOUTUBERS, INFLUENCERS & CREATORS ---
    "CarryMinati": ("YOUTUBER", "Top Gaming & Roast Creator", "male", "indian", 25, (0, 160, 255)),
    "Bhuvan Bam": ("YOUTUBER", "BB Ki Vines Creator & Actor", "male", "indian", 30, (255, 180, 50)),
    "Ashish Chanchlani": ("YOUTUBER", "Comedy Blockbuster Creator", "male", "indian", 30, (80, 220, 255)),
    "Prajakta Koli": ("INFLUENCER", "MostlySane Creator & Actress", "female", "indian", 31, (255, 150, 210)),
    "Kusha Kapila": ("INFLUENCER", "Digital Creator & Actress", "female", "indian", 34, (245, 140, 200)),
    "Dhruv Rathee": ("YOUTUBER", "Educational YouTube Creator", "male", "indian", 29, (100, 230, 180)),
    "Samay Raina": ("YOUTUBER", "Comedian & Chess Streamer", "male", "indian", 26, (120, 190, 255)),
    "Zakir Khan": ("INFLUENCER", "Stand-Up Comedy Icon", "male", "indian", 37, (220, 175, 90)),
    "Tanmay Bhat": ("YOUTUBER", "Comedy & YouTube Creator", "male", "indian", 37, (140, 200, 250)),
    "Ranveer Allahbadia": ("YOUTUBER", "BeerBiceps & Podcast Host", "male", "indian", 31, (100, 240, 160)),
    "Triggered Insaan": ("YOUTUBER", "Gaming & Commentary Creator", "male", "indian", 28, (255, 130, 90)),
    "Techno Gamerz": ("YOUTUBER", "Gaming YouTube Superstar", "male", "indian", 22, (80, 250, 160)),
    "Komal Pandey": ("INFLUENCER", "Fashion & Style Influencer", "female", "indian", 30, (255, 140, 190)),
    "Dolly Singh": ("INFLUENCER", "Digital Creator & Actress", "female", "indian", 31, (240, 165, 220)),
    "Mithila Palkar": ("INFLUENCER", "Actress & Digital Creator", "female", "indian", 31, (255, 175, 195)),
    "Jannat Zubair Rahmani": ("INFLUENCER", "Social Media & TV Star", "female", "indian", 23, (255, 160, 215)),
    "Avneet Kaur": ("INFLUENCER", "Actress & Social Media Icon", "female", "indian", 22, (250, 155, 205)),
    "Munawar Faruqui": ("INFLUENCER", "Stand-Up Artist & Creator", "male", "indian", 32, (160, 215, 255)),
    "Vir Das": ("INFLUENCER", "Emmy-Winning Comedian & Actor", "male", "indian", 45, (180, 210, 140)),

    # --- SPORTS & ATHLETES ---
    "Virat Kohli": ("ATHLETE", "World Champion Cricket Icon", "male", "indian", 35, (255, 140, 0)),
    "MS Dhoni": ("ATHLETE", "Championship Winning Captain", "male", "indian", 43, (0, 215, 255)),
    "Rohit Sharma": ("ATHLETE", "World Cup Winning Captain", "male", "indian", 37, (255, 175, 40)),
    "Sachin Tendulkar": ("ATHLETE", "Legendary Cricket Champion", "male", "indian", 51, (240, 200, 80)),
    "Hardik Pandya": ("ATHLETE", "Championship All-Rounder", "male", "indian", 30, (80, 200, 255)),
    "Jasprit Bumrah": ("ATHLETE", "World #1 Fast Bowler", "male", "indian", 30, (100, 230, 240)),
    "Shubman Gill": ("ATHLETE", "International Cricket Star", "male", "indian", 25, (120, 215, 255)),
    "KL Rahul": ("ATHLETE", "International Cricket Star", "male", "indian", 32, (150, 210, 240)),
    "Smriti Mandhana": ("ATHLETE", "International Cricket Superstar", "female", "indian", 28, (255, 160, 210)),
    "Neeraj Chopra": ("ATHLETE", "Olympic Gold Medalist", "male", "indian", 26, (0, 230, 180)),
    "P. V. Sindhu": ("ATHLETE", "Double Olympic Medalist", "female", "indian", 29, (0, 220, 170)),
    "PV Sindhu": ("ATHLETE", "Double Olympic Medalist", "female", "indian", 29, (0, 220, 170)),
    "Mary Kom": ("ATHLETE", "6x World Boxing Champion", "female", "indian", 41, (0, 210, 160)),
    "Sania Mirza": ("ATHLETE", "6x Grand Slam Champion", "female", "indian", 37, (240, 170, 210)),
    "Sunil Chhetri": ("ATHLETE", "International Football Legend", "male", "indian", 40, (80, 210, 255)),
    "Manu Bhaker": ("ATHLETE", "Double Olympic Medalist", "female", "indian", 22, (255, 180, 210)),
}


class CelebrityMatcher:
    """
    Deep Learning + Biometric Look-Alike Matcher.
    Combines 128-D SFace Deep Face Embeddings (on 5-point aligned faces) with
    12 anthropometric proportions and learned metric calibration weights.
    """

    FEATURE_NAMES = [
        "Face Ratio",
        "Eye Spacing",
        "Eye Height",
        "Eye Alignment",
        "Nose Height",
        "Mouth Width",
        "Mouth Height",
        "Mouth Alignment",
        "Philtrum Gap",
        "Eye-Mouth Span",
        "Complexion Undertone",
        "Texture Profile"
    ]

    FEATURE_TOLERANCES = np.array([
        0.22,  # Face ratio
        0.18,  # Eye spacing
        0.14,  # Eye height
        0.12,  # Eye alignment
        0.15,  # Nose height
        0.20,  # Mouth width
        0.18,  # Mouth height
        0.14,  # Mouth alignment
        0.16,  # Philtrum gap
        0.16,  # Eye-mouth span
        0.25,  # Complexion
        0.30   # Texture
    ], dtype=np.float32)

    EXPANDED_DATABASE = [
        {
            "id": "carryminati", "name": "CarryMinati",
            "title": "Top Gaming & Roast Creator", "category": "YOUTUBER",
            "gender": "male", "region": "indian", "color": (0, 160, 255),
            "vector": [0.80, 0.43, 0.39, 0.02, 0.57, 0.38, 0.75, 0.02, 0.18, 0.36, 0.56, 0.44]
        },
        {
            "id": "bhuvan_bam", "name": "Bhuvan Bam",
            "title": "BB Ki Vines Creator & Actor", "category": "YOUTUBER",
            "gender": "male", "region": "indian", "color": (255, 180, 50),
            "vector": [0.76, 0.42, 0.38, 0.02, 0.58, 0.37, 0.76, 0.02, 0.18, 0.38, 0.58, 0.46]
        },
        {
            "id": "ashish_chanchlani", "name": "Ashish Chanchlani",
            "title": "Comedy Blockbuster Creator", "category": "YOUTUBER",
            "gender": "male", "region": "indian", "color": (80, 220, 255),
            "vector": [0.84, 0.44, 0.39, 0.02, 0.57, 0.40, 0.75, 0.02, 0.18, 0.36, 0.57, 0.45]
        },
        {
            "id": "prajakta_koli", "name": "Prajakta Koli",
            "title": "MostlySane Creator & Actress", "category": "INFLUENCER",
            "gender": "female", "region": "indian", "color": (255, 150, 210),
            "vector": [0.78, 0.44, 0.38, 0.02, 0.56, 0.39, 0.74, 0.02, 0.18, 0.36, 0.55, 0.38]
        },
        {
            "id": "kusha_kapila", "name": "Kusha Kapila",
            "title": "Digital Creator & Actress", "category": "INFLUENCER",
            "gender": "female", "region": "indian", "color": (245, 140, 200),
            "vector": [0.79, 0.45, 0.38, 0.02, 0.56, 0.40, 0.74, 0.02, 0.18, 0.36, 0.54, 0.37]
        },
        {
            "id": "dhruv_rathee", "name": "Dhruv Rathee",
            "title": "Educational YouTube Creator", "category": "YOUTUBER",
            "gender": "male", "region": "indian", "color": (100, 230, 180),
            "vector": [0.78, 0.43, 0.38, 0.02, 0.57, 0.38, 0.75, 0.02, 0.18, 0.37, 0.53, 0.42]
        },
        {
            "id": "samay_raina", "name": "Samay Raina",
            "title": "Comedian & Chess Streamer", "category": "YOUTUBER",
            "gender": "male", "region": "indian", "color": (120, 190, 255),
            "vector": [0.85, 0.44, 0.39, 0.02, 0.57, 0.39, 0.75, 0.02, 0.18, 0.36, 0.54, 0.43]
        },
        {
            "id": "zakir_khan", "name": "Zakir Khan",
            "title": "Stand-Up Comedy Icon", "category": "INFLUENCER",
            "gender": "male", "region": "indian", "color": (220, 175, 90),
            "vector": [0.83, 0.44, 0.39, 0.02, 0.58, 0.40, 0.76, 0.02, 0.18, 0.37, 0.62, 0.47]
        },
        {
            "id": "ranveer_allahbadia", "name": "Ranveer Allahbadia",
            "title": "BeerBiceps & Podcast Host", "category": "YOUTUBER",
            "gender": "male", "region": "indian", "color": (100, 240, 160),
            "vector": [0.79, 0.43, 0.38, 0.02, 0.57, 0.39, 0.75, 0.02, 0.18, 0.37, 0.57, 0.45]
        },
        {
            "id": "triggered_insaan", "name": "Triggered Insaan",
            "title": "Gaming & Commentary Creator", "category": "YOUTUBER",
            "gender": "male", "region": "indian", "color": (255, 130, 90),
            "vector": [0.77, 0.43, 0.38, 0.02, 0.56, 0.38, 0.74, 0.02, 0.18, 0.36, 0.53, 0.40]
        },
        {
            "id": "techno_gamerz", "name": "Techno Gamerz",
            "title": "Gaming YouTube Superstar", "category": "YOUTUBER",
            "gender": "male", "region": "indian", "color": (80, 250, 160),
            "vector": [0.78, 0.42, 0.38, 0.02, 0.56, 0.37, 0.74, 0.02, 0.18, 0.36, 0.56, 0.39]
        },
        {
            "id": "komal_pandey", "name": "Komal Pandey",
            "title": "Fashion & Style Influencer", "category": "INFLUENCER",
            "gender": "female", "region": "indian", "color": (255, 140, 190),
            "vector": [0.76, 0.45, 0.38, 0.02, 0.55, 0.40, 0.74, 0.02, 0.19, 0.36, 0.56, 0.36]
        },
        {
            "id": "dolly_singh", "name": "Dolly Singh",
            "title": "Digital Creator & Actress", "category": "INFLUENCER",
            "gender": "female", "region": "indian", "color": (240, 165, 220),
            "vector": [0.74, 0.44, 0.38, 0.02, 0.56, 0.38, 0.74, 0.02, 0.18, 0.36, 0.56, 0.37]
        },
        {
            "id": "mithila_palkar", "name": "Mithila Palkar",
            "title": "Actress & Digital Creator", "category": "INFLUENCER",
            "gender": "female", "region": "indian", "color": (255, 175, 195),
            "vector": [0.77, 0.44, 0.38, 0.02, 0.55, 0.39, 0.73, 0.02, 0.18, 0.35, 0.55, 0.36]
        },
    ]

    def __init__(self, celebrities_dir="celebrities", target_probability=7.2):
        self.celebrities_dir = celebrities_dir
        self.target_probability = target_probability
        self.best_match_idx = 0
        self.lock_counter = 0
        self.lock_duration = 25
        self.gallery_base_dir = None
        self.celebrities = []
        self._yunet = None
        self.sface = PureSFaceRecognizer()
        self.calib_data = None

        self.indian_only = False
        self.gender_filter_enabled = True
        self.last_top_matches = []

        self._load_calibration()
        self._load_gallery_database()
        self._merge_curated_indian_profiles()

        if not self.celebrities:
            self.celebrities = list(self.EXPANDED_DATABASE)
            self._load_or_generate_images()

    def _load_calibration(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(base_dir, "models", "calibrated_models.npz"),
            os.path.join("models", "calibrated_models.npz"),
        ]
        for path in candidates:
            if os.path.exists(path):
                try:
                    self.calib_data = dict(np.load(path, allow_pickle=False))
                    print(f"[CelebrityMatcher] Loaded trained metric calibration ({path})")
                    break
                except Exception as e:
                    print(f"[CelebrityMatcher] Calibration load note: {e}")

    def _init_crop_yunet(self):
        """Lazy initialization of YuNet detector for face crops without pre-computed landmarks."""
        if self._yunet is not None:
            return self._yunet
        candidates = [
            os.path.join(os.path.dirname(__file__), "models", "face_detection_yunet_2023mar.onnx"),
            os.path.join(os.path.dirname(__file__), "..", "CV_Mini project", "models", "face_detection_yunet_2023mar.onnx")
        ]
        for yp in candidates:
            if os.path.exists(yp):
                try:
                    from face_detector import PureYuNetDetector
                    self._yunet = PureYuNetDetector(yp, score_threshold=0.50, nms_threshold=0.30)
                    break
                except Exception:
                    pass
        return self._yunet

    def _load_gallery_database(self):
        """Discovers and loads index.npz containing 181+ real celebrities + 128-D SFace embeddings."""
        candidates = [
            os.path.join(os.path.dirname(__file__), "celebrity_gallery", "index.npz"),
            os.path.join(os.path.dirname(__file__), "..", "CV_Mini project", "celebrity_gallery", "index.npz"),
            os.path.join(os.path.dirname(__file__), "celebrities", "index.npz"),
        ]
        index_file = None
        for cand in candidates:
            if os.path.exists(cand):
                index_file = cand
                self.gallery_base_dir = os.path.dirname(cand)
                break

        if not index_file:
            return

        try:
            data = np.load(index_file, allow_pickle=False)
            labels = data["labels"]
            feature_cards = data["feature_cards"]
            image_files = data["image_files"]
            embeddings = data["embeddings"] if "embeddings" in data else None
            genders = data["genders"] if "genders" in data else None
            regions = data["regions"] if "regions" in data else None

            entries = []
            for i, label in enumerate(labels):
                name = self._format_celebrity_name(str(label))
                category, title, gender, region, color = self._infer_metadata(name)
                if genders is not None and i < len(genders) and str(genders[i]) in ("male", "female"):
                    gender = str(genders[i])
                if regions is not None and i < len(regions) and str(regions[i]) in ("indian", "global"):
                    region = str(regions[i])

                celeb_id = str(label).lower().replace(" ", "_")
                img_rel = str(image_files[i])
                img_path = os.path.join(self.gallery_base_dir, img_rel)

                emb = None
                if embeddings is not None and i < len(embeddings):
                    vec_e = np.asarray(embeddings[i], dtype=np.float32).flatten()
                    if np.linalg.norm(vec_e) > 1e-6:
                        emb = vec_e / float(np.linalg.norm(vec_e))

                entries.append({
                    "id": celeb_id,
                    "name": name,
                    "title": title,
                    "category": category,
                    "gender": gender,
                    "region": region,
                    "color": color,
                    "vector": feature_cards[i].tolist(),
                    "embedding": emb,
                    "img_path": img_path if os.path.exists(img_path) else None,
                    "gallery_idx": i
                })

            if entries:
                self.celebrities = entries
                emb_count = sum(1 for e in entries if e.get("embedding") is not None)
                print(
                    f"[CelebrityMatcher] Loaded {len(self.celebrities)} verified celebrities "
                    f"({emb_count} with 128-D SFace embeddings) from {index_file}"
                )
        except Exception as e:
            print(f"[CelebrityMatcher] Note loading gallery index: {e}")

    def _merge_curated_indian_profiles(self):
        """Ensures curated creators/celebrities are always present."""
        existing_names = {c["name"].lower() for c in self.celebrities}
        for prof in self.EXPANDED_DATABASE:
            if prof["name"].lower() not in existing_names:
                entry = dict(prof)
                img_path = self._find_celeb_photo(entry)
                entry["img_path"] = img_path
                entry.setdefault("embedding", None)
                self.celebrities.append(entry)
                existing_names.add(entry["name"].lower())

    def _format_celebrity_name(self, label):
        """Formats label into professional name display."""
        special = {
            "A_AP_Rocky": "A$AP Rocky",
            "Chris_Evans_actor": "Chris Evans",
            "Prince_musician": "Prince",
            "Samuel_L_Jackson": "Samuel L. Jackson",
            "Conan_O_Brien": "Conan O'Brien",
            "PV_Sindhu": "P. V. Sindhu",
            "N_T_Rama_Rao_Jr": "N. T. Rama Rao Jr.",
            "A_R_Rahman": "A. R. Rahman",
            "Yash_actor": "Yash",
            "Vijay_actor": "Vijay",
            "Zakir_Khan_comedian": "Zakir Khan",
            "John_Green_author": "John Green",
            "KSI_YouTuber": "KSI",
            "Jay-Z": "Jay-Z"
        }
        if label in special:
            return special[label]
        return label.replace("_", " ")

    def _infer_metadata(self, name):
        """Infers region-neutral (category, title, gender, region, color) for any celebrity."""
        if name in CELEBRITY_METADATA:
            cat, title, gender, region, _, color = CELEBRITY_METADATA[name]
            return cat, title, gender, region, color

        female_global = {
            "Taylor Swift", "Beyonce", "Adele", "Rihanna", "Lady Gaga", "Billie Eilish",
            "Ariana Grande", "Dua Lipa", "Madonna", "Katy Perry", "Selena Gomez",
            "Miley Cyrus", "Shakira", "Jennifer Lopez", "Celine Dion", "Cher",
            "Billie Holiday", "Stevie Nicks", "Dolly Parton", "Angelina Jolie",
            "Meryl Streep", "Scarlett Johansson", "Jennifer Lawrence", "Zendaya",
            "Margot Robbie", "Sandra Bullock", "Julia Roberts", "Viola Davis",
            "Nicole Kidman", "Cate Blanchett", "Anne Hathaway", "Emma Stone",
            "Emma Watson", "Jenna Ortega", "Florence Pugh", "Anya Taylor-Joy",
            "Michelle Yeoh", "Lucy Liu", "Awkwafina", "Greta Gerwig", "Oprah Winfrey",
            "Ellen DeGeneres", "Lilly Singh", "Emma Chamberlain", "Michelle Phan",
            "Liza Koshy", "Rosanna Pansino", "NikkieTutorials", "Huda Kattan",
            "Chiara Ferragni", "Nara Smith", "Serena Williams", "Venus Williams",
            "Naomi Osaka", "Simone Biles", "Ronda Rousey", "Greta Thunberg",
            "Malala Yousafzai", "Michelle Obama", "Ruth Bader Ginsburg", "Jane Goodall",
            "Kylie Jenner", "Charli D'Amelio", "Addison Rae", "Khloe Kardashian"
        }
        gender = "female" if name in female_global else "male"

        actors = {
            "Keanu Reeves", "Leonardo DiCaprio", "Robert Downey Jr", "Robert Downey Jr.",
            "Scarlett Johansson", "Zendaya", "Timothee Chalamet", "Brad Pitt", "Tom Cruise",
            "Margot Robbie", "Dwayne Johnson", "Chris Evans", "Chris Hemsworth", "Emma Watson",
            "Emma Stone", "Jennifer Lawrence", "Jennifer Lopez", "Jenna Ortega", "Ryan Reynolds",
            "Tom Hanks", "Morgan Freeman", "Pedro Pascal", "Cillian Murphy", "Anne Hathaway",
            "Angelina Jolie", "George Clooney", "Matt Damon", "Meryl Streep", "Cate Blanchett",
            "Hugh Jackman", "Jackie Chan", "Florence Pugh", "Anya Taylor-Joy", "Ben Affleck",
            "Daniel Radcliffe", "Donnie Yen", "Harrison Ford", "Julia Roberts", "Sandra Bullock",
            "Viola Davis", "Michelle Yeoh", "Lucy Liu", "Chow Yun-fat", "Denzel Washington",
            "Samuel L. Jackson", "Nicole Kidman", "Bruce Lee"
        }
        youtubers = {
            "MrBeast", "PewDiePie", "Marques Brownlee", "Markiplier", "Lilly Singh",
            "Emma Chamberlain", "Casey Neistat", "Zach King", "Liza Koshy", "Logan Paul",
            "Jake Paul", "KSI"
        }
        musicians = {
            "Taylor Swift", "Billie Eilish", "Ariana Grande", "Dua Lipa", "Drake",
            "Rihanna", "Selena Gomez", "Adele", "Beyonce", "Ed Sheeran", "Justin Bieber",
            "Lady Gaga", "Katy Perry", "Harry Styles", "Kendrick Lamar", "Bruno Mars",
            "Shawn Mendes", "Snoop Dogg", "The Weeknd", "Bad Bunny", "Eminem", "Elton John",
            "Freddie Mercury", "John Lennon", "Paul McCartney", "Mick Jagger", "Madonna",
            "Shakira", "Miley Cyrus", "Bob Dylan", "Bruce Springsteen", "Kurt Cobain",
            "David Bowie", "Celine Dion", "Cher", "A$AP Rocky", "Prince", "Jay-Z",
            "Billie Holiday", "Dolly Parton", "Robert Plant", "Stevie Nicks", "Willie Nelson"
        }
        athletes = {
            "LeBron James", "Stephen Curry", "Naomi Osaka", "Rafael Nadal", "Tiger Woods",
            "Usain Bolt", "Conor McGregor", "Kobe Bryant", "Ronda Rousey", "Cristiano Ronaldo",
            "Lionel Messi", "Serena Williams", "Venus Williams", "Roger Federer", "Simone Biles"
        }

        if name in youtubers:
            return "YOUTUBER", "Top YouTube Creator", gender, "global", (255, 120, 0)
        elif name in musicians:
            return "POP STAR", "Music & Pop Icon", gender, "global", (255, 150, 200)
        elif name in athletes:
            return "ATHLETE", "World Champion Athlete", gender, "global", (0, 220, 160)
        elif name in actors:
            return "ACTOR", "Award-Winning Film Star", gender, "global", (255, 200, 80)
        else:
            return "ICON", "Cultural Icon", gender, "global", (140, 200, 240)

    def _find_celeb_photo(self, celeb):
        """Searches for celebrity photo across local folder and existing gallery."""
        candidates = [
            os.path.join(self.celebrities_dir, f"{celeb['id']}.jpg"),
            os.path.join(self.celebrities_dir, f"{celeb['id']}.png"),
        ]
        gallery_source_candidates = [
            os.path.join(os.path.dirname(__file__), "celebrity_gallery", "source"),
            os.path.join(os.path.dirname(__file__), "..", "CV_Mini project", "celebrity_gallery", "source")
        ]
        gallery_source = next((p for p in gallery_source_candidates if os.path.exists(p)), None)
        if gallery_source and os.path.exists(gallery_source):
            name_norm = celeb["name"].replace(" ", "_")
            for entry in os.listdir(gallery_source):
                if name_norm.lower() == entry.lower() or celeb["id"].lower() == entry.lower():
                    cand = os.path.join(gallery_source, entry, f"{entry}.jpg")
                    if os.path.exists(cand):
                        candidates.insert(0, cand)
                    folder_path = os.path.join(gallery_source, entry)
                    if os.path.isdir(folder_path):
                        for f in os.listdir(folder_path):
                            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                                candidates.append(os.path.join(folder_path, f))

        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    def _load_or_generate_images(self):
        """Loads celebrity images or generates fallback cards."""
        for celeb in self.celebrities:
            img_path = self._find_celeb_photo(celeb)
            if img_path:
                celeb["img_path"] = img_path
            else:
                local_path = os.path.join(self.celebrities_dir, f"{celeb['id']}.jpg")
                celeb["img_path"] = local_path
                if not os.path.exists(local_path):
                    img = self._create_stylized_celebrity_card(celeb)
                    try:
                        os.makedirs(self.celebrities_dir, exist_ok=True)
                        cv2.imwrite(local_path, img)
                    except Exception:
                        pass

    def _create_stylized_celebrity_card(self, celeb):
        """Creates an HD portrait avatar card for the figure."""
        size = 280
        card = np.zeros((size, size, 3), dtype=np.uint8)

        b_color = celeb.get("color", (200, 180, 100))
        for y in range(size):
            ratio = y / float(size)
            r = int(20 + (b_color[2] * 0.18) * ratio)
            g = int(24 + (b_color[1] * 0.18) * ratio)
            b = int(30 + (b_color[0] * 0.18) * ratio)
            card[y, :] = (b, g, r)

        center = (size // 2, int(size * 0.39))
        radius = int(size * 0.29)
        cv2.circle(card, center, radius + 4, b_color, 2, cv2.LINE_AA)
        cv2.circle(card, center, radius, (38, 42, 50), -1, cv2.LINE_AA)

        head_radius = int(radius * 0.44)
        head_center = (center[0], center[1] - int(radius * 0.15))
        cv2.circle(card, head_center, head_radius, (215, 220, 230), -1, cv2.LINE_AA)

        shoulder_center = (center[0], center[1] + int(radius * 0.85))
        cv2.ellipse(card, shoulder_center, (int(radius * 0.85), int(radius * 0.50)),
                    0, 0, 360, (215, 220, 230), -1, cv2.LINE_AA)

        parts = [p for p in celeb.get("name", "Star").split() if p]
        initials = "".join(p[0].upper() for p in parts[:2])
        cv2.putText(card, initials, (head_center[0] - 16, head_center[1] + 8),
                    cv2.FONT_HERSHEY_DUPLEX, 0.72, (40, 45, 55), 2, cv2.LINE_AA)

        category_tag = celeb.get("category", "STAR")[:15]
        cv2.rectangle(card, (center[0] - 60, center[1] + int(radius * 0.45)),
                      (center[0] + 60, center[1] + int(radius * 0.45) + 18), (20, 24, 30), -1)
        cv2.rectangle(card, (center[0] - 60, center[1] + int(radius * 0.45)),
                      (center[0] + 60, center[1] + int(radius * 0.45) + 18), b_color, 1)
        cv2.putText(card, category_tag, (center[0] - 54, center[1] + int(radius * 0.45) + 13),
                    cv2.FONT_HERSHEY_PLAIN, 0.82, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.rectangle(card, (10, size - 58), (size - 10, size - 10), (18, 20, 25), -1)
        cv2.rectangle(card, (10, size - 58), (size - 10, size - 10), (65, 75, 90), 1)

        cv2.putText(card, celeb["name"].upper()[:22],
                    (18, size - 34), cv2.FONT_HERSHEY_DUPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(card, celeb.get("title", "Celebrity")[:32],
                    (18, size - 16), cv2.FONT_HERSHEY_PLAIN, 0.82, (180, 190, 205), 1, cv2.LINE_AA)

        return card

    def extract_biometric_features(self, face_bgr, landmarks=None):
        """
        Extracts 12 accurate anthropometric facial proportions:
        1-10: Geometric facial landmark ratios matching index.npz
        11-12: Complexion undertone & micro-texture roughness
        """
        h, w = face_bgr.shape[:2]
        if h <= 2 or w <= 2:
            return np.zeros(12, dtype=np.float32)

        lm = None
        if landmarks is not None:
            lm_arr = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
            if lm_arr.shape[0] == 5:
                if np.max(lm_arr[:, 0]) > w or np.max(lm_arr[:, 1]) > h:
                    lm_arr[:, 0] -= np.min(lm_arr[:, 0])
                    lm_arr[:, 1] -= np.min(lm_arr[:, 1])
                lm = lm_arr

        if lm is None:
            detector = self._init_crop_yunet()
            if detector is not None:
                try:
                    res = detector.detect(face_bgr)
                    if res:
                        lm = res[0]["landmarks"]
                except Exception:
                    pass

        if lm is None:
            re_x, re_y = w * 0.32, h * 0.37
            le_x, le_y = w * 0.68, h * 0.37
            nose_x, nose_y = w * 0.50, h * 0.58
            rm_x, rm_y = w * 0.36, h * 0.75
            lm_x, lm_y = w * 0.64, h * 0.75
        else:
            re_x, re_y = float(lm[0, 0]), float(lm[0, 1])
            le_x, le_y = float(lm[1, 0]), float(lm[1, 1])
            nose_x, nose_y = float(lm[2, 0]), float(lm[2, 1])
            rm_x, rm_y = float(lm[3, 0]), float(lm[3, 1])
            lm_x, lm_y = float(lm[4, 0]), float(lm[4, 1])

        face_ratio = float(w) / float(h)
        eye_dist = max(1.0, math.hypot(le_x - re_x, le_y - re_y))
        eye_spacing = eye_dist / float(w)

        eye_mid_y = (re_y + le_y) * 0.5
        eye_height = eye_mid_y / float(h)
        eye_tilt = abs(le_y - re_y) / eye_dist

        nose_height = nose_y / float(h)

        mouth_dist = max(1.0, math.hypot(lm_x - rm_x, lm_y - rm_y))
        mouth_width = mouth_dist / float(w)

        mouth_mid_y = (rm_y + lm_y) * 0.5
        mouth_height = mouth_mid_y / float(h)
        mouth_tilt = abs(lm_y - rm_y) / mouth_dist

        nose_mouth_gap = max(0.01, (mouth_mid_y - nose_y) / float(h))
        eye_mouth_span = max(0.05, (mouth_mid_y - eye_mid_y) / float(h))

        ycrcb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)
        cr_mean = float(np.mean(ycrcb[:, :, 1]))
        cb_mean = float(np.mean(ycrcb[:, :, 2]))
        complexion = float(((cr_mean - 128.0) * 0.6 + (cb_mean - 128.0) * 0.4) / 50.0 + 0.5)

        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        texture_prof = float(min(1.0, math.log1p(lap_var) / 7.0))

        return np.array([
            face_ratio,
            eye_spacing,
            eye_height,
            eye_tilt,
            nose_height,
            mouth_width,
            mouth_height,
            mouth_tilt,
            nose_mouth_gap,
            eye_mouth_span,
            complexion,
            texture_prof
        ], dtype=np.float32)

    def extract_deep_embedding(self, face_bgr, landmarks=None):
        """Extracts 128-D SFace deep identity embedding on 5-point aligned face crop."""
        if self.sface is not None and self.sface.available:
            return self.sface.extract_embedding(face_bgr, landmarks=landmarks)
        return None

    def _infer_gender_from_embedding(self, query_embedding):
        """Infers gender from 128-D SFace embedding using trained logistic probe."""
        if self.calib_data is not None and query_embedding is not None and len(query_embedding) == 128:
            try:
                gw = self.calib_data["gender_weights"].flatten()
                gb = float(self.calib_data["gender_bias"].flatten()[0])
                logit = float(np.dot(query_embedding, gw) + gb)
                prob_female = 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, logit))))
                if prob_female >= 0.62:
                    return "female"
                elif prob_female <= 0.38:
                    return "male"
            except Exception:
                pass
        return None

    def match(self, face_bgr, landmarks=None, detected_gender=None, gender_confidence=0.0, query_embedding=None):
        """
        Evaluates all candidates using Hybrid 128-D Deep SFace Embedding + 12 Biometric Proportions.
        Returns: (top_celeb, final_score, celeb_img, best_trait_scores)
        """
        if face_bgr is None or face_bgr.size == 0 or len(self.celebrities) == 0:
            top_celeb = self.celebrities[self.best_match_idx] if self.celebrities else {
                "name": "Keanu Reeves", "title": "Award-Winning Film Star", "category": "ACTOR", "id": "keanu_reeves"
            }
            return top_celeb, 7.1, self.get_celebrity_image(top_celeb), [("Overall Geometry", 74), ("Feature Proportions", 72)]

        query_vector = self.extract_biometric_features(face_bgr, landmarks)
        if query_embedding is None:
            query_embedding = self.extract_deep_embedding(face_bgr, landmarks)

        trait_weights_learned = None
        emb_mean = None
        emb_proj = None
        if self.calib_data is not None:
            trait_weights_learned = self.calib_data.get("trait_weights")
            emb_mean = self.calib_data.get("emb_mean")
            emb_proj = self.calib_data.get("emb_proj")

        q_emb_calib = None
        if query_embedding is not None:
            q_emb_calib = query_embedding.copy()
            if emb_mean is not None and emb_proj is not None and len(emb_mean) == len(q_emb_calib):
                q_emb_calib = np.dot(q_emb_calib - emb_mean, emb_proj)
                n_q = float(np.linalg.norm(q_emb_calib))
                if n_q > 1e-6:
                    q_emb_calib /= n_q

        # Determine gender filter from embedding or passed gender
        target_gender = None
        if self.gender_filter_enabled:
            if detected_gender in ("Male", "Female") and gender_confidence >= 60.0:
                target_gender = detected_gender.lower()
            else:
                target_gender = self._infer_gender_from_embedding(query_embedding)

        candidate_indices = []
        for idx, c in enumerate(self.celebrities):
            if target_gender and c.get("gender") in ("male", "female") and c.get("gender") != target_gender:
                continue
            candidate_indices.append(idx)

        if not candidate_indices:
            candidate_indices = list(range(len(self.celebrities)))

        scored_candidates = []

        for idx in candidate_indices:
            celeb = self.celebrities[idx]
            ref_raw = np.array(celeb["vector"], dtype=np.float32)
            num_dim = min(len(query_vector), len(ref_raw))
            q_sub = query_vector[:num_dim]
            r_sub = ref_raw[:num_dim]
            tol_sub = self.FEATURE_TOLERANCES[:num_dim]

            differences = np.abs(q_sub - r_sub)
            check_scores = np.clip(100.0 * (1.0 - differences / tol_sub), 18.0, 99.0)

            if trait_weights_learned is not None and len(trait_weights_learned) >= num_dim:
                weights = np.asarray(trait_weights_learned[:num_dim], dtype=np.float32)
            else:
                weights = np.array([1.25, 1.25, 1.0, 0.8, 1.1, 1.1, 1.0, 0.8, 1.0, 1.1, 0.7, 0.7][:num_dim], dtype=np.float32)

            trait_avg = float(np.average(check_scores, weights=weights))

            norm_q = float(np.linalg.norm(q_sub))
            norm_r = float(np.linalg.norm(r_sub))
            geom_cos = float(np.dot(q_sub, r_sub) / (norm_q * norm_r)) if norm_q > 0 and norm_r > 0 else 0.75
            geom_composite = 0.75 * trait_avg + 0.25 * (geom_cos * 100.0)

            ref_emb = celeb.get("embedding")
            if query_embedding is not None and ref_emb is not None and len(ref_emb) == len(query_embedding):
                raw_cos = float(np.dot(query_embedding, ref_emb))
                calib_cos = raw_cos
                if q_emb_calib is not None and emb_mean is not None and emb_proj is not None:
                    r_calib = np.dot(ref_emb - emb_mean, emb_proj)
                    n_r = float(np.linalg.norm(r_calib))
                    if n_r > 1e-6:
                        calib_cos = float(np.dot(q_emb_calib, r_calib / n_r))

                eff_cos = 0.65 * raw_cos + 0.35 * calib_cos
                deep_score = float(np.clip(58.0 + eff_cos * 68.0, 35.0, 99.5))
                composite = 0.68 * deep_score + 0.32 * geom_composite
            else:
                composite = geom_composite

            top_trait_indices = np.argsort(check_scores)[::-1][:3]
            trait_list = [
                (self.FEATURE_NAMES[i], int(round(check_scores[i])))
                for i in top_trait_indices
            ]
            scored_candidates.append((composite, idx, trait_list))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        best_composite_score, best_candidate_idx, best_trait_scores = scored_candidates[0]

        if (
            self.lock_counter > 0
            and self.best_match_idx != best_candidate_idx
            and self.best_match_idx in candidate_indices
        ):
            self.lock_counter -= 1
            for comp, c_idx, t_list in scored_candidates:
                if c_idx == self.best_match_idx:
                    best_composite_score = comp
                    best_trait_scores = t_list
                    break
        else:
            self.best_match_idx = best_candidate_idx
            self.lock_counter = self.lock_duration

        top_celeb = self.celebrities[self.best_match_idx]
        final_score = self._composite_to_ten_scale(best_composite_score)

        self.last_top_matches = []
        for comp, c_idx, t_list in scored_candidates[:3]:
            c_obj = self.celebrities[c_idx]
            self.last_top_matches.append({
                "celeb": c_obj,
                "score": self._composite_to_ten_scale(comp),
                "traits": t_list
            })

        celeb_img = self.get_celebrity_image(top_celeb)
        return top_celeb, final_score, celeb_img, best_trait_scores

    @staticmethod
    def _composite_to_ten_scale(composite):
        """Maps composite similarity (0-100) to a realistic 10-point scale."""
        if composite >= 90.0:
            score = 8.4 + (composite - 90.0) * 0.14
        elif composite >= 72.0:
            score = 7.3 + ((composite - 72.0) / 18.0) * 1.1
        else:
            score = 6.8 + ((composite - 45.0) / 27.0) * 0.5
        return round(float(np.clip(score, 6.8, 9.8)), 1)

    def cycle_next(self):
        """Manually advances to the next celebrity in the database and resets debounce."""
        if not self.celebrities:
            return None
        self.best_match_idx = (self.best_match_idx + 1) % len(self.celebrities)
        self.lock_counter = self.lock_duration
        return self.celebrities[self.best_match_idx]

    def get_celebrity_image(self, celeb):
        """Loads and returns 3-channel BGR square portrait image (280x280)."""
        path = celeb.get("img_path")
        if not path or not os.path.exists(path):
            path = self._find_celeb_photo(celeb)
            if path:
                celeb["img_path"] = path

        if path and os.path.exists(path):
            img = cv2.imread(path)
            if img is not None and img.size > 0:
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                h, w = img.shape[:2]
                min_dim = min(h, w)
                top = (h - min_dim) // 2
                left = (w - min_dim) // 2
                cropped = img[top:top + min_dim, left:left + min_dim]
                return cv2.resize(cropped, (280, 280), interpolation=cv2.INTER_AREA)

        return self._create_stylized_celebrity_card(celeb)
