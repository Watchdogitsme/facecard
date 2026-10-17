"""
Face Skin Evaluator & Celebrity Matcher
Real-time OpenCV Application

Features:
- Live face detection with sleek biometric brackets
- Skin quality rating (Smoothness, Tone Uniformity, Blemish Clarity) out of 10
- Age estimation based on facial wrinkles, micro-texture, and geometry
- Celebrity look-alike portrait display with deep + biometric matching
- Side-by-side futuristic HUD dashboard
- Interactive keyboard shortcuts (Q to quit, S to snapshot, C to cycle celeb, M for mask mode)
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np

from skin_analyzer import SkinAnalyzer
from age_estimator import AgeEstimator
from celebrity_matcher import CelebrityMatcher
from face_detector import RobustFaceDetector


def find_cascade_path():
    """Locates or downloads haarcascade_frontalface_default.xml across platforms."""
    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        pip_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if os.path.exists(pip_path):
            return pip_path

    local_path = os.path.join(os.path.dirname(__file__), "models", "haarcascade_frontalface_default.xml")
    if os.path.exists(local_path):
        return local_path

    linux_paths = [
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    ]
    for lp in linux_paths:
        if os.path.exists(lp):
            return lp

    import urllib.request
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    try:
        try:
            os.makedirs(models_dir, exist_ok=True)
        except OSError:
            pass
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        print("[Setup] Downloading official OpenCV face cascade...")
        urllib.request.urlretrieve(url, local_path)
        if os.path.exists(local_path):
            return local_path
    except Exception as e:
        print(f"[Setup] Warning: Could not auto-download cascade ({e}).")

    return None


def draw_corner_brackets(img, pt1, pt2, color=(0, 225, 255), d=18, thickness=2):
    """Draws sleek sci-fi corner brackets around a bounding box."""
    x1, y1 = pt1
    x2, y2 = pt2

    # Top-Left
    cv2.line(img, (x1, y1), (x1 + d, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + d), color, thickness, cv2.LINE_AA)
    # Top-Right
    cv2.line(img, (x2, y1), (x2 - d, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + d), color, thickness, cv2.LINE_AA)
    # Bottom-Left
    cv2.line(img, (x1, y2), (x1 + d, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - d), color, thickness, cv2.LINE_AA)
    # Bottom-Right
    cv2.line(img, (x2, y2), (x2 - d, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - d), color, thickness, cv2.LINE_AA)


def draw_progress_bar(panel, x, y, width, height, value, max_val, fg_color, bg_color=(45, 50, 60)):
    """Draws a rounded modern progress bar."""
    cv2.rectangle(panel, (x, y), (x + width, y + height), bg_color, -1)
    fill_w = int(np.clip(value / float(max_val), 0.0, 1.0) * width)
    if fill_w > 0:
        cv2.rectangle(panel, (x, y), (x + fill_w, y + height), fg_color, -1)
    cv2.rectangle(panel, (x, y), (x + width, y + height), (70, 75, 88), 1)


class FaceEvaluationApp:
    def __init__(self, camera_index=0, width=640, height=480, target_prob=7.1, indian_only=False):
        self.camera_index = camera_index
        self.cam_w = width
        self.cam_h = height
        self.side_w = 420
        self.total_h = max(height, 620)
        self.total_w = self.cam_w + self.side_w

        # Engines
        self.skin_analyzer = SkinAnalyzer()
        self.age_estimator = AgeEstimator()
        self.celeb_matcher = CelebrityMatcher(target_probability=target_prob)
        self.celeb_matcher.indian_only = indian_only

        # Robust Face Detection Engine (Haar / YuNet / Color-Contour Fallback)
        self.face_detector = RobustFaceDetector()

        # UI State
        self.show_mask_mode = False
        self.paused = False
        self.snapshot_message_time = 0
        self.fps = 0.0
        self.last_face_time = 0
        self.cached_analysis = None

    def render_sidebar(self, celeb_dict, celeb_prob, celeb_img, skin_report, age_report, top_traits=None):
        """Renders the right-hand biometric sidebar panel showing top look-alike match."""
        panel = np.zeros((self.total_h, self.side_w, 3), dtype=np.uint8)
        # Background dark gradient
        for y in range(self.total_h):
            r = int(18 + (y / self.total_h) * 12)
            panel[y, :] = (r + 4, r + 2, r)

        # Top Header Banner
        cv2.rectangle(panel, (0, 0), (self.side_w, 42), (28, 32, 40), -1)
        cv2.putText(panel, "AI BIOMETRIC & SKIN MIRROR", (20, 26),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 215, 0), 1, cv2.LINE_AA)
        cv2.line(panel, (0, 42), (self.side_w, 42), (50, 60, 75), 1)

        # ----------------- SECTION 1: TOP LOOK-ALIKE MATCH -----------------
        y_sec1 = 50
        cv2.putText(panel, "★ TOP LOOK-ALIKE (12 CHECKS)", (20, y_sec1 + 14),
                    cv2.FONT_HERSHEY_DUPLEX, 0.46, (100, 200, 255), 1, cv2.LINE_AA)

        # Category Badge (ACTOR / YOUTUBER / INFLUENCER / POP STAR / ATHLETE)
        category_tag = celeb_dict.get('category', 'TOP MATCH')
        cv2.rectangle(panel, (self.side_w - 130, y_sec1), (self.side_w - 20, y_sec1 + 18), (35, 42, 55), -1)
        cv2.rectangle(panel, (self.side_w - 130, y_sec1), (self.side_w - 20, y_sec1 + 18), (80, 150, 240), 1)
        cv2.putText(panel, category_tag, (self.side_w - 124, y_sec1 + 13),
                    cv2.FONT_HERSHEY_PLAIN, 0.85, (220, 235, 255), 1, cv2.LINE_AA)

        # Celebrity Photo Box
        photo_size = 130
        photo_x = 20
        photo_y = y_sec1 + 25

        if celeb_img is not None:
            resized_celeb = cv2.resize(celeb_img, (photo_size, photo_size))
            panel[photo_y:photo_y + photo_size, photo_x:photo_x + photo_size] = resized_celeb

        # Photo border
        cv2.rectangle(panel, (photo_x - 1, photo_y - 1),
                      (photo_x + photo_size + 1, photo_y + photo_size + 1), (100, 180, 255), 2)

        # Celebrity Info text (Right of photo)
        text_x = photo_x + photo_size + 14
        cv2.putText(panel, celeb_dict['name'], (text_x, photo_y + 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, celeb_dict.get('title', 'Top Match'), (text_x, photo_y + 38),
                    cv2.FONT_HERSHEY_PLAIN, 0.85, (160, 175, 195), 1, cv2.LINE_AA)

        # Match Probability Display
        prob_str = f"{celeb_prob:.1f} / 10"
        percent_str = f"({int(celeb_prob * 10)}% Match)"
        cv2.putText(panel, f"Score: {prob_str}", (text_x, photo_y + 68),
                    cv2.FONT_HERSHEY_DUPLEX, 0.46, (0, 220, 180), 1, cv2.LINE_AA)
        cv2.putText(panel, percent_str, (text_x, photo_y + 84),
                    cv2.FONT_HERSHEY_PLAIN, 0.90, (180, 230, 210), 1, cv2.LINE_AA)

        # Progress bar for celebrity similarity
        bar_w = self.side_w - text_x - 15
        draw_progress_bar(panel, text_x, photo_y + 92, bar_w, 10, celeb_prob, 10.0, (0, 210, 170))

        # Top Matching Biometric Trait Highlights
        if top_traits:
            t1, t1_val = top_traits[0]
            t2, t2_val = top_traits[1] if len(top_traits) > 1 else ("", 0)
            traits_str = f"Top Traits: {t1} ({t1_val}%)"
            if t2:
                traits_str += f", {t2} ({t2_val}%)"
            cv2.putText(panel, traits_str, (text_x, photo_y + 118),
                        cv2.FONT_HERSHEY_PLAIN, 0.80, (140, 200, 240), 1, cv2.LINE_AA)

        # ----------------- SECTION 2: SKIN EVALUATION -----------------
        y_sec2 = photo_y + photo_size + 25
        cv2.line(panel, (15, y_sec2 - 10), (self.side_w - 15, y_sec2 - 10), (45, 52, 65), 1)

        cv2.putText(panel, "♥ SKIN HEALTH RATING", (20, y_sec2 + 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.48, (255, 120, 160), 1, cv2.LINE_AA)

        # Overall Score Pill / Badge
        score_val = skin_report.get('overall_score', 8.0)
        status_text = skin_report.get('status', 'Healthy & Smooth')
        glow_col = skin_report.get('glow_color', (0, 220, 120))

        cv2.rectangle(panel, (20, y_sec2 + 20), (145, y_sec2 + 68), (32, 38, 48), -1)
        cv2.rectangle(panel, (20, y_sec2 + 20), (145, y_sec2 + 68), glow_col, 2)
        cv2.putText(panel, f"{score_val:.1f}", (32, y_sec2 + 58),
                    cv2.FONT_HERSHEY_DUPLEX, 1.15, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, "/10", (98, y_sec2 + 56),
                    cv2.FONT_HERSHEY_PLAIN, 1.1, (170, 180, 195), 1, cv2.LINE_AA)

        # Status text beside score
        cv2.putText(panel, "Complexion Grade:", (160, y_sec2 + 38),
                    cv2.FONT_HERSHEY_PLAIN, 0.95, (160, 175, 190), 1, cv2.LINE_AA)
        cv2.putText(panel, status_text, (160, y_sec2 + 58),
                    cv2.FONT_HERSHEY_DUPLEX, 0.46, glow_col, 1, cv2.LINE_AA)

        # Detailed Sub-Metric Bars
        metrics = [
            ("Smoothness", skin_report.get('smoothness', 8.0), (80, 210, 255)),
            ("Even Tone", skin_report.get('uniformity', 8.2), (150, 230, 110)),
            ("Blemish Clarity", skin_report.get('clarity', 8.5), (255, 180, 100)),
        ]

        bar_y_start = y_sec2 + 82
        for idx, (label, val, bar_col) in enumerate(metrics):
            by = bar_y_start + idx * 28
            cv2.putText(panel, f"{label}: {val:.1f}/10", (22, by + 10),
                        cv2.FONT_HERSHEY_PLAIN, 0.95, (215, 220, 230), 1, cv2.LINE_AA)
            draw_progress_bar(panel, 180, by, self.side_w - 200, 10, val, 10.0, bar_col)

        # ----------------- SECTION 3: AGE ESTIMATION -----------------
        y_sec3 = bar_y_start + len(metrics) * 28 + 15
        cv2.line(panel, (15, y_sec3 - 8), (self.side_w - 15, y_sec3 - 8), (45, 52, 65), 1)

        cv2.putText(panel, "⌚ AGE ESTIMATION", (20, y_sec3 + 12),
                    cv2.FONT_HERSHEY_DUPLEX, 0.48, (240, 180, 90), 1, cv2.LINE_AA)

        age_val = age_report.get('estimated_age', 24)
        age_range = age_report.get('age_range', '22 - 26 yrs')
        age_desc = age_report.get('descriptor', 'Youthful & Fresh')

        # Age badge
        cv2.rectangle(panel, (20, y_sec3 + 22), (145, y_sec3 + 70), (32, 38, 48), -1)
        cv2.rectangle(panel, (20, y_sec3 + 22), (145, y_sec3 + 70), (240, 180, 90), 2)
        cv2.putText(panel, f"~{age_val}", (30, y_sec3 + 60),
                    cv2.FONT_HERSHEY_DUPLEX, 1.15, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, "yrs", (102, y_sec3 + 58),
                    cv2.FONT_HERSHEY_PLAIN, 1.05, (170, 180, 195), 1, cv2.LINE_AA)

        # Age details
        cv2.putText(panel, f"Bracket: {age_range}", (160, y_sec3 + 42),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (210, 220, 235), 1, cv2.LINE_AA)
        cv2.putText(panel, f"Texture: {age_desc}", (160, y_sec3 + 62),
                    cv2.FONT_HERSHEY_PLAIN, 0.95, (160, 210, 240), 1, cv2.LINE_AA)

        # ----------------- SECTION 4: CONTROLS & SNAPSHOT -----------------
        footer_y = self.total_h - 60
        cv2.line(panel, (15, footer_y - 6), (self.side_w - 15, footer_y - 6), (45, 52, 65), 1)

        # Snapshot banner if recently saved
        if time.time() - self.snapshot_message_time < 3.0:
            cv2.rectangle(panel, (15, footer_y), (self.side_w - 15, footer_y + 48), (20, 80, 40), -1)
            cv2.putText(panel, "✓ SNAPSHOT SAVED TO snapshots/!", (25, footer_y + 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.42, (100, 255, 150), 1, cv2.LINE_AA)
        else:
            cv2.putText(panel, "[Q] Quit  |  [S] Save Snapshot  |  [M] Mask", (20, footer_y + 20),
                        cv2.FONT_HERSHEY_PLAIN, 0.95, (180, 190, 205), 1, cv2.LINE_AA)
            cv2.putText(panel, "[Space] Pause Analysis  |  Auto-Lock Top Match", (20, footer_y + 40),
                        cv2.FONT_HERSHEY_PLAIN, 0.90, (140, 150, 170), 1, cv2.LINE_AA)

        return panel

    def process_frame(self, frame):
        """Processes a single video frame and produces the combined dashboard."""
        frame = cv2.resize(frame, (self.cam_w, self.cam_h))
        vis_frame = frame.copy()

        # High-Accuracy Face & Landmark Detection
        detailed_faces = self.face_detector.detect_faces_detailed(frame, scale_factor=1.12, min_neighbors=5, min_size=(70, 70))
        has_face = len(detailed_faces) > 0

        if has_face:
            primary_face = max(detailed_faces, key=lambda d: d["box"][2] * d["box"][3])
            fx, fy, fw, fh = primary_face["box"]
            landmarks = primary_face.get("landmarks")
            self.last_face_time = time.time()

            fx1, fy1 = max(0, fx), max(0, fy)
            fx2, fy2 = min(frame.shape[1], fx + fw), min(frame.shape[0], fy + fh)
            face_crop = frame[fy1:fy2, fx1:fx2]

            crop_landmarks = None
            if landmarks is not None:
                crop_landmarks = landmarks.copy()
                crop_landmarks[:, 0] -= fx1
                crop_landmarks[:, 1] -= fy1

            # Extract 128-D SFace Deep Identity Embedding on 5-point aligned face
            deep_embedding = self.celeb_matcher.extract_deep_embedding(face_crop, landmarks=crop_landmarks)

            # Run Biometric AI Engines
            skin_report = self.skin_analyzer.evaluate_face(face_crop, landmarks=crop_landmarks)
            age_report = self.age_estimator.estimate(face_crop)
            celeb_dict, celeb_prob, celeb_img, top_traits = self.celeb_matcher.match(
                face_crop,
                landmarks=crop_landmarks,
                query_embedding=deep_embedding
            )

            self.cached_analysis = (celeb_dict, celeb_prob, celeb_img, skin_report, age_report, top_traits)

            # Draw sci-fi corner brackets around face
            bracket_color = skin_report.get('glow_color', (0, 225, 255))
            draw_corner_brackets(vis_frame, (fx, fy), (fx + fw, fy + fh), bracket_color, d=20, thickness=2)

            # Draw subtle biometric landmark tracking points
            if landmarks is not None:
                for pt in landmarks:
                    px, py = int(round(pt[0])), int(round(pt[1]))
                    cv2.circle(vis_frame, (px, py), 2, (0, 240, 255), -1, cv2.LINE_AA)

            # Highlight skin diagnostic ROIs on camera feed
            rois = skin_report.get('rois', {})
            for roi_name, roi_data in rois.items():
                rx, ry, rw, rh = roi_data['box']
                rx_abs = fx1 + rx
                ry_abs = fy1 + ry
                cv2.rectangle(vis_frame, (rx_abs, ry_abs), (rx_abs + rw, ry_abs + rh), (0, 255, 200), 1)

            # Optional: mask mode overlay
            if self.show_mask_mode:
                mask = self.skin_analyzer.segment_skin(face_crop)
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                mask_bgr[:, :, 1] = np.clip(mask_bgr[:, :, 1] * 1.5, 0, 255).astype(np.uint8)
                vis_frame[fy1:fy2, fx1:fx2] = cv2.addWeighted(face_crop, 0.5, mask_bgr, 0.5, 0)

            # Face label tag
            cv2.putText(vis_frame, f"Skin: {skin_report['overall_score']}/10 | Age: ~{age_report['estimated_age']} ({age_report.get('gender', 'Adult')})",
                        (fx, max(25, fy - 10)), cv2.FONT_HERSHEY_DUPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        else:
            if self.cached_analysis is None:
                default_skin = self.skin_analyzer._default_result()
                default_age = {'estimated_age': 24, 'age_range': '22 - 26 yrs', 'descriptor': 'Analyzing...', 'wrinkle_level': 'Normal'}
                celeb_dict, celeb_prob, celeb_img, top_traits = self.celeb_matcher.match(None)
                self.cached_analysis = (celeb_dict, celeb_prob, celeb_img, default_skin, default_age, top_traits)

            cv2.putText(vis_frame, "Looking for face... Please look at camera",
                        (40, self.cam_h // 2), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 200, 255), 1, cv2.LINE_AA)

        celeb_dict, celeb_prob, celeb_img, skin_report, age_report, top_traits = self.cached_analysis

        # Overlay Camera HUD pill on top left
        hud_bg = (20, 24, 30)
        cv2.rectangle(vis_frame, (10, 10), (280, 42), hud_bg, -1)
        status_dot_col = (0, 240, 100) if has_face else (0, 160, 255)
        cv2.circle(vis_frame, (26, 26), 6, status_dot_col, -1, cv2.LINE_AA)
        status_label = "FACE DETECTED" if has_face else "SEARCHING..."
        cv2.putText(vis_frame, f"{status_label} | {self.fps:.0f} FPS", (40, 31),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, (230, 235, 245), 1, cv2.LINE_AA)

        # Generate Sidebar with top traits
        sidebar = self.render_sidebar(celeb_dict, celeb_prob, celeb_img, skin_report, age_report, top_traits=top_traits)

        # Match heights exactly before side-by-side concatenation
        if vis_frame.shape[0] != sidebar.shape[0]:
            target_h = sidebar.shape[0]
            new_w = int(vis_frame.shape[1] * (target_h / float(vis_frame.shape[0])))
            vis_frame = cv2.resize(vis_frame, (new_w, target_h))

        # Combine side-by-side
        combined = np.hstack([vis_frame, sidebar])
        return combined

    def save_snapshot(self, dashboard_img):
        """Saves current combined dashboard image to snapshots/ folder."""
        if dashboard_img is None:
            return
        base_dir = os.path.dirname(os.path.abspath(__file__))
        snapshots_dir = os.path.join(base_dir, "snapshots")
        try:
            os.makedirs(snapshots_dir, exist_ok=True)
        except OSError:
            pass
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(snapshots_dir, f"skin_celeb_eval_{timestamp}.jpg")
        try:
            cv2.imwrite(filename, dashboard_img)
            self.snapshot_message_time = time.time()
            print(f"[Snapshot] Successfully saved to {filename}")
        except Exception as e:
            print(f"[Snapshot] Error saving snapshot: {e}")

    def run_webcam(self):
        """Main real-time webcam loop."""
        print(f"[FaceEvaluationApp] Initializing camera index {self.camera_index}...")
        cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            print(f"[FaceEvaluationApp] Error: Camera index {self.camera_index} could not be opened.")
            print("[FaceEvaluationApp] Falling back to interactive DEMO mode with synthetic test feed...")
            self.run_demo_feed()
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_h)

        window_name = "Skin Rating & Celebrity Matcher"
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        except cv2.error as e:
            print(f"\n[Display Error] Could not create GUI window ({e}).")
            print("Tip: If you are running inside WSL or headless environment, ensure your X11/Wayland display server is running.\n")
            cap.release()
            return

        frame_count = 0
        t0 = time.time()
        dashboard = None

        print("[FaceEvaluationApp] Running real-time camera. Press 'Q' to quit, 'S' for snapshot, 'C' to cycle celeb.")

        try:
            while True:
                if not self.paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("[FaceEvaluationApp] Warning: Failed to grab camera frame.")
                        time.sleep(0.05)
                        continue

                    # FPS Calculation
                    frame_count += 1
                    elapsed = time.time() - t0
                    if elapsed >= 1.0:
                        self.fps = frame_count / elapsed
                        frame_count = 0
                        t0 = time.time()

                    dashboard = self.process_frame(frame)
                    try:
                        cv2.imshow(window_name, dashboard)
                    except cv2.error:
                        pass

                # Key handling
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), ord('Q'), 27]:
                    break
                elif key in [ord('s'), ord('S')]:
                    self.save_snapshot(dashboard)
                elif key in [ord('c'), ord('C')]:
                    next_celeb = self.celeb_matcher.cycle_next()
                    print(f"[Celebrity] Manually cycled to: {next_celeb['name']}")
                elif key in [ord('m'), ord('M')]:
                    self.show_mask_mode = not self.show_mask_mode
                    print(f"[Skin] Mask visualization mode: {'ON' if self.show_mask_mode else 'OFF'}")
                elif key == 32:
                    self.paused = not self.paused

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("[FaceEvaluationApp] Closed cleanly.")

    def run_demo_feed(self, image_path=None):
        """Runs the pipeline on a test image or synthetic animated test face."""
        if image_path and os.path.exists(image_path):
            base_frame = cv2.imread(image_path)
            if base_frame is None:
                base_frame = self._generate_synthetic_face()
        else:
            base_frame = self._generate_synthetic_face()

        window_name = "Skin Rating & Celebrity Matcher [DEMO MODE]"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        frame_count = 0
        t0 = time.time()
        print("[FaceEvaluationApp] Running in DEMO mode. Press 'Q' to quit, 'S' for snapshot, 'C' to cycle celeb.")

        try:
            while True:
                frame = base_frame.copy()

                pulse = np.sin(time.time() * 2.0) * 8
                frame = np.clip(frame.astype(np.float32) + pulse, 0, 255).astype(np.uint8)

                frame_count += 1
                elapsed = time.time() - t0
                if elapsed >= 1.0:
                    self.fps = frame_count / elapsed
                    frame_count = 0
                    t0 = time.time()

                dashboard = self.process_frame(frame)
                cv2.imshow(window_name, dashboard)

                key = cv2.waitKey(30) & 0xFF
                if key in [ord('q'), ord('Q'), 27]:
                    break
                elif key in [ord('s'), ord('S')]:
                    self.save_snapshot(dashboard)
                elif key in [ord('c'), ord('C')]:
                    self.celeb_matcher.cycle_next()
                elif key in [ord('m'), ord('M')]:
                    self.show_mask_mode = not self.show_mask_mode
                elif key == 32:
                    self.paused = not self.paused

        except Exception as e:
            print(f"[Demo] Headless display note: {e}")
        finally:
            cv2.destroyAllWindows()

    def _generate_synthetic_face(self):
        """Generates a realistic test face pattern for headless or test validation."""
        h, w = self.cam_h, self.cam_w
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:] = (35, 40, 48)

        cx, cy = w // 2, h // 2
        rx, ry = 95, 125

        skin_bgr = (175, 195, 235)
        cv2.ellipse(canvas, (cx, cy), (rx, ry), 0, 0, 360, skin_bgr, -1, cv2.LINE_AA)

        cv2.ellipse(canvas, (cx, cy - 40), (rx + 10, ry - 35), 0, 180, 360, (25, 30, 45), -1, cv2.LINE_AA)

        eye_y = cy - 20
        cv2.ellipse(canvas, (cx - 38, eye_y), (14, 8), 0, 0, 360, (245, 245, 245), -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx - 38, eye_y), 5, (70, 45, 30), -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (cx + 38, eye_y), (14, 8), 0, 0, 360, (245, 245, 245), -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx + 38, eye_y), 5, (70, 45, 30), -1, cv2.LINE_AA)

        cv2.ellipse(canvas, (cx - 38, eye_y - 12), (18, 5), 10, 180, 360, (40, 45, 55), 2, cv2.LINE_AA)
        cv2.ellipse(canvas, (cx + 38, eye_y - 12), (18, 5), -10, 180, 360, (40, 45, 55), 2, cv2.LINE_AA)

        cv2.line(canvas, (cx, cy - 10), (cx, cy + 22), (150, 170, 210), 2, cv2.LINE_AA)
        cv2.ellipse(canvas, (cx, cy + 22), (10, 4), 0, 0, 180, (140, 160, 200), -1, cv2.LINE_AA)

        cv2.ellipse(canvas, (cx, cy + 55), (24, 8), 0, 0, 360, (110, 115, 195), -1, cv2.LINE_AA)

        return canvas


def main():
    parser = argparse.ArgumentParser(description="OpenCV Face & Skin Evaluator + Celebrity Matcher")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--demo", action="store_true", help="Run with simulated test feed")
    parser.add_argument("--image", type=str, default=None, help="Process a single test image file")
    parser.add_argument("--indian-only", action="store_true", help="Filter to Indian celebrities & influencers")
    parser.add_argument("--target-prob", type=float, default=7.1, help="Target celebrity match score /10 (default: 7.1)")
    args = parser.parse_args()

    app = FaceEvaluationApp(camera_index=args.camera, target_prob=args.target_prob, indian_only=args.indian_only)

    if args.image:
        print(f"[FaceEvaluationApp] Processing static image: {args.image}")
        img = cv2.imread(args.image)
        if img is None:
            print(f"Error: Could not load image {args.image}")
            sys.exit(1)
        dashboard = app.process_frame(img)
        app.save_snapshot(dashboard)
        print("[FaceEvaluationApp] Static image evaluated and snapshot saved.")
    elif args.demo:
        app.run_demo_feed()
    else:
        app.run_webcam()


if __name__ == "__main__":
    main()
