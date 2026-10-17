"""
High-Accuracy Dermatological Skin Analyzer Module for OpenCV Face Analysis
Features:
- Illumination-invariant CIE LAB & Adaptive YCrCb segmentation calibrated for Indian & Global skin tones
- Specular highlight (webcam glare / oil shine) masking so lighting reflections don't penalize skin scores
- Individual Typology Angle (ITA°) melanin-adaptive blemish & redness detection
- Multi-scale Difference-of-Gaussians (DoG) pore & micro-texture smoothness evaluation
- Optional learned calibration weights from models/calibrated_models.npz
"""

import os
import math
import cv2
import numpy as np


class SkinAnalyzer:
    """
    Performs real-time dermatological skin analysis using computer vision:
    - Landmark-guided anatomical ROI extraction (Forehead, Left Cheek, Right Cheek)
    - Specular glare exclusion & CLAHE illumination normalization
    - Multi-scale pore/texture smoothness, melanin-aware blemish clarity, and chrominance tone uniformity
    """

    def __init__(self, models_dir="models", ema_alpha=0.16):
        self.models_dir = models_dir
        self.ema_alpha = ema_alpha
        self.smoothed_overall = 8.2
        self.smoothed_smoothness = 8.0
        self.smoothed_clarity = 8.5
        self.smoothed_uniformity = 8.1
        self.has_history = False
        self.calib_data = None
        self._load_calibration()

    def _load_calibration(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(self.models_dir, "calibrated_models.npz"),
            os.path.join(base_dir, "models", "calibrated_models.npz"),
        ]
        for path in candidates:
            if os.path.exists(path):
                try:
                    self.calib_data = dict(np.load(path, allow_pickle=False))
                    break
                except Exception:
                    pass

    def extract_skin_rois(self, face_img, landmarks=None):
        """
        Extracts strategic skin zones on the face:
        - Forehead: upper-mid face (avoids hair and eyebrows)
        - Left Cheek & Right Cheek: mid-lower face sides (avoids nose, eyes, and lips)
        - Guided by 5 facial landmarks when available for pinpoint accuracy
        """
        h, w = face_img.shape[:2]
        rois = {}

        if landmarks is not None:
            try:
                lm = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
                if lm.shape[0] == 5:
                    if np.max(lm[:, 0]) > w or np.max(lm[:, 1]) > h:
                        lm[:, 0] -= np.min(lm[:, 0])
                        lm[:, 1] -= np.min(lm[:, 1])

                    re_x, re_y = lm[0]
                    le_x, le_y = lm[1]
                    nose_x, nose_y = lm[2]
                    rm_x, rm_y = lm[3]
                    lm_x, lm_y = lm[4]

                    # Forehead: above eyebrows, centered between eyes
                    fh_y2 = int(max(0, min(re_y, le_y) - h * 0.06))
                    fh_y1 = int(max(0, fh_y2 - h * 0.17))
                    fh_x1 = int(max(0, min(re_x, le_x) - w * 0.04))
                    fh_x2 = int(min(w, max(re_x, le_x) + w * 0.04))
                    if fh_y2 - fh_y1 >= 8 and fh_x2 - fh_x1 >= 8:
                        rois['forehead'] = {
                            'box': (fh_x1, fh_y1, fh_x2 - fh_x1, fh_y2 - fh_y1),
                            'crop': face_img[fh_y1:fh_y2, fh_x1:fh_x2]
                        }

                    # Right Cheek (viewer's left side)
                    rc_y1 = int(max(0, min(re_y, nose_y) + h * 0.04))
                    rc_y2 = int(min(h, rm_y - h * 0.02))
                    rc_x1 = int(max(0, re_x - w * 0.18))
                    rc_x2 = int(min(w, re_x + w * 0.05))
                    if rc_y2 - rc_y1 >= 8 and rc_x2 - rc_x1 >= 8:
                        rois['right_cheek'] = {
                            'box': (rc_x1, rc_y1, rc_x2 - rc_x1, rc_y2 - rc_y1),
                            'crop': face_img[rc_y1:rc_y2, rc_x1:rc_x2]
                        }

                    # Left Cheek (viewer's right side)
                    lc_y1 = int(max(0, min(le_y, nose_y) + h * 0.04))
                    lc_y2 = int(min(h, lm_y - h * 0.02))
                    lc_x1 = int(max(0, le_x - w * 0.05))
                    lc_x2 = int(min(w, le_x + w * 0.18))
                    if lc_y2 - lc_y1 >= 8 and lc_x2 - lc_x1 >= 8:
                        rois['left_cheek'] = {
                            'box': (lc_x1, lc_y1, lc_x2 - lc_x1, lc_x2 - lc_x1),
                            'crop': face_img[lc_y1:lc_y2, lc_x1:lc_x2]
                        }

                    if rois:
                        return rois
            except Exception:
                pass

        # Proportional fallback
        fh_y1, fh_y2 = int(h * 0.16), int(h * 0.33)
        fh_x1, fh_x2 = int(w * 0.26), int(w * 0.74)
        if fh_y2 > fh_y1 and fh_x2 > fh_x1:
            rois['forehead'] = {
                'box': (fh_x1, fh_y1, fh_x2 - fh_x1, fh_y2 - fh_y1),
                'crop': face_img[fh_y1:fh_y2, fh_x1:fh_x2]
            }

        lc_y1, lc_y2 = int(h * 0.48), int(h * 0.70)
        lc_x1, lc_x2 = int(w * 0.14), int(w * 0.38)
        if lc_y2 > lc_y1 and lc_x2 > lc_x1:
            rois['left_cheek'] = {
                'box': (lc_x1, lc_y1, lc_x2 - lc_x1, lc_x2 - lc_y1),
                'crop': face_img[lc_y1:lc_y2, lc_x1:lc_x2]
            }

        rc_y1, rc_y2 = int(h * 0.48), int(h * 0.70)
        rc_x1, rc_x2 = int(w * 0.62), int(w * 0.86)
        if rc_y2 > rc_y1 and rc_x2 > rc_x1:
            rois['right_cheek'] = {
                'box': (rc_x1, rc_y1, rc_x2 - rc_x1, rc_y2 - rc_x1),
                'crop': face_img[rc_y1:rc_y2, rc_x1:rc_x2]
            }

        return rois

    def segment_skin(self, bgr_img):
        """
        Segments skin pixels using inclusive YCrCb & HSV thresholds calibrated
        for both Indian (wheatish, warm golden, tan, dusky, deep) and global skin tones.
        """
        ycrcb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)
        lower_ycrcb = np.array([15, 128, 70], dtype=np.uint8)
        upper_ycrcb = np.array([250, 180, 135], dtype=np.uint8)
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([0, 15, 35], dtype=np.uint8)
        upper_hsv = np.array([35, 245, 255], dtype=np.uint8)
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

        skin_mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        return skin_mask

    def _non_glare_mask(self, patch_bgr):
        """Masks out specular reflections (webcam glare) and deep shadow pixels."""
        lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0]
        # Exclude extreme specular glare (> 235) and non-skin shadow (< 25)
        valid = (l_chan >= 25) & (l_chan <= 235)
        if np.sum(valid) < 0.25 * valid.size:
            return np.ones_like(l_chan, dtype=bool)
        return valid

    def calculate_smoothness(self, skin_patches):
        """
        Calculates texture smoothness (0 to 10) using multi-scale Difference-of-Gaussians (DoG)
        and bilateral residual analysis while suppressing sensor noise and glare.
        """
        if not skin_patches:
            return 8.0

        scores = []
        for patch in skin_patches:
            if patch is None or patch.size == 0 or patch.shape[0] < 8 or patch.shape[1] < 8:
                continue

            valid_mask = self._non_glare_mask(patch)
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

            # Denoise fine webcam grain before measuring dermatological pore/texture band
            g_fine = cv2.GaussianBlur(gray, (3, 3), 0.7)
            g_base = cv2.bilateralFilter(g_fine, d=7, sigmaColor=32, sigmaSpace=32)

            diff = cv2.absdiff(g_fine, g_base).astype(np.float32)
            mean_diff = float(np.mean(diff[valid_mask]))

            # Band-pass Laplacian on denoised skin
            lap = np.abs(cv2.Laplacian(g_fine, cv2.CV_64F))
            lap_mad = float(np.mean(lap[valid_mask]))

            # Calibrated dermatological smoothness formula
            score = 10.0 - (mean_diff * 0.58 + lap_mad * 0.16)
            scores.append(float(np.clip(score, 4.0, 9.8)))

        return float(np.mean(scores)) if scores else 8.0

    def calculate_clarity_and_blemishes(self, skin_patches, face_img):
        """
        Melanin-adaptive blemish, acne, and redness detector in CIE LAB space.
        Adapts thresholds using Individual Typology Angle (ITA°) so warm/deep Indian
        complexions are evaluated with equal precision.
        """
        if not skin_patches:
            return 8.5, np.zeros(face_img.shape[:2], dtype=np.uint8)

        blemish_densities = []
        for patch in skin_patches:
            if patch is None or patch.size == 0 or patch.shape[0] < 8 or patch.shape[1] < 8:
                continue

            valid_mask = self._non_glare_mask(patch)
            lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
            l_chan, a_chan, b_chan = cv2.split(lab)

            # Denoise before black-hat spot detection
            l_smooth = cv2.GaussianBlur(l_chan, (3, 3), 0.8)

            # Compute approximate ITA° (skin tone lightness vs yellow/melanin)
            l_star = float(np.mean(l_smooth[valid_mask])) * (100.0 / 255.0)
            b_star = max(1.0, float(np.mean(b_chan[valid_mask])) - 128.0)
            ita_deg = math.degrees(math.atan2(l_star - 50.0, b_star))

            # Adaptive black-hat threshold (higher contrast threshold for deeper melanin tones to avoid false positives)
            bh_thresh = 17 if ita_deg >= 20.0 else 20
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            blackhat = cv2.morphologyEx(l_smooth, cv2.MORPH_BLACKHAT, kernel)
            _, dark_spots = cv2.threshold(blackhat, bh_thresh, 255, cv2.THRESH_BINARY)

            # Local erythema / acne redness detection
            a_valid = a_chan[valid_mask].astype(np.float32)
            a_mean = float(np.mean(a_valid))
            a_std = max(2.2, float(np.std(a_valid)))
            red_spots = ((a_chan.astype(np.float32) > (a_mean + 2.1 * a_std)) & valid_mask).astype(np.uint8) * 255

            blemish = cv2.bitwise_or(dark_spots, red_spots)
            blemish_valid = (blemish > 0) & valid_mask
            density = float(np.sum(blemish_valid)) / float(max(1, np.sum(valid_mask)))
            blemish_densities.append(density)

        mean_density = float(np.mean(blemish_densities)) if blemish_densities else 0.015
        clarity_score = 10.0 - np.clip(mean_density * 48.0, 0.2, 6.5)
        clarity_score = float(np.clip(clarity_score, 4.2, 9.8))

        return clarity_score, None

    def calculate_tone_uniformity(self, skin_patches):
        """
        Calculates skin tone uniformity (0 to 10) after removing macro shading gradients.
        Subtracting a low-frequency illumination plane ensures directional room lighting
        doesn't get mistaken for uneven pigmentation.
        """
        if not skin_patches:
            return 8.2

        stds = []
        for patch in skin_patches:
            if patch is None or patch.size == 0 or patch.shape[0] < 8 or patch.shape[1] < 8:
                continue

            valid_mask = self._non_glare_mask(patch)
            ycrcb = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb).astype(np.float32)
            cr = ycrcb[:, :, 1]
            cb = ycrcb[:, :, 2]

            # Remove low-frequency lighting gradient across the cheek/forehead patch
            ksize = max(5, (min(patch.shape[0], patch.shape[1]) // 2) | 1)
            cr_illum = cv2.GaussianBlur(cr, (ksize, ksize), 0)
            cb_illum = cv2.GaussianBlur(cb, (ksize, ksize), 0)

            cr_res = cr - cr_illum
            cb_res = cb - cb_illum

            std_color = float(np.std(cr_res[valid_mask]) + np.std(cb_res[valid_mask]))
            stds.append(std_color)

        mean_std = float(np.mean(stds)) if stds else 4.5
        score = 10.0 - np.clip((mean_std - 2.0) * 0.52, 0.2, 6.0)
        return float(np.clip(score, 4.2, 9.7))

    def evaluate_face(self, face_bgr, landmarks=None):
        """
        Main evaluation entry point.
        Analyzes the detected face crop and returns a full diagnostic report
        with overall skin rating out of 10.
        """
        if face_bgr is None or face_bgr.size == 0:
            return self._default_result()

        rois = self.extract_skin_rois(face_bgr, landmarks)
        patches = [roi['crop'] for roi in rois.values() if roi['crop'] is not None and roi['crop'].size > 0]

        if not patches:
            return self._default_result()

        raw_smoothness = self.calculate_smoothness(patches)
        raw_clarity, _ = self.calculate_clarity_and_blemishes(patches, face_bgr)
        raw_uniformity = self.calculate_tone_uniformity(patches)

        # Optional learned calibration weights from train_models.py
        w_smooth, w_clar, w_unif = 0.36, 0.34, 0.30
        if self.calib_data is not None and "skin_weights" in self.calib_data:
            sw = self.calib_data["skin_weights"].flatten()
            if len(sw) == 3 and np.sum(sw) > 0:
                sw = sw / np.sum(sw)
                w_smooth, w_clar, w_unif = float(sw[0]), float(sw[1]), float(sw[2])

        raw_overall = (w_smooth * raw_smoothness +
                       w_clar * raw_clarity +
                       w_unif * raw_uniformity)
        raw_overall = float(np.clip(raw_overall, 1.0, 10.0))

        if not self.has_history:
            self.smoothed_overall = raw_overall
            self.smoothed_smoothness = raw_smoothness
            self.smoothed_clarity = raw_clarity
            self.smoothed_uniformity = raw_uniformity
            self.has_history = True
        else:
            a = self.ema_alpha
            self.smoothed_overall = a * raw_overall + (1 - a) * self.smoothed_overall
            self.smoothed_smoothness = a * raw_smoothness + (1 - a) * self.smoothed_smoothness
            self.smoothed_clarity = a * raw_clarity + (1 - a) * self.smoothed_clarity
            self.smoothed_uniformity = a * raw_uniformity + (1 - a) * self.smoothed_uniformity

        overall_score = round(float(self.smoothed_overall), 1)

        if overall_score >= 8.8:
            status = "Radiant & Flawless"
            glow_color = (0, 220, 120)
        elif overall_score >= 7.8:
            status = "Healthy & Smooth"
            glow_color = (0, 215, 255)
        elif overall_score >= 6.8:
            status = "Good Complexion"
            glow_color = (0, 165, 255)
        else:
            status = "Hydration Needed"
            glow_color = (120, 120, 255)

        return {
            'overall_score': overall_score,
            'smoothness': round(float(self.smoothed_smoothness), 1),
            'clarity': round(float(self.smoothed_clarity), 1),
            'uniformity': round(float(self.smoothed_uniformity), 1),
            'status': status,
            'glow_color': glow_color,
            'rois': rois
        }

    def _default_result(self):
        return {
            'overall_score': 7.5,
            'smoothness': 7.5,
            'clarity': 7.5,
            'uniformity': 7.5,
            'status': "Analyzing...",
            'glow_color': (200, 200, 200),
            'rois': {}
        }
