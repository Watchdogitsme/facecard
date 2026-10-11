"""
Skin Analyzer Module for OpenCV Face Analysis
Evaluates skin quality, texture smoothness, blemishes, and tone uniformity.
Outputs a calibrated Skin Rating out of 10.
"""

import cv2
import numpy as np


class SkinAnalyzer:
    """
    Performs real-time biometric and dermatological skin analysis
    using computer vision techniques (YCrCb/LAB color segmentation,
    Laplacian gradient variance, Black-hat blemish filtering, and tone uniformity).
    """

    def __init__(self, ema_alpha=0.15):
        self.ema_alpha = ema_alpha
        # Smoothed scores for stable video rendering
        self.smoothed_overall = 8.2
        self.smoothed_smoothness = 8.0
        self.smoothed_clarity = 8.5
        self.smoothed_uniformity = 8.1
        self.has_history = False

    def extract_skin_rois(self, face_img, landmarks=None):
        """
        Extracts strategic skin zones on the face:
        - Forehead: upper-mid face (avoids hair and eyes)
        - Left Cheek & Right Cheek: mid-lower face sides (avoids nose and lips)
        - Guided by facial landmarks when available for pinpoint accuracy
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

                    # Forehead: above eyes, centered between eyes
                    fh_y2 = int(max(0, min(re_y, le_y) - h * 0.04))
                    fh_y1 = int(max(0, fh_y2 - h * 0.18))
                    fh_x1 = int(max(0, min(re_x, le_x) - w * 0.05))
                    fh_x2 = int(min(w, max(re_x, le_x) + w * 0.05))
                    if fh_y2 > fh_y1 and fh_x2 > fh_x1:
                        rois['forehead'] = {
                            'box': (fh_x1, fh_y1, fh_x2 - fh_x1, fh_y2 - fh_y1),
                            'crop': face_img[fh_y1:fh_y2, fh_x1:fh_x2]
                        }

                    # Right Cheek (viewer's left side): between right eye/nose and right mouth corner
                    rc_y1 = int(max(0, min(re_y, nose_y) + h * 0.02))
                    rc_y2 = int(min(h, rm_y))
                    rc_x1 = int(max(0, re_x - w * 0.20))
                    rc_x2 = int(min(w, re_x + w * 0.06))
                    if rc_y2 > rc_y1 and rc_x2 > rc_x1:
                        rois['right_cheek'] = {
                            'box': (rc_x1, rc_y1, rc_x2 - rc_x1, rc_y2 - rc_y1),
                            'crop': face_img[rc_y1:rc_y2, rc_x1:rc_x2]
                        }

                    # Left Cheek (viewer's right side): between left eye/nose and left mouth corner
                    lc_y1 = int(max(0, min(le_y, nose_y) + h * 0.02))
                    lc_y2 = int(min(h, lm_y))
                    lc_x1 = int(max(0, le_x - w * 0.06))
                    lc_x2 = int(min(w, le_x + w * 0.20))
                    if lc_y2 > lc_y1 and lc_x2 > lc_x1:
                        rois['left_cheek'] = {
                            'box': (lc_x1, lc_y1, lc_x2 - lc_x1, lc_y2 - lc_y1),
                            'crop': face_img[lc_y1:lc_y2, lc_x1:lc_x2]
                        }

                    if rois:
                        return rois
            except Exception:
                pass

        # Proportional fallback
        # Forehead (y: 18% to 35%, x: 25% to 75%)
        fh_y1, fh_y2 = int(h * 0.18), int(h * 0.35)
        fh_x1, fh_x2 = int(w * 0.25), int(w * 0.75)
        if fh_y2 > fh_y1 and fh_x2 > fh_x1:
            rois['forehead'] = {
                'box': (fh_x1, fh_y1, fh_x2 - fh_x1, fh_y2 - fh_y1),
                'crop': face_img[fh_y1:fh_y2, fh_x1:fh_x2]
            }

        # Left Cheek (y: 50% to 72%, x: 14% to 38%)
        lc_y1, lc_y2 = int(h * 0.50), int(h * 0.72)
        lc_x1, lc_x2 = int(w * 0.14), int(w * 0.38)
        if lc_y2 > lc_y1 and lc_x2 > lc_x1:
            rois['left_cheek'] = {
                'box': (lc_x1, lc_y1, lc_x2 - lc_x1, lc_y2 - lc_y1),
                'crop': face_img[lc_y1:lc_y2, lc_x1:lc_x2]
            }

        # Right Cheek (y: 50% to 72%, x: 62% to 86%)
        rc_y1, rc_y2 = int(h * 0.50), int(h * 0.72)
        rc_x1, rc_x2 = int(w * 0.62), int(w * 0.86)
        if rc_y2 > rc_y1 and rc_x2 > rc_x1:
            rois['right_cheek'] = {
                'box': (rc_x1, rc_y1, rc_x2 - rc_x1, rc_y2 - rc_y1),
                'crop': face_img[rc_y1:rc_y2, rc_x1:rc_x2]
            }

        return rois

    def segment_skin(self, bgr_img):
        """
        Segments skin pixels using YCrCb & HSV color spaces.
        Effective across diverse ethnicities and skin tones.
        """
        # Convert to YCrCb
        ycrcb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)
        # Standard skin chrominance threshold
        lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
        upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

        # Convert to HSV for auxiliary filtering (hue in skin range, moderate saturation)
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([0, 20, 50], dtype=np.uint8)
        upper_hsv = np.array([30, 240, 255], dtype=np.uint8)
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # Combine masks
        skin_mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)

        # Morphological cleanup: remove pepper noise, fill small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        return skin_mask

    def calculate_smoothness(self, skin_patches):
        """
        Calculates texture smoothness (0 to 10) on extracted skin patches.
        Smooth skin has lower high-frequency noise and lower Laplacian variance.
        """
        if not skin_patches:
            return 8.0

        scores = []
        for patch in skin_patches:
            if patch is None or patch.size == 0 or patch.shape[0] < 8 or patch.shape[1] < 8:
                continue
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            # Bilateral filter to simulate ideal smoothed skin
            filtered = cv2.bilateralFilter(gray, d=7, sigmaColor=35, sigmaSpace=35)
            # High-frequency texture is residual difference
            diff = cv2.absdiff(gray, filtered)
            mean_diff = np.mean(diff)

            # Also compute Laplacian variance of the patch
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            lap_var = lap.var()

            # Calibrated smoothness mapping:
            # Low difference (< 2.0) and low lap_var (< 50) means very smooth (~9.0+)
            # Higher roughness reduces the score gracefully
            score = 10.0 - (mean_diff * 0.7 + np.sqrt(max(0, lap_var)) * 0.12)
            score = np.clip(score, 3.0, 9.8)
            scores.append(score)

        return float(np.mean(scores)) if scores else 8.0

    def calculate_clarity_and_blemishes(self, skin_patches, face_img):
        """
        Detects blemishes, spots, and redness.
        Returns clarity score (0 to 10) and a blemish mask for visualization.
        """
        if not skin_patches:
            return 8.5, np.zeros(face_img.shape[:2], dtype=np.uint8)

        blemish_densities = []
        for patch in skin_patches:
            if patch is None or patch.size == 0 or patch.shape[0] < 8 or patch.shape[1] < 8:
                continue
            # Convert to LAB for luminance and redness inspection
            lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
            l_chan, a_chan, _ = cv2.split(lab)

            # Morphological black-hat highlights local dark spots / pigmentation
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            blackhat = cv2.morphologyEx(l_chan, cv2.MORPH_BLACKHAT, kernel)
            _, dark_spots = cv2.threshold(blackhat, 16, 255, cv2.THRESH_BINARY)

            # Redness / irritation detection: pixels significantly redder than mean
            a_mean = np.mean(a_chan)
            a_std = np.std(a_chan)
            red_spots = (a_chan > (a_mean + 1.8 * a_std)).astype(np.uint8) * 255

            # Combined blemish mask
            blemish = cv2.bitwise_or(dark_spots, red_spots)
            density = np.sum(blemish > 0) / float(patch.shape[0] * patch.shape[1])
            blemish_densities.append(density)

        mean_density = np.mean(blemish_densities) if blemish_densities else 0.02
        # Calibrated clarity: density around 0.01 -> score 9.3; density around 0.05 -> score 7.5
        clarity_score = 10.0 - np.clip(mean_density * 55.0, 0.2, 7.0)
        clarity_score = float(np.clip(clarity_score, 3.5, 9.7))

        return clarity_score, None

    def calculate_tone_uniformity(self, skin_patches):
        """
        Calculates skin tone uniformity (0 to 10) based on chrominance variance.
        Uniform complexion has lower color standard deviation across skin zones.
        """
        if not skin_patches:
            return 8.2

        stds = []
        for patch in skin_patches:
            if patch is None or patch.size == 0 or patch.shape[0] < 8 or patch.shape[1] < 8:
                continue
            ycrcb = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
            cr = ycrcb[:, :, 1]
            cb = ycrcb[:, :, 2]
            std_color = np.std(cr) + np.std(cb)
            stds.append(std_color)

        mean_std = np.mean(stds) if stds else 10.0
        # A standard deviation around 6-10 is typical smooth tone -> score ~8.5-9.2
        # Higher std (blotchiness/redness/tan unevenness) -> score drops
        score = 10.0 - np.clip((mean_std - 4.0) * 0.45, 0.3, 6.5)
        score = float(np.clip(score, 3.5, 9.6))

        return score

    def evaluate_face(self, face_bgr, landmarks=None):
        """
        Main evaluation entry point.
        Analyzes the detected face crop and returns a full diagnostic report
        with overall skin rating out of 10.
        """
        if face_bgr is None or face_bgr.size == 0:
            return self._default_result()

        # Extract skin ROIs (forehead, cheeks) guided by landmarks when present
        rois = self.extract_skin_rois(face_bgr, landmarks)
        patches = [roi['crop'] for roi in rois.values() if roi['crop'] is not None and roi['crop'].size > 0]

        if not patches:
            return self._default_result()

        # Compute individual metrics
        raw_smoothness = self.calculate_smoothness(patches)
        raw_clarity, _ = self.calculate_clarity_and_blemishes(patches, face_bgr)
        raw_uniformity = self.calculate_tone_uniformity(patches)

        # Composite overall rating (out of 10)
        # Smoothness: 35%, Clarity: 35%, Uniformity: 30%
        raw_overall = (0.35 * raw_smoothness +
                       0.35 * raw_clarity +
                       0.30 * raw_uniformity)
        raw_overall = float(np.clip(raw_overall, 1.0, 10.0))

        # Exponential Moving Average for smooth, jitter-free video display
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

        # Categorical description
        if overall_score >= 8.8:
            status = "Radiant & Flawless"
            glow_color = (0, 220, 120)  # Green
        elif overall_score >= 7.8:
            status = "Healthy & Smooth"
            glow_color = (0, 215, 255)  # Gold/Yellow
        elif overall_score >= 6.8:
            status = "Good Complexion"
            glow_color = (0, 165, 255)  # Orange
        else:
            status = "Hydration Needed"
            glow_color = (120, 120, 255)  # Soft red

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
