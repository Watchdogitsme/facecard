"""
Age Estimator Module for OpenCV Face Analysis
Estimates facial age using biometric wrinkle analysis, micro-texture gradients,
and facial feature geometry. Also supports OpenCV DNN Age models if available.
"""

import os
import cv2
import numpy as np


class AgeEstimator:
    """
    Real-time Age Estimator using OpenCV computer vision techniques:
    - Forehead horizontal wrinkle density
    - Crow's feet / periocular micro-crease analysis
    - Nasolabial fold prominence
    - Micro-texture Laplacian variance
    - Optional fallback to OpenCV DNN (Caffe) if model files exist
    """

    AGE_INTERVALS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

    def __init__(self, models_dir="models", ema_alpha=0.12):
        self.models_dir = models_dir
        self.ema_alpha = ema_alpha
        self.smoothed_age = 24.0
        self.has_history = False

        # Attempt to load OpenCV DNN Age model if available
        self.dnn_net = None
        self._init_dnn_model()

    def _init_dnn_model(self):
        # 1. Check for OpenVINO/ONNX Retail Age Model
        onnx_candidates = [
            os.path.join(self.models_dir, "age-gender-recognition-retail-0013.onnx"),
            os.path.join(os.path.dirname(__file__), "..", "CV_Mini project", "models", "age-gender-recognition-retail-0013.onnx")
        ]
        for op in onnx_candidates:
            if os.path.exists(op):
                try:
                    self.onnx_age_net = cv2.dnn.readNet(op)
                    print(f"[AgeEstimator] Loaded Deep Learning Age Model ({op})")
                    return
                except Exception as e:
                    print(f"[AgeEstimator] ONNX load note: {e}")

        self.onnx_age_net = None

        # 2. Check for Caffe Model
        prototxt = os.path.join(self.models_dir, "age_deploy.prototxt")
        caffemodel = os.path.join(self.models_dir, "age_net.caffemodel")
        if os.path.exists(prototxt) and os.path.exists(caffemodel):
            try:
                self.dnn_net = cv2.dnn.readNet(caffemodel, prototxt)
                print("[AgeEstimator] Successfully loaded OpenCV DNN Caffe model!")
            except Exception as e:
                print(f"[AgeEstimator] Note: Could not load DNN model ({e}), using CV Biometric Estimator.")

    def _analyze_forehead_wrinkles(self, gray_face):
        """Measures horizontal wrinkle lines in the forehead region."""
        h, w = gray_face.shape[:2]
        y1, y2 = int(h * 0.16), int(h * 0.36)
        x1, x2 = int(w * 0.22), int(w * 0.78)
        forehead = gray_face[y1:y2, x1:x2]
        if forehead.size == 0:
            return 0.0

        # Horizontal Sobel derivative specifically captures horizontal forehead lines
        sobel_y = cv2.Sobel(forehead, cv2.CV_64F, 0, 1, ksize=3)
        sobel_y = np.abs(sobel_y)

        # Canny edge density for distinct creases
        edges = cv2.Canny(forehead, 40, 110)
        edge_density = np.sum(edges > 0) / float(forehead.size)

        return float(edge_density * 100.0 + np.mean(sobel_y) * 0.05)

    def _analyze_eye_corners(self, gray_face):
        """Measures crow's feet and under-eye texture."""
        h, w = gray_face.shape[:2]
        # Left and right outer eye regions
        left_eye_roi = gray_face[int(h * 0.32):int(h * 0.48), int(w * 0.08):int(w * 0.28)]
        right_eye_roi = gray_face[int(h * 0.32):int(h * 0.48), int(w * 0.72):int(w * 0.92)]

        density_left = 0.0
        density_right = 0.0

        if left_eye_roi.size > 0:
            edges = cv2.Canny(left_eye_roi, 35, 100)
            density_left = np.sum(edges > 0) / float(left_eye_roi.size)

        if right_eye_roi.size > 0:
            edges = cv2.Canny(right_eye_roi, 35, 100)
            density_right = np.sum(edges > 0) / float(right_eye_roi.size)

        return float((density_left + density_right) * 50.0)

    def _analyze_nasolabial_folds(self, gray_face):
        """Measures smile lines / nasolabial creases around nose to mouth."""
        h, w = gray_face.shape[:2]
        nl_roi = gray_face[int(h * 0.55):int(h * 0.78), int(w * 0.25):int(w * 0.75)]
        if nl_roi.size == 0:
            return 0.0

        edges = cv2.Canny(nl_roi, 45, 120)
        return float(np.sum(edges > 0) / float(nl_roi.size) * 100.0)

    def _estimate_age_biometric(self, face_bgr):
        """
        Estimates age from biometric wrinkle, line, and texture density.
        Calibrated with young baseline and gradual aging indicators.
        Uses CLAHE for noise-resistant contrast enhancement.
        """
        h, w = face_bgr.shape[:2]
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Feature measurements
        forehead_score = self._analyze_forehead_wrinkles(gray)
        eye_score = self._analyze_eye_corners(gray)
        nasolabial_score = self._analyze_nasolabial_folds(gray)

        # Micro-texture roughness
        cheek = gray[int(h * 0.45):int(h * 0.70), int(w * 0.18):int(w * 0.40)]
        lap_var = cv2.Laplacian(cheek, cv2.CV_64F).var() if cheek.size > 0 else 50.0

        # Calibrated age mapping formula
        base_age = 20.0
        forehead_add = np.clip(forehead_score * 3.8, 0.0, 22.0)
        eye_add = np.clip(eye_score * 2.5, 0.0, 18.0)
        nasolabial_add = np.clip(nasolabial_score * 1.8, 0.0, 15.0)
        texture_add = np.clip((lap_var - 60.0) * 0.04, -3.0, 12.0)

        estimated_raw = base_age + forehead_add + eye_add + nasolabial_add + texture_add
        estimated_raw = float(np.clip(estimated_raw, 16.0, 68.0))

        return estimated_raw, (forehead_score + eye_score + nasolabial_score)

    def _estimate_age_onnx(self, face_bgr):
        """Runs inference using OpenVINO/ONNX retail age model (62x62 input)."""
        try:
            resized = cv2.resize(face_bgr, (62, 62), interpolation=cv2.INTER_AREA)
            blob = np.expand_dims(resized.astype(np.float32), axis=0)
            self.onnx_age_net.setInput(blob)
            outputs = self.onnx_age_net.forward(self.onnx_age_net.getUnconnectedOutLayersNames())
            gender_out = outputs[0].flatten() if outputs[0].size == 2 else outputs[1].flatten()
            age_out = outputs[1].flatten() if outputs[0].size == 2 else outputs[0].flatten()
            age = float(age_out[0]) * 100.0
            gender = "Female" if gender_out[0] > gender_out[1] else "Male"
            gender_conf = float(max(gender_out))
            return float(np.clip(age, 16.0, 95.0)), gender, gender_conf
        except Exception:
            raw, _ = self._estimate_age_biometric(face_bgr)
            return raw, "Unknown", 0.50

    def _estimate_age_dnn(self, face_bgr):
        """Runs inference with OpenCV DNN Age Model if loaded."""
        blob = cv2.dnn.blobFromImage(face_bgr, 1.0, (227, 227),
                                     (78.4263377603, 87.7689143744, 114.895847746),
                                     swapRB=False)
        self.dnn_net.setInput(blob)
        preds = self.dnn_net.forward()
        idx = preds[0].argmax()
        age_bin = self.AGE_INTERVALS[idx]

        # Convert bin midpoint to approximate scalar age
        bin_midpoints = [1.5, 5.0, 10.0, 18.0, 28.5, 40.5, 50.5, 70.0]
        scalar_age = bin_midpoints[idx]
        return scalar_age, age_bin

    def estimate(self, face_bgr, *args, **kwargs):
        """
        Main entry point for age estimation.
        Returns estimated age, range, gender, and vitality description.
        """
        if face_bgr is None or face_bgr.size == 0:
            return {
                'estimated_age': 23,
                'age_range': '21 - 25',
                'gender': 'Unknown',
                'gender_confidence': 50.0,
                'descriptor': 'Youthful',
                'wrinkle_level': 'Minimal'
            }

        gender = "Adult"
        gender_conf = 0.85
        if getattr(self, 'onnx_age_net', None) is not None:
            raw_age, gender, gender_conf = self._estimate_age_onnx(face_bgr)
        elif self.dnn_net is not None:
            raw_age, _ = self._estimate_age_dnn(face_bgr)
        else:
            raw_age, _ = self._estimate_age_biometric(face_bgr)

        # Smooth age across frames
        if not self.has_history:
            self.smoothed_age = raw_age
            self.has_history = True
        else:
            a = self.ema_alpha
            self.smoothed_age = a * raw_age + (1 - a) * self.smoothed_age

        final_age = int(round(self.smoothed_age))

        # Dynamic confidence range
        lower_bound = max(16, final_age - 2)
        upper_bound = final_age + 2
        age_range = f"{lower_bound} - {upper_bound} yrs"

        # Wrinkle / vitality descriptor
        if final_age < 25:
            descriptor = "Youthful & Fresh"
            wrinkle_level = "Very Low"
        elif final_age < 35:
            descriptor = "Prime Adult"
            wrinkle_level = "Low"
        elif final_age < 48:
            descriptor = "Mature & Distinguished"
            wrinkle_level = "Moderate"
        else:
            descriptor = "Experienced / Senior"
            wrinkle_level = "Noticeable"

        return {
            'estimated_age': final_age,
            'age_range': age_range,
            'gender': gender,
            'gender_confidence': round(gender_conf * 100, 1) if isinstance(gender_conf, float) else 85.0,
            'descriptor': descriptor,
            'wrinkle_level': wrinkle_level
        }
