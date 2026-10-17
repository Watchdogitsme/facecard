"""
Unified Robust Face Detector, 5-Point Affine Face Aligner & SFace Deep Embedding Engine
Handles multiple face detection and recognition backends gracefully:
1. PureYuNetDetector (YuNet ONNX DNN) for high-precision 2D boxes + 5 facial landmarks
2. 5-Point Similarity Affine Alignment (ArcFace / SFace standard 112x112 canonical warp)
3. PureSFaceRecognizer (SFace 128-D Deep Face Embeddings via FaceRecognizerSF or cv2.dnn)
4. cv2.CascadeClassifier (Haar Cascade) with multi-namespace resolution
5. Pure Computer Vision Skin-Color & Geometric Face Tracker (Bulletproof Fallback)
"""

import os
import cv2
import numpy as np


# Canonical 5-point landmarks for 112x112 aligned faces (ArcFace / SFace standard)
REFERENCE_FACIAL_POINTS_112 = np.array([
    [38.2946, 51.6963],  # Right eye (viewer's left)
    [73.5318, 51.5014],  # Left eye (viewer's right)
    [56.0252, 71.7366],  # Nose tip
    [41.5493, 92.3655],  # Right mouth corner
    [70.7299, 92.2041],  # Left mouth corner
], dtype=np.float32)


def align_face(image, landmarks=None, output_size=(112, 112)):
    """
    Warps a face image to a canonical upright coordinate frame using a 5-point
    similarity transform (rotation, scale, translation).
    Eliminates head tilt, scale variation, and translation jitter before DNN inference.
    """
    if image is None or image.size == 0:
        return np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)

    out_w, out_h = int(output_size[0]), int(output_size[1])
    if landmarks is not None:
        try:
            src_pts = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
            if src_pts.shape[0] == 5:
                dst_pts = REFERENCE_FACIAL_POINTS_112.copy()
                if out_w != 112 or out_h != 112:
                    dst_pts[:, 0] *= (out_w / 112.0)
                    dst_pts[:, 1] *= (out_h / 112.0)

                M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
                if M is not None:
                    aligned = cv2.warpAffine(
                        image,
                        M,
                        (out_w, out_h),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE
                    )
                    return aligned
        except Exception:
            pass

    return cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)


def get_cascade_classifier_class():
    """Safely retrieves CascadeClassifier class from cv2 regardless of build/namespace."""
    if hasattr(cv2, "CascadeClassifier"):
        return cv2.CascadeClassifier

    if hasattr(cv2, "cv2") and hasattr(cv2.cv2, "CascadeClassifier"):
        return cv2.cv2.CascadeClassifier

    try:
        import cv2.objdetect as objdetect
        if hasattr(objdetect, "CascadeClassifier"):
            return objdetect.CascadeClassifier
    except Exception:
        pass

    try:
        from cv2.objdetect import CascadeClassifier
        return CascadeClassifier
    except Exception:
        pass

    return None


def find_cascade_file():
    """Locates haarcascade_frontalface_default.xml across common paths."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "models", "haarcascade_frontalface_default.xml"),
        os.path.join(os.path.dirname(__file__), "..", "CV_Mini project", "models", "haarcascade_frontalface_default.xml"),
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
    ]
    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        candidates.insert(0, os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))

    for p in candidates:
        if os.path.exists(p):
            return p

    try:
        import urllib.request
        local_path = os.path.join(os.path.dirname(__file__), "models", "haarcascade_frontalface_default.xml")
        try:
            os.makedirs(os.path.join(os.path.dirname(__file__), "models"), exist_ok=True)
        except OSError:
            pass
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        urllib.request.urlretrieve(url, local_path)
        if os.path.exists(local_path):
            return local_path
    except Exception:
        pass

    return None


class PureSFaceRecognizer:
    """
    128-D Deep Face Recognition Embedding Engine powered by OpenCV Zoo's SFace ONNX model.
    Supports both cv2.FaceRecognizerSF and pure cv2.dnn.readNet + 5-point affine alignment.
    """

    def __init__(self, model_path=None):
        self.model_path = model_path or self._find_model()
        self.sf_recognizer = None
        self.dnn_net = None
        self.available = False

        if self.model_path and os.path.exists(self.model_path):
            # 1. Try native cv2.FaceRecognizerSF if available
            if hasattr(cv2, "FaceRecognizerSF"):
                try:
                    self.sf_recognizer = cv2.FaceRecognizerSF.create(self.model_path, "")
                    self.available = True
                except Exception:
                    self.sf_recognizer = None

            # 2. Also load cv2.dnn.readNet for direct aligned crop inference
            try:
                self.dnn_net = cv2.dnn.readNet(self.model_path)
                self.available = True
            except Exception:
                self.dnn_net = None

    @staticmethod
    def _find_model():
        candidates = [
            os.path.join(os.path.dirname(__file__), "models", "face_recognition_sface_2021dec.onnx"),
            os.path.join(os.path.dirname(__file__), "..", "CV_Mini project", "models", "face_recognition_sface_2021dec.onnx"),
        ]
        for c in candidates:
            if os.path.exists(c) and os.path.getsize(c) > 100 * 1024:
                return c
        return None

    def extract_embedding(self, image, landmarks=None, raw_row=None):
        """
        Extracts an L2-normalized 128-D deep facial identity embedding.
        Uses 5-point affine alignment for pose invariance.
        """
        if not self.available or image is None or image.size == 0:
            return None

        # Option A: Use FaceRecognizerSF.alignCrop if full image + raw_row (15 floats) provided
        if self.sf_recognizer is not None and raw_row is not None:
            try:
                face_row = np.asarray(raw_row, dtype=np.float32).reshape(1, -1)
                aligned = self.sf_recognizer.alignCrop(image, face_row)
                feat = self.sf_recognizer.feature(aligned)
                feat = feat.flatten().astype(np.float32)
                norm = float(np.linalg.norm(feat))
                if norm > 1e-6:
                    return feat / norm
            except Exception:
                pass

        # Option B: Align via 5-point similarity transform to 112x112
        aligned_112 = align_face(image, landmarks, output_size=(112, 112))

        if self.sf_recognizer is not None:
            try:
                feat = self.sf_recognizer.feature(aligned_112)
                feat = feat.flatten().astype(np.float32)
                norm = float(np.linalg.norm(feat))
                if norm > 1e-6:
                    return feat / norm
            except Exception:
                pass

        if self.dnn_net is not None:
            try:
                blob = cv2.dnn.blobFromImage(aligned_112, 1.0, (112, 112), (0, 0, 0), swapRB=True)
                self.dnn_net.setInput(blob)
                feat = self.dnn_net.forward().flatten().astype(np.float32)
                norm = float(np.linalg.norm(feat))
                if norm > 1e-6:
                    return feat / norm
            except Exception:
                pass

        return None


class PureYuNetDetector:
    """
    High-accuracy, real-time Deep Learning Face & Landmark Detector based on YuNet ONNX.
    Robustly handles all image dimensions with stride-32 padding and vectorized NMS.
    Outputs:
    - 2D Bounding Box: (x, y, w, h)
    - 5 Facial Keypoints: Right Eye, Left Eye, Nose Tip, Right Mouth Corner, Left Mouth Corner
    - Confidence Score
    """

    def __init__(self, model_path, score_threshold=0.55, nms_threshold=0.30):
        self.net = cv2.dnn.readNet(model_path)
        self.score_thresh = score_threshold
        self.nms_thresh = nms_threshold
        self.strides = [8, 16, 32]

    def detect(self, frame, max_dim=720):
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        scale = 1.0
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            target_w = int(round(w * scale))
            target_h = int(round(h * scale))
            proc_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            proc_frame = frame
            target_w, target_h = w, h

        # Pad to multiple of 32 to guarantee FPN layer compatibility
        pad_r = (32 - (target_w % 32)) % 32
        pad_b = (32 - (target_h % 32)) % 32
        if pad_r > 0 or pad_b > 0:
            padded = cv2.copyMakeBorder(proc_frame, 0, pad_b, 0, pad_r, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            padded = proc_frame

        pw, ph = padded.shape[1], padded.shape[0]
        blob = cv2.dnn.blobFromImage(padded, 1.0, (pw, ph), (0, 0, 0), swapRB=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        out_dict = dict(zip(self.net.getUnconnectedOutLayersNames(), outs))

        boxes = []
        scores = []
        landmarks = []

        for s in self.strides:
            cls_key = f"cls_{s}"
            obj_key = f"obj_{s}"
            bbox_key = f"bbox_{s}"
            kps_key = f"kps_{s}"

            if cls_key not in out_dict or obj_key not in out_dict:
                continue

            cls = out_dict[cls_key][0, :, 0]
            obj = out_dict[obj_key][0, :, 0]
            score = np.sqrt(np.maximum(0.0, cls * obj))
            mask = score >= self.score_thresh
            if not np.any(mask):
                continue

            indices = np.where(mask)[0]
            gw = pw // s
            gx = indices % gw
            gy = indices // gw

            raw_bbox = out_dict[bbox_key][0, indices]
            raw_kps = out_dict[kps_key][0, indices]

            cx = (gx + raw_bbox[:, 0]) * s
            cy = (gy + raw_bbox[:, 1]) * s
            bw = np.exp(raw_bbox[:, 2]) * s
            bh = np.exp(raw_bbox[:, 3]) * s

            bx = (cx - bw * 0.5) / scale
            by = (cy - bh * 0.5) / scale
            bw = bw / scale
            bh = bh / scale

            kps = np.zeros_like(raw_kps)
            for i in range(5):
                kps[:, i * 2] = ((gx + raw_kps[:, i * 2]) * s) / scale
                kps[:, i * 2 + 1] = ((gy + raw_kps[:, i * 2 + 1]) * s) / scale

            for i in range(len(indices)):
                boxes.append([float(bx[i]), float(by[i]), float(bw[i]), float(bh[i])])
                scores.append(float(score[indices[i]]))
                landmarks.append(kps[i].tolist())

        if not boxes:
            return []

        nms_indices = cv2.dnn.NMSBoxes(boxes, scores, self.score_thresh, self.nms_thresh)
        if len(nms_indices) == 0:
            return []

        if isinstance(nms_indices, tuple):
            nms_indices = list(nms_indices)
        else:
            nms_indices = nms_indices.flatten().tolist()

        results = []
        for i in nms_indices:
            bx, by, bw, bh = boxes[i]
            bx = max(0, min(w - 1, bx))
            by = max(0, min(h - 1, by))
            bw = max(1, min(w - bx, bw))
            bh = max(1, min(h - by, bh))
            lm = np.array(landmarks[i], dtype=np.float32).reshape(5, 2)
            results.append({
                "box": (int(round(bx)), int(round(by)), int(round(bw)), int(round(bh))),
                "score": float(scores[i]),
                "landmarks": lm,
                "raw_row": [float(bx), float(by), float(bw), float(bh)] + landmarks[i] + [float(scores[i])]
            })

        results.sort(key=lambda r: r["box"][2] * r["box"][3], reverse=True)
        return results


class RobustFaceDetector:
    """
    Unified face detector supporting high-accuracy YuNet DNN, Haar Cascade,
    and Color/Contour fallback, plus temporal bounding-box & landmark smoothing.
    """

    def __init__(self):
        self.backend = "unknown"
        self.cascade = None
        self.yunet = None
        self._prev_box = None
        self._prev_landmarks = None

        # 1. Prioritize YuNet DNN
        yunet_candidates = [
            os.path.join(os.path.dirname(__file__), "models", "face_detection_yunet_2023mar.onnx"),
            os.path.join(os.path.dirname(__file__), "..", "CV_Mini project", "models", "face_detection_yunet_2023mar.onnx")
        ]
        for yp in yunet_candidates:
            if os.path.exists(yp):
                try:
                    detector = PureYuNetDetector(yp, score_threshold=0.55, nms_threshold=0.30)
                    test_img = np.zeros((64, 64, 3), dtype=np.uint8)
                    detector.detect(test_img)
                    self.yunet = detector
                    self.backend = "yunet"
                    print(f"[FaceDetector] Loaded Deep Learning YuNet DNN ({yp})")
                    break
                except Exception as e:
                    print(f"[FaceDetector] YuNet init note: {e}")

        # 2. Fallback to Haar Cascade
        if self.backend == "unknown":
            cascade_cls = get_cascade_classifier_class()
            if cascade_cls is not None:
                xml_path = find_cascade_file()
                if xml_path and os.path.exists(xml_path):
                    try:
                        self.cascade = cascade_cls(xml_path)
                        if not self.cascade.empty():
                            self.backend = "haar_cascade"
                            print(f"[FaceDetector] Using Haar Cascade ({xml_path})")
                    except Exception as e:
                        print(f"[FaceDetector] Note: Could not initialize Haar cascade: {e}")

        # 3. Fallback to skin-color & geometric contour detector
        if self.backend == "unknown":
            self.backend = "color_contour_fallback"
            print("[FaceDetector] Warning: DNN/Haar classifiers unavailable.")
            print("[FaceDetector] Activated fallback Biometric Color & Contour Face Tracker.")

    def detect_faces(self, frame, scale_factor=1.12, min_neighbors=5, min_size=(70, 70)):
        """
        Detects faces in frame.
        Returns list of bounding boxes: [(x, y, w, h), ...]
        """
        detailed = self.detect_faces_detailed(frame, scale_factor, min_neighbors, min_size)
        return [d["box"] for d in detailed]

    def detect_faces_detailed(self, frame, scale_factor=1.12, min_neighbors=5, min_size=(70, 70)):
        """
        Detects faces and extracts facial landmark geometry when supported.
        Returns list of dicts with box, 5 landmarks, score, and raw_row.
        """
        if frame is None or frame.size == 0:
            return []

        # Backend 1: YuNet DNN (High Accuracy + 5 Landmarks)
        if self.backend == "yunet" and self.yunet is not None:
            try:
                results = self.yunet.detect(frame)
                filtered = [
                    r for r in results
                    if r["box"][2] >= min_size[0] and r["box"][3] >= min_size[1]
                ]
                if filtered:
                    return filtered
            except Exception as e:
                print(f"[FaceDetector] YuNet inference exception: {e}")

        # Backend 2: Haar Cascade Fallback
        if self.cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_size
            )
            detailed = []
            for (x, y, w, h) in faces:
                lm = self._estimate_landmarks_from_box(x, y, w, h)
                raw_row = [float(x), float(y), float(w), float(h)] + lm.flatten().tolist() + [0.85]
                detailed.append({
                    "box": (int(x), int(y), int(w), int(h)),
                    "landmarks": lm,
                    "score": 0.85,
                    "raw_row": raw_row
                })
            if detailed:
                return detailed

        # Backend 3: Color & Contour Fallback
        fallback_boxes = self._detect_fallback(frame, min_size)
        detailed = []
        for (x, y, w, h) in fallback_boxes:
            lm = self._estimate_landmarks_from_box(x, y, w, h)
            raw_row = [float(x), float(y), float(w), float(h)] + lm.flatten().tolist() + [0.60]
            detailed.append({
                "box": (int(x), int(y), int(w), int(h)),
                "landmarks": lm,
                "score": 0.60,
                "raw_row": raw_row
            })
        return detailed

    def _estimate_landmarks_from_box(self, x, y, w, h):
        """Estimates 5 anthropometric facial landmarks from face bounding box."""
        re_x = x + w * 0.32
        re_y = y + h * 0.37
        le_x = x + w * 0.68
        le_y = y + h * 0.37
        nose_x = x + w * 0.50
        nose_y = y + h * 0.58
        rm_x = x + w * 0.36
        rm_y = y + h * 0.76
        lm_x = x + w * 0.64
        lm_y = y + h * 0.76
        return np.array([
            [re_x, re_y],
            [le_x, le_y],
            [nose_x, nose_y],
            [rm_x, rm_y],
            [lm_x, lm_y]
        ], dtype=np.float32)

    def _detect_fallback(self, frame, min_size):
        """
        Fallback detector using skin chrominance segmentation and
        morphological aspect ratio filtering.
        """
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        min_area = min_size[0] * min_size[1]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = float(ch) / float(cw) if cw > 0 else 0
            if 0.9 <= aspect <= 2.2 and ch >= min_size[1] and cw >= min_size[0]:
                candidates.append((x, y, cw, ch))

        if candidates:
            candidates.sort(key=lambda b: b[2] * b[3], reverse=True)
            return candidates[:1]

        return []
