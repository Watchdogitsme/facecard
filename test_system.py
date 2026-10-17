"""
Pre-Launch File Integrity & System Readiness Verification Script
Checks that all core scripts, deep learning ONNX models, calibrated weights,
and the celebrity gallery index are intact and ready to use before launching.
"""

import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("==================================================================")
print("  FaceCard: Pre-Launch File Integrity & System Readiness Check")
print("==================================================================")

# 0. File & Model Integrity Check
REQUIRED_FILES = [
    ("app.py", 1024),
    ("face_detector.py", 1024),
    ("skin_analyzer.py", 1024),
    ("age_estimator.py", 1024),
    ("celebrity_matcher.py", 1024),
    (os.path.join("models", "face_detection_yunet_2023mar.onnx"), 100 * 1024),
    (os.path.join("models", "face_recognition_sface_2021dec.onnx"), 1000 * 1024),
    (os.path.join("models", "calibrated_models.npz"), 1024),
    (os.path.join("celebrity_gallery", "index.npz"), 1024),
]

missing_or_corrupt = []
for rel_path, min_bytes in REQUIRED_FILES:
    full_path = os.path.join(BASE_DIR, rel_path)
    if not os.path.exists(full_path) or os.path.getsize(full_path) < min_bytes:
        missing_or_corrupt.append(rel_path)

if missing_or_corrupt:
    print(f"✗ [FAIL] Missing or corrupted files: {missing_or_corrupt}")
    sys.exit(1)
else:
    print(f"✓ [PASS] File Integrity Verified: All {len(REQUIRED_FILES)} core modules, ONNX models, and gallery indexes are intact.")

# 1. Imports
try:
    from skin_analyzer import SkinAnalyzer
    from age_estimator import AgeEstimator
    from celebrity_matcher import CelebrityMatcher
    from face_detector import RobustFaceDetector, PureSFaceRecognizer
    from app import FaceEvaluationApp
    print("✓ [PASS] All Python modules imported successfully.")
except Exception as e:
    print(f"✗ [FAIL] Import error: {e}")
    sys.exit(1)

# 2. Face Detector & SFace 128-D Recognizer Backend
try:
    detector = RobustFaceDetector()
    sface = PureSFaceRecognizer()
    print(f"✓ [PASS] Robust Face Detector initialized (Backend: {detector.backend})")
    print(f"✓ [PASS] SFace 128-D Deep Recognizer available: {sface.available}")
except Exception as e:
    print(f"✗ [FAIL] Face detector initialization failed: {e}")
    sys.exit(1)

# 3. Engines
try:
    skin_analyzer = SkinAnalyzer()
    age_estimator = AgeEstimator()
    celeb_matcher = CelebrityMatcher(target_probability=7.1)
    print(f"✓ [PASS] AI engines initialized ({len(celeb_matcher.celebrities)} total celebrities ready).")
except Exception as e:
    print(f"✗ [FAIL] Engine initialization failed: {e}")
    sys.exit(1)

# 4. Pipeline Execution
try:
    app = FaceEvaluationApp(camera_index=0)
    synthetic_frame = app._generate_synthetic_face()
    result_dashboard = app.process_frame(synthetic_frame)
    h, w, c = result_dashboard.shape
    assert h >= 480 and w >= 640, f"Dashboard size unexpected: {w}x{h}"
    print(f"✓ [PASS] Dashboard rendered successfully: {w}x{h} px, {c} channels.")
except Exception as e:
    print(f"✗ [FAIL] Pipeline execution failed: {e}")
    sys.exit(1)

# 5. Celebrity & Influencer Deep + Biometric Matching Verification
try:
    emb = celeb_matcher.extract_deep_embedding(synthetic_frame)
    celeb, prob, img, top_traits = celeb_matcher.match(synthetic_frame, query_embedding=emb)
    print(f"✓ [PASS] Top Match: {celeb['name']} [{celeb.get('category', 'STAR')}] (Score: {prob:.1f}/10)")
    print(f"         Biometric traits: {top_traits}")
    assert 6.5 <= prob <= 9.9, f"Probability {prob} out of expected range"
    assert len(top_traits) >= 2, "Expected at least 2 top matching traits"
    assert img is not None and img.shape == (280, 280, 3), f"Image unexpected shape: {img.shape if img is not None else None}"

    next_celeb = celeb_matcher.cycle_next()
    print(f"✓ [PASS] Manual cycle works: {celeb['name']} -> {next_celeb['name']}")
    assert next_celeb is not None
except Exception as e:
    print(f"✗ [FAIL] Look-alike matcher check failed: {e}")
    sys.exit(1)

# 6. Age Estimation Check
try:
    age_report = age_estimator.estimate(synthetic_frame)
    print(f"✓ [PASS] Age & Demographics: ~{age_report['estimated_age']} yrs ({age_report.get('descriptor')})")
    assert 15 <= age_report['estimated_age'] <= 80, f"Age out of bounds: {age_report['estimated_age']}"
except Exception as e:
    print(f"✗ [FAIL] Age estimation check failed: {e}")
    sys.exit(1)

# 7. Stress Testing: Resolutions, Black Frame, Noise Frame
try:
    test_frames = [
        ("Black frame (no face)", np.zeros((480, 640, 3), dtype=np.uint8)),
        ("High-res 720p frame", np.zeros((720, 1280, 3), dtype=np.uint8)),
        ("Random noise frame", np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)),
    ]
    for label, tf in test_frames:
        out = app.process_frame(tf)
        assert out is not None and out.shape[0] >= 480
    print("✓ [PASS] Stress tests passed (black frames, HD feeds, noise feeds handled gracefully).")
except Exception as e:
    print(f"✗ [FAIL] Stress test failed: {e}")
    sys.exit(1)

print("==================================================================")
print("  🎉 ALL FILES INTACT & READY TO USE! You can now launch Option 1.")
print("==================================================================")
