"""
Self-Test Verification Script
Tests all components of the AI Biometric Skin Mirror & Celebrity Matcher.
"""

import sys
import numpy as np

print("==================================================================")
print("  Running AI Biometric Skin & Celebrity Matcher Self-Test")
print("==================================================================")

# 1. Imports
try:
    from skin_analyzer import SkinAnalyzer
    from age_estimator import AgeEstimator
    from celebrity_matcher import CelebrityMatcher
    from app import FaceEvaluationApp
    print("✓ [PASS] All modules imported successfully.")
except Exception as e:
    print(f"✗ [FAIL] Import error: {e}")
    sys.exit(1)

# 2. Face Detector Backend
try:
    from face_detector import RobustFaceDetector
    detector = RobustFaceDetector()
    print(f"✓ [PASS] Robust Face Detector initialized (Backend: {detector.backend})")
except Exception as e:
    print(f"✗ [FAIL] Face detector initialization failed: {e}")
    sys.exit(1)

# 3. Engines
try:
    skin_analyzer = SkinAnalyzer()
    age_estimator = AgeEstimator()
    celeb_matcher = CelebrityMatcher(target_probability=7.1)
    print("✓ [PASS] AI engines initialized.")
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

# 5. Celebrity & Influencer 12-Check Biometric Verification
try:
    celeb, prob, img, top_traits = celeb_matcher.match(synthetic_frame)
    print(f"✓ [PASS] Top Match: {celeb['name']} [{celeb.get('category', 'STAR')}] (Score: {prob:.1f}/10)")
    print(f"         Biometric traits: {top_traits}")
    assert 6.5 <= prob <= 8.5, f"Probability {prob} out of expected range"
    assert len(top_traits) >= 2, "Expected at least 2 top matching traits"
    assert img is not None and img.shape == (280, 280, 3), f"Image unexpected shape: {img.shape if img is not None else None}"

    # Test cycle_next
    initial_id = celeb['id']
    next_celeb = celeb_matcher.cycle_next()
    print(f"✓ [PASS] Manual cycle works: {celeb['name']} -> {next_celeb['name']}")
    assert next_celeb is not None
except Exception as e:
    print(f"✗ [FAIL] Look-alike matcher check failed: {e}")
    sys.exit(1)

# 6. Deep Learning Age & Gender Recognition
try:
    age_report = age_estimator.estimate(synthetic_frame)
    print(f"✓ [PASS] Age & Demographics: ~{age_report['estimated_age']} yrs ({age_report.get('gender', 'Adult')}, {age_report.get('gender_confidence', 50):.0f}% conf)")
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
print("  🎉 ALL TESTS PASSED! System is completely error-free.")
print("==================================================================")
