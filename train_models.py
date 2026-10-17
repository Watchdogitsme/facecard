"""
Model Training, Metric Learning & Deep Gallery Indexing Pipeline (train_models.py)
Trains and calibrates high-accuracy models on the local face & celebrity dataset:
1. Extracts 128-D SFace Deep Face Embeddings + 12 Anthropometric Proportions across
   multi-view augmented face crops (5-point affine aligned, mirrored, photometric jitter).
2. Learns a Regularized Discriminative Whitening / Metric Projection Matrix (emb_proj)
   and Fisher Discriminative Trait Weights (trait_weights) to maximize look-alike accuracy.
3. Trains a High-Accuracy L2-Regularized Logistic Gender Classifier on 128-D SFace embeddings.
4. Trains a Ridge-Regularized Age Regression Head on 128-D SFace embeddings + ONNX teacher.
5. Rebuilds celebrity_gallery/index.npz (with 128-D SFace embeddings, 12-D feature cards,
   gender labels, and Indian/Global region tags) and saves models/calibrated_models.npz.
"""

import argparse
import hashlib
import os
import time
from pathlib import Path

import cv2
import numpy as np

from face_detector import PureYuNetDetector, PureSFaceRecognizer, align_face
from celebrity_matcher import CelebrityMatcher, CELEBRITY_METADATA
from age_estimator import AgeEstimator
from skin_analyzer import SkinAnalyzer


ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
GALLERY_DIR = ROOT_DIR / "celebrity_gallery"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def generate_augmented_views(face_crop, landmarks=None):
    """
    Produces 5 multi-view training samples per face for robust metric learning:
    1. Canonical 5-point aligned crop
    2. Horizontally mirrored aligned crop
    3. Slight warm illumination boost (+8% contrast, +5 brightness)
    4. Slight cool/dim illumination (-6% contrast, -5 brightness)
    5. Slight 4% zoom crop
    """
    views = []
    aligned = align_face(face_crop, landmarks, output_size=(112, 112))
    views.append((aligned, None))

    # Mirrored view
    mirrored = cv2.flip(aligned, 1)
    views.append((mirrored, None))

    # Bright/high-contrast view
    bright = np.clip(aligned.astype(np.float32) * 1.07 + 5.0, 0, 255).astype(np.uint8)
    views.append((bright, None))

    # Dim/low-contrast view
    dim = np.clip(aligned.astype(np.float32) * 0.93 - 5.0, 0, 255).astype(np.uint8)
    views.append((dim, None))

    # Center zoom view
    h, w = aligned.shape[:2]
    pad = 4
    zoom = cv2.resize(aligned[pad:h - pad, pad:w - pad], (112, 112), interpolation=cv2.INTER_LINEAR)
    views.append((zoom, None))

    return views


def train_logistic_regression_l2(X: np.ndarray, y: np.ndarray, reg_lambda: float = 1.5, max_iter: int = 60):
    """
    Trains an L2-regularized binary logistic regression classifier via Newton-Raphson (IRLS).
    X: (N, D), y: (N,) in {0, 1}
    Returns: weights (D,), bias (1,)
    """
    N, D = X.shape
    X_aug = np.hstack([X, np.ones((N, 1), dtype=np.float64)])
    w = np.zeros(D + 1, dtype=np.float64)

    reg_mat = np.eye(D + 1, dtype=np.float64) * reg_lambda
    reg_mat[-1, -1] = 1e-4  # Don't penalize intercept

    for _ in range(max_iter):
        logits = np.clip(X_aug @ w, -20.0, 20.0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        grad = X_aug.T @ (probs - y) + reg_mat @ w
        r = np.maximum(probs * (1.0 - probs), 1e-5)
        H = (X_aug.T * r) @ X_aug + reg_mat
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        w -= step
        if np.linalg.norm(step) < 1e-6:
            break

    return w[:D].astype(np.float32), np.array([w[D]], dtype=np.float32)


def train_ridge_regression(X: np.ndarray, y: np.ndarray, alpha: float = 4.0):
    """
    Trains a closed-form L2 Ridge Regression model: min ||Xw + b - y||^2 + alpha ||w||^2.
    """
    N, D = X.shape
    X_mean = np.mean(X, axis=0)
    y_mean = float(np.mean(y))
    X_c = (X - X_mean).astype(np.float64)
    y_c = (y - y_mean).astype(np.float64)

    A = X_c.T @ X_c + alpha * np.eye(D, dtype=np.float64)
    b_vec = X_c.T @ y_c
    w = np.linalg.solve(A, b_vec)
    bias = y_mean - float(X_mean @ w)
    return w.astype(np.float32), np.array([bias], dtype=np.float32)


def learn_discriminative_whitening(embeddings_by_id: list[np.ndarray], shrinkage: float = 0.25):
    """
    Learns a Regularized Discriminative Metric Projection Matrix (W_proj) using
    Within-Class Covariance Whitening + Between-Class Fisher Subspace Enhancement.
    Maximizes cosine similarity between views of the same face while orthogonalizing different faces.
    """
    all_embs = np.vstack(embeddings_by_id).astype(np.float64)
    emb_mean = np.mean(all_embs, axis=0)
    D = all_embs.shape[1]

    # Within-class scatter S_W and Between-class scatter S_B
    S_W = np.zeros((D, D), dtype=np.float64)
    S_B = np.zeros((D, D), dtype=np.float64)

    for id_embs in embeddings_by_id:
        c_embs = id_embs.astype(np.float64)
        c_mean = np.mean(c_embs, axis=0)
        diffs = c_embs - c_mean
        S_W += (diffs.T @ diffs) / max(1, len(c_embs))
        mean_diff = (c_mean - emb_mean).reshape(-1, 1)
        S_B += mean_diff @ mean_diff.T

    S_W /= max(1, len(embeddings_by_id))
    S_B /= max(1, len(embeddings_by_id))

    # Regularized within-class covariance inverse square root (whitening)
    trace_sw = np.trace(S_W) / D
    S_W_reg = (1.0 - shrinkage) * S_W + shrinkage * trace_sw * np.eye(D)

    eigvals, eigvecs = np.linalg.eigh(S_W_reg)
    eigvals = np.maximum(eigvals, 1e-4)
    inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
    W_white = eigvecs @ inv_sqrt @ eigvecs.T

    # Blend identity with discriminative whitening for stability on unseen webcam faces
    W_proj = 0.60 * np.eye(D) + 0.40 * (W_white / (np.linalg.norm(W_white, ord=2) + 1e-6))
    return emb_mean.astype(np.float32), W_proj.astype(np.float32)


def learn_fisher_trait_weights(traits_by_id: list[np.ndarray]):
    """
    Computes Fisher Discriminative Ratios (Between-Identity Variance / Within-Identity Variance)
    for the 12 anthropometric facial traits so stable, identity-specific proportions get higher weight.
    """
    centroids = np.array([np.mean(t, axis=0) for t in traits_by_id], dtype=np.float64)
    between_var = np.var(centroids, axis=0) + 1e-5
    within_var = np.mean([np.var(t, axis=0) for t in traits_by_id], axis=0) + 1e-4

    fisher_ratio = between_var / within_var
    # Normalize weights to have mean 1.0 and clip within [0.65, 1.50]
    norm_weights = fisher_ratio / np.mean(fisher_ratio)
    norm_weights = np.clip(norm_weights, 0.65, 1.50)
    return norm_weights.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Train & Calibrate High-Accuracy Face, Age, Gender & Look-Alike Models")
    parser.add_argument("--source-dir", type=Path, default=GALLERY_DIR / "source", help="Directory of labeled celebrity portraits")
    parser.add_argument("--output-gallery", type=Path, default=GALLERY_DIR, help="Output directory for index.npz")
    args = parser.parse_args()

    t_start = time.time()
    print("===========================================================================")
    print("  FaceCard AI: High-Accuracy Model Training & Deep Gallery Indexing")
    print("===========================================================================")

    yunet_path = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
    sface_path = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
    if not yunet_path.exists():
        raise FileNotFoundError(f"Missing YuNet model: {yunet_path}. Run scripts/download_models.py first.")

    detector = PureYuNetDetector(str(yunet_path), score_threshold=0.55, nms_threshold=0.30)
    sface = PureSFaceRecognizer(str(sface_path) if sface_path.exists() else None)
    matcher_helper = CelebrityMatcher()
    age_helper = AgeEstimator()
    skin_helper = SkinAnalyzer()

    if not args.source_dir.is_dir():
        raise NotADirectoryError(f"Source directory not found: {args.source_dir}")

    identities = {}
    embeddings_by_id = []
    traits_by_id = []

    X_emb_all = []
    y_gender_all = []
    y_age_all = []
    skin_submetrics_all = []

    image_paths = sorted(
        p for p in args.source_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    print(f"[1/4] Scanning {len(image_paths)} reference images in {args.source_dir}...")

    usable_images = 0
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None or img.size == 0:
            continue

        label = img_path.parent.name if img_path.parent != args.source_dir else img_path.stem
        detections = detector.detect(img)
        if not detections:
            continue

        primary = detections[0]
        bx, by, bw, bh = primary["box"]
        landmarks = primary["landmarks"]

        h, w = img.shape[:2]
        x1, y1 = max(0, bx), max(0, by)
        x2, y2 = min(w, bx + bw), min(h, by + bh)
        face_crop = img[y1:y2, x1:x2].copy()
        if face_crop.size == 0:
            continue

        crop_lm = landmarks.copy()
        crop_lm[:, 0] -= x1
        crop_lm[:, 1] -= y1

        # Extract 12-D anthropometric vector on primary face
        feat_12 = matcher_helper.extract_biometric_features(face_crop, landmarks=crop_lm)

        # Generate 5 multi-view augmented crops for training
        aug_views = generate_augmented_views(face_crop, landmarks=crop_lm)
        view_embs = []
        view_traits = [feat_12]

        for v_img, v_lm in aug_views:
            emb = sface.extract_embedding(v_img, landmarks=v_lm)
            if emb is not None:
                view_embs.append(emb)
            t_vec = matcher_helper.extract_biometric_features(v_img, landmarks=v_lm)
            view_traits.append(t_vec)

        if not view_embs:
            continue

        canonical_emb = np.mean(view_embs, axis=0).astype(np.float32)
        canonical_emb /= max(1e-6, float(np.linalg.norm(canonical_emb)))

        # Metadata lookup for gender, region, and approximate age
        formatted_name = matcher_helper._format_celebrity_name(label)
        cat, title, gender, region, color = matcher_helper._infer_metadata(formatted_name)

        # Determine training age target (from metadata if known, else ONNX teacher prediction)
        onnx_age, prob_fem_onnx = age_helper._estimate_age_onnx(face_crop, landmarks=crop_lm)
        if formatted_name in CELEBRITY_METADATA:
            _, _, gender_meta, region_meta, approx_age, _ = CELEBRITY_METADATA[formatted_name]
            gender = gender_meta
            region = region_meta
            target_age = 0.65 * float(approx_age) + 0.35 * onnx_age
        else:
            target_age = onnx_age
            # Refine gender if ONNX is highly confident (> 85%) and not in explicit table
            if prob_fem_onnx > 0.85:
                gender = "female"
            elif prob_fem_onnx < 0.15:
                gender = "male"

        for v_emb in view_embs:
            X_emb_all.append(v_emb)
            y_gender_all.append(1.0 if gender == "female" else 0.0)
            y_age_all.append(target_age)

        # Skin metrics sample
        skin_rep = skin_helper.evaluate_face(face_crop, landmarks=crop_lm)
        skin_submetrics_all.append([
            skin_rep["smoothness"],
            skin_rep["clarity"],
            skin_rep["uniformity"]
        ])

        embeddings_by_id.append(np.array(view_embs, dtype=np.float32))
        traits_by_id.append(np.array(view_traits, dtype=np.float32))

        area = float(bw * bh)
        ident = identities.setdefault(
            label,
            {
                "feature_cards": [],
                "embeddings": [],
                "gender": gender,
                "region": region,
                "best_area": 0.0,
                "best_face": None,
            }
        )
        ident["feature_cards"].append(feat_12)
        ident["embeddings"].append(canonical_emb)
        if area > ident["best_area"]:
            ident["best_area"] = area
            ident["best_face"] = face_crop

        usable_images += 1

    print(f"  ✓ Extracted multi-view deep embeddings from {usable_images} portraits ({len(identities)} unique identities).")

    # ----------------- 2. TRAIN METRIC PROJECTION & TRAIT WEIGHTS -----------------
    print("[2/4] Training Discriminative Metric Projection & Fisher Biometric Weights...")
    emb_mean, emb_proj = learn_discriminative_whitening(embeddings_by_id, shrinkage=0.25)
    trait_weights = learn_fisher_trait_weights(traits_by_id)

    # Evaluate Rank-1 and Rank-3 Multi-View Identification Accuracy
    gallery_centroids = []
    for id_embs in embeddings_by_id:
        c = id_embs[0]  # Canonical view
        c_proj = np.dot(c - emb_mean, emb_proj)
        c_proj /= max(1e-6, float(np.linalg.norm(c_proj)))
        gallery_centroids.append(c_proj)
    gallery_mat = np.vstack(gallery_centroids)

    rank1_hits = 0
    rank3_hits = 0
    total_probes = 0
    for idx_id, id_embs in enumerate(embeddings_by_id):
        for probe in id_embs[1:]:  # Test on augmented/mirrored/jittered views
            p_proj = np.dot(probe - emb_mean, emb_proj)
            p_proj /= max(1e-6, float(np.linalg.norm(p_proj)))
            sims = gallery_mat @ p_proj
            top3 = np.argsort(sims)[::-1][:3]
            if top3[0] == idx_id:
                rank1_hits += 1
            if idx_id in top3:
                rank3_hits += 1
            total_probes += 1

    rank1_acc = (rank1_hits / max(1, total_probes)) * 100.0
    rank3_acc = (rank3_hits / max(1, total_probes)) * 100.0
    print(f"  ✓ Deep SFace Look-Alike Identification Accuracy (Augmented Test Views):")
    print(f"    • Rank-1 Accuracy: {rank1_acc:.2f}%  |  Rank-3 Accuracy: {rank3_acc:.2f}%")

    # ----------------- 3. TRAIN GENDER & AGE CALIBRATION PROBES -----------------
    print("[3/4] Training High-Accuracy Gender & Age Calibration Probes on 128-D Embeddings...")
    X_emb_np = np.array(X_emb_all, dtype=np.float32)
    y_gender_np = np.array(y_gender_all, dtype=np.float32)
    y_age_np = np.array(y_age_all, dtype=np.float32)

    # 5-fold cross-validation for Gender & Age
    n_samples = len(X_emb_np)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_samples)
    folds = np.array_split(perm, 5)

    val_gender_correct = 0
    val_age_errors = []
    for f_idx in range(5):
        val_idx = folds[f_idx]
        train_idx = np.hstack([folds[i] for i in range(5) if i != f_idx])

        gw_f, gb_f = train_logistic_regression_l2(X_emb_np[train_idx], y_gender_np[train_idx], reg_lambda=1.5)
        val_logits = X_emb_np[val_idx] @ gw_f + gb_f[0]
        val_preds = (val_logits >= 0.0).astype(np.float32)
        val_gender_correct += int(np.sum(val_preds == y_gender_np[val_idx]))

        aw_f, ab_f = train_ridge_regression(X_emb_np[train_idx], y_age_np[train_idx], alpha=4.0)
        val_age_pred = np.clip(X_emb_np[val_idx] @ aw_f + ab_f[0], 15.0, 80.0)
        val_age_errors.extend(np.abs(val_age_pred - y_age_np[val_idx]).tolist())

    gender_cv_acc = (val_gender_correct / max(1, n_samples)) * 100.0
    age_cv_mae = float(np.mean(val_age_errors))

    # Fit final probes on full training set
    gender_weights, gender_bias = train_logistic_regression_l2(X_emb_np, y_gender_np, reg_lambda=1.2)
    age_weights, age_bias = train_ridge_regression(X_emb_np, y_age_np, alpha=3.5)

    print(f"  ✓ 5-Fold Cross-Validation Demographic Results:")
    print(f"    • Gender Classification Accuracy: {gender_cv_acc:.2f}%")
    print(f"    • Age Estimation Mean Absolute Error (MAE): ±{age_cv_mae:.2f} years")

    # Calibrate skin sub-metric weights based on signal reliability (inverse coefficient of variation)
    skin_mat = np.array(skin_submetrics_all, dtype=np.float32)
    skin_std = np.std(skin_mat, axis=0) + 0.25
    skin_inv_var = 1.0 / skin_std
    skin_weights = (skin_inv_var / np.sum(skin_inv_var)).astype(np.float32)

    # Save calibrated models
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    calib_path = MODELS_DIR / "calibrated_models.npz"
    np.savez_compressed(
        calib_path,
        emb_mean=emb_mean,
        emb_proj=emb_proj,
        trait_weights=trait_weights,
        gender_weights=gender_weights,
        gender_bias=gender_bias,
        age_weights=age_weights,
        age_bias=age_bias,
        skin_weights=skin_weights,
        metrics=np.array([rank1_acc, rank3_acc, gender_cv_acc, age_cv_mae], dtype=np.float32),
    )
    print(f"  ✓ Saved trained model weights to {calib_path}")

    # ----------------- 4. BUILD UPDATED CELEBRITY GALLERY INDEX -----------------
    print("[4/4] Rebuilding celebrity_gallery/index.npz with 128-D SFace Embeddings & Indian Metadata...")
    references_dir = args.output_gallery / "references"
    references_dir.mkdir(parents=True, exist_ok=True)

    sorted_labels = sorted(identities.keys())
    feature_cards = []
    embeddings_list = []
    genders_list = []
    regions_list = []
    image_files = []

    for label in sorted_labels:
        info = identities[label]
        f_card = np.median(info["feature_cards"], axis=0).astype(np.float32)
        e_vec = np.mean(info["embeddings"], axis=0).astype(np.float32)
        e_vec /= max(1e-6, float(np.linalg.norm(e_vec)))

        rep_face = info["best_face"]
        ok, encoded = cv2.imencode(".jpg", rep_face)
        if not ok:
            continue
        digest = hashlib.sha256(label.encode("utf-8") + encoded.tobytes()).hexdigest()[:24]
        filename = f"{digest}.jpg"
        ref_path = references_dir / filename
        ref_path.write_bytes(encoded.tobytes())

        feature_cards.append(f_card)
        embeddings_list.append(e_vec)
        genders_list.append(info["gender"])
        regions_list.append(info["region"])
        image_files.append(f"references/{filename}")

    temp_index = args.output_gallery / "index.npz.tmp"
    with temp_index.open("wb") as f:
        np.savez_compressed(
            f,
            labels=np.asarray(sorted_labels, dtype=np.str_),
            feature_names=np.asarray(CelebrityMatcher.FEATURE_NAMES, dtype=np.str_),
            feature_cards=np.asarray(feature_cards, dtype=np.float32),
            embeddings=np.asarray(embeddings_list, dtype=np.float32),
            genders=np.asarray(genders_list, dtype=np.str_),
            regions=np.asarray(regions_list, dtype=np.str_),
            image_files=np.asarray(image_files, dtype=np.str_),
        )
    temp_index.replace(args.output_gallery / "index.npz")

    indian_total = sum(1 for r in regions_list if r == "indian")
    elapsed = time.time() - t_start
    print("===========================================================================")
    print(f"  🎉 TRAINING & INDEXING COMPLETE in {elapsed:.1f}s!")
    print(f"  • Total Indexed Celebrities: {len(sorted_labels)} ({indian_total} Indian photos + curated Indian creators)")
    print(f"  • Rank-1 Deep Identification: {rank1_acc:.2f}% | Rank-3: {rank3_acc:.2f}%")
    print(f"  • Gender Classification CV Accuracy: {gender_cv_acc:.2f}% | Age MAE: ±{age_cv_mae:.2f} yrs")
    print("===========================================================================")


if __name__ == "__main__":
    main()
