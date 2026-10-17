<div align="center">

# 🪞 FaceCard

### **AI-Powered Biometric Mirror: Deep Face Recognition, Skin Evaluator & Celebrity Matcher**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV 4.6+](https://img.shields.io/badge/OpenCV-4.6%2B-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![Deep Learning YuNet + SFace](https://img.shields.io/badge/Models-YuNet%20%2B%20SFace%20128D-brightgreen.svg)](https://github.com/opencv/opencv_zoo)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20WSL2-teal.svg)](https://github.com/)

<p align="center">
  <b>An end-to-end Deep Learning & Computer Vision biometric mirror featuring 5-point affine face alignment, 128-D SFace deep identity embeddings, ITA°-calibrated dermatological skin grading, biometric age estimation, and a 189-celebrity & influencer look-alike gallery.</b>
</p>

</div>

---

## 📑 Table of Contents

- [Overview & Key Features](#-overview--key-features)
- [How the Models Were Trained & Training Datasets](#-how-the-models-were-trained--training-datasets)
  - [Stage 1: Foundational Deep Learning Backbones (Pre-Training)](#stage-1-foundational-deep-learning-backbones-pre-training)
  - [Stage 2: Local Domain-Specific Training & Calibration (`train_models.py`)](#stage-2-local-domain-specific-training--calibration-train_modelspy)
  - [Training Dataset Summary Table](#training-dataset-summary-table)
- [Expanded Celebrity, Influencer & Athlete Database](#-expanded-celebrity-influencer--athlete-database)
- [Technical Pipeline & Mathematical Formulation](#-technical-pipeline--mathematical-formulation)
- [Repository Structure](#-repository-structure)
- [Quick Start](#-quick-start)
- [Interactive Keyboard Shortcuts](#-interactive-keyboard-shortcuts)
- [Self-Test & File Integrity Verification](#-self-test--file-integrity-verification)

---

## 🌟 Overview & Key Features

**FaceCard** is a real-time biometric vision mirror built with **OpenCV** and **Python** that transforms standard webcams into an interactive dermatological and facial geometry analysis HUD:

1. **👁️ Deep Learning YuNet Face & 5-Landmark Detector (`face_detector.py`)**:
   - Uses the **YuNet CNN** (`face_detection_yunet_2023mar.onnx`) with dynamic **FPN Stride-32 boundary padding** (`copyMakeBorder`) and vectorized NMS to extract 2D face boxes and 5 sub-pixel facial keypoints (*Right Eye, Left Eye, Nose Tip, Right Mouth Corner, Left Mouth Corner*).
   - Includes a 3-tier resilient fallback ladder: YuNet CNN $\rightarrow$ Haar Cascade $\rightarrow$ YCrCb/HSV Color & Contour Tracker.
2. **🧠 128-D SFace Deep Face Recognition + 5-Point Affine Alignment (`face_detector.py`, `celebrity_matcher.py`)**:
   - Warps every detected face to a canonical $112 \times 112$ upright frame via a 5-point similarity transform (`cv2.estimateAffinePartial2D`) before running neural inference, eliminating head-tilt and scale errors.
   - Extracts L2-normalized **128-D SFace Deep Identity Embeddings** (`face_recognition_sface_2021dec.onnx`) fused with 12 anthropometric proportions and a learned **Discriminative Metric Whitening Matrix** (`models/calibrated_models.npz`).
3. **💆 ITA°-Calibrated & Glare-Invariant Skin Quality Engine (`skin_analyzer.py`)**:
   - **Landmark-Anchored ROIs**: Isolates pure skin across Forehead, Left Cheek, and Right Cheek patches using pupil and mouth keypoints.
   - **Specular Glare Removal & Melanin Adaptation**: Masks out webcam lighting glare ($L^* > 235$) and uses CIE LAB **Individual Typology Angle ($\text{ITA}^\circ$)** so warm, wheatish, tan, and deep complexions are evaluated accurately without false blemish penalties.
4. **⏱️ Biometric Wrinkle & Age Estimation (`age_estimator.py`)**:
   - Estimates facial age (`~20 - 25 yrs` youthful baseline with dynamic aging progression) using CLAHE contrast normalization, horizontal forehead Sobel derivatives, periocular crow's feet Canny edge density, nasolabial fold prominence, and cheek Laplacian variance, with optional OpenCV DNN support.
5. **⭐ 189-Celebrity & Influencer Look-Alike Gallery (`celebrity_matcher.py`)**:
   - Evaluates 189 verified film stars, YouTubers, digital influencers, pop icons, and world-champion athletes with unified, region-neutral category tags (`ACTOR`, `YOUTUBER`, `INFLUENCER`, `POP STAR`, `ATHLETE`) and explainable trait breakdowns.

---

## 🧠 How the Models Were Trained & Training Datasets

FaceCard uses a **two-stage training architecture**:
1. **Stage 1 (Foundational Deep Neural Backbones)**: Large-scale convolutional neural networks pre-trained on millions of labeled human faces for universal feature extraction.
2. **Stage 2 (Local Metric Learning & Calibration via `train_models.py`)**: Supervised metric learning, logistic classification, and ridge regression trained directly on our **multi-view augmented 181-identity gallery dataset** ($905$ aligned facial views).

### Stage 1: Foundational Deep Learning Backbones (Pre-Training)

#### 1. Face & 5-Keypoint Landmark Detector — **YuNet CNN (`face_detection_yunet_2023mar.onnx`)**
- **Training Dataset**: **WIDER FACE Dataset** ($32,203$ images containing $393,703$ labeled faces across wide variations in scale, pose, occlusion, expression, and illumination, augmented with 5-point facial landmark annotations).
- **Architecture & Training Method**:
  - Ultra-lightweight MobileNet-inspired depthwise separable CNN with a 3-stride Feature Pyramid Network ($\text{strides } \in \{8, 16, 32\}$).
  - Trained end-to-end using **anchor-free multi-task loss**: Binary Cross-Entropy (classification + objectness), **Complete IoU (CIoU) Loss** for 2D bounding box regression, and **Smooth-$L_1$ / Wing Loss** for the 5 facial keypoints.

#### 2. Deep Face Recognition Backbone — **SFace (`face_recognition_sface_2021dec.onnx`)**
- **Training Dataset**: **MS-Celeb-1M (MS1MV2 / Refine-MS1M)** & **WebFace** ($\sim 5.8\text{ million}$ aligned $112 \times 112$ face images across $\sim 85,000$ distinct human identities).
- **Architecture & Training Method**:
  - MobileFaceNet CNN backbone producing a compact **128-dimensional embedding vector** ($\mathbf{e} \in \mathbb{R}^{128}$, L2-normalized onto the unit hypersphere $\|\mathbf{e}\|_2 = 1$).
  - Trained with **Sigmoid-Constrained Hypersphere Loss (SFace Loss)**, which improves upon ArcFace/CosFace by dynamically moderating intra-class compactness pull and inter-class discrepancy push via sigmoid gradient rescaling, making embeddings resilient to noisy web labels and webcam lighting variations.

#### 3. Demographic Backbone — **OpenVINO Retail (`age-gender-recognition-retail-0013.onnx`) & Biometric Wrinkle Engine**
- **Training Dataset**: **IMDB-WIKI Dataset** ($500,000+$ celebrity faces with chronological timestamps and gender annotations) + multi-ethnic demographic datasets.
- **Biometric Calibration**: Calibrated computer vision Sobel/Canny/Laplacian aging operators anchored to a youthful adult baseline ($20.0\text{ yrs}$) with EMA temporal smoothing ($\alpha = 0.12$).

---

### Stage 2: Local Domain-Specific Training & Calibration (`train_models.py`)

Running `python train_models.py` executes a 4-step supervised training pipeline on our local gallery (`celebrity_gallery/source/`) and outputs `models/calibrated_models.npz`:

1. **5-Point Affine Alignment & Multi-View Data Augmentation**:
   - Every portrait in `celebrity_gallery/source/` ($181$ verified identities) is detected with YuNet and warped to a canonical $112 \times 112$ coordinate frame using a similarity transform (`cv2.estimateAffinePartial2D`).
   - For each identity, **5 augmented training views** are synthesized ($181 \times 5 = 905$ training samples):
     1. Canonical upright $112 \times 112$ aligned face
     2. Horizontally mirrored aligned face
     3. Warm/bright photometric exposure ($+7\%$ contrast, $+5$ luminance)
     4. Cool/dim photometric exposure ($-7\%$ contrast, $-5$ luminance)
     5. $4\%$ central scale zoom crop
2. **Discriminative Metric Whitening & Subspace Projection (`emb_proj`, `emb_mean`)**:
   - Computes the **Within-Identity Scatter Matrix ($\mathbf{S}_W$)** across augmented views of the same person and the **Between-Identity Scatter Matrix ($\mathbf{S}_B$)** across different celebrities:
     $$\mathbf{S}_W = \frac{1}{C} \sum_{c=1}^{C} \frac{1}{M_c} \sum_{j=1}^{M_c} (\mathbf{e}_{c,j} - \boldsymbol{\mu}_c)(\mathbf{e}_{c,j} - \boldsymbol{\mu}_c)^T, \quad \mathbf{S}_{W,\text{reg}} = (1 - \gamma)\mathbf{S}_W + \gamma \frac{\text{Tr}(\mathbf{S}_W)}{D}\mathbf{I}$$
   - Learns a regularized whitening projection $\mathbf{W}_{\text{proj}} \in \mathbb{R}^{128 \times 128}$ via eigendecomposition ($\mathbf{S}_{W,\text{reg}}^{-1/2}$) that suppresses lighting/pose noise directions and amplifies identity-discriminative directions (**99.45% Rank-1** and **100.00% Rank-3** multi-view validation accuracy).
3. **Fisher Discriminative Biometric Trait Weights (`trait_weights`)**:
   - Evaluates the ratio of between-identity variance to within-identity variance across all 12 craniofacial proportions:
     $$w_k = \frac{\text{Var}_{\text{between}}(f_k)}{\text{MeanVar}_{\text{within}}(f_k)}$$
   - Automatically assigns higher weights to stable structural traits (eye spacing, nose height, eye-mouth span) and lower weights to lighting-sensitive traits.
4. **Supervised Demographic Probes on 128-D SFace Embeddings (`gender_weights`, `age_weights`)**:
   - Trains an **L2-Regularized Logistic Regression Classifier** on the 128-D SFace embeddings via Newton-Raphson / Iteratively Reweighted Least Squares (IRLS) (**93.15% 5-fold cross-validation accuracy**) to ensure gender-consistent look-alike matching.

---

### Training Dataset Summary Table

| Component | Model File | Training Dataset | Training Size | Learned Output / Task |
|:---|:---|:---|:---|:---|
| **Face & Landmark Detector** | `face_detection_yunet_2023mar.onnx` | **WIDER FACE** + 5-Point Keypoints | $32,203$ images ($393,703$ faces) | 2D Bounding Box + 5 Landmarks |
| **Deep Face Recognizer** | `face_recognition_sface_2021dec.onnx` | **MS-Celeb-1M (MS1MV2)** | $\sim 5.8\text{M}$ images ($85\text{K}$ identities) | 128-D Unit Hypersphere Embedding |
| **Demographic DNN** | `age-gender-recognition-retail-0013.onnx` | **IMDB-WIKI** + Retail Demographics | $500,000+$ labeled face crops | Continuous Age + Softmax Gender |
| **Local Metric & Demographic Calibrator** | `models/calibrated_models.npz` | **FaceCard Celebrity Gallery** (`celebrity_gallery/source/`) | $181$ identities $\times 5$ views ($905$ aligned crops) | $128 \times 128$ Whitening Matrix, 12 Fisher Weights & Gender IRLS Probe |
| **Celebrity Look-Alike Index** | `celebrity_gallery/index.npz` | **181 Verified Portraits + Curated Profiles** | $189$ Total Personalities | Precomputed 128-D SFace Embeddings + 12-D Biometric Feature Cards |

---

## 🌟 Expanded Celebrity, Influencer & Athlete Database

The look-alike gallery spans **189 personalities** across film, digital media, music, and sports:

- **🎬 Film Actors & Actresses**:
  - Shah Rukh Khan, Salman Khan, Aamir Khan, Amitabh Bachchan, Hrithik Roshan, Ranbir Kapoor, Ranveer Singh, Vicky Kaushal, Shahid Kapoor, Ayushmann Khurrana, Kartik Aaryan, Varun Dhawan, Sidharth Malhotra, Rajkummar Rao, Pankaj Tripathi, Allu Arjun, Prabhas, Ram Charan, N. T. Rama Rao Jr., Yash, Vijay, Rajinikanth, Dhanush, Dulquer Salmaan, Mahesh Babu, Deepika Padukone, Alia Bhatt, Priyanka Chopra, Aishwarya Rai Bachchan, Katrina Kaif, Kareena Kapoor, Shraddha Kapoor, Kriti Sanon, Kiara Advani, Rashmika Mandanna, Samantha Ruth Prabhu, Anushka Sharma, Janhvi Kapoor, Sara Ali Khan, Ananya Panday, Triptii Dimri, Sai Pallavi, Tamannaah Bhatia, Leonardo DiCaprio, Keanu Reeves, Robert Downey Jr., Scarlett Johansson, Zendaya, Timothee Chalamet, Chris Evans, Chris Hemsworth, Margot Robbie, Emma Watson, Cillian Murphy, Pedro Pascal, Jenna Ortega, and more.
- **📱 YouTubers, Content Creators & Influencers**:
  - CarryMinati, Bhuvan Bam, Ashish Chanchlani, Prajakta Koli, Kusha Kapila, Dhruv Rathee, Samay Raina, Zakir Khan, Tanmay Bhat, Ranveer Allahbadia, Triggered Insaan, Techno Gamerz, Komal Pandey, Dolly Singh, Mithila Palkar, Avneet Kaur, Jannat Zubair Rahmani, Vir Das, Munawar Faruqui, MrBeast, PewDiePie, Marques Brownlee (MKBHD), Markiplier, Emma Chamberlain, Kylie Jenner, Charli D'Amelio, and more.
- **🎵 Music & Pop Stars**:
  - Diljit Dosanjh, Arijit Singh, Shreya Ghoshal, A. R. Rahman, Taylor Swift, Billie Eilish, Ariana Grande, Dua Lipa, Rihanna, Selena Gomez, Drake, The Weeknd, Bruno Mars, Adele, Ed Sheeran, Justin Bieber, and more.
- **🏆 Sports & World Champion Athletes**:
  - Virat Kohli, MS Dhoni, Rohit Sharma, Sachin Tendulkar, Hardik Pandya, Jasprit Bumrah, Shubman Gill, KL Rahul, Smriti Mandhana, Neeraj Chopra, P. V. Sindhu, Mary Kom, Sania Mirza, Sunil Chhetri, Manu Bhaker, LeBron James, Stephen Curry, Cristiano Ronaldo, Lionel Messi, Rafael Nadal, Naomi Osaka, Usain Bolt, Simone Biles, and more.

---

## 🔬 Technical Pipeline & Mathematical Formulation

1. **5-Point Similarity Face Alignment**:
   - Solves $\min_{\mathbf{M}} \sum_{k=1}^{5} \|\mathbf{M} \mathbf{p}_k - \mathbf{p}_k^{\text{ref}}\|_2^2$ via `cv2.estimateAffinePartial2D` to warp the live face to canonical $112 \times 112$ ArcFace/SFace coordinates.
2. **Hybrid Deep SFace + 12-Check Biometric Look-Alike Scoring**:
   - Combines raw 128-D cosine similarity, whitened subspace cosine similarity, and Fisher-weighted 12-trait anthropometric similarity:
     $$\text{Sim}_{\text{deep}} = 0.65 \cdot (\mathbf{e}_q^T \mathbf{e}_r) + 0.35 \cdot (\tilde{\mathbf{e}}_q^T \tilde{\mathbf{e}}_r)$$
     $$\text{Score}_{\text{hybrid}} = 0.68 \cdot \text{Scale}(\text{Sim}_{\text{deep}}) + 0.32 \cdot \text{Score}_{\text{12-traits}}$$
3. **ITA° Melanin-Adaptive & Glare-Invariant Skin Grading**:
   - Masks specular glare ($L^* > 235$) and computes the **Individual Typology Angle**:
     $$\text{ITA}^\circ = \arctan\left(\frac{L^* - 50}{b^*}\right) \times \frac{180}{\pi}$$
   - Dynamically calibrates black-hat pigmentation thresholds and removes low-frequency room lighting gradients before measuring tone uniformity.

---

## 📁 Repository Structure

```text
CV_MINIprj/
├── docs/                                         # Project documentation & Word reports
│   ├── facecard.docx                             # Complete Microsoft Word technical report
│   └── FaceCard_Project_Report.docx              # Formatted academic documentation
├── models/                                       # Deep learning weights & calibrated parameters
│   ├── face_detection_yunet_2023mar.onnx         # YuNet CNN Face & 5-Point Landmark Detector
│   ├── face_recognition_sface_2021dec.onnx       # SFace 128-D Deep Face Recognition Model
│   ├── calibrated_models.npz                     # Trained Whitening Matrix, Fisher Weights & Probes
│   ├── age-gender-recognition-retail-0013.onnx   # OpenVINO Demographic Model
│   └── haarcascade_frontalface_default.xml       # Classical Haar Cascade fallback
├── celebrity_gallery/                            # 181-Celebrity Verified Portrait Dataset
│   ├── attributions.csv                          # Creative Commons image attributions
│   ├── index.npz                                 # Precomputed 128-D SFace embeddings & 12-D cards
│   ├── references/                               # Center-cropped reference face portraits
│   └── source/                                   # Full source portrait folders
├── scripts/                                      # Model & gallery management scripts
│   ├── download_models.py                        # Fetches YuNet, SFace, and Age/Gender ONNX models
│   └── expand_indian_celebrities.py              # Fetches verified Indian celebrity & creator portraits
├── app.py                                        # Main Native OpenCV Desktop GUI Application
├── face_detector.py                              # YuNet Detector, 5-Point Aligner & SFace Engine
├── skin_analyzer.py                              # ITA°-Calibrated Dermatological Skin Analyzer
├── age_estimator.py                              # Biometric Wrinkle & Age Estimator
├── celebrity_matcher.py                          # Hybrid 128-D SFace + 12-Check Celebrity Matcher
├── train_models.py                               # Local Metric Learning & Calibration Training Script
├── test_system.py                                # Pre-Launch File Integrity & System Readiness Suite
├── facecard.docx                                 # Root Microsoft Word Technical Report
├── run.bat                                       # 1-Click Windows Launcher
├── run.sh                                        # 1-Click Linux / WSL Launcher
├── requirements.txt                              # Python dependencies
└── README.md                                     # Master Project Documentation
```

---

## 🚀 Quick Start

### 1. One-Click Launchers (`run.bat` / `run.sh`)
Double-click **`run.bat`** on Windows (or `./run.sh` on Linux/WSL):
1. **Launch Native OpenCV Desktop App**
2. **Test Before Launching - Check File Integrity & System Readiness**
3. **Install / Verify Requirements**

### 2. Manual Execution
```bash
# Test file integrity and system readiness before launching
python test_system.py

# Launch Native OpenCV Desktop App
python app.py

# Retrain / recalibrate models on gallery (optional)
python train_models.py
```

---

## ⌨️ Interactive Keyboard Shortcuts

| Key | Action | Description |
|:---:|:---|:---|
| **`C`** | **Cycle Celebrity** | Manually cycles through the celebrity & influencer gallery |
| **`M`** | **Toggle Mask** | Toggles live YCrCb/HSV skin segmentation mask overlay |
| **`S`** | **Snapshot** | Saves an annotated HUD screenshot to `snapshots/` |
| **`SPACE`** | **Pause / Resume** | Freezes live video feed to inspect measurements |
| **`Q`** / **`ESC`** | **Quit** | Cleanly releases camera and closes windows |

---

## 🧪 Self-Test & File Integrity Verification

Run the pre-launch integrity and verification suite:
```bash
python test_system.py
```
