# Driver Attention Monitor

> **Real-time, on-device drowsiness and distraction detection system using computer vision**

A multi-signal driver monitoring system that fuses eye tracking (EAR/PERCLOS), mouth analysis (MAR/yawn detection), head pose estimation, iris-based gaze tracking, and hand/object detection into a unified drowsiness scoring pipeline. Built entirely with local processing — no cloud, no wearables, no data ever leaves the device.

---

## Problem Statement

Driver drowsiness is responsible for approximately 20% of all road accidents worldwide (WHO, 2023). Existing driver monitoring systems suffer from several critical limitations:

1. **Single-signal dependency** — Most systems rely on one indicator (e.g., only eye closure or only steering patterns), making them brittle to occlusion and individual variation
2. **Binary threshold alarms** — Systems trigger abrupt alerts that cause alarm fatigue, leading drivers to disable them
3. **Session-only calibration** — Every session starts from scratch, ignoring that a driver's baseline is relatively stable across days
4. **All-or-nothing monitoring** — Every sensor runs at full frequency regardless of risk level, wasting computation on a battery-constrained device
5. **Cloud/hardware dependency** — Many systems require wearables (EEG headbands) or cloud processing (video upload for server-side analysis)

This project addresses all five limitations with a multi-signal fusion pipeline featuring adaptive resource allocation, escalating alerts, cross-session learning, and fully on-device processing.

---

## Objectives

1. Detect driver drowsiness using multiple physiological signals (EAR, MAR, PERCLOS, head pose, blink duration)
2. Detect driver distraction via head pose deviation and iris-based gaze tracking
3. Detect handheld object usage (phone, cup) via YOLOv8 + MediaPipe Hands fusion
4. Fuse all signals into a weighted drowsiness score (0–100) with state machine (ALERT → WARNING → CRITICAL)
5. Implement adaptive sensor scheduling that scales monitoring intensity to actual risk level
6. Implement escalating alert response to prevent alarm fatigue
7. Implement cross-session calibration persistence that learns driver baselines over time
8. Run entirely on-device with zero cloud dependency — no video ever leaves the device

---

## Datasets Used

| Dataset | Purpose | Samples | Source |
|---|---|---|---|
| **CEW** (Closed Eyes in the Wild) | EAR threshold validation | 2,423 eye images | [Kaggle](https://www.kaggle.com/datasets/ahamedfarouk/cew-dataset) |
| **YawDD** | MAR threshold and yawn detection validation | 322 videos | [Kaggle](https://www.kaggle.com/datasets/enider/yawdd-dataset) |
| **NTHU-DDD** (Drowsy Driver Detection) | End-to-end evaluation across conditions (bare face, glasses, sunglasses, night) | 36 subjects, 4 conditions | [Kaggle](https://www.kaggle.com/datasets/samymesbah/nthu-dataset-ddd-multi-class) |

> **Note**: Dataset files are too large to include in the repository. Download them using the links above and place them in `data/cew/`, `data/yawdd/`, and `data/nthu/` respectively. See `shared/DATA_SETUP.md` for detailed instructions.

The YOLOv8 model (`yolov8n.pt`, ~6.5 MB) is included in the repository. It is the nano variant from [Ultralytics](https://docs.ultralytics.com/models/yolov8/) and is loaded locally — no download required at runtime.

---

## Technologies / Libraries Used

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core language |
| OpenCV | 4.8+ | Video capture, image processing, CLAHE, solvePnP |
| MediaPipe | 0.10+ | Face mesh (468+10 landmarks), hand detection, iris tracking |
| NumPy | 1.24+ | Numerical computation, signal processing |
| Ultralytics YOLOv8 | 8.0+ | Object detection (phone, cup, book) |
| scikit-learn | 1.3+ | Validation metrics (ROC, accuracy, confusion matrix) |
| Matplotlib/Seaborn | 3.7+ | Visualization, EAR/MAR distribution plots |
| pygame | 2.5+ | Audio alert generation (programmatic tones) |
| pyttsx3 | 2.90+ | Text-to-speech voice alerts |
| MLflow | 2.0+ | Experiment tracking for ablation studies |

---

## Methodology

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        WEBCAM INPUT (30 fps)                           │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              PREPROCESSING: CLAHE Enhancement                          │
│  LAB colorspace → CLAHE on L channel → uniform lighting                │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│           SHARED MEDIAPIPE FACEMESH (478 landmarks)                    │
│  Single inference → shared by all downstream modules                   │
└──┬──────────────┬──────────────┬──────────────┬────────────────────────┘
   │              │              │              │
   ▼              ▼              ▼              ▼
┌────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐
│PERCEPT.│  │HEAD POSE  │  │IRIS/GAZE │  │HAND/OBJECT    │
│EAR/MAR │  │solvePnP   │  │Iris lm   │  │MediaPipe Hands│
│Blink   │  │Nod detect │  │468-477   │  │+ YOLOv8 nano  │
│Yawn    │  │Distraction│  │Sustained │  │Grip+proximity │
└───┬────┘  └─────┬─────┘  └────┬─────┘  └──────┬────────┘
    │             │             │               │
    ▼             ▼             ▼               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    ADAPTIVE SENSOR SCHEDULER                           │
│  RELAXED → ATTENTIVE → ELEVATED → MAXIMUM (scales with risk)          │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              WEIGHTED FUSION & SCORING (0-100)                         │
│  PERCLOS 40% | Yawn Rate 25% | Blink Duration 20% | Nod Count 15%     │
│  Adaptive weights when eye confidence is low (glasses/dark)            │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│          STATE MACHINE: ALERT ←→ WARNING ←→ CRITICAL                   │
│  Hysteresis-based transitions (hold durations prevent flickering)       │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              ESCALATING ALERT RESPONSE                                 │
│  Subtle amber pulse → gentle TTS → urgent red flash + alarm            │
│  Prevents alarm fatigue through graduated sensory escalation           │
└─────────────────────────────────────────────────────────────────────────┘
```

### Novel Architectural Features

1. **Adaptive Sensor Scheduling**: Scales monitoring intensity to actual risk. In ALERT state (normal driving), hand/YOLO detection runs every 15th frame (~93% reduction), gaze every 5th frame (~80% reduction). Escalates to full frequency in CRITICAL state.

2. **Escalating Alert Response**: Replaces binary threshold→alarm with graduated responses: subtle amber pulsing border (WARNING onset) → gentle TTS reminders → urgent red flash + repeated voice alerts (CRITICAL). Prevents the well-documented problem of alarm fatigue/habituation.

3. **Cross-Session Calibration Persistence**: Saves EAR thresholds, head pose offsets, and gaze baselines to a local JSON file. Averages across sessions using exponential decay weighting (recent sessions weighted higher). Like a fitness tracker learning resting heart rate.

4. **On-Device Privacy Architecture**: Entire pipeline runs locally — webcam → MediaPipe → models → display. No video leaves the device, nothing is recorded or uploaded, no cloud APIs called. Explicit privacy manifest and startup verification.

---

## Steps to Execute

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/driver-attention-monitor.git
cd driver-attention-monitor
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r shared/requirements.txt
pip install ultralytics
```

### 4. Download Datasets (for validation only)
```bash
# Option A: Kaggle CLI
pip install kaggle
kaggle datasets download -d ahamedfarouk/cew-dataset -p data/cew --unzip
kaggle datasets download -d enider/yawdd-dataset -p data/yawdd --unzip
kaggle datasets download -d samymesbah/nthu-dataset-ddd-multi-class -p data/nthu --unzip

# Option B: Manual download from the Kaggle links above into data/ subfolders
```

### 5. Run the Application
```bash
# Default: webcam 0, all features enabled
python app.py

# Specify video source
python app.py --source path/to/video.mp4
python app.py --source 1  # webcam index

# Disable specific features
python app.py --no-hands     # disable hand/object detection
python app.py --no-mesh      # skip face mesh overlay
python app.py --no-adaptive  # disable adaptive scheduling (benchmark mode)
python app.py --no-persist   # disable cross-session calibration
python app.py --no-log       # disable CSV logging
```

### 6. Run Validation Scripts
```bash
# EAR threshold validation on CEW dataset
python perception/validate_cew.py

# MAR threshold validation on YawDD dataset
python perception/validate_yawdd.py

# Robustness test across NTHU conditions
python perception/validate_nthu.py

# Ablation study
python evaluation/ablation_study.py
```

### 7. Controls
- **Q** — Quit the application
- Calibration runs automatically for the first ~13 seconds (10s EAR + 3s pose)
- Look straight ahead during calibration for best results

---

## Results

### Live System Performance (11,689 frames, 10.6 min session)

| Metric | Value |
|---|---|
| Average FPS | 18.4 fps |
| Face detection rate | 99.4% |
| EAR mean ± std | 0.318 ± 0.081 |
| MAR mean ± std | 0.343 ± 0.071 |
| Average blink duration | 132.5 ms |
| Total blinks detected | 210 |
| Total yawns detected | 1 |
| Drowsy nods detected | 2 |
| PERCLOS mean / max | 11.0% / 28.9% |
| Drowsiness score mean / max | 6.7 / 24.1 |

### Dataset Benchmark Validation

| Benchmark Dataset | Samples Evaluated | Key Metric | Result | Output Plot |
|---|---|---|---|---|
| **CEW** (Eye State) | 3,700 images | ROC AUC / Optimal Acc | **0.9808** / **93.27%** (@ 0.185) | `results/cew_evaluation.png` |
| **YawDD** (Yawn Detection) | 9,367 frames (319 videos) | Accuracy / Optimal MAR | **67.26%** / **0.62** (@ 68.86%) | `results/yawdd_evaluation.png` |
| **NTHU-DDD** (Robustness) | 1,819 frames | Face Detection Rate (No Glasses / Glasses) | **94.2%** / **87.7%** (AUC: 0.6593) | `results/nthu_evaluation.png` |

To reproduce all evaluations:
```bash
python run_all_evaluations.py
```

### Scoring Weights

| Signal | Default Weight | Low-Confidence Fallback |
|---|---|---|
| PERCLOS | 0.40 | 0.20 |
| Yawn Rate | 0.25 | 0.35 |
| Blink Duration | 0.20 | 0.10 |
| Nod Count | 0.15 | 0.35 |

### State Machine Thresholds

| Transition | Threshold | Hold Duration |
|---|---|---|
| ALERT → WARNING | Score ≥ 30 | 5 seconds sustained |
| WARNING/ALERT → CRITICAL | Score ≥ 60 | 3 seconds sustained |
| WARNING/CRITICAL → ALERT | Score < 20 | 10 seconds sustained |

---

## Project Structure

```
driver-attention-monitor/
├── app.py                          # Main unified application
├── yolov8n.pt                      # YOLOv8 nano model (local)
├── output_log.csv                  # Session log (auto-generated)
├── perception/
│   ├── perception.py               # EAR, MAR, blink, yawn, CLAHE, calibration
│   ├── validate_cew.py             # CEW dataset validation
│   ├── validate_nthu.py            # NTHU robustness testing
│   ├── validate_yawdd.py           # YawDD dataset validation
│   └── test_perception.py          # Unit tests
├── decision/
│   ├── adaptive_scheduler.py       # Adaptive sensor scheduling
│   ├── alert_system.py             # Escalating alert response
│   ├── calibration_store.py        # Cross-session persistence
│   ├── session_reporter.py         # End-of-trip report & timeline generator
│   ├── calibrator.py               # Head pose calibration
│   ├── camera_offset.py            # Camera angle correction
│   ├── config.py                   # 3D model points, landmark IDs
│   ├── distraction_detector.py     # Sustained distraction detection
│   ├── face_loss_detector.py       # Face disappearance alerting
│   ├── face_selector.py            # Background face rejection
│   ├── head_pose.py                # solvePnP head pose estimation
│   ├── nod_detector.py             # Drowsy nod detection
│   ├── nod_rate.py                 # Rolling nod rate window
│   ├── output_contract.py          # Decision output schema
│   ├── perclos.py                  # Time-weighted PERCLOS window
│   ├── pose_sanity.py              # Pose plausibility filter
│   ├── privacy_guard.py            # On-device privacy architecture
│   ├── scoring.py                  # Weighted drowsiness scoring
│   ├── state_machine.py            # ALERT/WARNING/CRITICAL FSM
│   └── yawn_rate.py                # Rolling yawn rate window
├── evaluation/
│   ├── ablation_study.py           # Signal combination ablation
│   ├── hand_detector.py            # Hand/object detection (YOLO+MediaPipe)
│   ├── iris_tracker.py             # Iris-based gaze tracking
│   └── synthetic_stubs.py          # Synthetic data for early testing
├── shared/
│   ├── DATA_SETUP.md               # Dataset download instructions
│   ├── data_contract.md            # Inter-module interface contract
│   └── requirements.txt            # Python dependencies
└── data/                           # Datasets (gitignored)
    ├── cew/
    ├── yawdd/
    └── nthu/
```

---

## License

This project was developed as part of an academic course (Trimester 4). All code is original work by the project team.

## Acknowledgments

- [MediaPipe](https://google.github.io/mediapipe/) by Google for face mesh and hand detection
- [Ultralytics YOLOv8](https://docs.ultralytics.com/) for object detection
- NTHU Drowsy Driver Detection Dataset
- CEW (Closed Eyes in the Wild) Dataset
- YawDD Yawning Detection Dataset