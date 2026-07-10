

# Generic 3D facial model points (millimeters)

# MediaPipe landmark indices

NOSE = 1
CHIN = 152

LEFT_EYE = 33
RIGHT_EYE = 263

LEFT_MOUTH = 61
RIGHT_MOUTH = 291

# Store them together for convenience

LANDMARKS = {
    "Nose": NOSE,
    "Chin": CHIN,
    "Left Eye": LEFT_EYE,
    "Right Eye": RIGHT_EYE,
    "Left Mouth": LEFT_MOUTH,
    "Right Mouth": RIGHT_MOUTH
}

import numpy as np

# Generic 3D face model (millimetres)
# Order must match the landmark extraction order:
# Nose, Chin, Left Eye, Right Eye, Left Mouth, Right Mouth

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye outer corner
    (225.0, 170.0, -135.0),   # Right eye outer corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0)   # Right mouth corner
], dtype=np.float64)