import cv2 #used for webcam capture
import numpy as np #numerical operations
import joblib #loads trained model
import textwrap #wraps long sentence text nicely
import time #time for hold, cooldown, and spaces
import os #checks if files exist

import mediapipe as mp #detects the hand and gives landmark coordinates
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from landmarks import normalize_landmarks

# =========================
# CONFIG
# =========================
MODEL_PATH      = "hand_landmarker.task"
PKL_PATH        = "asl_model.pkl"
CONFIRM_SECONDS  = 1
COOLDOWN_SECONDS = 1   # cooldown after a successful confirmation, during which the same letter won't be confirmed again
SPACE_SECONDS    = 4   # seconds of inactivity before a space is inserted
MIN_CONFIDENCE   = 0.6
CAMERA_INDEX    = 0
PANEL_W         = 340

CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),
    (0,5), (5,6), (6,7), (7,8),
    (5,9), (9,10), (10,11), (11,12),
    (9,13), (13,14), (14,15), (15,16),
    (13,17), (17,18), (18,19), (19,20),
    (0,17)
]

# State machine states
IDLE       = "IDLE"
PREDICTING = "PREDICTING"
COOLDOWN   = "COOLDOWN"

STATE_COLORS = {
    IDLE:       (120, 120, 120),
    PREDICTING: (0, 200, 255),
    COOLDOWN:   (0, 255, 180),
}


# =========================
# LOAD MODEL
# =========================
if not os.path.exists(PKL_PATH):
    raise FileNotFoundError(
        f"Model file '{PKL_PATH}' not found.\n"
        "Run `python train_model.py` first to generate it."
    )
rf = joblib.load(PKL_PATH)
print(f"Loaded model from {PKL_PATH}")


# =========================
# SETUP MEDIAPIPE
# =========================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Missing MediaPipe model: {MODEL_PATH}\n"
        "Download hand_landmarker.task and place it next to this script."
    )

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)


# =========================
# DRAWING HELPERS
# =========================
def draw_text(img, text, pos, scale=0.6, color=(255,255,255), thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_panel(panel, pred_letter, confidence, state, sentence, hold_frac):
    panel[:] = (25, 25, 25)
    h, w = panel.shape[:2]

    # Large predicted letter
    letter_str = pred_letter if pred_letter else "-"
    cv2.putText(panel, letter_str, (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 220, 80), 5, cv2.LINE_AA)

    # Confidence
    conf_str = f"{confidence*100:.1f}%" if pred_letter else "---"
    draw_text(panel, f"Confidence: {conf_str}", (30, 130), scale=0.65)

    # Progress bar
    bar_x, bar_y, bar_w, bar_h = 30, 155, w - 60, 22
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
    fill = int(bar_w * min(hold_frac, 1.0))
    bar_color = (0, 255, 180) if hold_frac >= 1.0 else (0, 200, 80)
    if fill > 0:
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), bar_color, -1)
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (150, 150, 150), 1)

    # State indicator
    s_color = STATE_COLORS.get(state, (180, 180, 180))
    draw_text(panel, f"State: {state}", (30, 205), scale=0.6, color=s_color)

    # Sentence display (last 3 lines of word-wrapped text)
    cv2.rectangle(panel, (20, 225), (w - 20, 350), (45, 45, 45), -1)
    cv2.rectangle(panel, (20, 225), (w - 20, 350), (100, 100, 100), 1)
    draw_text(panel, "Sentence:", (30, 220), scale=0.5, color=(180, 180, 180))

    display_text = sentence if sentence else ""
    wrapped = textwrap.wrap(display_text, width=24) if display_text.strip() else [""]
    last3 = wrapped[-3:] if len(wrapped) >= 3 else wrapped
    for idx, line in enumerate(last3):
        draw_text(panel, line, (30, 255 + idx * 32), scale=0.7, color=(255, 255, 255))

    # Controls legend
    controls = [
        "Backspace: delete char",
        "c: clear sentence",
        "q: quit",
    ]
    y0 = h - len(controls) * 24 - 10
    for line in controls:
        draw_text(panel, line, (20, y0), scale=0.48, color=(160, 160, 160))
        y0 += 24


# =========================
# MAIN LOOP
# =========================
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera at index {CAMERA_INDEX}")

sentence          = ""
state             = IDLE
current_letter    = None
hold_start        = None
cooldown_start    = None
last_confirm_time = None
pred_letter           = None
confidence            = 0.0
frame_ts_ms           = 0

print("ASL Recognizer started. Press 'q' to quit.")

with vision.HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        frame_ts_ms += 33
        result = landmarker.detect_for_video(mp_img, frame_ts_ms)

        now = time.time()
        hold_frac = 0.0

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]

            # Draw skeleton on camera feed
            for a, b in CONNECTIONS:
                x1 = int(hand[a].x * w)
                y1 = int(hand[a].y * h)
                x2 = int(hand[b].x * w)
                y2 = int(hand[b].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 80, 0), 2)
            for lm in hand:
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            # Predict
            vec = normalize_landmarks(hand)
            if vec is not None:
                proba = rf.predict_proba([vec])[0]
                best_idx = np.argmax(proba)
                pred_letter = rf.classes_[best_idx]
                confidence  = proba[best_idx]

                # State machine
                if state == IDLE:
                    state          = PREDICTING
                    current_letter = pred_letter
                    hold_start     = now

                elif state == PREDICTING:
                    if pred_letter != current_letter:
                        current_letter = pred_letter
                        hold_start     = now
                    else:
                        held = now - (hold_start or now)
                        hold_frac = held / CONFIRM_SECONDS
                        if held >= CONFIRM_SECONDS and confidence >= MIN_CONFIDENCE:
                            sentence          += pred_letter
                            last_confirm_time  = now
                            cooldown_start     = now
                            state              = COOLDOWN

                elif state == COOLDOWN:
                    elapsed = now - (cooldown_start or now)
                    # Bar drains from full → empty during cooldown
                    hold_frac = 1.0 - (elapsed / COOLDOWN_SECONDS)
                    if elapsed >= COOLDOWN_SECONDS:
                        # Cooldown done — arm the same letter again
                        current_letter = pred_letter
                        hold_start     = now
                        state          = PREDICTING
                    elif pred_letter != current_letter:
                        # Different letter shown — exit cooldown immediately
                        current_letter = pred_letter
                        hold_start     = now
                        state          = PREDICTING
            else:
                pred_letter = None
                confidence  = 0.0

        else:
            # No hand detected — reset to IDLE but don't touch last_confirmed_letter
            pred_letter = None
            confidence  = 0.0
            hold_frac   = 0.0
            state       = IDLE
            current_letter = None
            hold_start     = None
            cooldown_start = None

        # Insert space after SPACE_SECONDS of no new confirmation
        if (last_confirm_time is not None
                and now - last_confirm_time >= SPACE_SECONDS
                and sentence
                and sentence[-1] != " "):
            sentence          += " "
            last_confirm_time  = now

        # Build composite frame: panel (left) + camera (right)
        canvas = np.zeros((h, PANEL_W + w, 3), dtype=np.uint8)
        canvas[:, PANEL_W:] = frame
q
        panel = canvas[:, :PANEL_W]
        draw_panel(panel, pred_letter, confidence, state, sentence, hold_frac)

        cv2.imshow("ASL Recognizer", canvas)

        key = cv2.waitKey(1) & 0xFF

        if key in (8, 127):  # Backspace
            sentence = sentence[:-1]
        elif key == ord("c"):
            sentence          = ""
            state             = IDLE
            current_letter    = None
            hold_start        = None
            cooldown_start    = None
            last_confirm_time = None
        elif key == ord("q"):
            print("Quitting...")
            break

cap.release()
cv2.destroyAllWindows()
