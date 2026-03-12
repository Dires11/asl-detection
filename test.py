import cv2
import csv
import os
import string
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================
# CONFIG
# =========================
MODEL_PATH = "hand_landmarker.task"
CSV_PATH = "asl_landmarks.csv"
SAMPLES_PER_LETTER = 40
CAMERA_INDEX = 0

# Static ASL letters only for now.
# J and Z involve motion, so they are usually handled differently.
LETTERS = [ch for ch in string.ascii_uppercase if ch not in {"J", "Z"}]

LANDMARK_NAMES = {
    0: "WRIST",
    1: "THUMB_CMC",
    2: "THUMB_MCP",
    3: "THUMB_IP",
    4: "THUMB_TIP",
    5: "INDEX_MCP",
    6: "INDEX_PIP",
    7: "INDEX_DIP",
    8: "INDEX_TIP",
    9: "MIDDLE_MCP",
    10: "MIDDLE_PIP",
    11: "MIDDLE_DIP",
    12: "MIDDLE_TIP",
    13: "RING_MCP",
    14: "RING_PIP",
    15: "RING_DIP",
    16: "RING_TIP",
    17: "PINKY_MCP",
    18: "PINKY_PIP",
    19: "PINKY_DIP",
    20: "PINKY_TIP",
}

CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),
    (0,5), (5,6), (6,7), (7,8),
    (5,9), (9,10), (10,11), (11,12),
    (9,13), (13,14), (14,15), (15,16),
    (13,17), (17,18), (18,19), (19,20),
    (0,17)
]

# =========================
# CSV HELPERS
# =========================
def create_csv_if_needed(csv_path: str):
    if os.path.exists(csv_path):
        return

    header = ["label"]
    for i in range(21):
        header += [f"x_{i}", f"y_{i}", f"z_{i}"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

def save_landmarks(csv_path: str, label: str, hand_landmarks):
    row = [label]
    for lm in hand_landmarks:
        row += [lm.x, lm.y, lm.z]

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

# =========================
# SETUP
# =========================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Missing model file: {MODEL_PATH}\n"
        "Download the MediaPipe Hand Landmarker model and place it next to this script."
    )

create_csv_if_needed(CSV_PATH)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera at index {CAMERA_INDEX}")

print("ASL Recorder started")
print("Controls:")
print("  s = save current sample")
print("  n = skip to next letter")
print("  ] = next landmark")
print("  [ = previous landmark")
print("  c = toggle all coordinate labels")
print("  q = quit")

letter_index = 0
current_count = 0
frame_timestamp_ms = 0
selected_idx = 8              # start with index fingertip
show_all_coords = False

with vision.HandLandmarker.create_from_options(options) as landmarker:
    while True:
        if letter_index >= len(LETTERS):
            print("Finished collecting all letters.")
            break

        current_letter = LETTERS[letter_index]

        success, frame = cap.read()
        if not success:
            print("Failed to read frame from camera.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Must be monotonically increasing in VIDEO mode
        frame_timestamp_ms += 33
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        detected_hand = None
        h, w, _ = frame.shape

        # dark info panel on left
        cv2.rectangle(frame, (0, 0), (380, h), (25, 25, 25), -1)

        if result.hand_landmarks:
            detected_hand = result.hand_landmarks[0]

            # skeleton
            for a, b in CONNECTIONS:
                x1 = int(detected_hand[a].x * w)
                y1 = int(detected_hand[a].y * h)
                x2 = int(detected_hand[b].x * w)
                y2 = int(detected_hand[b].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # points + labels
            for i, lm in enumerate(detected_hand):
                x = int(lm.x * w)
                y = int(lm.y * h)

                if i == selected_idx:
                    cv2.circle(frame, (x, y), 9, (0, 0, 255), -1)
                else:
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                cv2.putText(
                    frame,
                    str(i),
                    (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )

                if show_all_coords:
                    cv2.putText(
                        frame,
                        f"({lm.x:.2f},{lm.y:.2f})",
                        (x + 8, y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (0, 255, 255),
                        1
                    )

            # selected landmark info
            sel = detected_hand[selected_idx]
            px = int(sel.x * w)
            py = int(sel.y * h)

            info_lines = [
                f"Show letter: {current_letter}",
                f"Saved: {current_count}/{SAMPLES_PER_LETTER}",
                "",
                f"Selected point: {selected_idx}",
                f"Name: {LANDMARK_NAMES[selected_idx]}",
                f"x (norm): {sel.x:.4f}",
                f"y (norm): {sel.y:.4f}",
                f"z (norm): {sel.z:.4f}",
                f"x (pixel): {px}",
                f"y (pixel): {py}",
                "",
                "Controls:",
                "s = save sample",
                "n = next letter",
                "[ / ] = prev/next point",
                "c = toggle all coords",
                "q = quit",
            ]

            y0 = 35
            for line in info_lines:
                cv2.putText(
                    frame,
                    line,
                    (15, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
                y0 += 28

            # compact list of all coordinates
            y0 += 8
            for i, lm in enumerate(detected_hand):
                color = (0, 255, 255) if i == selected_idx else (180, 180, 180)
                text = f"{i:>2} {LANDMARK_NAMES[i]:<12} {lm.x:.2f} {lm.y:.2f} {lm.z:.2f}"
                cv2.putText(
                    frame,
                    text,
                    (15, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    color,
                    1
                )
                y0 += 18
                if y0 > h - 10:
                    break

        else:
            cv2.putText(
                frame,
                f"Show letter: {current_letter}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Saved: {current_count}/{SAMPLES_PER_LETTER}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "No hand detected",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                "s=save  n=next  [ ]=point  c=coords  q=quit",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        cv2.imshow("ASL Dataset Recorder", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            if detected_hand is not None:
                save_landmarks(CSV_PATH, current_letter, detected_hand)
                current_count += 1
                print(f"Saved {current_letter}: {current_count}/{SAMPLES_PER_LETTER}")

                if current_count >= SAMPLES_PER_LETTER:
                    print(f"Completed {current_letter}")
                    letter_index += 1
                    current_count = 0
            else:
                print("No hand detected. Sample not saved.")

        elif key == ord("n"):
            print(f"Skipping {current_letter}")
            letter_index += 1
            current_count = 0

        elif key == ord("]"):
            selected_idx = (selected_idx + 1) % 21

        elif key == ord("["):
            selected_idx = (selected_idx - 1) % 21

        elif key == ord("c"):
            show_all_coords = not show_all_coords

        elif key == ord("q"):
            print("Quitting...")
            break

cap.release()
cv2.destroyAllWindows()