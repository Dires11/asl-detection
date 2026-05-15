import cv2
import time
import os
import threading
import mediapipe as mp
import joblib
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from landmarks import normalize_landmarks
from groq import Groq
from dotenv import load_dotenv


load_dotenv()
MODEL_PATH = "hand_landmarker.task"
RF_MODEL_PATH = "asl_model.pkl"
CAMERA_INDEX = 0
API_KEY = os.getenv('GROQ_API_KEY')

# Minimum seconds between storing consecutive predictions to ensure deliberate signing and give user time to adjust hand pose
CADENCE_SECONDS = 3

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# BGR colors
WHITE  = (255, 255, 255)
GREEN  = (80, 220, 80)
RED    = (80, 80, 220)
ORANGE = (0, 165, 255)
YELLOW = (0, 220, 220)
CYAN   = (220, 200, 0)
GRAY   = (130, 130, 130)
DARK   = (15, 15, 15)


def overlay_rect(frame, x1, y1, x2, y2, color=DARK, alpha=0.65):
    """Draw a semi-transparent filled rectangle."""
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    sub = frame[y1:y2, x1:x2]
    filled = np.full_like(sub, color, dtype=np.uint8)
    cv2.addWeighted(filled, alpha, sub, 1 - alpha, 0, sub)
    frame[y1:y2, x1:x2] = sub


def sentence_format(letters):
    system_prompt = (
        "You are an English interpretation assistant that helps non-native English speakers "
        "by correcting their spelling. When a user inputs a word or sentence with spelling "
        "mistakes, your job is to interpret it into the most commonly used and natural English "
        "word or phrase.\n"
        "Rules you must follow:\n"
        "First, determine whether the input letters together infer a real life expression, "
        "phrase, or sentence that a native English speaker would commonly use.\n"
        "If the input does represent a real life expression, phrase, or sentence, you may freely "
        "adjust the number of letters, alter letters, add or remove letters as needed to produce "
        "the most natural and correct version of that expression.\n"
        "If the input does not represent a real life expression, phrase, or sentence, the total "
        "number of letters in your interpretation must remain exactly the same as the input. "
        "You may still alter, rearrange, reassign, or regroup individual letters — but cannot "
        "add or remove any.\n"
        "Spaces do not count as letters in either case.\n"
        "Always choose the interpretation that produces the most popular, most natural, and most "
        "commonly spoken English phrase or word.\n"
        "If multiple interpretations are possible, always select the one that a native English "
        "speaker would most likely say in everyday life.\n"
        "Prioritize grammar, spelling, and natural flow in all interpretations.\n"
        "Return only the interpretation. Do not explain, justify, or add any extra text."
    )
    client = Groq(api_key=API_KEY)
    try:
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": letters},
            ],
            model="openai/gpt-oss-120b",
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"API error: {e}")
        return letters


def draw_hand(frame, hand, width, height):
    for a, b in CONNECTIONS:
        x1, y1 = int(hand[a].x * width), int(hand[a].y * height)
        x2, y2 = int(hand[b].x * width), int(hand[b].y * height)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 150, 30), 2)
    for lm in hand:
        cx, cy = int(lm.x * width), int(lm.y * height)
        cv2.circle(frame, (cx, cy), 6, (50, 255, 130), -1)
        cv2.circle(frame, (cx, cy), 8, WHITE, 1)


def draw_ui(frame, pred_letter, hand_found, predictions, countdown, sentence, translating):
    h, w = frame.shape[:2]

    # ── Top bar ──────────────────────────────────────────────────────────────
    overlay_rect(frame, 0, 0, w, 60)
    cv2.putText(frame, "ASL LIVE DETECTOR",
                (w // 2 - 145, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, CYAN, 2)

    # Hand status dot + label (top-right)
    dot_color = GREEN if hand_found else RED
    status_label = "HAND DETECTED" if hand_found else "NO HAND"
    cv2.circle(frame, (w - 195, 30), 7, dot_color, -1)
    cv2.putText(frame, status_label,
                (w - 180, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, dot_color, 1)

    # ── Current letter box (left) ─────────────────────────────────────────────
    if hand_found and pred_letter:
        overlay_rect(frame, 10, 70, 120, 195, color=(10, 30, 10))
        cv2.rectangle(frame, (10, 70), (120, 195), GREEN, 1)
        cv2.putText(frame, "SIGN",
                    (33, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRAY, 1)
        cv2.putText(frame, pred_letter,
                    (22, 180), cv2.FONT_HERSHEY_DUPLEX, 3.8, GREEN, 4)
    elif hand_found:
        overlay_rect(frame, 10, 70, 200, 115, color=(30, 10, 10))
        cv2.rectangle(frame, (10, 70), (200, 115), RED, 1)
        cv2.putText(frame, "INVALID VECTOR",
                    (18, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED, 1)

    # ── Countdown box (right) ─────────────────────────────────────────────────
    if hand_found:
        overlay_rect(frame, w - 120, 70, w - 10, 195, color=(30, 20, 5))
        cv2.rectangle(frame, (w - 120, 70), (w - 10, 195), ORANGE, 1)
        cv2.putText(frame, "NEXT",
                    (w - 100, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRAY, 1)
        cv2.putText(frame, f"{countdown}s",
                    (w - 108, 178), cv2.FONT_HERSHEY_DUPLEX, 2.8, ORANGE, 3)

    # ── Bottom bar ────────────────────────────────────────────────────────────
    overlay_rect(frame, 0, h - 75, w, h)
    cv2.putText(frame, "CAPTURED:",
                (15, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRAY, 1)
    letters_str = "  ".join(predictions) if predictions else "(none)"
    # Truncate if too many letters to fit
    max_chars = (w - 260) // 18
    display_letters = letters_str[:max_chars] + ("…" if len(letters_str) > max_chars else "")
    cv2.putText(frame, display_letters,
                (15, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.75, YELLOW, 2)
    cv2.putText(frame, "[D] Translate   [Q] Quit",
                (w - 245, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRAY, 1)

    # ── Sentence / translating overlay (centre of frame) ─────────────────────
    if translating or sentence:
        cy = h // 2
        box_h = 90
        overlay_rect(frame, 30, cy - box_h // 2, w - 30, cy + box_h // 2,
                     color=(8, 25, 8), alpha=0.82)
        cv2.rectangle(frame, (30, cy - box_h // 2), (w - 30, cy + box_h // 2), GREEN, 2)

        if translating:
            cv2.putText(frame, "Translating...",
                        (w // 2 - 95, cy + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2)
        else:
            cv2.putText(frame, "TRANSLATION",
                        (48, cy - box_h // 2 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN, 1)
            # Truncate long sentences
            display = f'"{sentence}"' if len(sentence) <= 38 else f'"{sentence[:35]}..."'
            cv2.putText(frame, display,
                        (48, cy + 18), cv2.FONT_HERSHEY_DUPLEX, 0.85, WHITE, 2)


def run_hand_tracker():
    if not os.path.exists(MODEL_PATH):
        print(f"Missing MediaPipe model file: {MODEL_PATH}")
        return []
    if not os.path.exists(RF_MODEL_PATH):
        print(f"Missing trained classifier file: {RF_MODEL_PATH}")
        return []

    rf = joblib.load(RF_MODEL_PATH)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: could not open camera.")
        return []

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )

    predictions = []
    last_capture_time = time.time()
    frame_timestamp_ms = 0
    last_pred_letter = None

    # Shared translation state (accessed from main + worker thread)
    trans = {"sentence": "", "translating": False}

    def fetch_translation(letters):
        trans["translating"] = True
        trans["sentence"] = ""
        result = sentence_format(letters)
        trans["sentence"] = result
        trans["translating"] = False
        os.system(f'say "{result}" &')

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                print("Error: failed to read frame.")
                break

            now = time.time()
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            frame_timestamp_ms += 33
            result = landmarker.detect_for_video(mp_img, frame_timestamp_ms)

            pred_letter = None
            hand_found = bool(result.hand_landmarks)

            if not hand_found and last_pred_letter is not None:
                last_capture_time = now
                last_pred_letter = None

            if hand_found:
                hand = result.hand_landmarks[0]
                draw_hand(frame, hand, w, h)

                vec = normalize_landmarks(hand)
                if vec is not None:
                    pred_letter = rf.predict([vec])[0]

                    if pred_letter != last_pred_letter:
                        last_capture_time = now
                        last_pred_letter = pred_letter

                    if now - last_capture_time >= CADENCE_SECONDS:
                        predictions.append(pred_letter)
                        print(f"Stored: {pred_letter}")
                        last_capture_time = now

            countdown = max(0, int(CADENCE_SECONDS - (now - last_capture_time)))

            draw_ui(
                frame,
                pred_letter=pred_letter,
                hand_found=hand_found,
                predictions=predictions,
                countdown=countdown,
                sentence=trans["sentence"],
                translating=trans["translating"],
            )

            cv2.imshow("ASL Detector", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('d') and predictions and not trans["translating"]:
                letters = "".join(predictions)
                predictions = []
                trans["sentence"] = ""
                t = threading.Thread(target=fetch_translation, args=(letters,), daemon=True)
                t.start()

            if key == ord('q'):
                os.system('say "Thanks for using our platform, Goodbye!" &')
                print("DONE")
                break

    cap.release()
    cv2.destroyAllWindows()
    return predictions


if __name__ == "__main__":
    run_hand_tracker()
