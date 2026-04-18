import cv2
import time
import os
import mediapipe as mp
import joblib
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from landmarks import normalize_landmarks
import requests
from groq import Groq


MODEL_PATH = "hand_landmarker.task"
RF_MODEL_PATH = "asl_model.pkl"
CAMERA_INDEX = 0

# Rate Photos are taken of the vectors
CADENCE_SECONDS = 4

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

def sentence_format(letters):
    client = Groq(api_key='Insert-api-key')
    query = f"Using the string {letters}, apply the necessary spaces to form the sentence. Only respond with the sentence."
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": query}],
            model="llama-3.3-70b-versatile",
        )
        sentence = completion.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
    return sentence

def draw_hand(frame, hand, width, height):
    for a, b in CONNECTIONS:
        x1, y1 = int(hand[a].x * width), int(hand[a].y * height)
        x2, y2 = int(hand[b].x * width), int(hand[b].y * height)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 80, 0), 2)

    for lm in hand:
        cx = int(lm.x * width)
        cy = int(lm.y * height)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)


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
        num_hands=1
    )

    predictions = []
    last_capture_time = time.time()
    frame_timestamp_ms = 0

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

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                draw_hand(frame, hand, w, h)

                vec = normalize_landmarks(hand)

                if vec is not None:
                    pred_letter = rf.predict([vec])[0]

                    cv2.putText(
                        frame,
                        f"Predicted: {pred_letter}",
                        (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )
                    

                    if now - last_capture_time >= CADENCE_SECONDS:
                        predictions.append(pred_letter)
                        print(f"Stored: {pred_letter}")
                    

                        last_capture_time = now
                    
                else:
                    cv2.putText(
                        frame,
                        "INVALID VECTOR",
                        (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

                countdown = max(0, int(CADENCE_SECONDS - (now - last_capture_time)))
                cv2.putText(
                    frame,
                    f"Next: {countdown}s",
                    (15, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1
                )

            else:
                cv2.putText(
                    frame,
                    "NO HAND",
                    (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            cv2.putText(
                frame,
                f"Saved: {' '.join(predictions) if predictions else '(none)'}",
                (15, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

            cv2.imshow("ASL Prediction", frame)

            if cv2.waitKey(1) & 0xFF == ord('d'):
                # Create a black image and put text on it
                info_window = np.zeros((200, 400, 3), dtype="uint8")
                pred = predictions.pop(0)
                
                letters = "".join(pred)
                sentence = sentence_format(letters)
                cv2.putText(info_window, f"{sentence}!", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                os.system(f'say "{sentence}" &')
                cv2.imshow("String Output", info_window)
                predictions = []

            if cv2.waitKey(1) & 0xFF == ord("q"):
                info_window = np.zeros((200, 400, 3), dtype="uint8")
                os.system('say "Thanks for using our platform, Goodbye!" &')
                print('DONE')
                break

            

    cap.release()
    cv2.destroyAllWindows()

    return predictions



if __name__ == "__main__":
    predictions = run_hand_tracker()
