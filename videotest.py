import cv2
import time
import csv
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from landmarks import normalize_landmarks 
import urllib.request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import model
import joblib 
from preprocess import DataHandling, apply_data_handling




MODEL_PATH      = "hand_landmarker.task"
CSV_PATH        = "asl_landmarks_automated.csv"
CADENCE_SECONDS = 2
CAMERA_INDEX    = 0
LABEL_NAME      = "automated" # Change this to the letter you are practicing

CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),
    (5,9), (9,10), (10,11), (11,12), (9,13), (13,14), (14,15), (15,16),
    (13,17), (17,18), (18,19), (19,20), (0,17)
]

def create_csv(csv_path: str):
    header = ["label"]
    # Generates x_0, y_0, z_0... to match your training style
    for i in range(21):
        header += [f"x_{i}", f"y_{i}", f"z_{i}"]
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(header)

def run_hand_tracker():
    create_csv(CSV_PATH)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1
    )

    last_capture_time = time.time()
    frame_timestamp_ms = 0

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok: break

            # Mirror the frame just like the working recognizer
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            frame_timestamp_ms += 33
            result = landmarker.detect_for_video(mp_img, frame_timestamp_ms)

            # Visual Panel
            cv2.rectangle(frame, (0, 0), (280, h), (25, 25, 25), -1)

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]

                # Draw Visual Skeleton (Matching your working code)
                for a, b in CONNECTIONS:
                    x1, y1 = int(hand[a].x * w), int(hand[a].y * h)
                    x2, y2 = int(hand[b].x * w), int(hand[b].y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 80, 0), 2)
                for lm in hand:
                    cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 5, (0, 255, 0), -1)

                # --- THE FIX: USE THE WORKING NORMALIZATION ---
                vec = normalize_landmarks(hand)

                now = time.time()
                if now - last_capture_time >= CADENCE_SECONDS:
                    if vec is not None:
                        # Save the normalized vector, NOT raw landmarks
                        with open(CSV_PATH, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([LABEL_NAME] + list(vec))
                        
                        os.system('say "Logged" &')
                        print(f"Logged Normalized '{LABEL_NAME}'")
                        last_capture_time = now

                cv2.putText(frame, "READY", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                countdown = max(0, int(CADENCE_SECONDS - (now - last_capture_time)))
                cv2.putText(frame, f"Next: {countdown}s", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            else:
                cv2.putText(frame, "NO HAND", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow('Automated Normalized Capture', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


def run_model(csv, rf):
    df = pd.read_csv(csv)
    X = df.drop(columns = ['label'])
    print(df.columns)
    print('hi')
    return model.test_model(rf, X.values)

def clear_file(csv):
    with open(csv, "w") as f:
            pass
     
def apply_data(file, model, name):
    user_data = apply_data_handling(file, name)
    rf = joblib.load(model)
    ans = run_model(f"revised_asl_{name}.csv", rf)
    print(ans)


if __name__ == "__main__":
    if os.stat("asl_landmarks_automated.csv").st_size == 0:
        run_hand_tracker()
        #df = apply_data("asl_landmarks_automated.csv","rf_model.pkl", "user_auto")
        rf = joblib.load("rf_model.pkl")
        ans = run_model("revised_asl_user_auto.csv", rf)
        print(ans)
        clear_file("asl_landmarks_automated.csv")
    else:
        #df = apply_data("asl_landmarks_automated.csv","rf_model.pkl", "user_auto")
        rf = joblib.load("rf_model.pkl")
        ans = run_model("revised_asl_user_auto.csv", rf)
        print(ans)
        clear_file("asl_landmarks_automated.csv")

