import pandas as pd
import cv2
import mediapipe as mp
import csv
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore") 

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)


# Mediapipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

DATA_FILE = "gestures.csv"

# Step 1: Data collection
def collect_data():
    gesture_name = input("Enter gesture name (e.g., Hello): ")

    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    coords = []
                    for lm in hand_landmarks.landmark:
                        coords.append(lm.x)
                        coords.append(lm.y)

                    # Save to CSV
                    with open(DATA_FILE, mode='a', newline='') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow([gesture_name] + coords)

            cv2.putText(frame, f"Collecting: {gesture_name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Collect Data", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# Step 2: Train model
def train_model():
    # Load using pandas for safety
    df = pd.read_csv(DATA_FILE)
    
    # Drop rows with missing values
    df = df.dropna()
    
    # Convert to NumPy arrays
    labels = df.iloc[:, 0].values
    features = df.iloc[:, 1:].values.astype(np.float32)
    
    # Train model
    model = RandomForestClassifier()
    model.fit(features, labels)
    
    # Save model
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    print("✅ Model trained and saved as model.pkl")


if __name__ == "__main__":
    choice = input("Collect data (c) or Train model (t)? ")
    if choice.lower() == 'c':
        collect_data()
    elif choice.lower() == 't':
        train_model()
