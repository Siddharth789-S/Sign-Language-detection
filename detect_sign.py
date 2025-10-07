import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from collections import deque

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore") 

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Mediapipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Prediction smoothing buffer
PREDICTION_HISTORY = 5
prediction_buffer = deque(maxlen=PREDICTION_HISTORY)

# Start webcam
cap = cv2.VideoCapture(0)

prev_time = time.time()
fps = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip for selfie view
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract features (x, y coordinates)
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y])

            # Predict
            features = np.array(coords, dtype=np.float32).reshape(1, -1)
            pred_probs = model.predict_proba(features)[0]
            pred_label = model.classes_[np.argmax(pred_probs)]
            pred_conf = np.max(pred_probs)

            # Store prediction for smoothing
            prediction_buffer.append(pred_label)

            # Get smoothed prediction
            most_common = max(set(prediction_buffer), key=prediction_buffer.count)

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Show prediction with confidence
            cv2.putText(frame, f"{most_common} ({pred_conf*100:.1f}%)",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Sign Language Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
