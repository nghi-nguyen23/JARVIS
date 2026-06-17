#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import time
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pyautogui

from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# **Load model and normalize files**

# In[2]:


model = tf.keras.models.load_model("best_tcn_gesture_model.keras")

mean = np.load("landmark_mean.npy")
std = np.load("landmark_std.npy")

mean = np.squeeze(mean)
std = np.squeeze(std)

print("Model loaded")
print("Mean shape:", mean.shape)
print("Std shape:", std.shape)


# **Set up MediaPipe Tasks API**

# In[3]:


base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(options)

print("MediaPipe detector ready")


# **Constants and label mapping**

# In[4]:


SEQUENCE_LENGTH = 37
NUM_LANDMARKS = 21
FEATURES_PER_FRAME = NUM_LANDMARKS * 3  # 63

class_names = {
    0: "Doing other things",
    1: "Drumming Fingers",
    2: "No gesture",
    3: "Pulling Hand In",
    4: "Pulling Two Fingers In",
    5: "Pushing Hand Away",
    6: "Pushing Two Fingers Away",
    7: "Rolling Hand Backward",
    8: "Rolling Hand Forward",
    9: "Shaking Hand",
    10: "Sliding Two Fingers Down",
    11: "Sliding Two Fingers Left",
    12: "Sliding Two Fingers Right",
    13: "Sliding Two Fingers Up",
    14: "Stop Sign",
    15: "Swiping Down",
    16: "Swiping Left",
    17: "Swiping Right",
    18: "Swiping Up",
    19: "Thumb Down",
    20: "Thumb Up",
    21: "Turning Hand Clockwise",
    22: "Turning Hand Counterclockwise",
    23: "Zooming In With Full Hand",
    24: "Zooming In With Two Fingers",
    25: "Zooming Out With Full Hand",
    26: "Zooming Out With Two Fingers",
}


# **Action Map**

# In[11]:


ACTION_MAP = {
    16: "previous_tab",      # Swiping Left
    17: "next_tab",          # Swiping Right
    13: "scroll_up",         # Sliding Two Fingers Up
    10: "scroll_down",       # Sliding Two Fingers Down
    24: "fullscreen",        # Zooming In With two fingers
    26: "exit_fullscreen",   # Zooming Out With two fingers
    5: "open_new_tab",       # Pushing Hand Away
    19: "stop_listening",    # Thumb Down
    20: "start_listening",   # Thumb Up
}


# In[6]:


def extract_landmarks_from_image(rgb_image):
    """
    Takes one RGB webcam frame.
    Returns a flat landmark vector of shape (63,).
    If no hand is detected, returns zeros.
    """

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image
    )

    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:
        hand_landmarks = detection_result.hand_landmarks[0]

        landmarks = []
        for lm in hand_landmarks:
            landmarks.extend([lm.x, lm.y, lm.z])

        return np.array(landmarks, dtype=np.float32), True

    return np.zeros(FEATURES_PER_FRAME, dtype=np.float32), False


# In[7]:


def predict_gesture(landmark_sequence):
    """
    landmark_sequence shape: (37, 63)
    Returns class index and confidence.
    """

    x = landmark_sequence.astype("float32")

    # Apply same normalization as training
    x = (x - mean) / std

    # Add batch dimension: (1, 37, 63)
    x = np.expand_dims(x, axis=0)

    probs = model.predict(x, verbose=0)

    class_idx = int(np.argmax(probs, axis=1)[0])
    confidence = float(np.max(probs))

    return class_idx, confidence


# **PyAutoGUI action function**
# 
# has listening mode, confidence threshold, and cooldown so it does not spam actions.

# In[8]:


listening = True
last_action_time = 0

confidence_threshold = 0.75
cooldown_seconds = 2.0

def run_action(class_idx, confidence):
    global listening, last_action_time

    gesture_name = class_names.get(class_idx, f"Class {class_idx}")

    if confidence < confidence_threshold:
        print(f"Low confidence: {gesture_name} ({confidence:.2f}). No action.")
        return

    if class_idx not in ACTION_MAP:
        print(f"Unmapped gesture: {gesture_name} ({confidence:.2f}). No action.")
        return

    action = ACTION_MAP[class_idx]

    # Resume listening should work even if listening is off
    if action == "start_listening":
        listening = True
        print(f"Gesture: {gesture_name} | Action: START LISTENING")
        return

    # Stop listening
    if action == "stop_listening":
        listening = False
        print(f"Gesture: {gesture_name} | Action: STOP LISTENING")
        return

    # If paused, ignore everything except start_listening
    if not listening:
        print(f"Paused. Ignoring: {gesture_name}")
        return

    # Cooldown to prevent repeated spam
    current_time = time.time()
    if current_time - last_action_time < cooldown_seconds:
        print(f"Cooldown active. Ignoring: {gesture_name}")
        return

    print(f"Gesture: {gesture_name} | Confidence: {confidence:.2f} | Action: {action}")

    # Mac shortcuts
    if action == "open_new_tab":
        pyautogui.hotkey("command", "t")

    elif action == "next_tab":
        pyautogui.hotkey("ctrl", "shift", "tab")

    elif action == "previous_tab":
        pyautogui.hotkey("ctrl", "tab")

    elif action == "scroll_up":
        pyautogui.scroll(5)

    elif action == "scroll_down":
        pyautogui.scroll(-5)

    elif action == "fullscreen":
        pyautogui.hotkey("ctrl", "command", "f")

    elif action == "exit_fullscreen":
        pyautogui.press("esc")

    last_action_time = current_time


# In[13]:


camera_index = 1  

cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 15)

if not cap.isOpened():
    raise RuntimeError(f"Could not open camera {camera_index}. Try camera_index = 1.")

# Warm up camera
print("Warming up camera...")
for _ in range(20):
    ret, frame = cap.read()
    time.sleep(0.05)

landmark_buffer = deque(maxlen=SEQUENCE_LENGTH)

frame_count = 0
predict_every_n_frames = 10
max_seconds = 30
start_time = time.time()

# Reset listening state and cooldown
listening = True
last_action_time = 0

print("Starting REAL gesture control.")
print("No webcam window will open.")
print("Open/click Chrome first. Interrupt kernel to stop if needed.")

try:
    while True:
        success, frame = cap.read()

        if not success:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        landmarks, hand_detected = extract_landmarks_from_image(rgb_frame)
        landmark_buffer.append(landmarks)

        if len(landmark_buffer) == SEQUENCE_LENGTH and frame_count % predict_every_n_frames == 0:
            landmark_sequence = np.array(landmark_buffer)

            detected_frames = np.sum(~np.all(landmark_sequence == 0, axis=1))

            class_idx, confidence = predict_gesture(landmark_sequence)
            gesture_name = class_names.get(class_idx, f"Class {class_idx}")

            print(
                f"Prediction: {gesture_name} ({class_idx}) | "
                f"Confidence: {confidence:.2f} | "
                f"Hand frames: {detected_frames}/{SEQUENCE_LENGTH} | "
                f"Listening: {listening}"
            )

            run_action(class_idx, confidence)

        frame_count += 1

        ''' timer
        if time.time() - start_time > max_seconds:
            print("Auto-stopped.")
            break
        '''

except KeyboardInterrupt:
    #print("Stopped by user.")
    print("Stopped by Ctrl + C / kernel interrupt.")

finally:
    cap.release()
    print("Webcam released.")


# In[ ]:




