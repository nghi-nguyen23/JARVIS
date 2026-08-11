# 🖐️ JARVIS

JARVIS is a computer vision project inspired by Iron Man’s AI assistant. The project uses hand gesture recognition to classify real-time webcam gestures and map them to browser-control actions.

The system uses MediaPipe hand landmarks, a TensorFlow/Keras Temporal Convolutional Network model, and OpenCV webcam input to recognize gestures such as swiping, scrolling, zooming, and opening tabs.

## Project Overview

### Goal:

To build an AI-powered hand gesture recognition prototype that allows users to control basic browser actions through webcam-detected hand movements.

JARVIS was designed as a hands-free interaction tool that connects computer vision, machine learning, and automation. The project explores how AI can move beyond simple prediction tasks and be used in real-time technical workflows.

### Main Features:

- Detects hand landmarks from webcam input using MediaPipe
- Classifies hand gesture sequences with a trained deep learning model
- Maps recognized gestures to browser actions
- Supports actions such as:
  - Switching tabs
  - Scrolling up and down
  - Opening a new tab
  - Entering and exiting fullscreen mode
  - Pausing and resuming gesture listening

## Tech Stack:

- Python
- TensorFlow / Keras
- MediaPipe
- OpenCV
- NumPy
- Pandas
- PyAutoGUI
- Scikit-Learn

## Model

The gesture classifier uses a Temporal Convolutional Network model trained on MediaPipe hand landmark sequences. Each gesture sample is represented as a sequence of hand landmark coordinates, allowing the model to learn movement patterns over time instead of only analyzing a single frame.

## How It Works

1. Webcam video is captured using OpenCV.
2. MediaPipe detects hand landmarks from each frame.
3. Landmark sequences are normalized and passed into the trained model.
4. The model predicts the gesture class.
5. If the prediction confidence is high enough, the gesture is mapped to a browser-control action.

## Example Gesture Controls

| Gesture | Action |
|---|---|
| Sliding Two Fingers Right | Next browser tab |
| Sliding Two Fingers Left | Previous browser tab |
| Sliding Two Fingers Up | Scroll up |
| Sliding Two Fingers Down | Scroll down |
| Zooming In With Full Hand | Enter fullscreen |
| Zooming Out With Full Hand | Exit fullscreen |
| Thumb Up | Open new tab |
| Stop Sign | Pause listening |
| Shaking Hand | Resume listening |

## Project Status

This project is a working prototype. The model was trained and tested on processed hand landmark gesture data, and real-time webcam inference was implemented for browser-control actions.

## Authors

Created as part of an AI project focused on real-time hand gesture recognition and human-computer interaction.
