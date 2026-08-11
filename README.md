# 🖐️ JARVIS

JARVIS is a computer vision project inspired by Iron Man’s AI assistant. The current version is a working real-time demo that uses hand gesture recognition to classify webcam gestures and map selected gestures to browser-control actions.

The system uses MediaPipe hand landmarks, a TensorFlow/Keras Temporal Convolutional Network model, OpenCV webcam input, and PyAutoGUI automation to recognize gestures such as swiping, scrolling, zooming, and opening tabs.

This project is currently in the prototype/demo stage. Future updates may include improving model reliability, refining the gesture-action mapping, and building a more polished interface such as a web app or desktop-style demo.

## Repo Structure

```text
jarvis-demo/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── jarvis_prototype.py
├── notebooks/
│   ├── tcn_training.ipynb
│   └── webcam_inference_testing.ipynb
├── docs/
│   └── project_summary.md
└── models/
    └── README.md
```

## Project Overview

### Goal

To build an AI-powered hand gesture recognition prototype that allows users to control basic browser actions through webcam-detected hand movements.

JARVIS was designed as a hands-free interaction tool that connects computer vision, machine learning, and automation. The project explores how AI can move beyond simple prediction tasks and be used in real-time human-computer interaction workflows.

### Main Features

* Detects hand landmarks from webcam input using MediaPipe
* Classifies hand gesture sequences with a trained deep learning model
* Maps recognized gestures to browser-control actions using PyAutoGUI
* Supports actions such as:

  * Switching browser tabs
  * Scrolling up and down
  * Opening a new tab
  * Entering and exiting fullscreen mode
  * Pausing and resuming gesture listening

## Tech Stack

* Python
* TensorFlow / Keras
* MediaPipe
* OpenCV
* NumPy
* Pandas
* PyAutoGUI
* Scikit-Learn

## Model

The gesture classifier uses a Temporal Convolutional Network trained on MediaPipe hand landmark sequences. Each gesture sample is represented as a sequence of hand landmark coordinates, allowing the model to learn movement patterns over time instead of only analyzing a single frame.

Each processed sample is represented as a fixed-size landmark sequence with 37 frames and 63 features per frame.

## How It Works

1. Webcam video is captured using OpenCV.
2. MediaPipe detects hand landmarks from each frame.
3. Landmark sequences are normalized and passed into the trained TCN model.
4. The model predicts the gesture class.
5. If the prediction confidence is high enough, the gesture is mapped to a browser-control action.
6. Unmapped gestures are ignored to reduce accidental actions.

## Example Gesture Controls

| Gesture                    | Action               |
| -------------------------- | -------------------- |
| Sliding Two Fingers Right  | Next browser tab     |
| Sliding Two Fingers Left   | Previous browser tab |
| Sliding Two Fingers Up     | Scroll up            |
| Sliding Two Fingers Down   | Scroll down          |
| Zooming In With Full Hand  | Enter fullscreen     |
| Zooming Out With Full Hand | Exit fullscreen      |
| Thumb Up                   | Open new tab         |
| Stop Sign                  | Pause listening      |
| Shaking Hand               | Resume listening     |

## Project Status

JARVIS is currently a working prototype/demo. The model was trained and tested on processed hand landmark gesture data, and real-time webcam inference was implemented for browser-control actions.

The prototype works, but gesture predictions are not always perfect. Some gestures are more reliable than others, so the project uses confidence thresholds, cooldown logic, and gesture filtering to reduce accidental PyAutoGUI actions.

## Future Improvements

Potential future improvements include:

* Testing which gestures are most consistently recognized in real-time
* Improving preprocessing with better landmark normalization and missing-frame handling
* Comparing additional temporal models such as LSTM or GRU
* Tuning model architecture and hyperparameters
* Improving the real-time inference pipeline for lower latency
* Building a more polished user interface, such as a web app or desktop demo
* Adding clearer setup instructions and downloadable model files

## Authors

JARVIS demo was created as part of ACM AI Team 2.
