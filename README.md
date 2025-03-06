# Hand Detection Drawing App

This is a Streamlit-based web application that allows users to draw on the screen using hand gestures. The app uses a webcam to detect the user's hand, allowing them to draw using their index finger and clear the screen by making a fist.

## Features
- 🎨 Draw on the screen using your index finger.
- ✊ Make a fist to clear the entire canvas.
- 🖐️ Real-time hand tracking using MediaPipe.
- 🚀 Simple and interactive interface using Streamlit.

## Installation

### Prerequisites
- Python 3.7 or higher
- A working webcam

### Install Required Packages
```bash
pip install -r requirements.txt
```

#### Example `requirements.txt`
```
streamlit
opencv-python
mediapipe
Pillow
```

## Running the App
```bash
streamlit run code.py
```

## Usage
- Keep your hand visible to the webcam.
- Use your index finger to draw on the canvas.
- Make a fist to clear all drawings.
- Press 'q' to quit the app if running locally.

---
Made with ❤️ using Python, OpenCV, MediaPipe, and Streamlit.

