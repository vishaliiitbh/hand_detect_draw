import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
import os
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
st.title("Hand Detection Drawing App")
st.markdown("Use your index finger to draw. Make a fist to clear the canvas.")

mpHands = mp.solutions.hands
hands = None
mpDraw = mp.solutions.drawing_utils

pencilColor = (0, 255, 0)  # Green
brushThickness = 5

drawnPoints = []

cap = cv2.VideoCapture(0)

if hands is None:
    hands = mpHands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

canvas = st.empty()

while cap.isOpened():
    success, img = cap.read()
    if not success:
        st.warning("No webcam detected.")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            if lmList:
                x1, y1 = lmList[8][1:]  # Index Finger Tip

                # Check if the hand is showing a fist (all fingertips close to palm)
                fistDetected = True
                for tipId in [8, 12, 16, 20]:
                    if lmList[tipId][2] < lmList[tipId - 2][2]:
                        fistDetected = False

                if fistDetected:
                    drawnPoints = [] 
                else:
                    cv2.circle(img, (x1, y1), brushThickness, pencilColor, cv2.FILLED)
                    drawnPoints.append([x1, y1, pencilColor, brushThickness])

    for point in drawnPoints:
        cv2.circle(img, (point[0], point[1]), point[3], point[2], cv2.FILLED)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    canvas.image(imgRGB, channels="RGB")

cap.release()
