import pose_module as pm
import mediapipe as mp
import cv2
import time
import numpy as np
import pandas as pd


mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()
detector = pm.poseDetector()


def test_1(Image):
    results = pose.process(Image)
    lmList = []
    if results.pose_landmarks:
        mpDraw.draw_landmarks(Image, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = Image.shape
            cx, cy = int(lm.x * w), int(lm.y * h)  # get pixel values instead of ratio of picture width
            # print(id, cx,cy)
            lmList.append([id, cx, cy])  # list of id, x and y coords of all 33 landmarks
            cv2.circle(Image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)  # draw blue dots
        cv2.circle(Image, (lmList[14][1], lmList[14][2]), 5, (0, 255, 0),
                   cv2.FILLED)  # draw green dot on landmark 14



def test_2(Image):
    mask = cv2.imread('mask.png', 0)

    Image = detector.findPose(Image)
    lmList = detector.findPosition(Image, draw=True)
    print(lmList)
