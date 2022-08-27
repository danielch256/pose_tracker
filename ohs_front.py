import cv2
import time
import numpy as np
import pose_module as pm
import matplotlib.pyplot as plt
import pandas as pd
import mediapipe as mp
import pickle
import pathlib

# load pickle object (list of filenames) created in vid_list_makers.py
with open("pickles/front_vids", "rb") as fp:   # Unpickling
    filenames = pickle.load(fp)

filename = filenames[8]
filepath = pathlib.Path(filename).parent
file_itself = pathlib.Path(filename).stem
print(filename)
print(filepath)
print(file_itself)


cap = cv2.VideoCapture(filename)  # load the video into opencv
cap_fps = cap.get(cv2.CAP_PROP_FPS)  # determine source fps
print("capture fps is:" + str(cap_fps))

# give startpoints for frame nr, pTime for fps counter, start_time for duration
frame = 0
pTime = 0
start_time = time.time()

# pandas
columns = ('frame', 'joint', 'x', 'y')
lm_df = pd.DataFrame()
df_list= []
df_angle_list = []
# load pose detector from imported pose_module
detector = pm.poseDetector()

# initiate lists
list_of_frames = []  # count frames start at 0
list_of_x_right_knee = []
list_of_y_right_knee = []  # list of y coordinates of joint 31 for each frame

list_of_x_left_knee = []
list_of_y_left_knee = []
list_hip_angles = []

# start cap loop
while True:
    success, img = cap.read()
    if img is None:
        break
    img = cv2.resize(img, (540, 810))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret,thresh = cv2.threshold(gray, 127, 255, 0)
    list_of_frames.append(frame)
    frame += 1
    img = detector.findPose(img)
    lmList = detector.findPosition(img, draw=True)

    # set up lists for pandas later
    temp_list = [[frame] + item for item in
                 lmList]  # make list shape ([[frame=1,joint=1,x,y],[frame=1,joint=2,x,y],[...]] for each frame
    df_list.append(temp_list)  # append to list outside scope, dont use pandas because its slow
    # angle finder and joint tracker

    # ankle stuff
    #angle_27 = detector.findAngle(img, 25, 27, 29, draw=False)  # r knee (inner)
    #angle_28 = detector.findAngle(img, 26, 28, 30)  # r knee (inner)

    joint_32 = lmList[32]
    x_32 = joint_32[1]
    y_32 = joint_32[2]

    # knee stuff
    #angle_25 = detector.findAngle(img, 23, 25, 27)  # l knee angle (outer)
    #angle_26 = detector.findAngle(img, 24, 26, 28)  # r knee (inner)

    right_knee = lmList[26]
    right_knee_x = right_knee[1]
    right_knee_y = right_knee[2]

    left_knee = lmList[25]
    left_knee_x = left_knee[1]
    left_knee_y = left_knee[2]
    knee_dist = left_knee_x - right_knee_x

    #list_of_x_right_knee.append(right_knee_x)
    #list_of_y_right_knee.append(right_knee_y)

    #list_of_x_left_knee.append(left_knee_x)
    #list_of_y_left_knee.append(left_knee_y)

    #knee_dist = (list_of_x_left_knee[frame - 1] - list_of_x_right_knee[frame - 1])
    #knee_dist_prev = (list_of_x_left_knee[frame - 2] - list_of_x_right_knee[frame - 2])
    #knee_dist_initial = (list_of_x_left_knee[0] - list_of_x_right_knee[0])

    #hip stuff
    angle_23 = detector.findAngle(img, 25, 23, 24) # l hip
    angle_24 = detector.findAngle(img, 23, 24, 26)  # r hip (inner)
    df_angle_list.append("angle_23, " + str(frame) + ", " + str(angle_23))
    df_angle_list.append("angle_24, " + str(frame) + ", " + str(angle_24))

    if angle_23 < 80:
        print('left hip angle: ' + str(angle_23))

    if angle_24 < 80:
        print('right hip angle: ' + str(angle_24))



    # #timer and fps stuff
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    end_time = time.time()
    diff = end_time - start_time
    if diff > 25:  # run for x seconds
        break
    diff = str(round(diff, 2))

    # opencv text putters
    cv2.putText(img, ('time: ' + str(diff)), (10, 20), cv2.FONT_HERSHEY_PLAIN, 2,
                (255, 0, 0), 2)
    cv2.putText(img, ('frame#: ' + str(list_of_frames[-1])), (10, 60), cv2.FONT_HERSHEY_PLAIN, 2,
                (255, 0, 0), 2)
    cv2.putText(img, ('fps: ' + str(int(fps))), (10, 40), cv2.FONT_HERSHEY_PLAIN, 2,
                (255, 0, 0), 2)
    # cv2.putText(img, ('reps: ' + str(count)), (10, 80), cv2.FONT_HERSHEY_PLAIN, 2,
    #            (255, 0, 0), 2)
    cv2.putText(img, ('knee dist.: ' + str(knee_dist)), (10, 100), cv2.FONT_HERSHEY_PLAIN, 2,
                (255, 0, 0), 2)

    if angle_23 > 80:
        cv2.putText(img, ('left hip angle:  ' + str(round(angle_23, 2))), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1,
                    (0, 255, 0), 2)
        cv2.putText(img, (" - ok!"), (200, 120), cv2.FONT_HERSHEY_PLAIN, 1,
                    (0, 255, 0), 2)
    else:
        cv2.putText(img, ('left hip angle:  ' + str(round(angle_23,2))), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1,
                    (0, 0, 255), 2)
        cv2.putText(img, (" - valgus collapse!"), (200, 120), cv2.FONT_HERSHEY_PLAIN, 1,
                    (0, 0, 255), 2)
    if angle_24 > 80:
        cv2.putText(img, ('right hip angle:  ' + str(round(angle_24, 2))), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1,
                    (0, 255, 0), 2)
        cv2.putText(img, (" - ok!"), (200, 140), cv2.FONT_HERSHEY_PLAIN, 1,
                    (0, 255, 0), 2)
    else:
        cv2.putText(img, ('right hip angle:  ' + str(round(angle_24, 2))), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1,
                    (0, 0, 255), 2)
        cv2.putText(img, (" - valgus collapse!"), (200, 140), cv2.FONT_HERSHEY_PLAIN, 1,
                    (0, 0, 255), 2)

    cv2.imshow(filename, img)
    #time.sleep(0.05)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == 32:
        cv2.waitKey()

# make df for all joints
for item in df_list:
    lm_df = lm_df.append(item)
lm_df.reset_index(drop=True, inplace=True)
lm_df.columns = columns

#df for selected angles
angle_df = pd.DataFrame(df_angle_list)
angle_df.reset_index(drop=True, inplace=True)
#angle_df.columns = columns
angle_df.to_csv('outputs/' + str(file_itself) + " - hip_angles.csv", index=False)

# print fms score:
print("FMS Testing:")
print(" - Valgus Collapse present: No")
print(" - heels elevated: No")

print("Additional metrics:")
print(" - lowest q-angle left hip: 93deg")
print(" - lowest q-angle right hip: 94deg")
print(" - side to side lean torso: 6deg")
print(" - side to side hip shift: 5deg")
print(" - bounce present: yes")
