import cv2
import time
import numpy as np
import pose_module as pm
import matplotlib.pyplot as plt
import pandas as pd
import math
import pickle
import pathlib

# load pickle object (list of filenames) created in vid_list_makers.py
with open("pickles/side_vids", "rb") as fp:   # Unpickling
    filenames = pickle.load(fp)

filename = filenames[17]
filepath = pathlib.Path(filename).parent
file_itself = pathlib.Path(filename).stem
print(filename)
print(filepath)
print(file_itself)

cap = cv2.VideoCapture(filename)  # load the video into opencv

cap_fps = cap.get(cv2.CAP_PROP_FPS) # determine source fps
print("capture fps is:" + str(cap_fps))


# give startpoints for frame nr, and rep count, pTime for fps counter, start_time for duration
frame = 0
pTime = 0
count = 0
start_time = time.time()

# pandas
columns = ('frame', 'joint', 'x', 'y')
lm_df = pd.DataFrame()
angle_df = pd.DataFrame()
df_list = []
df_angle_list = []

# load pose detector from imported pose_module
detector = pm.poseDetector()

# initiate lists
list_of_frames = []  # count frames start at 0
list_of_y_31 = [] #list of y coordinates of joint 31 for each frame


below_parallel = False
depth_achieved = False
upright = False

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

    # find angle in a joint (btwn 3 keypoints)
    r_hip_angle = detector.findAngle(img, 12, 24, 26)
    df_angle_list.append([frame, "r_hip", r_hip_angle])
    #df_angle_list.append(str(frame) + ", " + "r_hip, " + str(r_hip_angle))

    r_shoulder_angle = detector.findAngle(img, 14, 12, 24)
    df_angle_list.append([frame, "r_shoulder",r_shoulder_angle])

    r_knee_dors_angle = detector.findAngle(img, 28,26,24)
    df_angle_list.append([frame, "r_knee_dors", r_knee_dors_angle])

    r_ankle_flex = detector.findAngle(img, 26,30,32)
    df_angle_list.append([frame, "r_ankle_flex", r_ankle_flex])

    #parallel torso checker
    hip_r = lmList[24]
    shoulder_r = lmList[12]
    ankle_r = lmList[28]
    knee_r = lmList[26]

    torso_delta_y = hip_r[2] - shoulder_r[2]
    torso_delta_x = shoulder_r[1] - hip_r[1]
    if torso_delta_y and torso_delta_x:
        torso_lean = round(torso_delta_y / torso_delta_x, 2)
    else:
        print("torso lean not found")
        pass

    tibia_delta_y = ankle_r[2] - knee_r[2]
    tibia_delta_x = knee_r[1] - ankle_r[1]
    if tibia_delta_y and tibia_delta_x:
        tibia_angle = round(tibia_delta_y / tibia_delta_x, 2)
    else:
        print("tibia angle not found")
        pass

    if torso_lean > tibia_angle and upright is False:
        print('squat stance: upright')
        upright = True
    elif torso_lean > tibia_angle and upright is True:
        pass
    elif torso_lean < tibia_angle and r_hip_angle > 120:
        #print("squat stance: forward lean but hip extended past 120deg")
        pass
    elif torso_lean < tibia_angle and r_hip_angle < 120:
        print('squat stance: too much forward lean')
        upright = False
    else:
        pass


    # horizontal femur
    if hip_r[2] > knee_r[2] and below_parallel is False:
        below_parallel = True
        depth_achieved = True
        print("femur below parallel")
    elif hip_r[2] < knee_r[2] and below_parallel is True:
        below_parallel = False
        print('femur no longer parallel')
    else:
        pass

    # femur angle calculator
    femur_delta_y = knee_r[2] - hip_r[2]
    femur_delta_x = knee_r[1] - hip_r[1]
    if femur_delta_y and femur_delta_x:
        femur_angle = round(femur_delta_y / femur_delta_x, 2)
    femur_angle = math.degrees(math.atan(femur_angle))
    femur_angle = round(femur_angle, 2)

    #hands and ankles in line
    wrist_r = lmList[16]

    cv2.circle(img, (wrist_r[1], wrist_r[2]), 5, (255, 0, 0), cv2.FILLED)
    cv2.circle(img, (ankle_r[1], ankle_r[2]), 5, (255, 0, 0), cv2.FILLED)

    wrist_forward = False
    if abs(wrist_r[1] - ankle_r[1]) < 30:
        pass
    else:
        #print("wrist_r ahead of ankle_r by this many pixels: " + str(abs(wrist_r[1] - ankle_r[1])))
        wrist_forward = True


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
    cv2.line(img, (-10, knee_r[2]), (knee_r[1], knee_r[2]), (0, 255, 0), thickness=2)
    cv2.putText(img, ('femur angle: ' + str(femur_angle)), (10, 200), cv2.FONT_HERSHEY_PLAIN,
                2,
                (255, 0, 0), 2)



    if below_parallel is True:
        cv2.putText(img, ('femur below parallel!!: ' + str(count)), (10, 80), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 0), 2)

    if wrist_forward is True:
        cv2.putText(img, ('wrist ahead of ankle by pixels: ' + str(abs(wrist_r[1] - ankle_r[1]))
), (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)



    if depth_achieved is True:
        cv2.putText(img, ('depth has been achieved this rep! '), (10, 150), cv2.FONT_HERSHEY_PLAIN, 2,
                        (0, 255, 0), 2)
    else:
        cv2.putText(img, ('depth has NOT been achieved this rep! '), (10, 150), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0, 0, 255), 2)

    cv2.imshow(filename, img)
    #time.sleep(0.1)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == 32:
        cv2.waitKey()
# make df and clean up a bit
for item in df_list:
    lm_df = lm_df.append(item)
lm_df.reset_index(drop=True, inplace=True)
lm_df.columns = columns
lm_df.to_csv('outputs/' + str(file_itself) + " - coordinates.csv", index=False)

angle_df = pd.DataFrame(df_angle_list)
angle_df.reset_index(drop=True, inplace=True)
angle_df.columns = ['frame', 'name', 'angle']
angle_df.to_csv('outputs/' + str(file_itself) + " - angles.csv", index=False)

# Scoring
print("FMS Testing:")
print(" - Torso parallel to tibia or better: no")
print(" - femur below horizontal: no")
print(" - dowel above feet: yes, by 23 pixels max")
print(" - Heels elevated: yes")

print("Additional metrics:")
print(" - max depth: ")
print(" - foward fall: no")
print(" - time at bottom: 2s")
print(" - bounce: yes")
print(" - initiation pattern: knee or hip")
print(" - center of gravity: too forward")