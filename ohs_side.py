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

filename = filenames[8]
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
    angle_13 = detector.findAngle(img, 12, 24, 26)  # l elbow angle
    #df_angle_list.append(angle_13)
    df_angle_list.append("angle_13, " + str(frame) + ", "+ str(angle_13))
    #angle_25 = detector.findAngle(img, 23, 25, 27)  # l knee angle (outer)
    #angle_26 = detector.findAngle(img, 24, 26, 28)  # r knee (inner)


    #parallel torso checker
    hip_r = lmList[24]
    shoulder_r = lmList[12]
    ankle_r = lmList[28]
    knee_r = lmList[26]

    torso_delta_y = hip_r[2] - shoulder_r[2]
    torso_delta_x = shoulder_r[1] - hip_r[1]
    if torso_delta_y and torso_delta_x:
        torso_lean = round(torso_delta_y / torso_delta_x, 2)

    tibia_delta_y = ankle_r[2] - knee_r[2]
    tibia_delta_x = knee_r[1] - ankle_r[1]
    if tibia_delta_y and tibia_delta_x:
        tibia_angle = round(tibia_delta_y / tibia_delta_x, 2)

    # print(shoulder_r)
    # print(hip_r)
    #
    # print(torso_delta_y)
    # print(torso_delta_x)
    #print("torso: "  + str(torso_lean))
    #print("tibia: " + str(tibia_angle))
    #torso_lean_deg = math.atan(torso_lean)
    #print("torsot lean deg: "  + str(torso_lean_deg))
    if torso_lean > tibia_angle and upright is False:
        print('squat stance: upright')
        upright = True
    elif torso_lean > tibia_angle and upright is True:
        pass
    elif torso_lean < tibia_angle and angle_13 > 120:
        #print("squat stance: forward lean but hip extended past 120deg")
        pass
    elif torso_lean < tibia_angle and angle_13 < 120:
        print('squat stance: too much forward lean')
        upright = False
    else:
        pass


    #horizontal femur
    if hip_r[2] > knee_r[2] and below_parallel is False:
        below_parallel = True
        depth_achieved = True
        print("femur below parallel")
    elif hip_r[2] < knee_r[2] and below_parallel is True:
        below_parallel = False
        print('femur no longer parallel')
    else:
        pass

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
#angle_df.columns = columns
angle_df.to_csv('outputs/' + str(file_itself) + " - angle_13.csv", index=False)

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