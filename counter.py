import cv2
import time
import numpy as np
import pose_module as pm
import matplotlib.pyplot as plt
import pandas as pd

filename = 'vids/kid.mp4'
cap = cv2.VideoCapture(filename)  # load the video into opencv
mask = cv2.imread('mask.png', 0)  # load the mask

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
df_list = []

# load pose detector from imported pose_module
detector = pm.poseDetector()

# initiate lists
list_of_frames = []  # count frames start at 0
list_of_y_31 = [] #list of y coordinates of joint 31 for each frame
list_of_y_32 = []
list_of_x_31 = []
list_of_x_32 = []

# set bools --> must start on the left side (from jumper pov)
inside_roi2 = True
inside_roi1 = False

# start cap loop
while True:
    success, img = cap.read()
    if img is None:
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret,thresh = cv2.threshold(gray, 127, 255, 0)
    list_of_frames.append(frame)
    frame += 1
    img = detector.findPose(img)
    lmList = detector.findPosition(img, draw=True)

    # set up lists for pandas later
    temp_list = [[frame] + item for item in
                 lmList]  # make list shape ([[frame=1,joint=1,x,y],[frame=1,joint=2,x,y],[...]] scope=inside while loop
    df_list.append(temp_list)  # append to list outside scope, dont use pandas because its slow

    # angle finder and joint tracker

    # find angle in a joint (btwn 3 keypoints)
    # angle_13 = detector.findAngle(img, 11, 13, 15)  # l elbow angle
    # angle_25 = detector.findAngle(img, 23, 25, 27)  # l knee angle (outer)
    angle_26 = detector.findAngle(img, 24, 26, 28)  # r knee (inner)

    # track right foot, get x and y
    joint_31 = lmList[31]
    x_31 = joint_31[1]
    y_31 = joint_31[2]

    # track left foot, get x and y
    joint_32 = lmList[32]
    x_32 = joint_32[1]
    y_32 = joint_32[2]

    list_of_x_31.append(x_31)
    list_of_x_32.append(x_32)
    list_of_y_31.append(y_31)
    list_of_y_32.append(y_31)

    # Counter and error detector

    # load mask and find contours, define roi1 and 2 (left and right landing areas)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[1:3]
    cv2.drawContours(img, cnt, -1, (0, 0, 255), 2)

    # get x,y,w and h values of bounding rectangles(!!!) of landing areas
    roi_1 = cv2.boundingRect(contours[1])
    [x, y, w, h] = roi_1
    roi_2 = cv2.boundingRect(contours[2])
    [x, y, w, h] = roi_2

    # use bools for rools (bules for rules)
    if x_31 < roi_1[0]:
        print('outside left')
    elif roi_1[0] < x_31 < roi_1[0] + roi_1[2] and roi_1[1] < y_31 < roi_1[1] + roi_1[3]:
        # print("inside roi_1")
        if inside_roi1 == False:
            inside_roi1 = True
            inside_roi2 = False
            count += 1
        else:
            pass

    elif roi_2[0] < x_31 < roi_2[0] + roi_2[2] and roi_2[1] < y_31 < roi_2[1] + roi_2[3]:
        # print('inside roi_2')
        if inside_roi2 == False:
            inside_roi2 = True
            inside_roi1 = False
            count += 1
    elif x_31 > roi_2[0] + roi_2[2]:
        print('outside right')

    # calculate velocity and accleration of a joint (not used rn because: 24 fps is not accurate enough to do anything with
    # vel_x = np.diff(list_of_x_31)
    # vel_y = np.diff(list_of_y_31)
    # acc_x = np.diff(vel_x)

    # #timer and fps stuff
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    end_time = time.time()
    diff = end_time - start_time
    if diff > 15:  # run for x seconds
        break
    diff = str(round(diff, 2))

    # opencv text putters
    cv2.putText(img, ('time: ' + str(diff)), (10, 20), cv2.FONT_HERSHEY_PLAIN, 2,
                (255, 0, 0), 2)
    cv2.putText(img, ('frame#: ' + str(list_of_frames[-1])), (10, 60), cv2.FONT_HERSHEY_PLAIN, 2,
                (255, 0, 0), 2)
    cv2.putText(img, ('fps: ' + str(int(fps))), (10, 40), cv2.FONT_HERSHEY_PLAIN, 2,
                (255, 0, 0), 2)
    cv2.putText(img, ('reps: ' + str(count)), (10, 80), cv2.FONT_HERSHEY_PLAIN, 2,
                (255, 0, 0), 2)

    cv2.imshow(filename, img)
    # time.sleep(0.2)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# make df and clean up a bit
for item in df_list:
    lm_df = lm_df.append(item)
lm_df.reset_index(drop=True, inplace=True)
lm_df.columns = columns
print(lm_df.to_markdown())
