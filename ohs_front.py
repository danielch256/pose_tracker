import cv2
import time
import numpy as np
import pose_module as pm
import matplotlib.pyplot as plt
import pandas as pd
import mediapipe as mp
import pickle
import pathlib
import math
import config

# load pickle object (list of filenames) created in vid_list_makers.py
with open("pickles/front_vids", "rb") as fp:   # Unpickling
    filenames = pickle.load(fp)
for filename in filenames:
#filename = filenames[10]
    filepath = pathlib.Path(filename).parent
    file_itself = pathlib.Path(filename).stem

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
    angle_df = pd.DataFrame()
    df_list= []
    df_extra_list = []

    # load pose detector from imported pose_module
    detector = pm.poseDetector()

    # initiate lists and bools
    list_of_frames = []  # count frames start at 0
    frames_with_parallel = []
    parallel = False
    valgus_left = False
    valgus_right = False
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
                     lmList]  # make list shape ([[frame=1,joint=1,x,y],[frame=1,joint=2,x,y],[...]] for each frame
        df_list.append(temp_list)  # append to list outside scope, dont use pandas because its slow

        # angle finders and joint trackers
        # knee stuff

        right_knee = lmList[26]
        right_knee_x = right_knee[1]
        right_knee_y = right_knee[2]
        left_knee = lmList[25]
        left_knee_x = left_knee[1]
        left_knee_y = left_knee[2]

        knee_dist = left_knee_x - right_knee_x

        df_extra_list.append([frame, "knee_dist_x",knee_dist])

        # hip stuff
        q_angle_l = detector.findAngle(img, 25, 23, 24) # l hip q-angle
        q_angle_r = detector.findAngle(img, 23, 24, 26)  # r hip q-angle
        df_extra_list.append([frame, "q_angle_l", round(q_angle_l,3)])
        df_extra_list.append([frame, "q_angle_r", round(q_angle_r,3)])

        hip_midpoint_x = (lmList[23][1] + lmList[24][1])/2
        knee_midpoint_x = (left_knee_x + right_knee_x)/2
        df_extra_list.append([frame, "hip_midpoint_x", hip_midpoint_x])
        df_extra_list.append([frame, "knee_midpoint_x", knee_midpoint_x])

        #shoulder angle
        shoulder_angle_l = detector.findAngle(img, 13, 11, 23)
        shoulder_angle_r = detector.findAngle(img, 24, 12, 14)
        shoulder_angle_r_upper = detector.findAngle(img, 12,11,13)
        shoulder_angle_l_upper = detector.findAngle(img, 14,12,11)
        df_extra_list.append([frame, "shoulder_angle_l", round(shoulder_angle_l, 3)])
        df_extra_list.append([frame, "shoulder_angle_r", round(shoulder_angle_r, 3)])
        df_extra_list.append([frame, "shoulder_angle_r_upper", round(shoulder_angle_r_upper, 3)])
        df_extra_list.append([frame, "shoulder_angle_l_upper", round(shoulder_angle_l_upper, 3)])


        # parallel lines checker
        hip_r = lmList[24]
        hip_l = lmList[23]
        shoulder_r = lmList[12]
        shoulder_l = lmList[11]
        knee_r = lmList[26]
        knee_l = lmList[25]
        wrist_r = lmList[16]
        wrist_l = lmList[15]
        ankle_r = lmList[28]
        ankle_l = lmList[27]
        # hip-line
        hips_delta_y = hip_l[2] - hip_r[2]
        hips_delta_x = hip_l[1] - hip_r[1]
        if hips_delta_y and hips_delta_x:
            hip_lean = hips_delta_y/hips_delta_x
            hip_lean = math.degrees(math.atan(hip_lean))
            hip_lean = round(hip_lean, 2)
            df_extra_list.append([frame, "hip_lean", hip_lean])
        else:
            print('no hip lean! filled with 0')
            df_extra_list.append([frame, "hip_lean", 0])

            #print("hip lean: " + str(hip_lean) + "\N{DEGREE SIGN}")
        # shoulder-line
        shoulder_delta_y = shoulder_l[2] - shoulder_r[2]
        shoulder_delta_x = shoulder_l[1] - shoulder_r[1]
        if shoulder_delta_y and shoulder_delta_x:
            shoulder_lean = shoulder_delta_y/shoulder_delta_x
            shoulder_lean = math.degrees(math.atan(shoulder_lean))
            shoulder_lean = round(shoulder_lean,2)
            df_extra_list.append([frame, "shoulder_lean", shoulder_lean])
        else:
            print('no shoulder lean, filled with 0')
            df_extra_list.append([frame, "shoulder_lean", 0])
            #print("shoulder lean: " + str(shoulder_lean) + "\N{DEGREE SIGN}")
        #wrist-line
        wrist_delta_y = wrist_l[2] - wrist_r[2]
        wrist_delta_x = wrist_l[1] - wrist_r[1]
        if wrist_delta_y and wrist_delta_x:
            wrist_lean = wrist_delta_y/wrist_delta_x
            wrist_lean = math.degrees(math.atan(wrist_lean))
            wrist_lean = round(wrist_lean, 2)
            df_extra_list.append([frame, "wrist_lean", wrist_lean])
            #print("wrist lean: " + str(wrist_lean) + "\N{DEGREE SIGN}")


        #valgus
        # if q_angle_l < 80:
        #     print('left hip angle: ' + str(q_angle_l))
        #
        # if q_angle_r < 80:
        #     print('right hip angle: ' + str(q_angle_r))
        if knee_r[1] > ankle_r[1] and q_angle_r > 125:
            #print("valgus right knee at frame: " + str(frame))
            #print("(" + str(knee_r[1]) + ", " + str(ankle_r[1]) + ')')
            valgus_right = True
        if knee_l[1] < ankle_l[1] and q_angle_l > 125:
            #print('valgus left knee at frame: ' + str(frame))
            #print("(" + str(knee_l[1]) + ", " + str(ankle_l[1]) + ')')
            valgus_left = True

        #shin angle finder
        ankle_angle_med_l = detector.findAngle(img, 28, 27, 25)
        ankle_angle_med_r = detector.findAngle(img, 26, 28, 27)
        df_extra_list.append([frame, "ankle_angle_med_l", round(ankle_angle_med_l,3)])
        df_extra_list.append([frame, "ankle_angle_med_r", round(ankle_angle_med_r,3)])

        # #timer and fps stuff
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        end_time = time.time()
        diff = end_time - start_time
        if diff > 60:  # run for x seconds
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

        if not valgus_left:
            cv2.putText(img, ('left hip angle:  ' + str(round(q_angle_l, 2))), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 255, 0), 2)
            cv2.putText(img, (" - ok!"), (200, 120), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 255, 0), 2)
        else:
            cv2.putText(img, ('left hip angle:  ' + str(round(q_angle_l, 2))), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 255), 2)
            cv2.putText(img, (" - suspected valgus collapse!"), (200, 120), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 255), 2)
        if not valgus_right:
            cv2.putText(img, ('right hip angle:  ' + str(round(q_angle_r, 2))), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 255, 0), 2)
            cv2.putText(img, (" - ok!"), (200, 140), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 255, 0), 2)
        else:
            cv2.putText(img, ('right hip angle:  ' + str(round(q_angle_r, 2))), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 255), 2)
            cv2.putText(img, (" - suspected valgus collapse!"), (200, 140), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 255), 2)
        img = cv2.resize(img, (540, 960))
        cv2.imshow(filename, img)
        #time.sleep(0.05)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == 32:
            cv2.waitKey()

    # make df for all joint coords
    for item in df_list:
        lm_df = lm_df.append(item)
    lm_df.reset_index(drop=True, inplace=True)
    lm_df.columns = columns
    lm_df.insert(0, 'id', 1)
    lm_df['id'] = file_itself
    lm_df.to_csv('outputs/' + str(file_itself) + "_joint_coords.csv", index=False)

    #df for selected extra
    extra_df = pd.DataFrame(df_extra_list)
    extra_df.reset_index(drop=True, inplace=True)
    extra_df.columns = ['frame', 'name', 'value']
    extra_df.insert(0, "id", 1)
    extra_df['id'] = file_itself
    extra_df.to_csv('outputs/' + str(file_itself) + "_extra.csv", index=False)

    # print fms score:
    print("Nr of frames: " + str(len(list_of_frames)))
    print('Successful run: analyzing')
    print("FMS Testing: " + file_itself)
    bools_list = []
    print(" - Knee valgus:")
    if valgus_right:
        print("     - suspected valgus right!")
        bools_list.append(['valgus_r_bool', 1])
    else:
        print('     - no suspected valgus right')
        bools_list.append(['valgus_r_bool', 0])

    if valgus_left:
        print('     - suspected valgus left!')
        bools_list.append(['valgus_l_bool', 1])
    else:
        print('     - no suspected valgus left')
        bools_list.append(['valgus_l_bool', 0])

    bools_df = pd.DataFrame(bools_list, columns=['name', 'value'])
    bools_df.to_csv('outputs/' + str(file_itself) + '_bools_front.csv', index=False)

# a good valgus check is hard to do in 2d. other way could be compare hip distance x to knee distance x.
    # the different ways are all imperfect and sensitive to different aspects of what knee valgus actually IS.


    print("Additional metrics:")

    metric_q_angle_l = extra_df.loc[extra_df['name'] == 'q_angle_l']
    metric_q_angle_r = extra_df.loc[extra_df['name'] == 'q_angle_r']
    print(" - q-angle left hip min,max: (" + str(metric_q_angle_l.min()[3]) + ", " + str(metric_q_angle_l.max()[3]) + ")")
    print(" - q-angle right hip min,max: (" + str(metric_q_angle_r.min()[3]) + ", " + str(metric_q_angle_r.max()[3]) + ")")

    metric_ankle_angle_med_l = extra_df.loc[extra_df['name'] == 'ankle_angle_med_l']
    metric_ankle_angle_med_r = extra_df.loc[extra_df['name'] == 'ankle_angle_med_r']
    print(" - medial shin angle l min,max: (" + str(metric_ankle_angle_med_l.min()[3]) + ", " + str(metric_ankle_angle_med_l.max()[3]) + ")")
    print(" - medial shin angle r min,max: (" + str(metric_ankle_angle_med_r.min()[3]) + ", " + str(metric_ankle_angle_med_r.max()[3]) + ")")

    metric_shoulder_angle_l = extra_df.loc[extra_df['name'] == 'shoulder_angle_l']
    metric_shoulder_angle_r = extra_df.loc[extra_df['name'] == 'shoulder_angle_r']
    print(" - shoulder angle l min,max: (" + str(metric_shoulder_angle_l.min()[3]) + ", " + str(metric_shoulder_angle_l.max()[3]) + ")")
    print(" - shoulder angle r min,max: (" + str(metric_shoulder_angle_r.min()[3]) + ", " + str(metric_shoulder_angle_r.max()[3]) + ")")

    metric_shoulder_lean = extra_df.loc[extra_df['name'] == 'shoulder_lean']
    print(" - shoulder lean min, max: ("
          + str(metric_shoulder_lean.min()[3]) + ", " +
          str(metric_shoulder_lean.max()[3]) + ")")
          #"," + str(metric_shoulder_lean.mean()[3]) + ")")

    metric_hip_lean = extra_df.loc[extra_df['name']== 'hip_lean']
    print(" - hip lean min, max: ("
          + str(metric_hip_lean.min()[3]) + ", " +
          str(metric_hip_lean.max()[3]) + ")")

    metric_wrist_lean = extra_df.loc[extra_df['name']== 'wrist_lean']
    print(" - bar lean min, max: ("
          + str(metric_wrist_lean.min()[3]) + ", " +
          str(metric_wrist_lean.max()[3]) + ")")

    metric_hip_shift_x = extra_df.loc[extra_df['name']== 'hip_midpoint_x']
    hip_shift = (metric_hip_shift_x.max()[3] - metric_hip_shift_x.min()[3]) / metric_hip_shift_x.min()[3] *100
    print(" - hip shift on x axis: (" + str(round(hip_shift,3)) + " %)")


    knee_midpoint_shift_x = extra_df.loc[extra_df['name']== 'knee_midpoint_x']
    knee_shift = (knee_midpoint_shift_x.max()[3] - knee_midpoint_shift_x.min()[3]) / knee_midpoint_shift_x.min()[3] *100
    print(" - knee shift on x axis: (" + str(round(knee_shift,3)) + " %)")

    #screengrabber
    quarter_of_the_way_frame = len(list_of_frames)/4
    cap = cv2.VideoCapture(filename)  # video_name is the video being called
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(quarter_of_the_way_frame))  # Where frame_no is the frame you want
    ret, img = cap.read()  #

    quarter_of_the_way_frm_list = df_list[int(quarter_of_the_way_frame)]
    circle_list = [12,14,24,26,28,11,13,23,25,27]
    joint_line_list = [(26,28), (26,24), (24,12), (23,24), (12,14), (27,25), (25,23), (23,11), (11,13), (11,12), (27,28)]

    for i in circle_list:
        img = cv2.circle(img, (quarter_of_the_way_frm_list[i][2], quarter_of_the_way_frm_list[i][3]), radius=10, color=(0, 0, 255), thickness=-1)
    for a,b in joint_line_list:
        img= img= cv2.line(img, (quarter_of_the_way_frm_list[a][2], quarter_of_the_way_frm_list[a][3]), (quarter_of_the_way_frm_list[b][2], quarter_of_the_way_frm_list[b][3]), (255, 0, 0), thickness=2)
    #draw midpoints
    midpoint_hip_circle_x =int((quarter_of_the_way_frm_list[23][2]+ quarter_of_the_way_frm_list[24][2])/2)
    midpoint_hip_circle_y =int((quarter_of_the_way_frm_list[23][3]+ quarter_of_the_way_frm_list[24][3])/2)
    img = cv2.circle(img, (midpoint_hip_circle_x ,midpoint_hip_circle_y ), radius=10,
                     color=(0, 255, 0), thickness=-1)

    midpoint_knee_circle_x =int((quarter_of_the_way_frm_list[26][2]+ quarter_of_the_way_frm_list[25][2])/2)
    midpoint_knee_circle_y =int((quarter_of_the_way_frm_list[26][3]+ quarter_of_the_way_frm_list[25][3])/2)
    img = cv2.circle(img, (midpoint_knee_circle_x ,midpoint_knee_circle_y ), radius=10,
                     color=(0, 255, 0), thickness=-1)

    cv2.putText(img, '<<', (midpoint_knee_circle_x - 60, midpoint_knee_circle_y+10), cv2.FONT_HERSHEY_PLAIN, 2,
                        (0, 255, 0), 2)
    cv2.putText(img, '>>', (midpoint_knee_circle_x + 20, midpoint_knee_circle_y +10), cv2.FONT_HERSHEY_PLAIN, 2,
                        (0, 255, 0), 2)

    angles_for_ellipsens = extra_df.loc[extra_df['frame']== int(quarter_of_the_way_frame)]
    # print(angles_for_ellipsens)
    angle_for_ell_r_ankle = angles_for_ellipsens.loc[angles_for_ellipsens['name']== 'ankle_angle_med_r']
    angle_for_ell_l_ankle = angles_for_ellipsens.loc[angles_for_ellipsens['name']== 'ankle_angle_med_l']
    angle_for_ell_r_ankle = int(angle_for_ell_r_ankle['value'])
    angle_for_ell_l_ankle = int(angle_for_ell_l_ankle['value'])

    angle_for_shoulder_lean = angles_for_ellipsens.loc[angles_for_ellipsens['name']== 'shoulder_lean']
    angle_for_shoulder_lean = int(angle_for_shoulder_lean['value'])
    angle_for_hip_lean = angles_for_ellipsens.loc[angles_for_ellipsens['name']== 'hip_lean']
    angle_for_hip_lean = int(angle_for_hip_lean['value'])


    angle_for_l_shoulder = angles_for_ellipsens.loc[angles_for_ellipsens['name']== 'shoulder_angle_l']
    angle_for_l_shoulder = int(angle_for_l_shoulder['value'])

    angle_for_l_shoulder_upper = angles_for_ellipsens.loc[angles_for_ellipsens['name']== 'shoulder_angle_l_upper']
    angle_for_l_shoulder_upper = int(angle_for_l_shoulder_upper['value'])

    angle_for_r_shoulder = angles_for_ellipsens.loc[angles_for_ellipsens['name']== 'shoulder_angle_r']
    angle_for_r_shoulder = int(angle_for_r_shoulder['value'])

    angle_for_r_shoulder_upper = angles_for_ellipsens.loc[angles_for_ellipsens['name']== 'shoulder_angle_r_upper']
    angle_for_r_shoulder_upper = int(angle_for_r_shoulder_upper['value'])

    angle_for_q_angle_r = angles_for_ellipsens.loc[angles_for_ellipsens['name']== 'q_angle_r']
    angle_for_q_angle_r = int(angle_for_q_angle_r['value'])

    angle_for_q_angle_l = angles_for_ellipsens.loc[angles_for_ellipsens['name']== 'q_angle_l']
    angle_for_q_angle_l = int(angle_for_q_angle_l['value'])

    # Ellipse parameters
    radius = 75
    axes = (radius, radius)
    thickness = 3

    #ankle r ellipse
    center = (quarter_of_the_way_frm_list[28][2], quarter_of_the_way_frm_list[28][3])
    angle = - angle_for_ell_r_ankle
    startAngle = 0
    endAngle = angle_for_ell_r_ankle
    cv2.ellipse(img, center, axes, angle, startAngle, endAngle, (0,255,0), thickness)

    #ankle l
    center = (quarter_of_the_way_frm_list[27][2], quarter_of_the_way_frm_list[27][3])
    angle = 180
    startAngle = 0
    endAngle = angle_for_ell_l_ankle
    cv2.ellipse(img, center, axes, angle, startAngle, endAngle, (0,255,0), thickness)

    #shoulder angle l
    center = (quarter_of_the_way_frm_list[11][2], quarter_of_the_way_frm_list[11][3])
    angle = 180 - angle_for_shoulder_lean + angle_for_l_shoulder_upper
    startAngle = 0
    endAngle = angle_for_l_shoulder
    cv2.ellipse(img, center, axes, angle, startAngle, endAngle, (0,255,0), thickness)

    #shoulder angle r
    center = (quarter_of_the_way_frm_list[12][2], quarter_of_the_way_frm_list[12][3])
    angle = angle_for_shoulder_lean - angle_for_r_shoulder_upper
    startAngle = 0
    endAngle = - angle_for_r_shoulder
    cv2.ellipse(img, center, axes, angle, startAngle, endAngle, (0,255,0), thickness)

    #q-angle l
    radius = 45
    axes = (radius, radius)
    center = (quarter_of_the_way_frm_list[23][2], quarter_of_the_way_frm_list[23][3])
    angle = 180 + angle_for_hip_lean
    startAngle = 0
    endAngle = - angle_for_q_angle_l
    cv2.ellipse(img, center, axes, angle, startAngle, endAngle, (0,255,0), thickness)

    #q-angle l
    radius = 45
    axes = (radius, radius)
    center = (quarter_of_the_way_frm_list[24][2], quarter_of_the_way_frm_list[24][3])
    angle = angle_for_hip_lean
    startAngle = 0
    endAngle = angle_for_q_angle_r
    cv2.ellipse(img, center, axes, angle, startAngle, endAngle, (0,255,0), thickness)


    cv2.putText(img, 'a', (quarter_of_the_way_frm_list[23][2]-20, quarter_of_the_way_frm_list[23][3]+20), cv2.FONT_HERSHEY_PLAIN,3,
                        (0, 255, 0), 2)
    cv2.putText(img, 'b', (quarter_of_the_way_frm_list[27][2] -20, quarter_of_the_way_frm_list[27][3]), cv2.FONT_HERSHEY_PLAIN,3,
                        (0, 255, 0), 2)
    cv2.putText(img, 'c', (quarter_of_the_way_frm_list[11][2], quarter_of_the_way_frm_list[11][3]+20), cv2.FONT_HERSHEY_PLAIN,3,
                        (0, 255, 0), 2)
    cv2.putText(img, 'd', (midpoint_hip_circle_x ,midpoint_hip_circle_y -10 ), cv2.FONT_HERSHEY_PLAIN,3,
                        (0, 255, 0), 2)
    cv2.putText(img, 'e', (midpoint_knee_circle_x ,midpoint_knee_circle_y -10 ), cv2.FONT_HERSHEY_PLAIN,3,
                        (0, 255, 0), 2)

    #crop and save
    cropped = img[int(df_list[1][16][3]-200):int(df_list[1][28][3]+ 200),int(df_list[1][16][2] - 100):int(df_list[1][15][2] + 100)] # y,x for some reason
    cv2.imwrite(f'./outputs/images/{file_itself}_quarter_way_front.jpg', cropped)


