import cv2
import time
import numpy as np
import pose_module as pm
import matplotlib.pyplot as plt
import pandas as pd
import math
import pickle
import pathlib
import config
from PIL import Image
import io

# load pickle object (list of filenames) created in vid_list_makers.py
with open("pickles/side_vids", "rb") as fp:   # Unpickling
    filenames = pickle.load(fp)

# filename = filenames[0]
for filename in filenames:
    filepath = pathlib.Path(filename).parent
    file_itself = pathlib.Path(filename).stem
    print(filename)
    print(filepath)
    print(file_itself)

    cap = cv2.VideoCapture(filename)  # load the video into opencv

    cap_fps = cap.get(cv2.CAP_PROP_FPS) # determine source fps
    print("capture fps is:" + str(cap_fps))
    ##config stuff


    #
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
    df_extra_list = []

    # instantiate pose detector from imported pose_module
    detector = pm.poseDetector()

    # initiate lists and bools
    list_of_frames = []  # count frames start at 0
    #list_of_y_31 = [] #list of y coordinates of joint 31 for each frame
    frames_with_forward_fall = []
    frames_with_hands_forward = []
    below_parallel = False
    depth_achieved = False
    upright = False
    hands_forward = False
    hands_back = False
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

        # find angle in a joint (btwn 3 keypoints)
        r_hip_angle = detector.findAngle(img, 12, 24, 26)
        r_hip_angle = round(r_hip_angle,2)
        df_extra_list.append([frame, "r_hip", r_hip_angle])
        #df_extra_list.append(str(frame) + ", " + "r_hip, " + str(r_hip_angle))

        r_shoulder_angle = detector.findAngle(img, 14, 12, 24)
        r_shoulder_angle = round(r_shoulder_angle,2)
        df_extra_list.append([frame, "r_shoulder", r_shoulder_angle])

        r_knee_dors_angle = detector.findAngle(img, 28,26,24)
        r_knee_dors_angle = round(r_knee_dors_angle,2)
        df_extra_list.append([frame, "r_knee_dors", r_knee_dors_angle])

        r_ankle_flex = detector.findAngle(img, 26,30,32)
        r_ankle_flex = round(r_ankle_flex,2)
        df_extra_list.append([frame, "r_ankle_flex", r_ankle_flex])

        #parallel torso checker
        hip_r = lmList[24]
        shoulder_r = lmList[12]
        ankle_r = lmList[28]
        knee_r = lmList[26]

        torso_delta_y = hip_r[2] - shoulder_r[2]
        torso_delta_x = shoulder_r[1] - hip_r[1]
        if torso_delta_y and torso_delta_x:
            if torso_delta_x <0:
                torso_lean = torso_delta_y / torso_delta_x
                torso_lean = math.degrees(math.atan(torso_lean))
                torso_lean = round(180+ torso_lean, 2)
                df_extra_list.append([frame, "torso_lean", torso_lean])
            else:
                torso_lean = torso_delta_y / torso_delta_x
                torso_lean = math.degrees(math.atan(torso_lean))
                torso_lean = round(torso_lean, 2)
                df_extra_list.append([frame, "torso_lean", torso_lean])
        else:
            #print("frame " + str(frame) + " - torso lean not found")
            #df_extra_list.append([frame, "torso_lean", 'n/a'])
            pass

        tibia_delta_y = ankle_r[2] - knee_r[2]
        tibia_delta_x = knee_r[1] - ankle_r[1]
        if tibia_delta_y and tibia_delta_x:
            if tibia_delta_x < 0:
                tibia_angle = tibia_delta_y / tibia_delta_x
                tibia_angle = math.degrees(math.atan(tibia_angle))
                tibia_angle = round(180 + tibia_angle, 2)
                df_extra_list.append([frame, "tibia_angle", tibia_angle])
            else:
                tibia_angle = tibia_delta_y / tibia_delta_x
                tibia_angle = math.degrees(math.atan(tibia_angle))
                tibia_angle = round(tibia_angle, 2)
                df_extra_list.append([frame, "tibia_angle", tibia_angle])
        else:
            # print("frame " + str(frame) +" - tibia angle not found")
            # df_extra_list.append([frame, "tibia_angle", 'n/a'])
            tibia_angle = 90


        if torso_lean > tibia_angle and upright is False:
            #print('squat stance: upright')
            upright = True
        elif torso_lean > tibia_angle and upright is True:
            pass
        elif torso_lean < tibia_angle and r_hip_angle > 140:
            #print("squat stance: forward lean but hip extended past 120deg")
            pass
        elif torso_lean < tibia_angle and r_hip_angle < 140:
            print('squat stance: too much forward lean')
            upright = False
            frames_with_forward_fall.append([frame])
        else:
            pass


        # horizontal femur
        if hip_r[2] > knee_r[2] and below_parallel is False:
            below_parallel = True
            depth_achieved = True
            #print("femur below parallel")
        elif hip_r[2] < knee_r[2] and below_parallel is True:
            below_parallel = False
            #print('femur no longer parallel')
        else:
            pass

        # femur angle calculator
        femur_delta_y = knee_r[2] - hip_r[2]
        femur_delta_x = knee_r[1] - hip_r[1]
        if femur_delta_y and femur_delta_x:
            if femur_delta_x < 0:
                femur_angle = femur_delta_y / femur_delta_x
                femur_angle = math.degrees(math.atan(femur_angle))
                femur_angle = round(180+femur_angle, 2)
            else:
                femur_angle = femur_delta_y / femur_delta_x
                femur_angle = math.degrees(math.atan(femur_angle))
                femur_angle = round(femur_angle, 2)
            df_extra_list.append([frame, "femur_angle", femur_angle])
        else:
        #     print("frame " + str(frame) +" - femur angle not found")
            #df_extra_list.append([frame, "femur_angle", 'n/a'])
            pass



        #hands and ankles in line
        wrist_r = lmList[16]

        cv2.circle(img, (wrist_r[1], wrist_r[2]), 5, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (ankle_r[1], ankle_r[2]), 5, (255, 0, 0), cv2.FILLED)

        if (ankle_r[1]*0.9) < wrist_r[1] < (ankle_r[1]*1.1):
            pass
            # df_extra_list.append([frame, "hands_back", 0])
            # df_extra_list.append([frame, "hands_forward", 0])

        elif wrist_r[1] < (ankle_r[1]*0.9):
            #print("hands too far back!")
            hands_back = True
            # df_extra_list.append([frame, "hands_back", 1])
            # df_extra_list.append([frame, "hands_forward", 0])
            frames_with_hands_forward.append([frame])
        elif wrist_r[1] > (ankle_r[1]*1.1):
            #print("hands too far forward!")
            hands_forward = True
            # df_extra_list.append([frame, "hands_forward", 1])
            # df_extra_list.append([frame, "hands_back", 0])
            frames_with_hands_forward.append([frame])

        # hands and ankles angle calc
        p1x, p1y = ankle_r[1], (ankle_r[2]-100)
        p2x, p2y = ankle_r[1], ankle_r[2]
        p3x, p3y = wrist_r[1], wrist_r[2]

        hands_forward_angle = math.degrees(math.atan2(p3y - p2y, p3x - p2x) - math.atan2(p1y - p2y, p1x - p2x))
        df_extra_list.append([frame, "hands_forward_angle", round(hands_forward_angle,2)])

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
        cv2.line(img, (-10, knee_r[2]), (1000, knee_r[2]), (0, 255, 0), thickness=2)
        cv2.putText(img, ('femur angle: ' + str(femur_angle)), (10, 200), cv2.FONT_HERSHEY_PLAIN,
                    2,(255, 0, 0), 2)
        cv2.putText(img, ('torso angle: ' + str(torso_lean)), (10, 220), cv2.FONT_HERSHEY_PLAIN,
                    2,(255, 0, 0), 2)
        cv2.putText(img, ('tibia angle: ' + str(tibia_angle)), (10, 240), cv2.FONT_HERSHEY_PLAIN,
                    2,(255, 0, 0), 2)


        if below_parallel is True:
            cv2.putText(img, ('femur below parallel!!: ' + str(count)), (10, 80), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 0, 0), 2)

        if hands_forward is True:
            cv2.putText(img, ('hands ahead of ankle by pixels: ' + str(wrist_r[1] - ankle_r[1])
                              ), (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        if hands_back is True:
            cv2.putText(img, ('hands behind ankle by pixels: ' + str(ankle_r[1] - wrist_r[1])
                              ), (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        if depth_achieved is True:
            cv2.putText(img, ('depth has been achieved this rep! '), (10, 150), cv2.FONT_HERSHEY_PLAIN, 2,
                            (0, 255, 0), 2)
        else:
            cv2.putText(img, ('depth has NOT been achieved this rep! '), (10, 150), cv2.FONT_HERSHEY_PLAIN, 2,
                        (0, 0, 255), 2)
        resized_img = cv2.resize(img, (540, 960)) #was 540,810 in case i break anythhing

        cv2.imshow(file_itself, resized_img)
        #time.sleep(0.1)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == 32:
            cv2.waitKey()



    # make coords df and clean up a bit
    for item in df_list:
        lm_df = lm_df.append(item)
    lm_df.reset_index(drop=True, inplace=True)
    lm_df.columns = columns
    lm_df.insert(0, 'id', 1)
    lm_df['id'] = file_itself
    lm_df.to_csv('outputs/' + str(file_itself) + "_joint_coords.csv", index=False)

    # make extra df
    extra_df = pd.DataFrame(df_extra_list)
    extra_df.reset_index(drop=True, inplace=True)
    extra_df.columns = ['frame', 'name', 'value']
    extra_df.insert(0, "id", 1)
    extra_df['id'] = file_itself
    extra_df.to_csv('outputs/' + str(file_itself) + "_extra.csv", index=False)

    bools_list =[]
    # Scoring
    print("FMS Testing: " + file_itself)
    print("Nr of frames: " + str(len(list_of_frames)))
    print('succesful run: analyzing')
    if len(frames_with_forward_fall) == 0:
        print(" - Torso parallel to tibia or better: yes")
        bools_list.append(['torso_bool',1])
        bools_list.append(['torso_fall_pc',0])
    else:
        print(" - nr of frames with too much forward lean: " + str(len(frames_with_forward_fall)) + ", " +
              str(round(len(frames_with_forward_fall)/len(list_of_frames),3)*100) + " %.")
        print(frames_with_forward_fall)
        bools_list.append(['torso_bool',0])
        bools_list.append(['torso_fall_pc',str(round(len(frames_with_forward_fall)/len(list_of_frames),3)*100)])

    if len(frames_with_hands_forward) == 0:
        bools_list.append(['hands_forward_pc', 0])

    else:
        bools_list.append(['hands_forward_pc',str(round(len(frames_with_hands_forward)/len(list_of_frames),3)*100)])

    if depth_achieved:
        print(" - femur below horizontal: yes")
        bools_list.append(['squat_bool',1])

    else:
        print(' - depth not achieved!')
        bools_list.append(['squat_bool',0])


    if hands_forward:
        print(" - hands too far ahead!")
        bools_list.append(['hands_forward',1])
    else:
        bools_list.append(['hands_forward',0])

    if hands_back:
        print(" - hands too far back!")
        bools_list.append(['hands_back',1])
    else:
        bools_list.append(['hands_back',0])


    # if not hands_forward and hands_back:
    #     print(" - bar position ok")
    #     bools_list.append(['hands_back',0])
    #     bools_list.append(['hands_forward',0])


    if hands_forward or hands_back or (depth_achieved==False) or len(frames_with_forward_fall):
        print("suggested fms score: 1")
        bools_list.append(['fms_zero',0])
        bools_list.append(['fms_one',1])
        bools_list.append(['fms_two',0])
        bools_list.append(['fms_three',0])
    else:
        print("suggested fms score: 2 or 3")
        bools_list.append(['fms_zero',0])
        bools_list.append(['fms_one',0])
        bools_list.append(['fms_two',0])
        bools_list.append(['fms_thre',1])


    #limb length
    first_frame = df_list[1]
    limb_length_shoulder_x, limb_length_shoulder_y = first_frame[12][2], first_frame[12][3]
    limb_length_hip_x, limb_length_hip_y = first_frame[24][2], first_frame[24][3]
    limb_length_knee_x, limb_length_knee_y = first_frame[26][2], first_frame[26][3]
    limb_length_ankle_x, limb_length_ankle_y = first_frame[28][2], first_frame[28][3]

    torso_length = round(math.dist([limb_length_shoulder_x, limb_length_shoulder_y], [limb_length_hip_x, limb_length_hip_y]),2)
    index_value = torso_length
    torso_length = torso_length/index_value

    femur_length = round(math.dist([limb_length_hip_x,limb_length_hip_y],[limb_length_knee_x, limb_length_knee_y]),2)
    femur_length = round(femur_length/index_value,2)

    tib_length = round(math.dist([limb_length_knee_x, limb_length_knee_y],[limb_length_ankle_x, limb_length_ankle_y]),2)
    tib_length = round(tib_length/index_value,2)
    bools_list.append(['torso_length', torso_length])
    bools_list.append(['femur_length', femur_length])
    bools_list.append(['tib_length', tib_length])


    bools_df = pd.DataFrame(bools_list, columns=['name', 'value'])
    bools_df.to_csv('outputs/'+ str(file_itself) + '_bools_side.csv', index=False)

    print("Additional metrics:")
    metric_sqt_depth = extra_df.loc[extra_df['name']== 'femur_angle']
    print(" - lowest squat angle: " + str(metric_sqt_depth.min()[3]))
    metric_r_shoulder_angle = extra_df.loc[extra_df['name']== 'r_shoulder']
    print(" - shoulder flexion angle min, max: (" + str(metric_r_shoulder_angle.min()[3]) +", "+ str(metric_r_shoulder_angle.max()[3])+ ")")
    metric_hip_angle = extra_df.loc[extra_df['name']== 'r_hip']
    print(" - hip angle min, max: (" + str(metric_hip_angle.min()[3]) +", "+ str(metric_hip_angle.max()[3])+ ")")
    metric_dorsal_knee = extra_df.loc[extra_df['name']== 'r_knee_dors']
    print(" - dorsal knee angle min, max: ("+ str(metric_dorsal_knee.min()[3])+", "+ str(metric_dorsal_knee.max()[3])+ ")")
    metric_ankle_flexion = extra_df.loc[extra_df['name']== 'r_ankle_flex']
    print(" - ankle flexion min, max: (" + str(metric_ankle_flexion.min()[3])+", " + str(metric_ankle_flexion.max()[3])+ ")")
    metric_tibia_lean = extra_df.loc[extra_df['name']== 'tibia_angle']
    print(" - shin angle min, max: (" + str(metric_tibia_lean.min()[3]) +", "+ str(metric_tibia_lean.max()[3])+ ")")
    metric_torso_lean = extra_df.loc[extra_df['name']== 'torso_lean']
    print(" - torso forward lean min, max: (" + str(metric_torso_lean.min()[3]) +", " + str(metric_torso_lean.max()[3])+ ")")

    #find frame with lowest squat angle
    lowest_sqt_index = metric_sqt_depth[['value']].idxmin()
    lowest_sqt_frame = metric_sqt_depth.loc[lowest_sqt_index, 'frame'].to_string(index=False)
    print("lowest squat angle @ frame: " + lowest_sqt_frame)
    quarter_of_the_way_frame = len(list_of_frames)/4

    #screengrabber
    cap = cv2.VideoCapture(filename)  # video_name is the video being called
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(quarter_of_the_way_frame))  # Where frame_no is the frame you want
    ret, img = cap.read()  #

    quarter_of_the_way_frm_list = df_list[int(quarter_of_the_way_frame)]

    circle_list = [12,14,16,24,26,28]
    joint_line_list = [(26,28), (26,24), (24,12), (12,14), (14,16)]
    for i in circle_list:
        img = cv2.circle(img, (quarter_of_the_way_frm_list[i][2], quarter_of_the_way_frm_list[i][3]), radius=10, color=(0, 0, 255), thickness=-1)
    img= cv2.line(img, (quarter_of_the_way_frm_list[26][2] - 150, quarter_of_the_way_frm_list[26][3]), (quarter_of_the_way_frm_list[26][2] + 150, quarter_of_the_way_frm_list[26][3]), (0, 255, 0), thickness=2)
    img= cv2.line(img, (quarter_of_the_way_frm_list[24][2] - 150, quarter_of_the_way_frm_list[24][3]), (quarter_of_the_way_frm_list[24][2] + 150, quarter_of_the_way_frm_list[24][3]), (0, 255, 0), thickness=2)
    img= cv2.line(img, (quarter_of_the_way_frm_list[28][2] - 150, quarter_of_the_way_frm_list[28][3]), (quarter_of_the_way_frm_list[28][2] + 150, quarter_of_the_way_frm_list[28][3]), (0, 255, 0), thickness=2)
    for a,b in joint_line_list:
        img= img= cv2.line(img, (quarter_of_the_way_frm_list[a][2], quarter_of_the_way_frm_list[a][3]), (quarter_of_the_way_frm_list[b][2], quarter_of_the_way_frm_list[b][3]), (255, 0, 0), thickness=2)

    ## angles
    angles_for_ellipsens = extra_df.loc[extra_df['frame']== int(quarter_of_the_way_frame)]
    print(angles_for_ellipsens)
    angle_for_ell_r_hip = angles_for_ellipsens.loc[angles_for_ellipsens['name']== 'r_hip']
    angle_for_ell_r_hip = int(angle_for_ell_r_hip['value'])

    angle_for_ell_torso_lean = angles_for_ellipsens.loc[angles_for_ellipsens['name']== 'torso_lean']
    angle_for_ell_torso_lean = int(angle_for_ell_torso_lean['value'])

    angle_for_ell_r_shoulder = angles_for_ellipsens.loc[angles_for_ellipsens['name']== 'r_shoulder']
    angle_for_ell_r_shoulder = int(angle_for_ell_r_shoulder['value'])

    angle_for_ell_femur = angles_for_ellipsens.loc[angles_for_ellipsens['name']== 'femur_angle']
    angle_for_ell_femur = int(angle_for_ell_femur['value'])

    angle_for_ell_tibia_angle = angles_for_ellipsens.loc[angles_for_ellipsens['name']== 'tibia_angle']
    angle_for_ell_tibia_angle = int(angle_for_ell_tibia_angle['value'])

    # Ellipse parameters
    radius = 75
    axes = (radius, radius)
    thickness = 3

    #hip ellipse
    center = (quarter_of_the_way_frm_list[24][2], quarter_of_the_way_frm_list[24][3])
    angle = - angle_for_ell_torso_lean
    startAngle = 0
    endAngle = angle_for_ell_r_hip
    cv2.ellipse(img, center, axes, angle, startAngle, endAngle, (0,255,0), thickness)

    #shoulder flexion ellipse
    center = (quarter_of_the_way_frm_list[12][2], quarter_of_the_way_frm_list[12][3])
    angle =  180 - angle_for_ell_torso_lean
    startAngle =  0
    endAngle = -angle_for_ell_r_shoulder
    cv2.ellipse(img, center, axes, angle, startAngle, endAngle, (0,255,0), thickness)

    #femur angle
    center = (quarter_of_the_way_frm_list[26][2], quarter_of_the_way_frm_list[26][3])
    angle = 180
    startAngle =  0
    endAngle = angle_for_ell_femur
    cv2.ellipse(img, center, axes, angle, startAngle, endAngle, (0,255,0), thickness)

    #tibia angle
    center = (quarter_of_the_way_frm_list[28][2], quarter_of_the_way_frm_list[28][3])
    angle = 0
    startAngle =  0
    endAngle = -angle_for_ell_tibia_angle
    cv2.ellipse(img, center, axes, angle, startAngle, endAngle, (0,255,0), thickness)

    #torso lean
    center = (quarter_of_the_way_frm_list[24][2], quarter_of_the_way_frm_list[24][3])
    angle = 0
    radius=100
    axes = (radius, radius)
    startAngle =  0
    endAngle = -angle_for_ell_torso_lean
    cv2.ellipse(img, center, axes, angle, startAngle, endAngle, (0,255,0), thickness)
    #put text for angles
    cv2.putText(img, 'b', (quarter_of_the_way_frm_list[24][2] +10, quarter_of_the_way_frm_list[24][3] -20 ), cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 255, 0), 2)
    cv2.putText(img, 'a', (quarter_of_the_way_frm_list[12][2]+10, quarter_of_the_way_frm_list[12][3]), cv2.FONT_HERSHEY_PLAIN,3,
                        (0, 255, 0), 2)
    cv2.putText(img, 'c', (quarter_of_the_way_frm_list[24][2] +70, quarter_of_the_way_frm_list[24][3] -70 ), cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 255, 0), 2)
    cv2.putText(img, 'e', (quarter_of_the_way_frm_list[26][2] -30 , quarter_of_the_way_frm_list[26][3] -20), cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 255, 0), 2)
    cv2.putText(img, 'd', (quarter_of_the_way_frm_list[28][2] +30, quarter_of_the_way_frm_list[28][3]-30), cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 255, 0), 2)

    print(file_itself)
    #crop and save
    cropped = img[int(df_list[1][16][3]-200):int(df_list[1][28][3]+ 200),int(df_list[1][24][2] - 300):int(df_list[1][24][2] + 300)]
    cv2.imwrite(f'./outputs/images/{file_itself}_quarter_way_screen.jpg', cropped)

    ##lowest
    cap = cv2.VideoCapture(filename)  # video_name is the video being called
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(lowest_sqt_frame))  # Where frame_no is the frame you want
    ret, img = cap.read()  #
    lowest_sqt_frm_list = df_list[int(lowest_sqt_frame) - 1]
    circle_list = [12,14,16,24,26,28]
    joint_line_list = [(26,28), (26,24), (24,12), (12,14), (14,16)]
    for i in circle_list:
        img = cv2.circle(img, (lowest_sqt_frm_list[i][2], lowest_sqt_frm_list[i][3]), radius=10, color=(0, 0, 255), thickness=-1)
    img= cv2.line(img, (lowest_sqt_frm_list[26][2] - 250, lowest_sqt_frm_list[26][3]), (lowest_sqt_frm_list[26][2] + 150, lowest_sqt_frm_list[26][3]), (0, 255, 0), thickness=2)
    #img= cv2.line(img, (lowest_sqt_frm_list[24][2] - 150, lowest_sqt_frm_list[24][3]), (lowest_sqt_frm_list[24][2] + 150, lowest_sqt_frm_list[24][3]), (0, 255, 0), thickness=2)
    #img= cv2.line(img, (lowest_sqt_frm_list[28][2] - 150, lowest_sqt_frm_list[28][3]), (lowest_sqt_frm_list[28][2] + 150, lowest_sqt_frm_list[28][3]), (0, 255, 0), thickness=2)
    for a,b in joint_line_list:
        img= img= cv2.line(img, (lowest_sqt_frm_list[a][2], lowest_sqt_frm_list[a][3]), (lowest_sqt_frm_list[b][2], lowest_sqt_frm_list[b][3]), (255, 0, 0), thickness=2)

    angles_for_ellipsens = extra_df.loc[extra_df['frame']== int(lowest_sqt_frame)]
    angle_for_ell_femur = angles_for_ellipsens.loc[angles_for_ellipsens['name']== 'femur_angle']
    angle_for_ell_femur = int(angle_for_ell_femur['value'])

    #femur angle
    radius = 75
    axes = (radius, radius)
    thickness = 3
    center = (lowest_sqt_frm_list[26][2], lowest_sqt_frm_list[26][3])
    angle = 180
    startAngle =  0
    endAngle = angle_for_ell_femur
    cv2.ellipse(img, center, axes, angle, startAngle, endAngle, (0,255,0), thickness)


    #cv2.putText(img,str(torso_length)+':' +str(femur_length)+':'+ str(tib_length) , (lowest_sqt_frm_list[28][2] -100, lowest_sqt_frm_list[28][3]-30), cv2.FONT_HERSHEY_PLAIN, 3,
                        # (0, 255, 0), 2)

    #crop and save
    cropped = img[int(df_list[1][16][3]-200):int(df_list[1][28][3]+ 200),int(df_list[1][24][2] - 300):int(df_list[1][24][2] + 300)]
    cv2.imwrite(f'./outputs/images/{file_itself}_lowest_screen.jpg', cropped)
