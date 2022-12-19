import pickle
import pathlib
import pandas as pd
import cv2
from PIL import Image
import config
import matplotlib.pyplot as plt
import numpy as np
def save_var_latex(key, value):
    import csv
    import os

    dict_var = {}

    file_path = os.path.join(os.getcwd(), "mydata.dat")

    try:
        with open(file_path, newline="") as file:
            reader = csv.reader(file)
            for row in reader:
                dict_var[row[0]] = row[1]
    except FileNotFoundError:
        pass

    dict_var[key] = value

    with open(file_path, "w") as f:
        for key in dict_var.keys():
            f.write(f"{key},{dict_var[key]}\n")
# load side view stuff
with open("pickles/side_vids", "rb") as fps:   # Unpickling
    full_path_filenames_side = pickle.load(fps)
with open("pickles/front_vids", "rb") as fpf:   # Unpickling
    full_path_filenames_front = pickle.load(fpf)
# for filename in full_path_filenames:
#     filename = pathlib.Path(filename).name
#     file_itself = pathlib.Path(filename).stem
id_nr=43
filename_side = pathlib.Path(full_path_filenames_side[id_nr]).name
file_itself_side = pathlib.Path(full_path_filenames_side[id_nr]).stem
print(file_itself_side)
filename_front = pathlib.Path(full_path_filenames_front[id_nr]).name
file_itself_front = pathlib.Path(full_path_filenames_front[id_nr]).stem
print(file_itself_front)


extra_df = pd.read_csv(f'outputs/static_csv/{file_itself_side}_extra.csv')
joint_coords = pd.read_csv(f'outputs/static_csv/{file_itself_side}_joint_coords.csv')
info = pd.read_csv('info.csv')
bools_side = pd.read_csv(f'outputs/static_csv/{file_itself_side}_bools_side.csv')
bools_front = pd.read_csv(f'outputs/static_csv/{file_itself_front}_bools_front.csv')

#info stuff
# id = int(info.loc[info['name']== 'id']['value'].iloc[0])
id = file_itself_side[:3]

first = info.loc[info['name']== 'first']['value'].iloc[0]
last = info.loc[info['name']== 'last']['value'].iloc[0]
height = (info.loc[info['name']== 'height']['value'].iloc[0])
age = info.loc[info['name']== 'age']['value'].iloc[0]
date = info.loc[info['name']== 'date']['value'].iloc[0]
event = info.loc[info['name']== 'event']['value'].iloc[0]
location = info.loc[info['name']== 'location']['value'].iloc[0]

#bools
squat_bool = int(bools_side.loc[bools_side['name']== 'squat_bool']['value'].iloc[0])
hands_forward = int(bools_side.loc[bools_side['name']== 'hands_forward']['value'].iloc[0])
hands_back = int(bools_side.loc[bools_side['name']== 'hands_back']['value'].iloc[0])

torso_bool = int(bools_side.loc[bools_side['name']== 'torso_bool']['value'].iloc[0])
fms_zero = bools_side.loc[bools_side['name']== 'fms_zero']['value'].iloc[0]
fms_one = bools_side.loc[bools_side['name']== 'fms_one']['value'].iloc[0]
fms_two = bools_side.loc[bools_side['name']== 'fms_two']['value'].iloc[0]
fms_three = bools_side.loc[bools_side['name']== 'fms_three']['value'].iloc[0]
torso_fall_pc = bools_side.loc[bools_side['name']== 'torso_fall_pc']['value'].iloc[0]
hands_forward_pc = bools_side.loc[bools_side['name']== 'hands_forward_pc']['value'].iloc[0]
torso_length = bools_side.loc[bools_side['name']== 'torso_length']['value'].iloc[0]
femur_length = bools_side.loc[bools_side['name']== 'femur_length']['value'].iloc[0]
tib_length = bools_side.loc[bools_side['name']== 'tib_length']['value'].iloc[0]




valgus_l_bool = int(bools_front.loc[bools_front['name']== 'valgus_l_bool']['value'].iloc[0])
valgus_r_bool = int(bools_front.loc[bools_front['name']== 'valgus_r_bool']['value'].iloc[0])



print(height)

#side videw stuff
metric_sqt_depth = round(extra_df.loc[extra_df['name']== 'femur_angle'],1)
lowest_sqt_index = metric_sqt_depth[['value']].idxmin()
lowest_sqt_frame = metric_sqt_depth.loc[lowest_sqt_index, 'frame'].to_string(index=False)

metric_r_shoulder_angle = round(extra_df.loc[extra_df['name']== 'r_shoulder'],1)
metric_hip_angle = round(extra_df.loc[extra_df['name']== 'r_hip'],1)
metric_dorsal_knee = round(extra_df.loc[extra_df['name']== 'r_knee_dors'],1)
metric_ankle_flexion = round(extra_df.loc[extra_df['name']== 'r_ankle_flex'],1)
metric_tibia_lean = round(extra_df.loc[extra_df['name']== 'tibia_angle'],1)
metric_torso_lean = round(extra_df.loc[extra_df['name']== 'torso_lean'],1)
metric_hands_forward_angle = round(extra_df.loc[extra_df['name']== 'hands_forward_angle'],1)

# screenshots
save_var_latex('report_img_1', str("\includegraphics[width=0.25\linewidth]{../outputs/images/" + str(file_itself_side) + "_quarter_way_screen.jpg}"))
save_var_latex('report_img_2', str("\includegraphics[width=0.25\linewidth]{../outputs/images/" + str(file_itself_side) + "_lowest_screen.jpg}"))

# make graphs
list_to_graph = [metric_sqt_depth, metric_hands_forward_angle, metric_tibia_lean, metric_torso_lean]
fig, ax = plt.subplots()
fig.set_size_inches(11, 3)

for a in list_to_graph:
    label = a['name'].iloc[0]
    x = a['frame']
    y= a['value']
    poly = np.polyfit(x, y, 20)
    poly_y = np.poly1d(poly)(x)
    ax.plot(x, poly_y, label=str(label))
    fig.set_tight_layout(True)
    ax.set_xlabel('Frame nr.')
    ax.set_ylabel('Angle')
    fig.suptitle('FMS parameters, side view')

    plt.legend(bbox_to_anchor=(1.04, 0.5), loc= 'center left')
plt.show()
fig.savefig(f'outputs/graphs/{file_itself_side}_graph.png', format='png')
save_var_latex('side_view_graph', str("\includegraphics[width=.9\linewidth]{../outputs/graphs/" + str(file_itself_side) + "_graph.png}"))

# load front view files
# with open("pickles/front_vids", "rb") as fp:   # Unpickling
#     full_path_filenames = pickle.load(fp)
# filename = pathlib.Path(full_path_filenames[10]).name
# file_itself = pathlib.Path(full_path_filenames[10]).stem
# print(file_itself)
extra_df = pd.read_csv(f'outputs/static_csv/{file_itself_front}_extra.csv')
joint_coords = pd.read_csv(f'outputs/static_csv/{file_itself_front}_joint_coords.csv')

metric_hip_shift_x = extra_df.loc[extra_df['name']== 'hip_midpoint_x']
hip_shift = (metric_hip_shift_x.max()[3] - metric_hip_shift_x.min()[3]) / metric_hip_shift_x.min()[3] *100
hip_shift = round(hip_shift,2)

knee_midpoint_shift_x = extra_df.loc[extra_df['name']== 'knee_midpoint_x']
knee_shift = (knee_midpoint_shift_x.max()[3] - knee_midpoint_shift_x.min()[3]) / knee_midpoint_shift_x.min()[3] *100
knee_shift = round(knee_shift,2)

metric_q_angle_l = round(extra_df.loc[extra_df['name']== 'q_angle_l'],1)
metric_q_angle_r = round(extra_df.loc[extra_df['name']== 'q_angle_r'],1)
metric_ankle_l = round(extra_df.loc[extra_df['name']== 'ankle_angle_med_l'],1)
metric_ankle_r = round(extra_df.loc[extra_df['name']== 'ankle_angle_med_r'],1)
metric_shoulder_angle_l = round(extra_df.loc[extra_df['name'] == 'shoulder_angle_l'],1)
metric_shoulder_angle_r = round(extra_df.loc[extra_df['name'] == 'shoulder_angle_r'],1)

#screenshots
save_var_latex('report_img_3', str("\includegraphics[width=0.25\linewidth]{../outputs/images/" + str(file_itself_front) + "_quarter_way_front.jpg}"))


## cleanup and make vars to pass
latex_filename = filename_side.replace("_", "\\_")
save_var_latex('filename', latex_filename)

#info
save_var_latex('id', id)
save_var_latex('first', first)
save_var_latex('last', last)
save_var_latex('height', height)
save_var_latex('date', date)
save_var_latex('age', age)
save_var_latex('event', event)
save_var_latex('location', location)

if fms_zero ==1:
    save_var_latex('fms_score', '0')
elif fms_one == 1:
    save_var_latex('fms_score', '1')
elif fms_two ==1:
    save_var_latex('fms_score', '2')
elif fms_three == 1:
    save_var_latex('fms_score', '3')

if squat_bool == 1:
    save_var_latex('squat_bool', 'Pass')
else:
    save_var_latex('squat_bool', 'Fail')

if hands_forward == 1:
    save_var_latex('hands_forward', 'Fail')
else:
    save_var_latex('hands_forward', 'Pass')

if hands_back == 1:
    save_var_latex('hands_back', 'Fail')
else:
    save_var_latex('hands_back', 'Pass')


if torso_bool == 1:
    save_var_latex('torso_bool', 'Pass')
else:
    save_var_latex('torso_bool', 'Fail')

if valgus_l_bool == 0:
    save_var_latex('valgus_l_bool', 'Pass')
else:
    save_var_latex('valgus_l_bool', 'Fail')

if valgus_r_bool == 0:
    save_var_latex('valgus_r_bool', 'Pass')
else:
    save_var_latex('valgus_r_bool', 'Fail')

# make graphs
list_to_graph = [metric_ankle_l, metric_ankle_r, metric_q_angle_l, metric_q_angle_r]
fig, ax = plt.subplots()
fig.set_size_inches(11, 3)

for a in list_to_graph:
    label = a['name'].iloc[0]
    print(label)
    x = a['frame']
    y= a['value']
    poly = np.polyfit(x, y, 20)
    poly_y = np.poly1d(poly)(x)
    ax.plot(x, poly_y, label=str(label))
    fig.set_tight_layout(True)

    #ax.plot(x,y)
    ax.set_xlabel('Frame nr.')
    ax.set_ylabel('Angle')
    fig.suptitle('FMS parameters, front view')

    plt.legend(bbox_to_anchor=(1.04, 0.5), loc= 'center left')
    #fig.legend()
plt.show()
fig.savefig(f'outputs/graphs/{file_itself_front}_graph.png', format='png')


#side view
save_var_latex('lsa', str(metric_sqt_depth.min()[3]))
save_var_latex('sfa_min', str(metric_r_shoulder_angle.min()[3]))
save_var_latex('sfa_max', str(metric_r_shoulder_angle.max()[3]))
save_var_latex('hfa_min', str(metric_hip_angle.min()[3]))
save_var_latex('hfa_max', str(metric_hip_angle.max()[3]))
save_var_latex('dors_kfa_min', str(metric_dorsal_knee.min()[3]))
save_var_latex('dors_kfa_max', str(metric_dorsal_knee.max()[3]))
save_var_latex('afa_min', str(metric_ankle_flexion.min()[3]))
save_var_latex('afa_max', str(metric_ankle_flexion.max()[3]))
save_var_latex('tib_lean_min', str(metric_tibia_lean.min()[3]))
save_var_latex('tib_lean_max', str(metric_tibia_lean.max()[3]))
save_var_latex('torso_lean_min', str(metric_torso_lean.min()[3]))
save_var_latex('torso_lean_max', str(metric_torso_lean.max()[3]))
save_var_latex('torso_fall_pct', str(round(torso_fall_pc,2)))
save_var_latex('hands_forward_angle', str(metric_hands_forward_angle.max()[3]))
save_var_latex('hands_forward_pct', str(round(hands_forward_pc,2)))
save_var_latex('torso_length', '1') #this should always be one actually but we will see
save_var_latex('femur_length', str(femur_length))
save_var_latex('tib_length', str(tib_length))


#front view
save_var_latex('hip_shift', str(hip_shift))
save_var_latex("knee_shift", str(knee_shift))
save_var_latex('q_angle_l_min', str(metric_q_angle_l.min()[3]))
save_var_latex('q_angle_l_max', str(metric_q_angle_l.max()[3]))
save_var_latex('q_angle_r_min',str(metric_q_angle_r.min()[3]))
save_var_latex('q_angle_r_max',str(metric_q_angle_r.max()[3]))
save_var_latex('ankle_l_min', str(metric_ankle_l.min()[3]))
save_var_latex('ankle_l_max', str(metric_ankle_l.max()[3]))
save_var_latex('ankle_r_min', str(metric_ankle_r.min()[3]))
save_var_latex('ankle_r_max', str(metric_ankle_r.max()[3]))
save_var_latex('shoulder_l_min',str(metric_shoulder_angle_l.min()[3]))
save_var_latex('shoulder_l_max',str(metric_shoulder_angle_l.max()[3]))
save_var_latex('shoulder_r_min', str(metric_shoulder_angle_r.min()[3]))
save_var_latex('shoulder_r_max', str(metric_shoulder_angle_r.max()[3]))

##graphs
save_var_latex('front_view_graph', str("\includegraphics[width=.9\linewidth]{../outputs/graphs/" + str(file_itself_front) + "_graph.png}"))


