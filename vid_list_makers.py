import os
import pandas as pd
import csv
import pickle


# front view walker

path_front = 'C:/Users/danny/PycharmProjects/pose/vids/cfbs/sorted/front'
vid_list_front = []

for root, dirs, files in os.walk(path_front):
    for file in files:
        # append the file name to the list
        vid_list_front.append(os.path.join(root,file))

# print all the file names
for name in vid_list_front:
    print(name)
print('nr. of front perspective vids: ' + str(len(vid_list_front)))

#side view walker

path_side = 'C:/Users/danny/PycharmProjects/pose/vids/cfbs/sorted/side'
vid_list_side = []

for root, dirs, files in os.walk(path_side):
    for file in files:
        # append the file name to the list
        vid_list_side.append(os.path.join(root,file))

# print all the file namesx
for name in vid_list_side:
    print(name)
print('nr. of side perspective vids: ' + str(len(vid_list_side)))


# save lists as pickles

with open("pickles/front_vids", "wb") as fp:   #Pickling
    pickle.dump(vid_list_front, fp)

with open("pickles/side_vids", "wb") as fp:   #Pickling
    pickle.dump(vid_list_side, fp)
