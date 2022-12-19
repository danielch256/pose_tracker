import cv2
import pathlib
import os
path = 'C:/Users/danny/Desktop/unblur/rotate'
filepath = pathlib.Path(path).parent
vid_list = []

#check the chosen path and add all files to list
for root, dirs, files in os.walk(path):
    for file in files:
        vid_list.append(os.path.join(root,file))

for vid in vid_list:
    filename = pathlib.Path(vid).stem
    cap = cv2.VideoCapture(vid)
    frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get width and height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #write rotated video: remember to flip height and width if its not 180deg
    newvideoR = cv2.VideoWriter(str(path) + '/done/' + str(filename) + '.mp4', cv2.VideoWriter_fourcc(*"mp4v"), 50, (frame_height, frame_width))

    # Original Frames
    for i in range(frame_number):
        ret, frame = cap.read()

        # do the rotation
        new = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        cv2.imshow('output', new)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        newvideoR.write(new)

    newvideoR.release()
    cap.release()