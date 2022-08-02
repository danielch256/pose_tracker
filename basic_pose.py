import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture('vids/kid.mp4')
pTime = 0
cap_fps = cap.get(cv2.CAP_PROP_FPS)
print("capture fps is:" + str(cap_fps))
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    lmList = []
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = img.shape
            cx, cy = int(lm.x*w), int(lm.y*h) # get pixel values instead of ratio of picture width
            #print(id, cx,cy)
            lmList.append([id,cx,cy]) # list of id, x and y coords of all 33 landmarks
            cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED) #draw blue dots
        cv2.circle(img, (lmList[14][1], lmList[14][2]), 5, (0, 255, 0), cv2.FILLED)  # draw green dot on landmark 14
    print(lmList[14][1],lmList[14][2]) # print x,y coords of landmark 14
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255,0,0), 3)
    cv2.imshow("Image", img)


    # Press 'q' to quit
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, end loop
    if key == ord('q'):
        break