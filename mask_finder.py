import cv2
import time
import numpy as np


frm=0

def nothing(x): #dummy function
    pass

def coords_mouse_disp(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK: #left mouse double click
         print("Orginal BGR:",frame[x,y])
         print("HSV values:", hsv[x,y])

cap = cv2.VideoCapture('vids/kid.mp4')
fps = int(cap.get(cv2.CAP_PROP_FPS))
print("This is the fps ", fps)

cv2.namedWindow('trackbars')
cv2.createTrackbar("lower hue", 'trackbars', 0, 255, nothing)
cv2.createTrackbar("lower sat", 'trackbars', 22, 255, nothing)
cv2.createTrackbar("lower val", 'trackbars', 100, 255, nothing)
cv2.createTrackbar("upper hue", 'trackbars', 30, 255, nothing)
cv2.createTrackbar("upper sat", 'trackbars', 82, 255, nothing)
cv2.createTrackbar("upper val", 'trackbars', 139, 255, nothing)
while True:
    ret, frame = cap.read()
    #frame = frame[630:700, 0:-1]
    if frame is None:
        break

    if ret == True:
        time.sleep(1 / fps)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos('lower hue', 'trackbars')
    l_s = cv2.getTrackbarPos('lower sat', 'trackbars')
    l_v = cv2.getTrackbarPos('lower val', 'trackbars')

    u_h = cv2.getTrackbarPos('upper hue', 'trackbars')
    u_s = cv2.getTrackbarPos('upper sat', 'trackbars')
    u_v = cv2.getTrackbarPos('upper val', 'trackbars')

    lower_b = np.array([l_h,l_s,l_v])
    upper_b = np.array([u_h,u_s,u_v])

    mask = cv2.inRange(hsv, lower_b, upper_b)
    invert_mask = cv2.bitwise_not(mask)
    res = cv2.bitwise_and(frame, frame, mask=mask)




    cv2.imshow("frame", frame)
    #cv2.imshow("mask", mask)
    cv2.imshow("res", res)
    #cv2.imshow('hsv', hsv)
    #cv2.setMouseCallback("hsv", coords_mouse_disp)
    frm += 1
    filename = 'scr_frm_' + str(frm) + '.png'

    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord('q'):
        break
    if key == ord('c'):  # calls screenshot function when 'c' is pressed
        cv2.imwrite(filename,mask) # or saves it to disk
