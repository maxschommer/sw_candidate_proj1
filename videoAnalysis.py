import numpy as np
import cv2
import time
cap = cv2.VideoCapture('sample_weld_video.mp4')
frameNum = 0
while(1):
    frameNum += 1
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    time.sleep(.1)
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()