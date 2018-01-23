#!/usr/bin/env python

'''
example to show optical flow
USAGE: opt_flow.py [<video_source>]
Keys:
 1 - toggle HSV flow visualization
 2 - toggle glitch
Keys:
    ESC    - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
from OpticalFlow import video

trailAvg = -30
avgSize = 10
def preProcess(image):
    res = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # print(res.mean())
    res = cv.equalizeHist(res)
    # gray = sharpen(gray)
    return res


def draw_flow(img, flow, trailAvg,  step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    fxSort = np.sort(fx)
    trailAvg = trailAvg * (avgSize - 1)/avgSize + np.mean(fxSort[0:10])/avgSize
    print(trailAvg)

    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis, trailAvg


if __name__ == '__main__':
    optMinX = 0
    optMaxX = 700
    optMinY = 0
    optMaxY = 2000

    cam = cv.VideoCapture("sample_weld_video.mp4")
    ret, prev = cam.read()

    prev = prev[optMinY:optMaxY, optMinX:optMaxX]

    prevgray = preProcess(prev)
    show_hsv = False
    show_glitch = False
    cur_glitch = prev.copy()

    while True:
        ret, img = cam.read()
        gray = preProcess(img)
        gray = gray[optMinY:optMaxY, optMinX:optMaxX]
        unProcessed = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        unProcessed = unProcessed[optMinY:optMaxY, optMinX:optMaxX]
        flow = cv.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prevgray = gray
        vis, trailAvg = draw_flow(gray, flow, trailAvg)
        print(vis.shape)
        cv.imshow('flow', np.hstack((vis, unProcessed)))

        ch = cv.waitKey(5)
        if ch == 27:
            break
    cv.destroyAllWindows()