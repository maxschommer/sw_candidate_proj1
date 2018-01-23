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


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return bgr


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv.remap(img, flow, None, cv.INTER_LINEAR)
    return res

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
        if show_hsv:
            cv.imshow('flow HSV', draw_hsv(flow))
        if show_glitch:
            cur_glitch = warp_flow(cur_glitch, flow)
            cv.imshow('glitch', cur_glitch)

        ch = cv.waitKey(5)
        if ch == 27:
            break
        if ch == ord('1'):
            show_hsv = not show_hsv
            print('HSV flow visualization is', ['off', 'on'][show_hsv])
        if ch == ord('2'):
            show_glitch = not show_glitch
            if show_glitch:
                cur_glitch = img.copy()
            print('glitch is', ['off', 'on'][show_glitch])
    cv.destroyAllWindows()