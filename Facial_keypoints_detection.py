#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 02:41:39 2020

@author: nagaraj
"""

import cv2
import numpy as np
from time import sleep

refPt = []
num_points = int(input('Enter the number of points to track: '))
landmarks = np.zeros((num_points,2),dtype='float32')

cap = cv2.VideoCapture(0)

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 4,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20000, 0.001))

def click(event, x, y, flags, param):
    # grab references to the global variables
    global refPt
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        print('clicked')
        refPt.append((int(x), int(y)))
    
    # draw a rectangle around the region of interest
    for x,y in refPt:
        cv2.circle(frame,(x,y),1,(0, 0, 255), 3)
        cv2.imshow("image", frame)
        
while True:
    ret, frame = cap.read()
    img_x = frame.copy()
    img_x = cv2.cvtColor(img_x,cv2.COLOR_BGR2GRAY)
    cv2.namedWindow("frame")
    cv2.setMouseCallback("frame", click, param = num_points)
    
    cv2.imshow('frame',frame)
    
    if(len(refPt)==num_points):
        landmarks = np.array(refPt,dtype='float32')
        cv2.destroyWindow('frame')
        cap.release()
        print('Exiting if')
        cap.open(0)
        while True:
            ret,new_frame = cap.read()
            #print(ret)
            #cv2.imshow('new_frame',new_frame)
            frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
            p1, st, err = cv2.calcOpticalFlowPyrLK(img_x, frame_gray, landmarks, None, **lk_params)
            tracked_points = p1.reshape((num_points,2))
            # Indicate the tracked points
            for i,(x,y) in enumerate(tracked_points):
                #print('xxxxx')
                new_frame = cv2.circle(new_frame,(x,y),1,(0,0,255),4)

            cv2.imshow('Landmarks tracking',new_frame)

            # Now update the previous frame and previous points
            img_x = frame_gray.copy()
            landmarks = p1.reshape(-1,1,2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                cap.release()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        cap.release()
        
    