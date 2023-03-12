# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 23:11:00 2022

@author: Ivan
"""

import cv2
import depthai as dai
import numpy as np
import dlib
from math import hypot

# Create pipeline
pipeline = dai.Pipeline()

# Define source and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
xoutVideo = pipeline.create(dai.node.XLinkOut)
xoutPreview = pipeline.create(dai.node.XLinkOut)

xoutVideo.setStreamName("video")
xoutPreview.setStreamName("preview")

# Properties
camRgb.setPreviewSize(400, 400)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(True)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# Linking
camRgb.video.link(xoutVideo.input)
camRgb.preview.link(xoutPreview.input)




detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

font=cv2.FONT_HERSHEY_SIMPLEX

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def get_blinking_ratio(eye_points, facial_landmarks ):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    #hor_line = cv2.line(previewFrame.getFrame(), left_point, right_point, (0, 255, 0), 1)
    #ver_line = cv2.line(previewFrame.getFrame(), center_top, center_bottom, (0, 255, 0), 1)
    
    ver_line_length= hypot((center_top[0]- center_bottom[0]),(center_top[1]- center_bottom[1]))
    hor_line_length= hypot((left_point[0]- right_point[0]),(left_point[1]- right_point[1]))
    ratio= hor_line_length/ver_line_length
    return ratio

def get_gaze_ratio(eye_points, facial_landmarks):
    left_eye_region=np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                              (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                              (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                              (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                              (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                              (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
    #cv2.polylines(previewFrame.getFrame(),[left_eye_region],True, (0,0,255),1)
    
    height,width,_=previewFrame.getFrame().shape
    mask=np.zeros((height,width), np.uint8)
    
    cv2.polylines(mask,[left_eye_region],True, 255,1)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye= cv2.bitwise_and(gray,gray,mask=mask)
    
    min_x=np.min(left_eye_region[:,0])
    max_x=np.max(left_eye_region[:,0])
    min_y=np.min(left_eye_region[:,1])
    max_y=np.max(left_eye_region[:,1])
    
    gray_eye=eye[min_y: max_y, min_x: max_x]
    eye=cv2.resize(gray_eye,None,fx=5,fy=5)
    _,threshold_eye = cv2.threshold(eye,35, 255,cv2.THRESH_BINARY)
    #threshold_eye=cv2.resize(threshold_eye,None,fx=5,fy=5)
    height,width=threshold_eye.shape
    left_side_threshold = threshold_eye[0:height, 0:int(width/2)]
    left_side_white=cv2.countNonZero(left_side_threshold)
    
    right_side_threshold = threshold_eye[0:height, int(width/2):width]
    right_side_white=cv2.countNonZero(right_side_threshold)
    
    #gaze_ratio = left_side_white/right_side_white
    print(left_side_white/ right_side_white)
    return left_side_white

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    video = device.getOutputQueue('video')
    preview = device.getOutputQueue('preview')

    while True:
        videoFrame = video.get()
        previewFrame = preview.get()
        gray = cv2.cvtColor(previewFrame.getFrame(), cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for face in faces:
        
            landmarks = predictor(gray, face)
            left_eye_ratio=get_blinking_ratio([36,37,38,39,40,41],landmarks)
            right_eye_ratio=get_blinking_ratio([42,43,44,45,46,47],landmarks)
            blinking_ratio=(left_eye_ratio+right_eye_ratio)/2
# =============================================================================
#             if blinking_ratio>5:
#                 cv2.putText(previewFrame.getFrame(),"BLINKING", (50,150), font, 3,(255,50,50))
#         
# =============================================================================
            gaze_ratio_left_eye=get_gaze_ratio([36,37,38,39,40,41],landmarks)
            gaze_ratio_right_eye=get_gaze_ratio([42,43,44,45,46,47],landmarks)
            
            gaze_ratio=((gaze_ratio_left_eye+gaze_ratio_right_eye)/2)
            #print(gaze_ratio_left_eye)
        # Get BGR frame from NV12 encoded video frame to show with opencv
        #cv2.imshow("video", videoFrame.getCvFrame())
        # Show 'preview' frame as is (already in correct format, no copy is made)
        cv2.imshow("preview", previewFrame.getFrame())

        if cv2.waitKey(1) == ord('q'):
            break
        
cv2.destroyAllWindows()