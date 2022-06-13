# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 01:11:44 2019

@author: vidya
"""

import pickle
import numpy as np
import cv2
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import time
from datetime import datetime

#face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade.load('haarcascade_frontalface_default.xml')
with open('emotion_detector_NN', 'rb') as f:
    model = pickle.load(f)

def detect_emotion(frames):
    frm_Norm = frames / 255.0
    #print(frm_Norm.shape, frm_Norm[0:1].shape)
    emotion = model.predict(frm_Norm, batch_size = 20)
    
    k = emotion[0].max()
    if (emotion[0][0] == k):
        return "Neutral", emotion[0]
    elif (emotion[0][1] == k):
        return "Sad", emotion[0]
    elif (emotion[0][2] == k):
        return "Happy", emotion[0]
        
def capturePicture():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Emotion Detection")
    frame_cnt = 0
    test_pos = (0, 0)
    emotion = 'Invalid'
    
    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_cnt += 1;        
        
        if not ret:
            break
        
        k = cv2.waitKey(1)

        if k % 256 == 27:
       
            print("Escape hit, closing...")
            break
        else:
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                cv2.putText(frame, emotion, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 4)
               
                fceImage = cv2.resize(gray[y:y + h, x:x + w], (64, 64))
                faceimage = np.asarray(fceImage)
                fi = np.reshape(faceimage, (-1, 4096))
                start = datetime.now()
                emotion, emotion_vector = detect_emotion(fi)
                print('Time:', datetime.now() - start)
                
        cv2.imshow("Emotion Detection", frame)
        
    cam.release()
    cv2.destroyWindow("Emotion Detection")
    
capturePicture()
            
