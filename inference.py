#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 23:06:50 2022

@author: 21059299
"""

import cv2
from time import sleep
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np


model = tf.keras.models.load_model('/Users/21059299/model.h5')

video_capture = cv2.VideoCapture(0)

labels = ['A','B','C','D','E','F',
          'G','H','I','J','K','L',
          'M','N','O','P','Q','R',
          'S','T','U','V','W','X',
          'Y','Z']

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    
    ret, frame = video_capture.read()
    frame = cv2.resize(frame,(256,256))
    
    img = cv2.resize(frame,(128,128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype("float") / 255.0
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    result = model.predict(img)
    label = labels[np.argmax(result[0])]
    print(label)
    
    cv2.putText(img=frame, text=label, org=(0, 80), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(0, 255, 0),thickness=3)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow('Video', frame)


video_capture.release()
cv2.destroyAllWindows()