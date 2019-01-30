import cv2
from cnn_model import build_tflearn_convnet_1
import numpy as np

# loading the trained model
MODEL = build_tflearn_convnet_1()
MODEL.load('./cnn_model_save/CCN')
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprised", "neutral"]

# loading the haar file for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        test_file = img[y:y + h, x:x + w]
        test_file = cv2.resize(test_file, (48, 48))
        test_file = cv2.cvtColor(test_file, cv2.COLOR_BGR2GRAY) / 255.0
        label = EMOTIONS[int(np.argmax(MODEL.predict(test_file.reshape([-1, 48, 48, 1]))))]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, label, (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow('img', img)
    cv2.waitKey(30)

# https://docs.opencv.org/3.4/d7/d8b/tutorial_py_face_detection.html
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
# https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
