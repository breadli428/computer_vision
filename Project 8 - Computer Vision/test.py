import cv2
import matplotlib.pyplot as plt
import numpy

vc = cv2.VideoCapture("test_video.mp4")
while vc.isOpened():
    ret, frame = vc.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('result', gray)
        if cv2.waitKey(10) & 0xFF == 27:
            break
vc.release()
cv2.destroyAllWindows()
