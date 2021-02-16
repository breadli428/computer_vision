import cv2
import numpy as np


cap = cv2.VideoCapture('test.avi')

feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7)
lk_params = dict(winSize=(15, 15), maxLevel=2)

color = np.random.randint(0, 255, (100, 3))

ret, frame_0 = cap.read()
frame_0_gray = cv2.cvtColor(frame_0, cv2.COLOR_BGR2GRAY)
pt_0 = cv2.goodFeaturesToTrack(frame_0_gray, mask=None, **feature_params)
mask = np.zeros_like(frame_0)

while True:
    ret, frame_1 = cap.read()
    frame_1_gray = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
    pt_1, status, err = cv2.calcOpticalFlowPyrLK(frame_0_gray, frame_1_gray, pt_0, None, **lk_params)
    good_pt_0 = pt_0[status == 1]
    good_pt_1 = pt_1[status == 1]

    for i, (new, old) in enumerate(zip(good_pt_1, good_pt_0)):
        new = tuple(new)
        old = tuple(old)
        mask = cv2.line(mask, new, old, color[i].tolist(), 2)
        frame_1 = cv2.circle(frame_1, new, 5, color[i].tolist(), -1)
    img = cv2.add(frame_1, mask)
    cv2.imshow('img', img)

    k = cv2.waitKey(150) & 0xff
    if k == 27:
        break
    frame_0_gray = frame_1_gray.copy()
    pt_0 = good_pt_1.reshape(-1, 1, 2)

cap.release()
cv2.destroyAllWindows()







