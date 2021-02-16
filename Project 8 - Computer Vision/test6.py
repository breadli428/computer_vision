import cv2
import numpy as np

img_sm = cv2.imread("sm.jpg")
img_mon = cv2.imread("mon.jpg")
img_sm_gray = cv2.cvtColor(img_sm, cv2.COLOR_BGR2GRAY)

h, w = img_mon.shape[0:2]

res = cv2.matchTemplate(img_sm, img_mon, cv2.TM_CCOEFF_NORMED)

threshold = 0.8
loc = np.where(res > threshold)

for pt in zip(*loc[::-1]):
    corner = (pt[0] + w, pt[1] + h)
    cv2.rectangle(img_sm, pt, corner, (0, 0, 255), 2)

cv2.imshow('result', img_sm)
cv2.waitKey(0)
cv2.destroyAllWindows()
