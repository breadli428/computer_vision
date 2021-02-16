import cv2
import numpy as np

img = cv2.imread("cat.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


img_draw = img.copy()
res = cv2.drawContours(img_draw, contours, -1, (0, 0, 255), 2)

cv2.imshow('result', res)
cv2.waitKey(0)
cv2.destroyAllWindows()

