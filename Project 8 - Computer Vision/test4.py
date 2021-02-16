import cv2
import numpy as np

img = cv2.imread("cat.jpg", cv2.IMREAD_GRAYSCALE)
v1 = cv2.Canny(img, 200, 300)
v2 = cv2.Canny(img, 50, 100)
res = np.hstack((v1, v2))
cv2.imshow('result', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
