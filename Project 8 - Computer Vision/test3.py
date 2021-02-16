import cv2
import numpy as np

img = cv2.imread("cat.jpg", cv2.IMREAD_GRAYSCALE)

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobelx_abs = cv2.convertScaleAbs(sobelx)

sobely = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely_abs = cv2.convertScaleAbs(sobely)

sobelxy = cv2.addWeighted(sobelx_abs, 0.5, sobely_abs, 0.5, 0)

cv2.imshow('result', sobelxy)
cv2.waitKey(0)
cv2.destroyAllWindows()

