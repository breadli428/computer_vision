import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("cat.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
mask = np.zeros(img.shape, np.uint8)
mask[100:300, 200:400] = 255
cv2.imshow('mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
img_masked = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow('img_masked', img_masked)
cv2.waitKey(0)
cv2.destroyAllWindows()
hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
hist_masked = cv2.calcHist([img_masked], [0], mask, [256], [0, 256])


plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask, 'gray')
plt.subplot(223), plt.imshow(img_masked, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_masked)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()




