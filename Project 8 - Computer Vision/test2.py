import cv2
import matplotlib.pyplot as plt

img = cv2.imread("cat.jpg")
top_size, bottom_size, left_size, right_size = (50, 50, 50, 50)
replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT101)
wrap = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_WRAP)
constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_CONSTANT, value=0)


plt.subplot(231), plt.imshow(img)
plt.subplot(232), plt.imshow(replicate)
plt.subplot(233), plt.imshow(reflect)
plt.subplot(234), plt.imshow(reflect101)
plt.subplot(235), plt.imshow(wrap)
plt.subplot(236), plt.imshow(constant)

plt.show()

