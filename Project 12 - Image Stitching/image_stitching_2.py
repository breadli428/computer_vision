from stitch import Stitcher
import cv2

img_1 = cv2.imread('co_1.jpg')
img_2 = cv2.imread('co_2.jpg')

stitcher = Stitcher()

result, vis = stitcher.stitch([img_1, img_2], showMatches=True)

cv2.imshow('img_1', img_1)
cv2.imshow('img_2', img_2)
cv2.imshow('Key point matches', vis)
cv2.imshow('result', result)
cv2.imwrite('co_stitched.jpg', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

