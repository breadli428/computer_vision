import cv2
import numpy as np


img_1 = cv2.imread('image_1.jpg')
img_2 = cv2.imread('image_2.jpg')

img_1 = cv2.resize(img_1, (int(img_1.shape[1] / 2), int(img_1.shape[0] / 2)))
img_2 = cv2.resize(img_2, (int(img_2.shape[1] / 2), int(img_2.shape[0] / 2)))

img_1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
img_2_gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d_SIFT.create()

kp1, ftr1 = sift.detectAndCompute(img_1_gray, None)
kp2, ftr2 = sift.detectAndCompute(img_2_gray, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(ftr1, ftr2, k=2)

match_idx = []
good = []
for m, n in matches:
    if m.distance < n.distance * 0.75:
        good.append([m])
        match_idx.append((m.trainIdx, m.queryIdx))

img_matched = cv2.drawMatchesKnn(img_1, kp1, img_2, kp2, good[:10], None, flags=2)
cv2.imshow('img_matched', img_matched)


kp1 = np.float32([kp.pt for kp in kp1])
kp2 = np.float32([kp.pt for kp in kp2])

if len(match_idx) > 4:
    kpts1 = np.float32([kp1[i] for (_, i) in match_idx])
    kpts2 = np.float32([kp2[i] for (i, _) in match_idx])
    H, status = cv2.findHomography(kpts1, kpts2, cv2.RANSAC, ransacReprojThreshold=4.0)

    result = cv2.warpPerspective(img_1, H, (img_1.shape[1] + img_2.shape[1], img_2.shape[0]))
    cv2.imshow('warped', result)

    result[0: img_2.shape[0], 0: img_2.shape[1]] = img_2
    cv2.imshow('result', result)

cv2.waitKey(0)
cv2.destroyAllWindows()




