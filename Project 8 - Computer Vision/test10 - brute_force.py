import cv2


def cv_show(img_name, img):
    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img1 = cv2.imread('img1.png')
img2 = cv2.imread('img2.png')

img1 = cv2.resize(img1, (800, 400))
img2 = cv2.resize(img2, (800, 400))


img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d_SIFT.create()

kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

bf_1 = cv2.BFMatcher(crossCheck=True)
matches_1 = bf_1.match(des1, des2)
matches_1 = sorted(matches_1, key=lambda x: x.distance)

img_matched_1 = cv2.drawMatches(img1_gray, kp1, img2_gray, kp2, matches_1[:10], None, flags=2)
cv_show('img_matched_1', img_matched_1)

bf_2 = cv2.BFMatcher()
matches_2 = bf_2.knnMatch(des1, des2, k=2)

good = []
for m, n in matches_2:
    if m.distance < 0.75 * n.distance:
        good.append([m])

img_matched_2 = cv2.drawMatchesKnn(img1_gray, kp1, img2_gray, kp2, good[:10], None, flags=2)
cv_show('img_matched_2', img_matched_2)
