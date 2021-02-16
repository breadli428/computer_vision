import cv2


def cv_show(img_name, img):
    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread('cat.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d_SIFT.create()
kp = sift.detect(gray, None)

img = cv2.drawKeypoints(gray, kp, img)
cv_show('img', img)

kp, des = sift.compute(gray, kp)
