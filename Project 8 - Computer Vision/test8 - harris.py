import cv2


def cv_show(img_name, img):
    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


harris = cv2.imread('harris.jpg')
harris = cv2.resize(harris, (750, 750))
gray = cv2.cvtColor(harris, cv2.COLOR_BGR2GRAY)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

harris[dst > 0.01 * dst.max()] = [0, 0, 255]

cv_show('harris', harris)
