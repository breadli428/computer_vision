import cv2
import numpy as np
import math


def resize(img, width=None, height=None):
    h, w = img.shape[:2]
    if width is None and height is None:
        return img

    if width is None:
        r = w / float(h)
        width = int(height * r)
        dim = (width, height)
    else:
        r = h / float(w)
        height = int(width * r)
        dim = (width, height)
    resized = cv2.resize(img, dim)
    return resized


def order_points(pts):
    rect = np.zeros((4, 2), dtype='float32')
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def four_point_transform(img, pts):
    rect = order_points(pts)

    width_top = math.hypot((rect[1] - rect[0])[0], (rect[1] - rect[0])[1])
    width_bottom = math.hypot((rect[2] - rect[3])[0], (rect[2] - rect[3])[1])
    max_width = max(int(width_top), int(width_bottom))

    height_left = math.hypot((rect[3] - rect[0])[0], (rect[3] - rect[0])[1])
    height_right = math.hypot((rect[2] - rect[1])[0], (rect[2] - rect[1])[1])
    max_height = max(int(height_left), int(height_right))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype='float32')

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (max_width, max_height))
    return warped


def cv_show(img_name, img):
    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


receipt = cv2.imread('receipt.jpg')
scale = receipt.shape[0] / 500.0
receipt_copy = receipt.copy()

receipt_resized = resize(receipt, height=500)
receipt_gray = cv2.cvtColor(receipt_resized, cv2.COLOR_BGR2GRAY)

receipt_gauss = cv2.GaussianBlur(receipt_gray, (5, 5), 0)
receipt_canny = cv2.Canny(receipt_gauss, 75, 200)
cv_show('canny', receipt_canny)

contours = cv2.findContours(receipt_canny.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
print(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02*peri, True)
    if len(approx) == 4:
        screen_contour = approx
        break

cv2.drawContours(receipt_resized, screen_contour, -1, (0, 255, 0), 2)
cv_show('contour', receipt_resized)

receipt_warped = four_point_transform(receipt_copy, screen_contour.reshape(4, 2) * scale)
cv_show('warped', receipt_warped)

receipt_warped = cv2.cvtColor(receipt_warped, cv2.COLOR_BGR2GRAY)
receipt_bin = cv2.threshold(receipt_warped, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv_show('bin', receipt_bin)

cv2.imwrite('scanned_copy.jpg', receipt_bin)


