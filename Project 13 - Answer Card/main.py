import cv2
import numpy as np
import math

ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}


def cv_show(img_name, img):
    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def order_points(vertex):
    rect = np.zeros((4, 2), dtype='float32')
    s = vertex.sum(axis=1)
    rect[0] = vertex[np.argmin(s)]
    rect[2] = vertex[np.argmax(s)]
    d = np.diff(vertex, axis=1)
    rect[1] = vertex[np.argmin(d)]
    rect[3] = vertex[np.argmax(d)]
    return rect


def four_point_transform(img, vertex):

    rect = order_points(vertex)
    tl, tr, br, bl = rect
    tw = math.hypot(tr[0] - tl[0], tr[1] - tl[1])
    bw = math.hypot(br[0] - bl[0], br[1] - bl[1])
    w = max(int(tw), int(bw))
    lh = math.hypot(bl[0] - tl[0], bl[1] - tl[1])
    rh = math.hypot(br[0] - tr[0], br[1] - tr[1])
    h = max(int(lh), int(rh))
    dst = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]], dtype='float32')
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (w, h))
    return warped


def sort_contours(cnt, method='left-to-right'):
    reverse = False
    i = 0
    if method == 'right-to-left' or method == 'bottom-to-top':
        reverse = True
    if method == 'top-to-bottom' or method == 'bottom-to-top':
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnt]
    cnt, boundingBoxes = zip(*sorted(zip(cnt, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    return cnt, boundingBoxes


card = cv2.imread('./images/test_01.png')
cv_show('card', card)

card_cnt = card.copy()
card_gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
card_blurred = cv2.GaussianBlur(card_gray, (5, 5), 0)
cv_show('blurred', card_blurred)

card_edge = cv2.Canny(card_blurred, 75, 200)
cv_show('edged', card_edge)

contours = cv2.findContours(card_edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
cv2.drawContours(card_cnt, contours, -1, (0, 0, 255), 3)
cv_show('contours', card_cnt)

cnt_vertex = None
if len(contours) > 0:
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            cnt_vertex = approx
            break

card_warped = four_point_transform(card_gray, cnt_vertex.reshape(4, 2))
card_warped_BGR = four_point_transform(card, cnt_vertex.reshape(4, 2))

cv_show('warped', card_warped)

card_thresh = cv2.threshold(card_warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cv_show('thresh', card_thresh)

card_thresh_cnt = card_thresh.copy()
thresh_contours = cv2.findContours(card_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
cv2.drawContours(card_thresh_cnt, thresh_contours, -1, (0, 0, 255), 3)
cv_show('thresh_contours', card_thresh_cnt)

letter_cnt = []
for c in thresh_contours:
    x, y, w, h = cv2.boundingRect(c)
    ar = w / float(h)
    if w >= 20 and h >= 20 and 0.9 <= ar <= 1.1:
        letter_cnt.append(c)

letter_cnt = sort_contours(letter_cnt, method='top-to-bottom')[0]

correct = 0
for q, i in enumerate(np.arange(0, len(letter_cnt), 5)):
    q_cnt = sort_contours(letter_cnt[i: i + 5])[0]
    bubbled = None
    for j, c in enumerate(q_cnt):
        mask = np.zeros(card_thresh.shape, dtype='uint8')
        cv2.drawContours(mask, [c], -1, 255, -1)
        # cv_show('mask', mask)

        mask = cv2.bitwise_and(card_thresh, card_thresh, mask=mask)
        # cv_show('mask', mask)

        total = cv2.countNonZero(mask)
        if bubbled is None or total > bubbled[0]:
            bubbled = (total, j)

    color = (0, 0, 255)
    k = ANSWER_KEY[q]
    if k == bubbled[1]:
        color = (0, 255, 0)
        correct = correct + 1
    cv2.drawContours(card_warped_BGR, [q_cnt[k]], -1, color, 3)

score = correct / 5.0 * 100
cv2.putText(card_warped_BGR, "{:.2f}".format(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.imshow('Original', card)
cv2.imshow('Exam', card_warped_BGR)
cv2.waitKey(0)
cv2.destroyAllWindows()
