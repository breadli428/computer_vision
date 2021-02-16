import cv2
import numpy as np


def cv_show(img_name, img):
    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


template = cv2.imread('template.png')
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
ret, template_bin = cv2.threshold(template_gray, 127, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(template_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(template, contours, -1, (0, 0, 255), 2)

BoundingBoxes = [cv2.boundingRect(each) for each in contours]
BoundingBoxes, contours = zip(*sorted(zip(BoundingBoxes, contours)))
digits = {}
for index, template_cnt in enumerate(contours):
    x, y, w, h = cv2.boundingRect(template_cnt)
    pt1 = (x, y)
    pt2 = (x + w, y + h)
    cv2.rectangle(template, pt1, pt2, (0, 0, 255), 2)
    roi = template_gray[y:y + h, x:x + w]
    roi = cv2.resize(roi, (57, 88))
    digits[index] = roi

cv_show('template', template)

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

card = cv2.imread('card.png')
card = cv2.resize(card, (250, 200))
card_gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
card_tophat = cv2.morphologyEx(card_gray, cv2.MORPH_TOPHAT, rectKernel)

card_sobelx = cv2.Sobel(card_tophat, cv2.CV_32F, 1, 0, ksize=3)
card_sobelx = np.absolute(card_sobelx)

minVal, maxVal = (np.min(card_sobelx), np.max(card_sobelx))
card_gradx = 255 * (card_sobelx - minVal) / (maxVal - minVal)

card_gradx = card_gradx.astype('uint8')


card_close = cv2.morphologyEx(card_gradx, cv2.MORPH_CLOSE, rectKernel)

ret, card_bin = cv2.threshold(card_close, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
card_close2 = cv2.morphologyEx(card_bin, cv2.MORPH_CLOSE, sqKernel)

contours, hierarchy = cv2.findContours(card_close2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
card_copy = card.copy()
cv2.drawContours(card_copy, contours, -1, (0, 0, 255), 2)

loc = []
for index, template_cnt in enumerate(contours):
    x, y, w, h = cv2.boundingRect(template_cnt)
    ar = w / float(h)
    if 2.5 < ar < 4.0:
        if (40 < w < 55) and (10 < h < 20):
            loc.append((x, y, w, h))
loc = sorted(loc)
cv_show('card', card_copy)

output = []
for i, (xi, yi, wi, hi) in enumerate(loc):
    group_output = []
    group = card_gray[yi - 5:yi + hi + 5, xi - 5:xi + wi + 5]
    # cv_show('card_gray', card_gray)
    # cv_show('group', group)
    ret, group_bin = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv_show('group_bin', group_bin)
    contours, hierarchy = cv2.findContours(group_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    BoundingBoxes = [cv2.boundingRect(each) for each in contours]
    BoundingBoxes, contours = zip(*sorted(zip(BoundingBoxes, contours)))
    for index, group_cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(group_cnt)
        roi = group_bin[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))
        # cv_show('roi', roi)

        scores = []
        for index, digits_roi in digits.items():
            # cv_show('digits_roi', digits_roi)
            result = cv2.matchTemplate(roi, digits_roi, cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(np.absolute(result))
            scores.append(score)
        group_output.append(str(np.argmax(scores)))
    output.extend(group_output)
    pt1 = (xi - 2, yi - 2)
    pt2 = (xi + wi + 2, yi + hi + 2)
    cv2.rectangle(card, pt1, pt2, (0, 255, 0), 1)
    cv2.putText(card, "".join(group_output), (xi, yi - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv_show('card', card)
print("Credit Card #: {}".format("".join(output)))









