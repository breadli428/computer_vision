import cv2
import numpy as np
import os.path
import glob


def cv_show(img_name, img):
    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


template = {}
for jpgfile in glob.glob(r'./templates/*.jpg'):
    key = os.path.basename(jpgfile)
    key = os.path.splitext(key)[0]
    template_digit = cv2.imread(jpgfile)
    template_digit = cv2.resize(template_digit, (57, 88))
    template_gray = cv2.cvtColor(template_digit, cv2.COLOR_BGR2GRAY)
    template[key] = cv2.threshold(template_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

plate = cv2.imread('plate.jpg')
plate_identity = plate.copy()
cv_show('plate', plate)

plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
cv_show('gray', plate_gray)

rectkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
sqkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


plate_sobelx = cv2.Sobel(plate_gray, cv2.CV_32F, 1, 0, ksize=3)
plate_sobelx = np.absolute(plate_sobelx)

minVal, maxVal = (np.min(plate_sobelx), np.max(plate_sobelx))
plate_gradx = 255 * (plate_sobelx - minVal) / (maxVal - minVal)
plate_gradx = plate_gradx.astype('uint8')

plate_sobely = cv2.Sobel(plate_gray, cv2.CV_32F, 0, 1, ksize=3)
plate_sobely = np.absolute(plate_sobely)

minVal, maxVal = (np.min(plate_sobely), np.max(plate_sobely))
plate_grady = 255 * (plate_sobely - minVal) / (maxVal - minVal)
plate_grady = plate_grady.astype('uint8')

plate_grad = cv2.addWeighted(plate_gradx, 0.5, plate_grady, 0.5, 1)

cv_show('gradx', plate_grad)


plate_bin = cv2.threshold(plate_grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv_show('bin', plate_bin)


contours = cv2.findContours(plate_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
cv2.drawContours(plate, contours, -1, (0, 0, 255), 2)
cv_show('plate', plate)

loc = []
for i, plate_cnt in enumerate(contours):
    x, y, w, h = cv2.boundingRect(plate_cnt)
    ar = w / float(h)
    if 2 < ar < 4 and 500 < w < 800:
        loc.append((x, y, w, h))
        pt1 = (x - 5, y - 5)
        pt2 = (x + w + 5, y + h - 5)
        cv2.rectangle(plate, pt1, pt2, (0, 255, 0), 2)
        cv2.rectangle(plate_identity, pt1, pt2, (0, 255, 0), 2)
loc = sorted(loc, key=lambda x: x[1])
cv_show('plate', plate)

for i, (xi, yi, wi, hi) in enumerate(loc):
    digits_loc = []
    group = plate[yi + 10:yi + hi - 10, xi + 10:xi + wi - 10]
    group_gray = cv2.cvtColor(group, cv2.COLOR_BGR2GRAY)
    cv_show('group_gray', group_gray)

    group_bin = cv2.threshold(group_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv_show('group_bin', group_bin)

    contours = cv2.findContours(group_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    BoundingBoxes = [cv2.boundingRect(each) for each in contours]
    BoundingBoxes, contours = zip(*sorted(zip(BoundingBoxes, contours)))
    cv2.drawContours(group, contours, -1, (0, 0, 255), 2)
    cv_show('group_contours', group)
    for i, digit_cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(digit_cnt)
        ar = w / float(h)
        if 0.4 < ar < 0.6:
            digits_loc.append((x, y, w, h))
            pt1 = (x - 5, y - 5)
            pt2 = (x + w + 5, y + h + 5)
            cv2.rectangle(group, pt1, pt2, (0, 255, 0), 2)
    group_output = []
    for i, (x, y, w, h) in enumerate(digits_loc):
        roi = group_bin[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))
        scores = []
        keys = []

        for j, template_roi in template.items():

            result = cv2.matchTemplate(roi, template_roi, cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            keys.append(j)
            scores.append(score)
        key_max = np.argmax(scores)
        identity = keys[key_max]
        cv2.putText(plate_identity, identity, (x + 30, yi + 40), cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 255, 0), 2)
        group_output.append(identity)
    print("License Plate Number : {}".format("".join(group_output)))

cv_show('plate_identity', plate_identity)

