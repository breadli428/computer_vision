import cv2

cap = cv2.VideoCapture('test.avi')

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    fgmask_open = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    contours = cv2.findContours(fgmask_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        if peri > 188:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    cv2.imshow('fgmask', fgmask_open)
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
