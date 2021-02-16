import cv2


trackers = cv2.MultiTracker_create()
vc = cv2.VideoCapture('./videos/los_angeles.mp4')

while True:
    ret, frame = vc.read()
    if frame is None:
        break
    h, w = frame.shape[:2]
    width = 600
    ar = w / float(h)
    height = int(width / ar)
    dim = (width, height)
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    success, boxes = trackers.update(frame)

    for box in boxes:
        x, y, w, h = [int(d) for d in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    k = cv2.waitKey(50) & 0xFF

    if k == ord('s'):
        box = cv2.selectROI('frame', frame, showCrosshair=False, fromCenter=False)
        tracker = cv2.TrackerKCF_create()
        trackers.add(tracker, frame, box)

    elif k == 27:
        break

vc.release()
cv2.destroyAllWindows()
