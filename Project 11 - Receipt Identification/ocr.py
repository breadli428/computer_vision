import cv2
import pytesseract


def cv_show(img_name, img):
    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


scanned_copy = cv2.imread('scanned_copy.jpg')
scanned_copy_gray = cv2.cvtColor(scanned_copy, cv2.COLOR_BGR2GRAY)

scanned_copy_bin = cv2.threshold(scanned_copy_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

cv_show('scanned_copy_bin', scanned_copy_bin)
text = pytesseract.image_to_string(scanned_copy_bin)
print(text)
