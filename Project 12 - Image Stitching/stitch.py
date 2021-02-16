import cv2
import numpy as np


class Stitcher:

    def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
        img_1, img_2 = images
        kpts_1, features_1 = self.detectAndDescribe(img_1)
        kpts_2, features_2 = self.detectAndDescribe(img_2)
        M = self.matchKeypoints(kpts_1, kpts_2, features_1, features_2, ratio, reprojThresh)
        if M is None:
            return None
        matches, H, status = M
        result = cv2.warpPerspective(img_1, H, (img_1.shape[1] + img_2.shape[1], img_2.shape[0]))
        self.cv_show('warped', result)
        result[0: img_2.shape[0], 0: img_2.shape[1]] = img_2
        self.cv_show('result', result)
        if showMatches:
            vis = self.drawMatches(img_1, img_2, kpts_1, kpts_2, matches, status)
            return result, vis
        return result

    def detectAndDescribe(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d_SIFT.create()
        kpts, features = sift.detectAndCompute(img_gray, None)
        kpts = np.float32([kpt.pt for kpt in kpts])
        return kpts, features

    def matchKeypoints(self, kpts_1, kpts_2, features_1, features_2, ratio, reprojThresh):
        matcher = cv2.BFMatcher()
        rawMatches = matcher.knnMatch(features_1, features_2, k=2)
        matches = []
        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
        if len(matches) > 4:
            kpts_1_good = np.float32([kpts_1[i] for (_, i) in matches])
            kpts_2_good = np.float32([kpts_2[i] for (i, _) in matches])
            H, status = cv2.findHomography(kpts_1_good, kpts_2_good, cv2.RANSAC, reprojThresh)
            return matches, H, status
        else:
            return None

    def drawMatches(self, img_1, img_2, kpts_1, kpts_2, matches, status):

        (h1, w1) = img_1.shape[:2]
        (h2, w2) = img_2.shape[:2]
        vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype="uint8")
        vis[0:h1, 0:w1] = img_1
        vis[0:h2, w1:] = img_2

        for ((trainIdx, queryIdx), s) in zip(matches, status):
            if s == 1:
                pt1 = (int(kpts_1[queryIdx][0]), int(kpts_1[queryIdx][1]))
                pt2 = (int(kpts_2[trainIdx][0]) + w1, int(kpts_2[trainIdx][1]))
                cv2.line(vis, pt1, pt2, (0, 255, 0), 1)

        return vis

    def cv_show(self, img_name, img):
        cv2.imshow(img_name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
