
import cv2 as cv
import numpy as np

def CameraClib(imgs):
    pattern_h = 11
    pattern_w = 11

    per_objpoints = np.zeros([pattern_h * pattern_w, 3])
    per_objpoints[:, :2] = np.mgrid[:pattern_h, :pattern_w] \
                           .transpose().reshape([-1, 2])

    objpoints = list()
    imgpoints = list()

    for img in imgs:
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

        ret, corners = cv.findChessboardCorners(
            gray, [pattern_h, pattern_w], None)

        if ret == False:
            continue

        objpoints.append(per_objpoints)

        corners = cv.cornerSubPix(gray, corners, [11, 11], [-1, -1],
            [cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 40, 0.001])

        imgpoints.append(corners)

        uimg = cv.UMat(img)
        cv.drawChessboardCorners(uimg, [pattern_h, pattern_w], corners, True)
        cv.imshow(uimg)
        cv.waitKey(45)

    cv.destryoAllWindows()

    print(f"gray.shape = {gray.shape}")

    ret, camera_mat, camera_distort, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[:-1], None, None)

def main():
    imgs = list()

    # for i in range()

    CameraClib([])


    # cv.findChessboardCorners(gray )

if __name__ == "__main__":
    main()
