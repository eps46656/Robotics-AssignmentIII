
import cv2 as cv
import numpy as np
import os
import random

DIR = os.path.dirname(__file__).replace("\\", "/")

def CameraClib(imgs):
    pattern_h = 8
    pattern_w = 6

    per_objpoints = np.zeros([pattern_h * pattern_w, 3], dtype=np.float32)
    per_objpoints[:, :2] = np.mgrid[:pattern_h, :pattern_w] \
                           .transpose().reshape([-1, 2])

    objpoints = list()
    imgpoints = list()

    for img in imgs:
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

        ret, corners = cv.findChessboardCorners(
            gray, (pattern_h, pattern_w), None)

        if not ret:
            continue

        objpoints.append(per_objpoints)

        corners = cv.cornerSubPix(gray, corners, [11, 11], [-1, -1],
            [cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 40, 0.001])

        imgpoints.append(corners)

        # uimg = cv.UMat(img)
        # cv.drawChessboardCorners(uimg, [pattern_h, pattern_w], corners, True)
        # cv.imshow("uimg", uimg)
        # cv.waitKey(45)

    # cv.destroyAllWindows()

    ret, camera_mat, camera_distort, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    return camera_mat, camera_distort

def main():
    imgs = list()

    # for i in range()

    num_of_frames = 1211

    frame_idxes = sorted(random.choices(range(num_of_frames), k=64))

    print(f"frame_idxes = {frame_idxes}")

    for frame_i in frame_idxes:
        img = cv.imread(f"{DIR}/vedio_frames/frame-{frame_i}.png")

        imgs.append(img)

    camera_mat, camera_distort = CameraClib(imgs)

    print("camera_mat = ")
    print(camera_mat)
    print("camera_distort = ")
    print(camera_distort)

    np.save(f"{DIR}/camera_params.npy", {"camera_mat": camera_mat,
                                         "camera_distort": camera_distort})

if __name__ == "__main__":
    main()
