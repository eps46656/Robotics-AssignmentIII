
import cv2 as cv
import numpy as np
import os
import glob

DIR = os.path.dirname(__file__).replace("\\", "/")

def CameraCalib(imgs):
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

    ret, camera_mat, camera_distort, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    return camera_mat, camera_distort

def UndistortImages(imgs, camera_mat, camera_distort):
    undistorted_imgs = list()

    for img in imgs:
        h, w = img.shape[:2]

        new_camera_mat, roi = cv.getOptimalNewCameraMatrix(
            camera_mat, camera_distort, (w, h), 1, (w, h))

        undistorted_img = cv.undistort(
            img, camera_mat, camera_distort, None, new_camera_mat)

        x, y, w, h = roi

        undistorted_img = undistorted_img[y:y+h, x:x+w]
        undistorted_imgs.append(undistorted_img)

    return undistorted_imgs

def main():
    dir1 = f"frames"
    dir2 = f"frames_extra"

    dir = dir1
    path = f"{DIR}/{dir}/frame-*.png"

    filenames = glob.glob(path)

    imgs = [cv.imread(filename) for filename in filenames]

    camera_mat, camera_distort = CameraCalib(imgs)

    print(f"camera_mat =\n{camera_mat}")
    print(f"camera_distort =\n{camera_distort}")

    np.save(f"{DIR}/camera_params.npy", {"camera_mat": camera_mat,
                                         "camera_distort": camera_distort})

    undistorted_imgs = UndistortImages(imgs, camera_mat, camera_distort)

    for filename, undistorted_img in zip(filenames, undistorted_imgs):
        filename = f"{DIR}/undistorted_{dir}/undistorted-{os.path.basename(filename)}"

        print(filename)

        cv.imwrite(filename, undistorted_img)

if __name__ == "__main__":
    main()
