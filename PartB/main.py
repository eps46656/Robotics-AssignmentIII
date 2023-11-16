
import cv2 as cv
import math
import numpy as np
import os
import sys

DIR = os.path.dirname(__file__).replace("\\", "/")

def FindObjs(img):
    # img[H, W, 1]

    assert len(img.shape) == 3

    H, W, C = img.shape

    if 1 < C:
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    threshold = 196

    img = cv.medianBlur(img, 11)

    m = np.zeros([H, W], dtype=np.int32)
    obj_i = 0

    q = list()

    dx = [1, 0, -1, 0]
    dy = [0, 1, 0, -1]

    for i in range(H):
        for j in range(W):
            if 0 < m[i, j] or img[i, j] < threshold:
                continue

            obj_i += 1
            m[i, j] = obj_i
            q.append((i, j))

            while 0 < len(q):
                cur_i, cur_j = q[-1]
                q.pop()

                for d in range(4):
                    nxt_i, nxt_j = cur_i + dx[d], cur_j + dy[d]

                    if 0 <= nxt_i and nxt_i < H and \
                       0 <= nxt_j and nxt_j < W and \
                       m[nxt_i, nxt_j] == 0 and \
                       threshold <= img[nxt_i, nxt_j]:
                        m[nxt_i, nxt_j] = obj_i
                        q.append((nxt_i, nxt_j))

    return m

def FindCL(points):
    # points[N, 2]

    N = points.shape[0]

    assert points.shape == (N, 2)

    center = points.mean(axis=0)
    norm_points = points - center

    u11 = (norm_points[:, 0] * norm_points[:, 1]).sum()
    u20 = (norm_points[:, 0]**2).sum()
    u02 = (norm_points[:, 1]**2).sum()

    phi = 0.5 * math.atan2(2 * u11, u20 - u02)

    return center, phi

def DrawLine(img, center, phi, color):
    c = math.cos(phi)
    s = math.sin(phi)

    H, W, C = img.shape

    def ok(p):
        return 0 <= p[0] and p[0] < H and 0 <= p[1] and p[1] < W

    ps = [[int((  0-center[0]) / c * s + center[1]),   0],
          [int((H-1-center[0]) / c * s + center[1]), H-1],
          [0  , int((  0-center[1]) / s * c + center[0])],
          [W-1, int((W-1-center[1]) / s * c + center[0])],]

    k = []

    for p in ps:
        if 0 <= p[0] and p[0] < W and 0 <= p[1] and p[1] < H:
            k.append(p)

    cv.line(img, k[0], k[1], color, 1)

def DrawCircle(img, center, radius, color):
    cv.circle(img, (int(center[1]), int(center[0])), radius, color, -1)

def main():
    img_path = sys.argv[1]
    img = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2RGB)

    H, W, C = img.shape

    colors =[[  0,   0,   0],
             [255,   0,   0],
             [  0, 255,   0],
             [  0,   0, 255],
             [255, 255,   0],
             [255,   0, 255],
             [  0, 255, 255],
             [255, 255, 255],]

    m = FindObjs(img)

    num_of_objs = m.max()
    print(f"num_of_objs = {num_of_objs}")

    for obj_i in range(1, num_of_objs+1):
        obj_points = np.stack(np.where(m == obj_i), axis=-1)

        center, phi = FindCL(obj_points)

        print(f"obj {obj_i}: center = {center} phi = {phi}")

        DrawLine(img, center, phi, colors[obj_i])
        DrawCircle(img, center, 3, colors[obj_i])

    cv.imshow("obj_img", cv.cvtColor(img, cv.COLOR_BGR2RGB))
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
