#!/usr/bin/env python3
import cv2
import numpy as np
import glob

CHESS = (9, 6)
SQUARE = 0.026
DATA = "/catkin_ws/calibrationdata_yuvy"

MTX_L = np.array([[732.479736, 0, 637.814896],
                  [0, 737.052297, 361.012880],
                  [0, 0, 1]], dtype=np.float64)
DIST_L = np.array([0.080362, -0.133627, 0.000570, -0.002039, 0.0])
MTX_R = np.array([[914.355384, 0, 596.620406],
                  [0, 919.686148, 273.383170],
                  [0, 0, 1]], dtype=np.float64)
DIST_R = np.array([0.153708, -0.313073, -0.000604, -0.001081, 0.0])

objp = np.zeros((CHESS[0] * CHESS[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESS[0], 0:CHESS[1]].T.reshape(-1, 2) * SQUARE

def find_corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHESS, None)
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return ret, corners

left_imgs = sorted(glob.glob(DATA + "/left-*.png"))
right_imgs = sorted(glob.glob(DATA + "/right-*.png"))

objpoints = []
imgpoints_l = []
imgpoints_r = []
dys = []
dxs = []
for lp, rp in zip(left_imgs, right_imgs):
    limg = cv2.imread(lp)
    rimg = cv2.imread(rp)
    retl, cl = find_corners(limg)
    retr, cr = find_corners(rimg)
    if retl and retr:
        if np.linalg.norm(cl[0] - cr[0]) > np.linalg.norm(cl[0] - cr[-1]):
            cr = cr[::-1]
        objpoints.append(objp)
        imgpoints_l.append(cl)
        imgpoints_r.append(cr)
        dys.append(float(np.mean(cl[:, :, 1]) - np.mean(cr[:, :, 1])))
        dxs.append(float(np.mean(cl[:, :, 0]) - np.mean(cr[:, :, 0])))

gray = cv2.cvtColor(cv2.imread(left_imgs[0]), cv2.COLOR_BGR2GRAY)
print("有效对: %d" % len(objpoints))

dys = np.array(dys)
dxs = np.array(dxs)
print("棋盘中心水平视差 dx: 均值 %.1f, 标准差 %.1f" % (dxs.mean(), dxs.std()))
print("棋盘中心垂直差 dy:   均值 %.1f, 标准差 %.1f" % (dys.mean(), dys.std()))
print("dy 超过 40px 的对数: %d / %d" % (int(np.sum(np.abs(dys) > 40)), len(dys)))

rms_free, m1, d1, m2, d2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_l, imgpoints_r,
    None, None, None, None, gray.shape[::-1], flags=0)
print("--- 全自由联合优化（camera_calibration 同款算法）---")
print("RMS = %.4f px" % rms_free)
print("优化后左: fx=%.2f fy=%.2f cx=%.2f cy=%.2f" % (m1[0, 0], m1[1, 1], m1[0, 2], m1[1, 2]))
print("优化后右: fx=%.2f fy=%.2f cx=%.2f cy=%.2f" % (m2[0, 0], m2[1, 1], m2[0, 2], m2[1, 2]))
print("基线 Tx = %.4f m" % T[0, 0])

rms_fix, _, _, _, _, _, _, _, _ = cv2.stereoCalibrate(
    objpoints, imgpoints_l, imgpoints_r,
    MTX_L, DIST_L.copy(), MTX_R, DIST_R.copy(), gray.shape[::-1],
    flags=cv2.CALIB_FIX_INTRINSIC)
print("--- FIX_INTRINSIC（固定 ost 内参只求外参）---")
print("RMS = %.4f px" % rms_fix)