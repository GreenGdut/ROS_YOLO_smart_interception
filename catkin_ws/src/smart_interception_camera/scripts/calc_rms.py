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

def rms_for(images, mtx, dist):
    errors = []
    used = 0
    for path in images:
        img = cv2.imread(path)
        ret, corners = find_corners(img)
        if not ret:
            continue
        ok, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(objp, rvec, tvec, mtx, dist)
        err = np.linalg.norm(corners.reshape(-1, 2) - proj.reshape(-1, 2), axis=1)
        errors.append(err)
        used += 1
    all_err = np.concatenate(errors)
    return np.sqrt(np.mean(all_err ** 2)), used

def collect_pairs(left_imgs, right_imgs):
    objpoints = []
    imgpoints_l = []
    imgpoints_r = []
    flipped = 0
    for lp, rp in zip(left_imgs, right_imgs):
        limg = cv2.imread(lp)
        rimg = cv2.imread(rp)
        retl, cl = find_corners(limg)
        retr, cr = find_corners(rimg)
        if retl and retr:
            if np.linalg.norm(cl[0] - cr[0]) > np.linalg.norm(cl[0] - cr[-1]):
                cr = cr[::-1]
                flipped += 1
            objpoints.append(objp)
            imgpoints_l.append(cl)
            imgpoints_r.append(cr)
    print("右图角点顺序翻转修正: %d 对" % flipped)
    return objpoints, imgpoints_l, imgpoints_r

left_imgs = sorted(glob.glob(DATA + "/left-*.png"))
right_imgs = sorted(glob.glob(DATA + "/right-*.png"))
print("发现左图 %d 张, 右图 %d 张" % (len(left_imgs), len(right_imgs)))

objpoints, imgpoints_l, imgpoints_r = collect_pairs(left_imgs, right_imgs)
print("左右都检测到角点的有效对: %d 对" % len(objpoints))

gray = cv2.cvtColor(cv2.imread(left_imgs[0]), cv2.COLOR_BGR2GRAY)
rms_stereo, _, _, _, _, _, _, _, _ = cv2.stereoCalibrate(
    objpoints, imgpoints_l, imgpoints_r, MTX_L, DIST_L, MTX_R, DIST_R,
    gray.shape[::-1], flags=cv2.CALIB_FIX_INTRINSIC)
print("双目联合标定 RMS（与 GUI 显示同源） = %.4f px" % rms_stereo)

rms_l, n_l = rms_for(left_imgs, MTX_L, DIST_L)
rms_r, n_r = rms_for(right_imgs, MTX_R, DIST_R)
print("左相机(cam_left/USB Camera) 逐张重投影 RMS = %.4f px（%d 张有效）" % (rms_l, n_l))
print("右相机(cam_right/HD 720P)  逐张重投影 RMS = %.4f px（%d 张有效）" % (rms_r, n_r))