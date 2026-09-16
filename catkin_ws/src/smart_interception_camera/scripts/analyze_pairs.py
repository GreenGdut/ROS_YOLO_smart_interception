#!/usr/bin/env python3
import cv2
import numpy as np
import glob

CHESS = (9, 6)
SQUARE = 0.026
DATA = "/catkin_ws/calib_pairs"
EXT = "/catkin_ws/src/smart_interception_camera/scripts/stereo_ext.npz"

ext = np.load(EXT)
mtx_l, dist_l = ext["mtx_l"], ext["dist_l"]
mtx_r, dist_r = ext["mtx_r"], ext["dist_r"]
R, T = ext["R"], ext["T"].flatten()

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
print("共 %d 对图" % len(left_imgs))
print("%-6s %10s %10s %12s" % ("对号", "左RMS(px)", "右RMS(px)", "跨目误差(cm)"))

rows = []
for i, (lp, rp) in enumerate(zip(left_imgs, right_imgs)):
    limg = cv2.imread(lp)
    rimg = cv2.imread(rp)
    retl, cl = find_corners(limg)
    retr, cr = find_corners(rimg)
    if not (retl and retr):
        rows.append((i, -1, -1, -1))
        continue
    if np.linalg.norm(cl[0] - cr[0]) > np.linalg.norm(cl[0] - cr[-1]):
        cr = cr[::-1]
    ok_l, rvec_l, tvec_l = cv2.solvePnP(objp, cl, mtx_l, dist_l)
    ok_r, rvec_r, tvec_r = cv2.solvePnP(objp, cr, mtx_r, dist_r)
    proj_l = cv2.projectPoints(objp, rvec_l, tvec_l, mtx_l, dist_l)[0]
    proj_r = cv2.projectPoints(objp, rvec_r, tvec_r, mtx_r, dist_r)[0]
    err_l = np.sqrt(np.mean(np.sum((cl.reshape(-1, 2) - proj_l.reshape(-1, 2)) ** 2, axis=1)))
    err_r = np.sqrt(np.mean(np.sum((cr.reshape(-1, 2) - proj_r.reshape(-1, 2)) ** 2, axis=1)))
    t_pred = R @ tvec_r.flatten() + T
    cross_err = np.linalg.norm(tvec_l.flatten() - t_pred) * 100
    rows.append((i, err_l, err_r, cross_err))

rows.sort(key=lambda r: -(r[3] if r[3] > 0 else 999))
print("=== 跨目误差最大的 15 对（不同步/棋盘在动的嫌疑对）===")
for i, el, er, ce in rows[:15]:
    print("pair %02d  左RMS=%.2f  右RMS=%.2f  跨目=%.2f cm" % (i, el, er, ce))

good = [r for r in rows if r[1] >= 0 and r[3] < 2.0]
print("=== 跨目误差 < 2cm 的对数: %d / %d ===" % (len(good), len(rows)))
if good:
    print("这些好对的单目 RMS: 左均值 %.2f, 右均值 %.2f" % (
        np.mean([r[1] for r in good]), np.mean([r[2] for r in good])))