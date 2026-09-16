#!/usr/bin/env python3
import numpy as np
import cv2
import yaml

W, H = 1280, 720
EXT = "stereo_ext.npz"

def save_yaml(name, mtx, dist, R, P):
    data = {
        "image_width": W,
        "image_height": H,
        "camera_name": name,
        "camera_matrix": {"rows": 3, "cols": 3, "data": np.asarray(mtx, dtype=np.float64).flatten().tolist()},
        "distortion_model": "plumb_bob",
        "distortion_coefficients": {"rows": 1, "cols": 5, "data": np.asarray(dist, dtype=np.float64).flatten().tolist()},
        "rectification_matrix": {"rows": 3, "cols": 3, "data": np.asarray(R, dtype=np.float64).flatten().tolist()},
        "projection_matrix": {"rows": 3, "cols": 4, "data": np.asarray(P, dtype=np.float64).flatten().tolist()},
    }
    with open(name + "_camera_info.yaml", "w") as f:
        yaml.safe_dump(data, f, default_flow_style=None, allow_unicode=True)

ext = np.load(EXT)
mtx_l, dist_l = ext["mtx_l"].astype(np.float64), ext["dist_l"].astype(np.float64)
mtx_r, dist_r = ext["mtx_r"].astype(np.float64), ext["dist_r"].astype(np.float64)
R, T = ext["R"].astype(np.float64), ext["T"].astype(np.float64).reshape(3, 1)

print("原标定: 左=USB Camera fx=%.2f, 右=HD 720P fx=%.2f" % (mtx_l[0, 0], mtx_r[0, 0]))

R_inv = R.T
T_inv = -R.T @ T

R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    mtx_r, dist_r,
    mtx_l, dist_l,
    (W, H), R_inv, T_inv, alpha=0)

baseline = -P2[0][3] / P2[0][0]
print("交换后基线 Tx = %.4f m（应为正）" % baseline)
if baseline <= 0:
    print("!! 基线仍为负，请勿使用，贴给我排查")
    raise SystemExit(1)

save_yaml("cam_left", mtx_r, dist_r, R1, P1)
save_yaml("cam_right", mtx_l, dist_l, R2, P2)
np.savez(EXT, R=R_inv, T=T_inv, Q=Q, P1=P1, P2=P2,
         mtx_l=mtx_r, dist_l=dist_r, mtx_r=mtx_l, dist_r=dist_l)
print("cam_left  = HD 720P  fx=%.2f fy=%.2f cx=%.2f cy=%.2f" % (mtx_r[0, 0], mtx_r[1, 1], mtx_r[0, 2], mtx_r[1, 2]))
print("cam_right = USB Camera fx=%.2f fy=%.2f cx=%.2f cy=%.2f" % (mtx_l[0, 0], mtx_l[1, 1], mtx_l[0, 2], mtx_l[1, 2]))
print("已更新 cam_left_camera_info.yaml / cam_right_camera_info.yaml / stereo_ext.npz")