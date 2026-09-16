#!/usr/bin/env python3
import os
import glob
import rospy
import cv2
import numpy as np
import yaml
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

CHESS = (9, 6)
SQUARE = 0.026
TARGET = 40
WIDTH = 1280
HEIGHT = 720
SAVE_DIR = "/catkin_ws/calib_pairs"
STABLE_PX = 2.0
NEW_POSE_PX = 40.0
HISTORY = 6

class StereoCalib:
    def __init__(self):
        rospy.init_node("stereo_calib", anonymous=True)
        self.bridge = CvBridge()
        self.pair = None
        self.pair_seq = 0
        self.processed_seq = 0
        self.sync_dt = None
        self.frames = []
        self.history = []
        self.last_saved_l = None
        self.status = "WAIT"
        self.objp = np.zeros((CHESS[0] * CHESS[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:CHESS[0], 0:CHESS[1]].T.reshape(-1, 2) * SQUARE
        for f in glob.glob(SAVE_DIR + "/left-*.png") + glob.glob(SAVE_DIR + "/right-*.png"):
            os.remove(f)
        sub_l = message_filters.Subscriber("/cam_left/image_raw", Image)
        sub_r = message_filters.Subscriber("/cam_right/image_raw", Image)
        ts = message_filters.ApproximateTimeSynchronizer([sub_l, sub_r], 10, slop=0.05)
        ts.registerCallback(self.sync_cb)

    def sync_cb(self, lmsg, rmsg):
        limg = self.bridge.imgmsg_to_cv2(lmsg, "bgr8")
        rimg = self.bridge.imgmsg_to_cv2(rmsg, "bgr8")
        dt = abs((lmsg.header.stamp - rmsg.header.stamp).to_sec())
        self.pair_seq += 1
        self.pair = (limg, rimg, self.pair_seq, dt)

    def find(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHESS, None)
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return ret, corners

    def do_capture(self, limg, rimg, cl, cr):
        if np.linalg.norm(cl[0] - cr[0]) > np.linalg.norm(cl[0] - cr[-1]):
            cr = cr[::-1]
        self.frames.append((limg.copy(), rimg.copy(), cl, cr))
        idx = len(self.frames) - 1
        os.makedirs(SAVE_DIR, exist_ok=True)
        cv2.imwrite("%s/left-%04d.png" % (SAVE_DIR, idx), limg)
        cv2.imwrite("%s/right-%04d.png" % (SAVE_DIR, idx), rimg)
        self.last_saved_l = cl.copy()
        print("自动采集 %d/%d（sync %.1f ms）" % (len(self.frames), TARGET, self.sync_dt * 1000))

    def process(self):
        if self.pair is None or self.pair[2] == self.processed_seq:
            return
        self.processed_seq = self.pair[2]
        limg, rimg, _, dt = self.pair
        self.sync_dt = dt
        retl, cl = self.find(limg)
        retr, cr = self.find(rimg)
        self.history.append((cl if retl else None, cr if retr else None))
        if len(self.history) > HISTORY:
            self.history.pop(0)
        stable = (len(self.history) == HISTORY and
                  all(h[0] is not None and h[1] is not None for h in self.history) and
                  np.mean(np.linalg.norm(self.history[-1][0] - self.history[0][0], axis=2)) < STABLE_PX)
        if not (retl and retr):
            self.status = "NO CORNERS"
        elif not stable:
            self.status = "MOVING"
        elif self.last_saved_l is not None and \
                np.mean(np.linalg.norm(cl - self.last_saved_l, axis=2)) < NEW_POSE_PX:
            self.status = "DUP POSE"
        else:
            self.status = "CAPTURED"
            self.do_capture(limg, rimg, cl, cr)
        disp = limg.copy()
        if retl:
            cv2.drawChessboardCorners(disp, CHESS, cl, retl)
        color = {"CAPTURED": (0, 255, 0), "NO CORNERS": (0, 0, 255),
                 "MOVING": (0, 165, 255), "DUP POSE": (128, 128, 128)}.get(self.status, (255, 0, 0))
        cv2.putText(disp, "%d/%d  %s  sync %.1fms" % (len(self.frames), TARGET, self.status, dt * 1000),
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.imshow("left", disp)

    def calibrate(self):
        if len(self.frames) < 10:
            print("有效图太少，至少 10 对")
            return
        objpoints = []
        imgpoints_l = []
        imgpoints_r = []
        for limg, rimg, cl, cr in self.frames:
            objpoints.append(self.objp)
            imgpoints_l.append(cl)
            imgpoints_r.append(cr)
        gray = cv2.cvtColor(self.frames[0][0], cv2.COLOR_BGR2GRAY)
        rms, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints_l, imgpoints_r,
            None, None, None, None, gray.shape[::-1], flags=0)
        print("标定完成，RMS = %.4f px（<0.5 为佳，>1.0 建议重采）" % rms)
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            mtx_l, dist_l, mtx_r, dist_r, gray.shape[::-1], R, T, alpha=0)
        print("基线 Tx = %.4f m" % (-P2[0][3] / P2[0][0]))
        print("左内参 fx=%.2f fy=%.2f cx=%.2f cy=%.2f" % (mtx_l[0, 0], mtx_l[1, 1], mtx_l[0, 2], mtx_l[1, 2]))
        print("右内参 fx=%.2f fy=%.2f cx=%.2f cy=%.2f" % (mtx_r[0, 0], mtx_r[1, 1], mtx_r[0, 2], mtx_r[1, 2]))
        self.save_yaml("cam_left", mtx_l, dist_l, R1, P1)
        self.save_yaml("cam_right", mtx_r, dist_r, R2, P2)
        np.savez("stereo_ext.npz", R=R, T=T, Q=Q, P1=P1, P2=P2,
                 mtx_l=mtx_l, dist_l=dist_l, mtx_r=mtx_r, dist_r=dist_r)
        print("已输出 cam_left_camera_info.yaml / cam_right_camera_info.yaml / stereo_ext.npz")

    def save_yaml(self, name, mtx, dist, R, P):
        data = {
            "image_width": WIDTH,
            "image_height": HEIGHT,
            "camera_name": name,
            "camera_matrix": {"rows": 3, "cols": 3, "data": np.asarray(mtx, dtype=np.float64).flatten().tolist()},
            "distortion_model": "plumb_bob",
            "distortion_coefficients": {"rows": 1, "cols": 5, "data": np.asarray(dist, dtype=np.float64).flatten().tolist()},
            "rectification_matrix": {"rows": 3, "cols": 3, "data": np.asarray(R, dtype=np.float64).flatten().tolist()},
            "projection_matrix": {"rows": 3, "cols": 4, "data": np.asarray(P, dtype=np.float64).flatten().tolist()},
        }
        with open(name + "_camera_info.yaml", "w") as f:
            yaml.safe_dump(data, f, default_flow_style=None, allow_unicode=True)

    def run(self):
        print("=== 双目自动采集标定 ===")
        print("流程: 移动棋盘 -> 停稳约0.6秒 -> 自动采集；采够 %d 对自动标定" % TARGET)
        print("操作: 移动中不采；同一位置只采一次；按 c 手动标定，按 q 退出")
        while not rospy.is_shutdown():
            self.process()
            k = cv2.waitKey(20) & 0xFF
            if k == ord("c"):
                self.calibrate()
            elif k == 27 or k == ord("q"):
                break
            if len(self.frames) >= TARGET and self.status != "DONE":
                self.status = "DONE"
                self.calibrate()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    StereoCalib().run()