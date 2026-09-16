#!/usr/bin/env python3
import glob
import os
import cv2
import numpy as np
from collections import namedtuple

PHOTO_DIR = "/catkin_ws/test_photo/"
EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# ==== 阈值/参数（初始放宽值，后续逐帧离线微调，再挪 config/ball_hsv.yaml）====
HSV_LOWER = np.array([5, 120, 140], dtype=np.uint8)
HSV_UPPER = np.array([28, 255, 255], dtype=np.uint8)
MIN_AREA = 1500
MAX_AREA = 400000
MIN_CIRCULARITY = 0.6
MIN_FILL = 0.6
MORPH_KERNEL = 7
DISPLAY_SCALE = 0.45   # 仅影响弹窗显示，不影响检测输入与坐标

Detection = namedtuple("Detection", ["cx", "cy", "r", "area", "circularity", "fill"])


# ==== 检测核心（纯 OpenCV，后续 ROS 节点直接复用）====
def detect_bgr(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA or area > MAX_AREA:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < MIN_CIRCULARITY:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        circle_area = np.pi * r * r
        fill = area / circle_area if circle_area > 0 else 0.0
        if fill < MIN_FILL:
            continue
        if best is None or area > best.area:
            best = Detection(cx, cy, r, area, circularity, fill)

    return best, mask


# ==== 临时离线测试（后续整体注释掉，换成发布 /ball_pix_* + /ball_debug_*）====
def show_debug(name, bgr, mask, det):
    vis = cv2.resize(bgr, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
                     interpolation=cv2.INTER_AREA)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_bgr = cv2.resize(mask_bgr, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
                          interpolation=cv2.INTER_AREA)

    if det is not None:
        cx = int(det.cx * DISPLAY_SCALE)
        cy = int(det.cy * DISPLAY_SCALE)
        r = max(1, int(det.r * DISPLAY_SCALE))
        cv2.circle(vis, (cx, cy), r, (0, 255, 0), 2)
        cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)
        text = "cx=%.1f cy=%.1f r=%.1f circ=%.2f fill=%.2f" % (
            det.cx, det.cy, det.r, det.circularity, det.fill)
    else:
        text = "no detection"
    cv2.putText(vis, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    canvas = np.hstack([vis, mask_bgr])
    cv2.imshow(name, canvas)


def main():
    paths = []
    for ext in EXTS:
        paths += glob.glob(os.path.join(PHOTO_DIR, "*" + ext))
    paths = sorted(paths)
    if not paths:
        print("no images in", PHOTO_DIR)
        return

    for path in paths:
        bgr = cv2.imread(path)
        if bgr is None:
            print("fail to read", path)
            continue
        det, mask = detect_bgr(bgr)
        print(os.path.basename(path), det)
        show_debug(os.path.basename(path), bgr, mask, det)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
