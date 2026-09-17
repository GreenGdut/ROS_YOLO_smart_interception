#include "ball_detector.hpp"

// 检测核心（纯 OpenCV，不依赖 ROS）。
// 流水线：BGR->HSV -> inRange 色度阈值 -> 开/闭形态学 -> findContours
//        -> 面积/圆度/填充率筛 -> 取最大合格者，输出最小外接圆。
// 参数含义见 common.h 的 Params；各阶段耗时写回 Timings（可选）。

#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace smart_interception {

namespace {

using Clock = std::chrono::steady_clock;

inline double msSince(const Clock::time_point& a, const Clock::time_point& b) {
  return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(b - a).count();
}

}  // namespace

// 对单帧 BGR 图做球检测。
// 输入  bgr     原图（检测吃原始分辨率，不做任何缩放）
// 输出  out     成功时为最优候选（球心/半径为原图像素坐标，M3 直接用）
//       mask    调试用：最终二值掩膜（形态学之后）
//       timings 可选，各阶段耗时(ms)
//       rejects 可选，被筛掉轮廓的原因（简短英文，仅调试窗口用，终端不打印）
// 返回  是否检出
bool BallDetector::detect(const cv::Mat& bgr, Detection& out, cv::Mat& mask,
                          Timings* timings, std::vector<std::string>* rejects) const {
  Clock::time_point t0 = Clock::now();

  cv::Mat hsv;
  cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
  Clock::time_point t1 = Clock::now();

  cv::inRange(hsv, cv::Scalar(p_.h_min, p_.s_min, p_.v_min),
              cv::Scalar(p_.h_max, p_.s_max, p_.v_max), mask);
  Clock::time_point t2 = Clock::now();

  cv::Mat kernel =
      cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(p_.morph_kernel, p_.morph_kernel));
  cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
  cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
  Clock::time_point t3 = Clock::now();

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  Clock::time_point t4 = Clock::now();

  bool found = false;
  char buf[64];
  // 被筛掉的原因写入 rejects（未传则零开销），供调试窗口显示，便于定位调参方向。
  const bool want_rejects = (rejects != nullptr);
  auto note = [&](const char* text) {
    if (want_rejects) rejects->push_back(text);
  };

  if (contours.empty()) note("no contour");

  for (const auto& c : contours) {
    double area = cv::contourArea(c);
    if (area < p_.min_area || area > p_.max_area) {
      snprintf(buf, sizeof(buf), "A %.0f %s %.0f", area, area < p_.min_area ? "<" : ">",
               area < p_.min_area ? p_.min_area : p_.max_area);
      note(buf);
      continue;
    }

    double perimeter = cv::arcLength(c, true);
    if (perimeter <= 0.0) {
      note("P<=0");
      continue;
    }

    double circularity = 4.0 * M_PI * area / (perimeter * perimeter);
    if (circularity < p_.min_circularity) {
      snprintf(buf, sizeof(buf), "C %.2f < %.2f", circularity, p_.min_circularity);
      note(buf);
      continue;
    }

    cv::Point2f center;
    float radius = 0.0f;
    cv::minEnclosingCircle(c, center, radius);
    double circle_area = M_PI * static_cast<double>(radius) * static_cast<double>(radius);
    double fill = circle_area > 0.0 ? area / circle_area : 0.0;
    if (fill < p_.min_fill) {
      snprintf(buf, sizeof(buf), "F %.2f < %.2f", fill, p_.min_fill);
      note(buf);
      continue;
    }

    if (!found || area > out.area) {
      out.cx = center.x;
      out.cy = center.y;
      out.r = radius;
      out.area = area;
      out.circularity = circularity;
      out.fill = fill;
      found = true;
    }
  }
  Clock::time_point t5 = Clock::now();

  if (timings != nullptr) {
    timings->cvt = msSince(t0, t1);
    timings->inrange = msSince(t1, t2);
    timings->morph = msSince(t2, t3);
    timings->contours = msSince(t3, t4);
    timings->select = msSince(t4, t5);
  }
  return found;
}

}  // namespace smart_interception
