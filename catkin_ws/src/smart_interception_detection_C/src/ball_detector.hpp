#pragma once

#include "common.h"
#include <opencv2/core.hpp>

namespace smart_interception {

// 检测核心：对单帧 BGR 图跑 HSV+几何筛选，输出球心。
// 数据结构契约见 common.h；实现见 ball_detector.cpp。
class BallDetector {
 public:
  explicit BallDetector(const Params& params) : p_(params) {}

  bool detect(const cv::Mat& bgr, Detection& out, cv::Mat& mask, Timings* timings = nullptr) const;

 private:
  Params p_;
};

}  // namespace smart_interception
