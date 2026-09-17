#pragma once

#include "common.h"
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace smart_interception {

// 检测核心：对单帧 BGR 图跑 HSV+几何筛选，输出球心。
// 数据结构契约见 common.h；实现见 ball_detector.cpp。
class BallDetector {
 public:
  explicit BallDetector(const Params& params) : p_(params) {}

  // rejects 非空时，逐条写入被筛掉的轮廓原因（简短英文，供调试窗口显示；终端不打印）。
  bool detect(const cv::Mat& bgr, Detection& out, cv::Mat& mask, Timings* timings = nullptr,
              std::vector<std::string>* rejects = nullptr) const;

 private:
  Params p_;
};

}  // namespace smart_interception
