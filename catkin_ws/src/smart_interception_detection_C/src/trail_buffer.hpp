#pragma once

// 点迹缓冲：仅存球心点（不存整帧），FIFO 环形——满了丢最旧。
#include <cstddef>
#include <deque>

namespace smart_interception {

struct TrailPoint {
  long frame_id = 0;  // 进程内递增帧号，仅用于保持先后顺序（离线无 roscore 时时间戳不可用）
  double cx = 0.0;    // 球心 x（原图像素坐标）
  double cy = 0.0;    // 球心 y（原图像素坐标）
  double r = 0.0;     // 当帧外接圆半径(px)
};

class TrailBuffer {
 public:
  explicit TrailBuffer(size_t capacity) : capacity_(capacity) {}

  // 追加一个点；超出容量即丢最旧（容量 0 视为关闭，不缓存）。
  void push(long frame_id, double cx, double cy, double r) {
    if (capacity_ == 0) return;
    points_.push_back(TrailPoint{frame_id, cx, cy, r});
    while (points_.size() > capacity_) {
      points_.pop_front();
    }
  }

  void clear() { points_.clear(); }

  size_t size() const { return points_.size(); }
  const std::deque<TrailPoint>& points() const { return points_; }

 private:
  size_t capacity_;
  std::deque<TrailPoint> points_;
};

}  // namespace smart_interception
