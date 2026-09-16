#include "trail_drawer.hpp"

// 点迹绘制：把 TrailBuffer 里的历史球心画成"折线 + 小圆点"，最新点加大高亮。
// 只负责画，不管缓冲（缓存见 trail_buffer.hpp）；坐标为原图像素，按 scale 缩放。

#include <opencv2/imgproc.hpp>

#include <vector>

namespace smart_interception {

void drawTrail(cv::Mat& vis, const TrailBuffer& trail, double scale) {
  const std::deque<TrailPoint>& pts = trail.points();
  if (pts.empty()) return;

  std::vector<cv::Point> line;
  line.reserve(pts.size());
  for (const TrailPoint& p : pts) {
    line.emplace_back(cvRound(p.cx * scale), cvRound(p.cy * scale));
  }

  if (line.size() >= 2) {
    cv::polylines(vis, line, false, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
  }
  for (size_t i = 0; i + 1 < line.size(); ++i) {
    cv::circle(vis, line[i], 2, cv::Scalar(0, 255, 255), -1);
  }
  cv::circle(vis, line.back(), 4, cv::Scalar(0, 255, 255), -1);
}

}  // namespace smart_interception
