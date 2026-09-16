#pragma once

#include <opencv2/core.hpp>

#include "trail_buffer.hpp"

namespace smart_interception {

// 在 vis 上绘制点迹：折线连线 + 各点小圆点，最新点高亮。
// trail 内坐标为原图像素，按 scale 缩放后绘制（与调试窗口一致）。
void drawTrail(cv::Mat& vis, const TrailBuffer& trail, double scale);

}  // namespace smart_interception
