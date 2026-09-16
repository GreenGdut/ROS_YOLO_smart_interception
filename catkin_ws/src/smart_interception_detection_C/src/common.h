#pragma once

// 检测核心与节点层共用的数据结构/参数契约。
// 只放"两边都要认"的东西，不放实现。

#include <opencv2/core.hpp>

namespace smart_interception {

// 检测参数：来源 config/ball_hsv.yaml（launch 加载为私有参数），
// 与 Python 版 ball_detector.py 同一套值。
struct Params {
  // HSV 阈值（OpenCV H 范围 0-179）。s/v 下限用于挡地面高光(低S)与暗斑/阴影(低V)。
  int h_min = 5;
  int h_max = 28;
  int s_min = 120;
  int s_max = 255;
  int v_min = 140;
  int v_max = 255;

  // 轮廓几何筛选。
  double min_area = 1500.0;        // 轮廓面积下限(px^2)，隐含球最小半径
  double max_area = 400000.0;      // 轮廓面积上限
  double min_circularity = 0.6;    // 4*pi*A/P^2，1 为正圆
  double min_fill = 0.6;           // 轮廓面积 / 最小外接圆面积，实心盘≈1
  int morph_kernel = 7;            // 开/闭运算核尺寸(px)，补高光与 logo 断裂
};

// 一次成功检测的结果；坐标为输入图像原图像素（供 M3 三角化直接使用）。
struct Detection {
  double cx = 0.0;               // 球心 x（原图坐标）
  double cy = 0.0;               // 球心 y（原图坐标）
  double r = 0.0;                // 最小外接圆半径(px)
  double area = 0.0;             // 轮廓面积
  double circularity = 0.0;      // 圆度
  double fill = 0.0;             // 填充率
};

// detect() 各阶段耗时(ms)，纯测量仪表，不参与检测逻辑。
// 用于定位耗时大头、判断是否需要并行/C++ 优化，与 FPS 一起打印。
struct Timings {
  double cvt = 0.0;       // BGR -> HSV
  double inrange = 0.0;   // inRange 色度阈值
  double morph = 0.0;     // 开/闭运算
  double contours = 0.0;  // findContours
  double select = 0.0;    // 面积/圆度/填充率筛选 + minEnclosingCircle
};

}  // namespace smart_interception
