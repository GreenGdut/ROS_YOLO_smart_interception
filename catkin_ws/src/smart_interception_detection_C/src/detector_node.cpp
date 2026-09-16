#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <deque>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "ball_detector.hpp"
#include "trail_buffer.hpp"
#include "trail_drawer.hpp"

// 检测节点：吃一路视频流（ROS 话题 或 本地视频/图片文件），逐帧检测球，
// 打印各阶段耗时与 FPS，开调试窗口（左=标注图+点迹，右=掩膜）。
// 入口约定：
//   detector_node --source <路径|话题名>
//   --source 为存在的普通文件 -> runFile 离线逐帧；否则按 ROS 话题订阅（runTopic）。
//   未传 --source 时取私有参数 ~source（默认 /cam_left/image_raw，见 config/ball_hsv.yaml）。

namespace {

using smart_interception::BallDetector;
using smart_interception::Detection;
using smart_interception::Params;
using smart_interception::Timings;
using smart_interception::TrailBuffer;
using smart_interception::drawTrail;

// 判断 --source 是不是本地文件（自动判别"离线文件 vs ROS 话题"的依据）。
bool isRegularFile(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

class DetectorNode {
 public:
  // 读私有参数（launch 把 config/ball_hsv.yaml 加载进 ~）构造检测器/点迹缓冲。
  // 无 roscore 时参数读不到，走代码内默认值（与 yaml 同值），离线模式仍可用。
  DetectorNode(ros::NodeHandle& pnh) {
    Params p;
    pnh.param("hsv/h_min", p.h_min, p.h_min);
    pnh.param("hsv/h_max", p.h_max, p.h_max);
    pnh.param("hsv/s_min", p.s_min, p.s_min);
    pnh.param("hsv/s_max", p.s_max, p.s_max);
    pnh.param("hsv/v_min", p.v_min, p.v_min);
    pnh.param("hsv/v_max", p.v_max, p.v_max);
    pnh.param("shape/min_area", p.min_area, p.min_area);
    pnh.param("shape/max_area", p.max_area, p.max_area);
    pnh.param("shape/min_circularity", p.min_circularity, p.min_circularity);
    pnh.param("shape/min_fill", p.min_fill, p.min_fill);
    pnh.param("shape/morph_kernel", p.morph_kernel, p.morph_kernel);
    detector_ = BallDetector(p);

    pnh.param("display/scale", display_scale_, 0.45);
    pnh.param("display/show", show_, true);

    int trail_max = 60;
    pnh.param("trail/max_points", trail_max, 60);
    pnh.param("trail/show", trail_show_, true);
    trail_ = TrailBuffer(trail_max > 0 ? static_cast<size_t>(trail_max) : 0u);
  }

  // 在线模式：订阅话题，回调里逐帧处理；窗口名即话题名。
  void runTopic(const std::string& topic) {
    ROS_INFO("[ball_detector_C] 订阅话题: %s", topic.c_str());
    window_ = topic;
    sub_ = nh_.subscribe(topic, 1, &DetectorNode::imageCb, this);
    ros::spin();
  }

  // 离线模式：先按视频文件逐帧读（ESC 可中断），VideoCapture 打不开再按单张图片读。
  // 播放结束保留最后一帧，按键才关窗（离线调阈值时方便盯着结果看）。
  void runFile(const std::string& path) {
    ROS_INFO("[ball_detector_C] 离线文件: %s", path.c_str());
    window_ = path;

    cv::VideoCapture cap(path);
    if (cap.isOpened()) {
      cv::Mat frame;
      while (ros::ok() && cap.read(frame)) {
        if (frame.empty()) break;
        process(frame);
        int key = cv::waitKey(1);
        if (key == 27) break;
      }
      if (window_created_) {
        ROS_INFO("[ball_detector_C] 播放结束，按任意键关闭窗口");
        cv::waitKey(0);
      }
      return;
    }

    cv::Mat img = cv::imread(path);
    if (img.empty()) {
      ROS_ERROR("[ball_detector_C] 无法打开文件: %s", path.c_str());
      return;
    }
    process(img);
    cv::waitKey(0);
  }

 private:
  // 话题回调：ROS 图像消息 -> BGR 帧 -> process()。
  void imageCb(const sensor_msgs::ImageConstPtr& msg) {
    try {
      cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
      process(cv_ptr->image);
    } catch (const cv_bridge::Exception& e) {
      ROS_ERROR_THROTTLE(1.0, "[ball_detector_C] cv_bridge 转换失败: %s", e.what());
    }
  }

  // 每帧主流程：检测 -> 统计耗时/FPS -> 点迹入缓冲 -> 调试窗口。
  void process(const cv::Mat& bgr) {
    Detection det;
    cv::Mat mask;
    Timings t;
    bool found = detector_.detect(bgr, det, mask, &t);
    double total = t.cvt + t.inrange + t.morph + t.contours + t.select;

    frame_count_++;
    // FPS = 最近 kFpsWindow 帧平均检测耗时的倒数（检测段吞吐，不含取流/显示）。
    frame_times_ms_.push_back(total);
    if (frame_times_ms_.size() > kFpsWindow) frame_times_ms_.pop_front();
    double avg_ms = 0.0;
    for (double v : frame_times_ms_) avg_ms += v;
    avg_ms /= static_cast<double>(frame_times_ms_.size());
    double fps = avg_ms > 0.0 ? 1000.0 / avg_ms : 0.0;

    if (found) {
      ROS_INFO(
          "#%ld det (cx=%.1f cy=%.1f r=%.1f circ=%.2f fill=%.2f) | "
          "cvt=%.1f inRange=%.1f morph=%.1f contours=%.1f select=%.1f total=%.1fms fps=%.1f",
          frame_count_, det.cx, det.cy, det.r, det.circularity, det.fill, t.cvt, t.inrange, t.morph,
          t.contours, t.select, total, fps);
    } else {
      ROS_INFO(
          "#%ld no detection | cvt=%.1f inRange=%.1f morph=%.1f contours=%.1f select=%.1f "
          "total=%.1fms fps=%.1f",
          frame_count_, t.cvt, t.inrange, t.morph, t.contours, t.select, total, fps);
    }

    if (found) {
      // 检出点入点迹缓冲；缓冲满则丢最旧（trail_buffer.hpp）。
      trail_.push(frame_count_, det.cx, det.cy, det.r);
    }

    if (show_) {
      showDebug(bgr, mask, det, found);
      // 必须泵一次 GUI 事件循环，否则在线模式窗口不渲染（黑屏/弹不出）。
      cv::waitKey(1);
    }
  }

  // 调试窗口：显示画布 = [标注图 | 掩膜] 左右拼接，整体按 display/scale 缩放。
  // 缩放只影响显示；检测与打印的坐标始终是原图像素。
  void showDebug(const cv::Mat& bgr, const cv::Mat& mask, const Detection& det, bool found) {
    cv::Mat vis = bgr.clone();
    cv::Mat mask_bgr;
    cv::cvtColor(mask, mask_bgr, cv::COLOR_GRAY2BGR);

    const double s = display_scale_;
    if (s != 1.0) {
      cv::Mat vis_s, mask_s;
      cv::resize(vis, vis_s, cv::Size(), s, s, cv::INTER_AREA);
      cv::resize(mask_bgr, mask_s, cv::Size(), s, s, cv::INTER_AREA);
      vis = vis_s;
      mask_bgr = mask_s;
    }

    if (trail_show_) {
      drawTrail(vis, trail_, s);
    }

    if (found) {
      cv::Point c(cvRound(det.cx * s), cvRound(det.cy * s));
      int r = std::max(1, cvRound(det.r * s));
      cv::circle(vis, c, r, cv::Scalar(0, 255, 0), 2);
      cv::circle(vis, c, 3, cv::Scalar(0, 0, 255), -1);
    }

    std::string text;
    if (found) {
      char buf[160];
      snprintf(buf, sizeof(buf), "cx=%.1f cy=%.1f r=%.1f circ=%.2f fill=%.2f", det.cx, det.cy,
               det.r, det.circularity, det.fill);
      text = buf;
    } else {
      text = "no detection";
    }
    cv::putText(vis, text, cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255),
                2);

    cv::Mat canvas;
    cv::hconcat(vis, mask_bgr, canvas);
    if (!window_created_) {
      // 参考 camera 包 image_view 的窗口配置：命名窗口 + 固定初始尺寸（autosize=false）。
      cv::namedWindow(window_, cv::WINDOW_NORMAL);
      cv::resizeWindow(window_, canvas.cols, canvas.rows);
      window_created_ = true;
    }
    cv::imshow(window_, canvas);
  }

  ros::NodeHandle nh_;
  ros::Subscriber sub_;
  BallDetector detector_{Params()};
  TrailBuffer trail_{0u};
  bool trail_show_ = true;
  double display_scale_ = 0.45;
  bool show_ = true;
  long frame_count_ = 0;
  bool window_created_ = false;
  std::string window_;
  static constexpr size_t kFpsWindow = 30;
  std::deque<double> frame_times_ms_;
};

}  // namespace

int main(int argc, char** argv) {
  // --source 是自设参数而非 ROS 重映射参数，须在 ros::init 前剥出来，
  // 否则会被 ROS 当作非法 remapping 参数。
  std::vector<std::string> clean_args;
  std::string source;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--source" && i + 1 < argc) {
      source = argv[++i];
    } else {
      clean_args.push_back(a);
    }
  }

  std::vector<char*> ros_argv;
  ros_argv.push_back(argv[0]);
  for (auto& s : clean_args) ros_argv.push_back(const_cast<char*>(s.c_str()));
  int ros_argc = static_cast<int>(ros_argv.size());
  ros::init(ros_argc, ros_argv.data(), "ball_detector_C");

  ros::NodeHandle pnh("~");
  if (source.empty()) {
    pnh.param<std::string>("source", source, std::string("/cam_left/image_raw"));
  }

  DetectorNode node(pnh);
  // 自动判别输入源：本地文件 -> 离线逐帧；否则按 ROS 话题订阅。
  if (isRegularFile(source)) {
    node.runFile(source);
  } else {
    node.runTopic(source);
  }
  return 0;
}
