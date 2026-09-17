#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cmath>
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
//
// 自设命令行参数（大小写敏感，值空格分隔；未给的项回落到 config/ball_hsv.yaml）：
//   --source <路径|话题名>   存在的普通文件 -> runFile 离线逐帧；否则按 ROS 话题订阅。
//                            未传时取私有参数 ~source（默认 /cam_left/image_raw）。
//   --V_scale <x>            BGR 亮度线性增益（须 >0，1.0=不变），覆盖 preproc/V_scale。
//   --H_min/--H_max <v>      色相阈值（0-179），覆盖 hsv/h_*。默认H_min=5,max=28;
//   --S_min/--S_max <v>      饱和度阈值（0-255），覆盖 hsv/s_*。默认S_min=120,S_max=255;
//   --V_min/--V_max <v>      明度阈值（0-255），覆盖 hsv/v_*。默认V_min=140,V_max=255
//   --A_min/--A_max <x>      轮廓面积下限/上限(px^2)，覆盖 shape/min_area|max_area。默认1500,400000
//   --C_min <x>              圆度下限（0-1，1=正圆），覆盖 shape/min_circularity。默认0.60
//   --F_min <x>              填充率下限（0-1，1=实心盘），覆盖 shape/min_fill。默认0.60
// 全部自设参数须在 ros::init 前剥出（见 parseCli）；启动时打印最终生效配置。
//
// 终端日志说明（每帧一条；被筛掉轮廓的原因只画在窗口左上角黄字，不打进终端）：
//   #<帧号> det (cx= cy= r= circ= fill=) | cvt= inRange= morph= contours= select= total=ms fps=
//   #<帧号> no detection | cvt= inRange= morph= contours= select= total=ms fps=
//     det 行       ：检出时的球心/半径（原图像素）与圆度 circ、填充率 fill
//     no detection ：本帧未检出（具体原因看窗口左上角）
//     cvt/inRange/morph/contours/select：各阶段耗时(ms)；total=五段之和
//     fps          ：最近 30 帧平均检测耗时的倒数（检测段吞吐，不含取流/显示）

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

// 命令行解析结果：未指定的字段留哨兵（source 为空串，其余 <0），由节点层用 yaml 值兜底。
struct CliOptions {
  std::string source;
  double V_scale = -1.0;
  double a_min = -1.0, a_max = -1.0, c_min = -1.0, f_min = -1.0;
  int h_min = -1, h_max = -1, s_min = -1, s_max = -1, v_min = -1, v_max = -1;
};

// 取 argv 里紧跟的下一个 token 作为值；缺值则告警。此刻还没 ros::init，只能用 stderr。
bool takeValue(int argc, char** argv, int& i, const char* name, std::string& out) {
  if (i + 1 >= argc) {
    std::fprintf(stderr, "[ball_detector_C] %s 缺少取值，已忽略\n", name);
    return false;
  }
  out = argv[++i];
  return true;
}

// 严格解析整数（拒绝 "120abc" 这类带尾巴的输入），失败告警返回 false。
bool parseIntStrict(const char* name, const std::string& text, int& out) {
  bool ok = false;
  try {
    size_t used = 0;
    int v = std::stoi(text, &used);
    if (used == text.size()) {
      out = v;
      ok = true;
    }
  } catch (const std::exception&) {
  }
  if (!ok) {
    std::fprintf(stderr, "[ball_detector_C] %s 的值 '%s' 非法，已忽略\n", name, text.c_str());
  }
  return ok;
}

// 严格解析浮点数（拒绝带尾巴），失败告警返回 false；allow_zero=false 时要求 >0。
bool parseDoubleStrict(const char* name, const std::string& text, double& out, bool allow_zero) {
  bool ok = false;
  try {
    size_t used = 0;
    double v = std::stod(text, &used);
    if (used == text.size() && (v > 0.0 || (allow_zero && v == 0.0))) {
      out = v;
      ok = true;
    }
  } catch (const std::exception&) {
  }
  if (!ok) {
    std::fprintf(stderr, "[ball_detector_C] %s 的值 '%s' 非法（须 %s），已忽略\n", name,
                 text.c_str(), allow_zero ? ">=0" : ">0");
  }
  return ok;
}

// HSV 阈值解析：合法才写入（负值视作未指定）。
void parseHsvArg(const char* name, const std::string& text, int& dst) {
  int v = 0;
  if (!parseIntStrict(name, text, v)) return;
  if (v < 0) {
    std::fprintf(stderr, "[ball_detector_C] %s 的值 %d 为负，已忽略\n", name, v);
    return;
  }
  dst = v;
}

// 剥出所有自设命令行参数，其余原样留在 clean_args 里交给 ROS（保留 := 重映射能力）。
// 支持 "--K v" 与 "--K=v" 两种写法；不认识的 "--xxx" 会告警（避免静默失效）。
CliOptions parseCli(int argc, char** argv, std::vector<std::string>& clean_args) {
  CliOptions cli;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    std::string key = a;
    std::string v;
    bool has_inline = false;
    if (a.size() > 2 && a[0] == '-' && a[1] == '-') {
      size_t eq = a.find('=');
      if (eq != std::string::npos) {
        key = a.substr(0, eq);
        v = a.substr(eq + 1);
        has_inline = true;
      }
    }
    // 取该参数的值：等号写法已带值，否则吃掉下一个 token。
    auto get_val = [&](const char* name) -> bool {
      return has_inline ? true : takeValue(argc, argv, i, name, v);
    };

    if (key == "--source") {
      if (get_val("--source")) cli.source = v;
    } else if (key == "--V_scale") {
      if (get_val("--V_scale")) parseDoubleStrict("--V_scale", v, cli.V_scale, false);
    } else if (key == "--C_min") {
      if (get_val("--C_min")) parseDoubleStrict("--C_min", v, cli.c_min, true);
    } else if (key == "--A_min") {
      if (get_val("--A_min")) parseDoubleStrict("--A_min", v, cli.a_min, true);
    } else if (key == "--A_max") {
      if (get_val("--A_max")) parseDoubleStrict("--A_max", v, cli.a_max, true);
    } else if (key == "--F_min") {
      if (get_val("--F_min")) parseDoubleStrict("--F_min", v, cli.f_min, true);
    } else if (key == "--H_min") {
      if (get_val("--H_min")) parseHsvArg("--H_min", v, cli.h_min);
    } else if (key == "--H_max") {
      if (get_val("--H_max")) parseHsvArg("--H_max", v, cli.h_max);
    } else if (key == "--S_min") {
      if (get_val("--S_min")) parseHsvArg("--S_min", v, cli.s_min);
    } else if (key == "--S_max") {
      if (get_val("--S_max")) parseHsvArg("--S_max", v, cli.s_max);
    } else if (key == "--V_min") {
      if (get_val("--V_min")) parseHsvArg("--V_min", v, cli.v_min);
    } else if (key == "--V_max") {
      if (get_val("--V_max")) parseHsvArg("--V_max", v, cli.v_max);
    } else {
      if (a.size() > 2 && a[0] == '-' && a[1] == '-') {
        std::fprintf(stderr,
                     "[ball_detector_C] 未知参数 '%s' 已忽略。可用：--source/--V_scale/"
                     "--H_min/--H_max/--S_min/--S_max/--V_min/--V_max/--A_min/--A_max/"
                     "--C_min/--F_min（大小写敏感，值空格分隔）\n",
                     a.c_str());
      }
      clean_args.push_back(a);
    }
  }
  return cli;
}

// 把命令行给的 HSV 阈值覆盖进 Params（负值=未指定）；越界截断（H 0-179，S/V 0-255）并告警。
void applyHsvOverrides(const CliOptions& cli, Params& p) {
  const struct {
    int value;
    int lo;
    int hi;
    int* dst;
    const char* name;
  } items[] = {
      {cli.h_min, 0, 179, &p.h_min, "--H_min"}, {cli.h_max, 0, 179, &p.h_max, "--H_max"},
      {cli.s_min, 0, 255, &p.s_min, "--S_min"}, {cli.s_max, 0, 255, &p.s_max, "--S_max"},
      {cli.v_min, 0, 255, &p.v_min, "--V_min"}, {cli.v_max, 0, 255, &p.v_max, "--V_max"},
  };
  for (const auto& it : items) {
    if (it.value < 0) continue;
    int clamped = std::min(std::max(it.value, it.lo), it.hi);
    if (clamped != it.value) {
      ROS_WARN("[ball_detector_C] %s=%d 越界，截断为 %d", it.name, it.value, clamped);
    }
    *it.dst = clamped;
  }
}

// 把命令行给的形状参数覆盖进 Params（负值=未指定）；越界截断并告警。
// 面积上限用 1e9 兜底（仅防手滑输入天文数字），圆度/填充率截到 [0,1]。
void applyShapeOverrides(const CliOptions& cli, Params& p) {
  const struct {
    double value;
    double lo;
    double hi;
    double* dst;
    const char* name;
  } items[] = {
      {cli.a_min, 0.0, 1e9, &p.min_area, "--A_min"},
      {cli.a_max, 0.0, 1e9, &p.max_area, "--A_max"},
      {cli.c_min, 0.0, 1.0, &p.min_circularity, "--C_min"},
      {cli.f_min, 0.0, 1.0, &p.min_fill, "--F_min"},
  };
  for (const auto& it : items) {
    if (it.value < 0.0) continue;
    double clamped = std::min(std::max(it.value, it.lo), it.hi);
    if (clamped != it.value) {
      ROS_WARN("[ball_detector_C] %s=%.2f 越界，截断为 %.2f", it.name, it.value, clamped);
    }
    *it.dst = clamped;
  }
}

class DetectorNode {
 public:
  // 读私有参数（launch 把 config/ball_hsv.yaml 加载进 ~）构造检测器/点迹缓冲，
  // 再用命令行参数覆盖。优先级：命令行 > yaml > 代码默认。
  // 无 roscore 时参数读不到，走代码内默认值（与 yaml 同值），离线模式仍可用。
  DetectorNode(ros::NodeHandle& pnh, const CliOptions& cli) {
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
    applyHsvOverrides(cli, p);
    applyShapeOverrides(cli, p);
    detector_ = BallDetector(p);

    // 亮度增益：yaml 兜底，命令行覆盖；非法值（<=0 或非有限）回退 1.0。
    pnh.param("preproc/V_scale", V_scale_, 1.0);
    if (cli.V_scale > 0.0) V_scale_ = cli.V_scale;
    if (!(V_scale_ > 0.0) || !std::isfinite(V_scale_)) {
      ROS_WARN("[ball_detector_C] V_scale 非法（须 >0 且有限），回退 1.0");
      V_scale_ = 1.0;
    }

    pnh.param("display/scale", display_scale_, 0.45);
    pnh.param("display/show", show_, true);

    int trail_max = 60;
    pnh.param("trail/max_points", trail_max, 60);
    pnh.param("trail/show", trail_show_, true);
    trail_ = TrailBuffer(trail_max > 0 ? static_cast<size_t>(trail_max) : 0u);

    // 启动即打印生效配置，手调场景时对账/记录用。
    ROS_INFO(
        "[ball_detector_C] 生效配置: V_scale=%.2f H=[%d,%d] S=[%d,%d] V=[%d,%d] "
        "A=[%.0f,%.0f] C_min=%.2f F_min=%.2f",
        V_scale_, p.h_min, p.h_max, p.s_min, p.s_max, p.v_min, p.v_max, p.min_area, p.max_area,
        p.min_circularity, p.min_fill);
    if (p.h_min > p.h_max || p.s_min > p.s_max || p.v_min > p.v_max) {
      ROS_WARN("[ball_detector_C] 阈值 min>max，掩膜必为空，请检查参数");
    }
    if (p.min_area > p.max_area) {
      ROS_WARN("[ball_detector_C] 面积 min>max，掩膜必为空，请检查参数");
    }
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

  // 每帧主流程：亮度增益 -> 检测 -> 统计耗时/FPS -> 点迹入缓冲 -> 调试窗口。
  void process(const cv::Mat& bgr) {
    // 亮度增益：对 BGR 逐像素线性缩放（255 饱和）；处理后的帧同时用于
    // 检测与显示（所见即所检）；V_scale==1.0 时零开销、不拷贝。
    cv::Mat scaled;
    const cv::Mat* input = &bgr;
    if (V_scale_ != 1.0) {
      bgr.convertTo(scaled, -1, V_scale_);
      input = &scaled;
    }

    Detection det;
    cv::Mat mask;
    Timings t;
    std::vector<std::string> rejects;  // 被筛掉轮廓的原因（只画到窗口，不进终端日志）
    bool found = detector_.detect(*input, det, mask, &t, &rejects);
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
      showDebug(*input, mask, det, found, rejects);
      // 必须泵一次 GUI 事件循环，否则在线模式窗口不渲染（黑屏/弹不出）。
      cv::waitKey(1);
    }
  }

  // 调试窗口：显示画布 = [标注图 | 掩膜] 左右拼接，整体按 display/scale 缩放。
  // 缩放只影响显示；检测与打印的坐标始终是原图像素。
  // rejects 为各轮廓被筛掉的原因（简短英文），逐行画在左上角。
  void showDebug(const cv::Mat& bgr, const cv::Mat& mask, const Detection& det, bool found,
                 const std::vector<std::string>& rejects) {
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

    // 被筛掉轮廓的原因（简短英文），最多 kMaxRejectLines 行，便于定位调参方向。
    const size_t kMaxRejectLines = 8;
    int ry = 60;
    size_t shown = 0;
    for (const std::string& r : rejects) {
      if (shown >= kMaxRejectLines) break;
      cv::putText(vis, r, cv::Point(15, ry), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                  cv::Scalar(0, 255, 255), 1);
      ry += 22;
      ++shown;
    }
    if (rejects.size() > kMaxRejectLines) {
      char more[64];
      snprintf(more, sizeof(more), "... +%lu more",
               static_cast<unsigned long>(rejects.size() - kMaxRejectLines));
      cv::putText(vis, more, cv::Point(15, ry), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                  cv::Scalar(0, 255, 255), 1);
    }

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
  double V_scale_ = 1.0;  // BGR 亮度增益（--V_scale / preproc.V_scale），1.0=不变
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
  // 自设参数（--source/--V_scale/--H_min ...）不是 ROS 重映射参数，
  // 须在 ros::init 前剥干净，否则会被 ROS 报非法参数。
  std::vector<std::string> clean_args;
  CliOptions cli = parseCli(argc, argv, clean_args);

  std::vector<char*> ros_argv;
  ros_argv.push_back(argv[0]);
  for (auto& s : clean_args) ros_argv.push_back(const_cast<char*>(s.c_str()));
  int ros_argc = static_cast<int>(ros_argv.size());
  ros::init(ros_argc, ros_argv.data(), "ball_detector_C");

  // 原样回显命令行，调参时用于核对参数是否真的传进来（与"生效配置"对照）。
  std::string cmdline;
  for (int i = 0; i < argc; ++i) {
    if (i > 0) cmdline += ' ';
    cmdline += argv[i];
  }
  ROS_INFO("[ball_detector_C] 命令行: %s", cmdline.c_str());

  ros::NodeHandle pnh("~");
  std::string source = cli.source;
  if (source.empty()) {
    pnh.param<std::string>("source", source, std::string("/cam_left/image_raw"));
  }

  DetectorNode node(pnh, cli);
  // 自动判别输入源：本地文件 -> 离线逐帧；否则按 ROS 话题订阅。
  if (isRegularFile(source)) {
    node.runFile(source);
  } else {
    node.runTopic(source);
  }
  return 0;
}
