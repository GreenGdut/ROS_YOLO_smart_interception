cmake_minimum_required(VERSION 3.0.2)
project(smart_interception_detection_C)

set(CMAKE_CXX_STANDARD 14)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  cv_bridge
)
find_package(OpenCV REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp sensor_msgs cv_bridge
  DEPENDS OpenCV
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(detector_node
  src/detector_node.cpp
  src/ball_detector.cpp
  src/trail_drawer.cpp
)
target_link_libraries(detector_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
