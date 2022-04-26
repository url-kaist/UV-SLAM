
cmake_minimum_required(VERSION 2.8.3)
project(uv_feature_tracker)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    sensor_msgs
    cv_bridge
    camera_model
    )

find_package(OpenCV REQUIRED)

catkin_package()

include_directories(
    ${catkin_INCLUDE_DIRS}
    )

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

set(ELSED_PATH ${PROJECT_SOURCE_DIR}/../ELSED)
include_directories(${ELSED_PATH}/src)
add_library(ELSED_LIB
  ${ELSED_PATH}/src/EdgeDrawer.cpp
  ${ELSED_PATH}/src/ELSED.cpp
  ${ELSED_PATH}/src/FullSegmentInfo.cpp
)

add_executable(feature_tracker
    src/feature_tracker_node.cpp
    src/parameters.cpp
    src/feature_tracker.cpp
    src/line_feature_tracker.cpp
    src/utility.cpp
    )

target_link_libraries(feature_tracker ${catkin_LIBRARIES} ${OpenCV_LIBS} ${ELSED_LIB})

