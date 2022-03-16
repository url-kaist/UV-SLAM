#pragma once
#include <ros/ros.h>
#include <opencv2/highgui/highgui.hpp>

extern int ROW;
extern int COL;
extern double FOCAL_LENGTH;
const int NUM_OF_CAM = 1;


extern std::string IMAGE_TOPIC;
extern std::string IMU_TOPIC;
extern std::string FISHEYE_MASK;
extern std::vector<std::string> CAM_NAMES;
extern int MAX_CNT;
extern int MIN_DIST;
extern int WINDOW_SIZE;
extern int FREQ;
extern double F_THRESHOLD;
extern int SHOW_TRACK;
extern int STEREO_TRACK;
extern int EQUALIZE;
extern int FISHEYE;
extern bool PUB_THIS_FRAME;

extern int CANNY_DETECT;


void readParameters(ros::NodeHandle &n);

extern double PROJ_FX;
extern double PROJ_FY;
extern double PROJ_CX;
extern double PROJ_CY;

extern double DIST_K1;
extern double DIST_K2;
extern double DIST_P1;
extern double DIST_P2;

