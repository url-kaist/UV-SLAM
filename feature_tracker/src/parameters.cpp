#include "parameters.h"

std::string IMAGE_TOPIC;
std::string IMU_TOPIC;
std::vector<std::string> CAM_NAMES;
std::string FISHEYE_MASK;
int MAX_CNT;
int MIN_DIST;
int WINDOW_SIZE;
int FREQ;
double F_THRESHOLD;
int SHOW_TRACK;
int STEREO_TRACK;
int EQUALIZE;
int ROW;
int COL;
double FOCAL_LENGTH;
int FISHEYE;
bool PUB_THIS_FRAME;

double PROJ_FX;
double PROJ_FY;
double PROJ_CX;
double PROJ_CY;

double DIST_K1;
double DIST_K2;
double DIST_P1;
double DIST_P2;

int CANNY_DETECT;

cv::Mat PROJ;

template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

void readParameters(ros::NodeHandle &n)
{
    std::string config_file;
    config_file = readParam<std::string>(n, "config_file");
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }
    std::string VINS_FOLDER_PATH = readParam<std::string>(n, "vins_folder");

    CANNY_DETECT = fsSettings["canny_detect"];

    fsSettings["image_topic"] >> IMAGE_TOPIC;
    fsSettings["imu_topic"] >> IMU_TOPIC;
    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    FREQ = fsSettings["freq"];
    F_THRESHOLD = fsSettings["F_threshold"];
    SHOW_TRACK = fsSettings["show_track"];
    EQUALIZE = fsSettings["equalize"];
    FISHEYE = fsSettings["fisheye"];
    if (FISHEYE == 1)
        FISHEYE_MASK = VINS_FOLDER_PATH + "config/fisheye_mask.jpg";
    CAM_NAMES.push_back(config_file);

    WINDOW_SIZE = 10;
    STEREO_TRACK = false;
    PUB_THIS_FRAME = false;

    if (FREQ == 0)
        FREQ = 100;

    cv::FileNode PROJ = fsSettings["projection_parameters"];
    PROJ_FX = PROJ["fx"];
    PROJ_FY = PROJ["fy"];
    PROJ_CX = PROJ["cx"];
    PROJ_CY = PROJ["cy"];
    FOCAL_LENGTH = PROJ["fx"];

    cv::FileNode DIST = fsSettings["distortion_parameters"];
    DIST_K1 = DIST["k1"];
    DIST_K2 = DIST["k2"];
    DIST_P1 = DIST["p1"];
    DIST_P2 = DIST["p2"];


    fsSettings.release();


}
