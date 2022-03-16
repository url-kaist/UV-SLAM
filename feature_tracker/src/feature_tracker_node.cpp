#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>

#include "feature_tracker.h"
#include "line_feature_tracker.h"

#include <chrono>

#define SHOW_UNDISTORTION 0

vector<uchar> r_status;
vector<float> r_err;
queue<sensor_msgs::ImageConstPtr> img_buf;

ros::Publisher pub_img, pub_match, pub_linematch, pub_latest_img;
ros::Publisher pub_restart;

FeatureTracker trackerData[NUM_OF_CAM];
LineFeatureTracker lineTrackerData;

double first_image_time;
int pub_count = 1;
bool first_image_flag = true;
double last_image_time = 0;
bool init_pub = 0;

cv_bridge::CvImageConstPtr cam1_ptr;

void img1_callback(const sensor_msgs::ImageConstPtr &img_msg){

    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        cam1_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        cam1_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
}

void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    if(first_image_flag)
    {
        first_image_flag = false;
        first_image_time = img_msg->header.stamp.toSec();
        last_image_time = img_msg->header.stamp.toSec();
        return;
    }
    // detect unstable camera stream
    if (img_msg->header.stamp.toSec() - last_image_time > 1.0 || img_msg->header.stamp.toSec() < last_image_time)
    {
        ROS_WARN("image discontinue! reset the feature tracker!");
        first_image_flag = true;
        last_image_time = 0;
        pub_count = 1;
        std_msgs::Bool restart_flag;
        restart_flag.data = true;
        pub_restart.publish(restart_flag);
        return;
    }
    last_image_time = img_msg->header.stamp.toSec();
    // frequency control
    if (round(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time)) <= FREQ)
    {
        PUB_THIS_FRAME = true;
        // reset the frequency control
        if (abs(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time) - FREQ) < 0.01 * FREQ)
        {
            first_image_time = img_msg->header.stamp.toSec();
            pub_count = 0;
        }
    }
    else
        PUB_THIS_FRAME = false;

    cv_bridge::CvImageConstPtr ptr;
    cv_bridge::CvImagePtr ptr_line;
    cv_bridge::CvImagePtr ptr_img;

    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat show_img = ptr->image;
    TicToc t_r;
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ROS_DEBUG("processing camera %d", i);
        if (i != 1 || !STEREO_TRACK)
        {
            trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)), img_msg->header.stamp.toSec());
            double t_point = t_r.toc();
            // Image undistortion and extract line
            //std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
            lineTrackerData.readImage4Line(ptr->image, img_msg->header.stamp.toSec());
            double t_line = t_r.toc();
            //std::chrono::duration<double> sec = std::chrono::system_clock::now() - start;
            //std::cout << "readStereoImage4Line processing time : " << sec.count()*1000 << " ms" << std::endl;

//            std::string OUTPUT_PATH = "/home/hyunjun/time/frontend.txt";
//            ofstream foutC(OUTPUT_PATH, ios::app);
//            foutC.setf(ios::fixed, ios::floatfield);
//            foutC.precision(3);
//            foutC << t_point << " "
//                  << t_line  << " "
//                  << t_point + t_line << endl;
//            foutC.close();
        }
        else
        {
            if (EQUALIZE)
            {
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img);
            }
            else
                trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
        }

#if SHOW_UNDISTORTION
        trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
#endif
    }


    for (unsigned int i = 0;; i++)
    {
        bool completed = false;
        for (int j = 0; j < NUM_OF_CAM; j++)
            if (j != 1 || !STEREO_TRACK)
                completed |= trackerData[j].updateID(i);
        if (!completed)
            break;
    }
    //TODO updateID for line
    for (unsigned int i = 0;; i++)
    {
       // cout << "index i" << i << endl;
        bool completed = false;
        for (int j = 0; j < NUM_OF_CAM; j++)
            if (j != 1 || !STEREO_TRACK)
                completed |= lineTrackerData.updateID(i);
        if (!completed){
            break;
        }
    }

   if (PUB_THIS_FRAME)
   {
        pub_count++;
        sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
        sensor_msgs::ChannelFloat32 id_of_point;
        sensor_msgs::ChannelFloat32 u_of_point;
        sensor_msgs::ChannelFloat32 v_of_point;
        sensor_msgs::ChannelFloat32 velocity_x_of_point;
        sensor_msgs::ChannelFloat32 velocity_y_of_point;

        //line information
        sensor_msgs::ChannelFloat32 start_x_of_line;
        sensor_msgs::ChannelFloat32 start_y_of_line;
        sensor_msgs::ChannelFloat32 start_z_of_line;
        sensor_msgs::ChannelFloat32 end_x_of_line;
        sensor_msgs::ChannelFloat32 end_y_of_line;
        sensor_msgs::ChannelFloat32 end_z_of_line;
        sensor_msgs::ChannelFloat32 id_of_line;
        sensor_msgs::ChannelFloat32 start_u_of_line;
        sensor_msgs::ChannelFloat32 start_v_of_line;
        sensor_msgs::ChannelFloat32 end_u_of_line;
        sensor_msgs::ChannelFloat32 end_v_of_line;
        sensor_msgs::ChannelFloat32 start_velocity_x_of_line;
        sensor_msgs::ChannelFloat32 start_velocity_y_of_line;
        sensor_msgs::ChannelFloat32 end_velocity_x_of_line;
        sensor_msgs::ChannelFloat32 end_velocity_y_of_line;
        sensor_msgs::ChannelFloat32 vp_x;
        sensor_msgs::ChannelFloat32 vp_y;
        sensor_msgs::ChannelFloat32 vp_z;

        feature_points->header = img_msg->header;
        feature_points->header.frame_id = "world";

        vector<set<int>> hash_ids(NUM_OF_CAM);
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            //for points
            auto &un_pts = trackerData[i].cur_un_pts;
            auto &cur_pts = trackerData[i].cur_pts;
            auto &ids = trackerData[i].ids;
            auto &pts_velocity = trackerData[i].pts_velocity;

            for (unsigned int j = 0; j < ids.size(); j++)
            {
                if (trackerData[i].track_cnt[j] > 1)
                {
                    int p_id = ids[j];
                    hash_ids[i].insert(p_id);
                    geometry_msgs::Point32 p;
                    p.x = un_pts[j].x;
                    p.y = un_pts[j].y;
                    p.z = 1;

                    feature_points->points.push_back(p);
                    id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
                    u_of_point.values.push_back(cur_pts[j].x);
                    v_of_point.values.push_back(cur_pts[j].y);
                    velocity_x_of_point.values.push_back(pts_velocity[j].x);
                    velocity_y_of_point.values.push_back(pts_velocity[j].y);
                }
            }
            //for lines
            auto &start_un_pts = lineTrackerData.curr_start_un_pts;
            auto &end_un_pts = lineTrackerData.curr_end_un_pts;
            auto &start_pts = lineTrackerData.curr_start_pts;
            auto &end_pts = lineTrackerData.curr_end_pts;
            auto &line_ids = lineTrackerData.ids;
            auto &start_velocity = lineTrackerData.start_pts_velocity;
            auto &end_velocity = lineTrackerData.end_pts_velocity;
            auto &vpts = lineTrackerData.vps;

            unsigned int num_line_cnt = 0;
            for (unsigned int j = 0; j < line_ids.size(); j++)
            {

                if (lineTrackerData.track_cnt[j] > 1)
                {
                    num_line_cnt++;

                    int p_id = line_ids[j];

                    //TODO
                    start_x_of_line.values.push_back(start_un_pts[j].x);
                    start_y_of_line.values.push_back(start_un_pts[j].y);
                    end_x_of_line.values.push_back(end_un_pts[j].x);
                    end_y_of_line.values.push_back(end_un_pts[j].y);
                    id_of_line.values.push_back(p_id);
                    start_u_of_line.values.push_back(start_pts[j].x);
                    start_v_of_line.values.push_back(start_pts[j].y);
                    end_u_of_line.values.push_back(end_pts[j].x);
                    end_v_of_line.values.push_back(end_pts[j].y);
                    start_velocity_x_of_line.values.push_back(start_velocity[j].x);
                    start_velocity_y_of_line.values.push_back(start_velocity[j].y);
                    end_velocity_x_of_line.values.push_back(end_velocity[j].x);
                    end_velocity_y_of_line.values.push_back(end_velocity[j].y);
                    vp_x.values.push_back(vpts[j](0));
                    vp_y.values.push_back(vpts[j](1));
                    vp_z.values.push_back(vpts[j](2));
                }
            }
        }
        //for points
        feature_points->channels.push_back(id_of_point); //0
        feature_points->channels.push_back(u_of_point); //1
        feature_points->channels.push_back(v_of_point); //2
        feature_points->channels.push_back(velocity_x_of_point); //3
        feature_points->channels.push_back(velocity_y_of_point); // 4

        //for lines
        feature_points->channels.push_back(id_of_line); //5
        feature_points->channels.push_back(start_x_of_line); //6
        feature_points->channels.push_back(start_y_of_line); //7
        feature_points->channels.push_back(end_x_of_line); //8
        feature_points->channels.push_back(end_y_of_line); //9
        feature_points->channels.push_back(start_u_of_line); //10
        feature_points->channels.push_back(start_v_of_line); //11
        feature_points->channels.push_back(end_u_of_line); //12
        feature_points->channels.push_back(end_v_of_line); //13
        feature_points->channels.push_back(start_velocity_x_of_line); //14
        feature_points->channels.push_back(start_velocity_y_of_line); //15
        feature_points->channels.push_back(end_velocity_x_of_line); //16
        feature_points->channels.push_back(end_velocity_y_of_line); //17
        feature_points->channels.push_back(vp_x); //18
        feature_points->channels.push_back(vp_y); //19
        feature_points->channels.push_back(vp_z); //20

        ROS_DEBUG("publish %f, at %f", feature_points->header.stamp.toSec(), ros::Time::now().toSec());
        // skip the first image; since no optical speed on frist image
        if (!init_pub)
        {
            init_pub = 1;
        }
        else
            pub_img.publish(feature_points);

        if (SHOW_TRACK)
        {
            ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);

            //cv::Mat stereo_img(ROW * NUM_OF_CAM, COL, CV_8UC3);
            cv::Mat stereo_img = ptr->image;
            ptr_img = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
            ptr_img->image = lineTrackerData.forw_img.clone();
            cv::cvtColor(ptr_img->image, ptr_img->image, CV_GRAY2RGB);
            pub_latest_img.publish(ptr_img->toImageMsg());

//            imshow("1", ptr_img->image);
//            waitKey(1);

            for (int i = 0; i < NUM_OF_CAM; i++)
            {
                cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
                cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);
                ptr_line = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
                ptr_line->image = lineTrackerData.forw_img.clone();
                cv::cvtColor(ptr_line->image, ptr_line->image, CV_GRAY2RGB);

                for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
                {
                    double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
                    cv::circle(tmp_img, trackerData[i].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
                }

                for (unsigned int j = 0; j < lineTrackerData.curr_keyLine.size(); j++)
                {
                    double len = std::min(1.0, 1.0 * lineTrackerData.track_cnt[j] / WINDOW_SIZE);
                    cv::Point sp = Point(lineTrackerData.curr_keyLine[j].startPointX, lineTrackerData.curr_keyLine[j].startPointY);
                    cv::Point ep = Point(lineTrackerData.curr_keyLine[j].endPointX, lineTrackerData.curr_keyLine[j].endPointY);
                    line(ptr_line->image, sp, ep, Scalar(255*(1-len), 0, 255*len), 2);
                }

            }
            pub_match.publish(ptr->toImageMsg());
            pub_linematch.publish(ptr_line->toImageMsg());
        }
    }
    //ROS_INFO("whole feature tracker processing costs: %f", t_r.toc());
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "feature_tracker");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);

    for (int i = 0; i < NUM_OF_CAM; i++)
        trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);

    lineTrackerData.readIntrinsicParameter(CAM_NAMES[0]);

    if(FISHEYE)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
            if(!trackerData[i].fisheye_mask.data)
            {
                ROS_INFO("load mask fail");
                ROS_BREAK();
            }
            else
                ROS_INFO("load mask success");
        }
    }

    ros::Subscriber sub_img = n.subscribe(IMAGE_TOPIC, 100, img_callback);
    ros::Subscriber sub_img1 = n.subscribe("/cam1/image_raw", 100, img1_callback);

    pub_img = n.advertise<sensor_msgs::PointCloud>("feature", 1000);
    pub_match = n.advertise<sensor_msgs::Image>("feature_img",1000);
    pub_restart = n.advertise<std_msgs::Bool>("restart",1000);


    // lineTrackerData.pub_undistort_img = n.advertise<sensor_msgs::Image>("raw_undistort_img",1000);
    // lineTrackerData.pub_lineExtract_img = n.advertise<sensor_msgs::Image>("lineExtract_img",1000);
    pub_linematch = n.advertise<sensor_msgs::Image>("line_feature_img",1000);

    pub_latest_img = n.advertise<sensor_msgs::Image>("latest_img", 1000);


    /*
    if (SHOW_TRACK)
        cv::namedWindow("vis", cv::WINDOW_NORMAL);
    */
    ros::spin();
    return 0;
}


// new points velocity is 0, pub or not?
// track cnt > 1 pub?
