#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseArray.h>


Estimator estimator;

std::condition_variable con;
double current_time = -1;
queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::PointCloudConstPtr> relo_buf;
int sum_of_wait = 0;

std::mutex m_buf;
std::mutex m_state;
std::mutex i_buf;
std::mutex m_estimator;

double latest_time;
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;
bool init_feature = 0;
bool init_imu = 1;
double last_imu_t = 0;

ros::Publisher pub_sync_gt_vins;
ros::Publisher pub_only_no_loop_gt;
ros::Publisher pub_only_no_loop_pose;

void no_loop_gt(const nav_msgs::PathConstPtr &gt_path)
{
  geometry_msgs::PoseArray sync_data_;
  geometry_msgs::Pose gt_pose;
  if(gt_path->poses.size() > 0 )
  {
    //for(int i=0; gt_path->poses.size(); i++)
    {
      std::cout << gt_path->poses.at(gt_path->poses.size()-1).header.stamp.toSec()*1e9 << ", " <<
                   gt_path->poses.at(gt_path->poses.size()-1).pose.position.x << ", " <<
                   gt_path->poses.at(gt_path->poses.size()-1).pose.position.y << ", " <<
                   gt_path->poses.at(gt_path->poses.size()-1).pose.position.z << std::endl;

      sync_data_.header.stamp = gt_path->poses.at(gt_path->poses.size()-1).header.stamp;
      gt_pose.position.x = gt_path->poses.at(gt_path->poses.size()-1).pose.position.x;
      gt_pose.position.y = gt_path->poses.at(gt_path->poses.size()-1).pose.position.y;
      gt_pose.position.z = gt_path->poses.at(gt_path->poses.size()-1).pose.position.z;
      sync_data_.poses.push_back(gt_pose);
    }
  }
  if(sync_data_.poses.size()>0)
  {
    pub_only_no_loop_gt.publish(sync_data_);
  }
}

void no_loop_pose(const nav_msgs::PathConstPtr &vins_path)
{
  geometry_msgs::PoseArray sync_data_;
  geometry_msgs::Pose vins_pose;
  if(vins_path->poses.size() > 0)
  {
    //for(int i=0; vins_path->poses.size(); i++)
    {
//      std::cout
//                << " VINSTimeStamp: " <<vins_path->poses.at(vins_path->poses.size()-1).header.stamp
//                << " VINS_x:" <<vins_path->poses.at(vins_path->poses.size()-1).pose.position.x
//                << " VINS_y:" <<vins_path->poses.at(vins_path->poses.size()-1).pose.position.y
//                << " VINS_z:" <<vins_path->poses.at(vins_path->poses.size()-1).pose.position.z

//                << std::endl;
      sync_data_.header.stamp = vins_path->poses.at(vins_path->poses.size()-1).header.stamp;
      vins_pose.position.x = vins_path->poses.at(vins_path->poses.size()-1).pose.position.x;
      vins_pose.position.y = vins_path->poses.at(vins_path->poses.size()-1).pose.position.y;
      vins_pose.position.z = vins_path->poses.at(vins_path->poses.size()-1).pose.position.z;
      sync_data_.poses.push_back(vins_pose);
    }
  }
  if(sync_data_.poses.size()>0)
  {
    pub_only_no_loop_pose.publish(sync_data_);
  }
}

void compare_callback(const nav_msgs::PathConstPtr &gt_path, const nav_msgs::PathConstPtr &vins_path)
{

  geometry_msgs::PoseArray sync_data_;
  geometry_msgs::Pose gt_pose;
  geometry_msgs::Pose vins_pose;

  std::cout << "SIZE GT POSE: " << gt_path->poses.size() << std::endl;
  std::cout << "SIZE VINS POSE: " << vins_path->poses.size() << std::endl;

  if(gt_path->poses.size() > 0 )
  {
    //for(int i=0; gt_path->poses.size(); i++)
    {
      std::cout << "GTTimeStamp: " <<gt_path->poses.at(gt_path->poses.size()-1).header.stamp
                << " GT_x:" <<gt_path->poses.at(gt_path->poses.size()-1).pose.position.x
                << " GT_y:" <<gt_path->poses.at(gt_path->poses.size()-1).pose.position.y
                << " GT_z:" <<gt_path->poses.at(gt_path->poses.size()-1).pose.position.z
                << " GT_r_x:" <<gt_path->poses.at(gt_path->poses.size()-1).pose.orientation.x
                << " GT_r_y:" <<gt_path->poses.at(gt_path->poses.size()-1).pose.orientation.y
                << " GT_r_z:" <<gt_path->poses.at(gt_path->poses.size()-1).pose.orientation.z
                << " GT_r_w:" <<gt_path->poses.at(gt_path->poses.size()-1).pose.orientation.w
                << std::endl;

      sync_data_.header.stamp = gt_path->poses.at(gt_path->poses.size()-1).header.stamp;
      gt_pose.position.x = gt_path->poses.at(gt_path->poses.size()-1).pose.position.x;
      gt_pose.position.y = gt_path->poses.at(gt_path->poses.size()-1).pose.position.y;
      gt_pose.position.z = gt_path->poses.at(gt_path->poses.size()-1).pose.position.z;
      sync_data_.poses.push_back(gt_pose);
    }
  }
  if(vins_path->poses.size() > 0)
  {
    //for(int i=0; vins_path->poses.size(); i++)
    {
      std::cout
                << " VINSTimeStamp: " <<vins_path->poses.at(vins_path->poses.size()-1).header.stamp
                << " VINS_x:" <<vins_path->poses.at(vins_path->poses.size()-1).pose.position.x
                << " VINS_y:" <<vins_path->poses.at(vins_path->poses.size()-1).pose.position.y
                << " VINS_z:" <<vins_path->poses.at(vins_path->poses.size()-1).pose.position.z
                << " VINS_r_x:" <<vins_path->poses.at(vins_path->poses.size()-1).pose.orientation.x
                << " VINS_r_y:" <<vins_path->poses.at(vins_path->poses.size()-1).pose.orientation.y
                << " VINS_r_z:" <<vins_path->poses.at(vins_path->poses.size()-1).pose.orientation.z
                << " VINS_r_w:" <<vins_path->poses.at(vins_path->poses.size()-1).pose.orientation.w
                << std::endl;

      vins_pose.position.x = vins_path->poses.at(vins_path->poses.size()-1).pose.position.x;
      vins_pose.position.y = vins_path->poses.at(vins_path->poses.size()-1).pose.position.y;
      vins_pose.position.z = vins_path->poses.at(vins_path->poses.size()-1).pose.position.z;
      sync_data_.poses.push_back(vins_pose);
    }
  }

  if(sync_data_.poses.size()>0)
  {
    pub_sync_gt_vins.publish(sync_data_);
  }
}

void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    if (init_imu)
    {
        latest_time = t;
        init_imu = 0;
        return;
    }
    double dt = t - latest_time;
    latest_time = t;

    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};

    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g;

    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);

    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;

    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

void update()
{
    TicToc t_predict;
    latest_time = current_time;
    tmp_P = estimator.Ps[WINDOW_SIZE];
    tmp_Q = estimator.Rs[WINDOW_SIZE];
    tmp_V = estimator.Vs[WINDOW_SIZE];
    tmp_Ba = estimator.Bas[WINDOW_SIZE];
    tmp_Bg = estimator.Bgs[WINDOW_SIZE];
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;

    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());

}

//TODO
std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements()
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

    while (true)
    {
        if (imu_buf.empty() || feature_buf.empty())
            return measurements;

        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            //ROS_WARN("wait for imu, only should happen at the beginning");
            sum_of_wait++;
            return measurements;
        }

        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            ROS_WARN("throw img, only should happen at the beginning");
            feature_buf.pop();
            continue;
        }
        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
        feature_buf.pop();

        std::vector<sensor_msgs::ImuConstPtr> IMUs;
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td)
        {
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }
        IMUs.emplace_back(imu_buf.front());
        if (IMUs.empty())
            ROS_WARN("no imu between two image");
        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}

void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    if (imu_msg->header.stamp.toSec() <= last_imu_t)
    {
        ROS_WARN("imu message in disorder!");
        return;
    }
    last_imu_t = imu_msg->header.stamp.toSec();

    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();
    con.notify_one();

    last_imu_t = imu_msg->header.stamp.toSec();

    {
        std::lock_guard<std::mutex> lg(m_state);
        predict(imu_msg);
        std_msgs::Header header = imu_msg->header;
        header.frame_id = "world";
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);
    }
}


void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    if (!init_feature)
    {
        //skip the first detected feature, which doesn't contain optical flow speed
        init_feature = 1;
        return;
    }
    m_buf.lock();
    feature_buf.push(feature_msg);
    m_buf.unlock();
    con.notify_one();
}

void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        m_buf.lock();
        while(!feature_buf.empty())
            feature_buf.pop();
        while(!imu_buf.empty())
            imu_buf.pop();
        m_buf.unlock();
        m_estimator.lock();
        estimator.clearState();
        estimator.setParameter();
        m_estimator.unlock();
        current_time = -1;
        last_imu_t = 0;
    }
    return;
}

void relocalization_callback(const sensor_msgs::PointCloudConstPtr &points_msg)
{
    //printf("relocalization callback! \n");
    m_buf.lock();
    relo_buf.push(points_msg);
    m_buf.unlock();
}

void latest_callback(const sensor_msgs::ImagePtr &img_msg)
{
    m_buf.lock();
    cv_bridge::CvImagePtr ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
    estimator.latest_img = ptr->image;
    m_buf.unlock();
}

// thread: visual-inertial odometry
void process()
{
    while (true)
    {
        TicToc t_process;
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
        std::unique_lock<std::mutex> lk(m_buf);
        con.wait(lk, [&]
                 {
            return (measurements = getMeasurements()).size() != 0;
                 });
        lk.unlock();
        m_estimator.lock();
        for (auto &measurement : measurements)
        {
            auto img_msg = measurement.second;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            for (auto &imu_msg : measurement.first)
            {
                double t = imu_msg->header.stamp.toSec();
                double img_t = img_msg->header.stamp.toSec() + estimator.td;
                if (t <= img_t)
                {
                    if (current_time < 0)
                        current_time = t;
                    double dt = t - current_time;
                    ROS_ASSERT(dt >= 0);
                    current_time = t;
                    dx = imu_msg->linear_acceleration.x;
                    dy = imu_msg->linear_acceleration.y;
                    dz = imu_msg->linear_acceleration.z;
                    rx = imu_msg->angular_velocity.x;
                    ry = imu_msg->angular_velocity.y;
                    rz = imu_msg->angular_velocity.z;
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);

                }
                else
                {
                    double dt_1 = img_t - current_time;
                    double dt_2 = t - img_t;
                    current_time = img_t;
                    ROS_ASSERT(dt_1 >= 0);
                    ROS_ASSERT(dt_2 >= 0);
                    ROS_ASSERT(dt_1 + dt_2 > 0);
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                }
            }
            // set relocalization frame
            sensor_msgs::PointCloudConstPtr relo_msg = NULL;
            while (!relo_buf.empty())
            {
                relo_msg = relo_buf.front();
                relo_buf.pop();
            }
            if (relo_msg != NULL)
            {
                vector<Vector3d> match_points;
                double frame_stamp = relo_msg->header.stamp.toSec();
                for (unsigned int i = 0; i < relo_msg->points.size(); i++)
                {
                    Vector3d u_v_id;
                    u_v_id.x() = relo_msg->points[i].x;
                    u_v_id.y() = relo_msg->points[i].y;
                    u_v_id.z() = relo_msg->points[i].z;
                    match_points.push_back(u_v_id);
                }
                Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]);
                Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4], relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]);
                Matrix3d relo_r = relo_q.toRotationMatrix();
                int frame_index;
                frame_index = relo_msg->channels[0].values[7];
                estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);
            }

            ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());
            // image
            // TODO!
            TicToc t_s;
            //points
            map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;

            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {
                int v = img_msg->channels[0].values[i] + 0.5;
                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;

                //std::cout << "------------" << std::endl;
                //std::cout << "feature_id: " << feature_id << std::endl;
                //std::cout << "camera_id: " << camera_id << std::endl;

                double x = img_msg->points[i].x;
                double y = img_msg->points[i].y;
                double z = img_msg->points[i].z;
                double p_u = img_msg->channels[1].values[i];
                double p_v = img_msg->channels[2].values[i];
                double velocity_x = img_msg->channels[3].values[i];
                double velocity_y = img_msg->channels[4].values[i];
                ROS_ASSERT(z == 1);
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                image[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
            }

            // lines
            map<int, vector<Eigen::Matrix<double, 15, 1>>> image_line;  //14 not sure
            for (unsigned int i = 0; i < img_msg->channels[5].values.size(); i++)
            {
                int line_id = img_msg->channels[5].values[i] + 0.5;;
                double start_x = img_msg->channels[6].values[i];
                double start_y = img_msg->channels[7].values[i];
                double end_x = img_msg->channels[8].values[i];
                double end_y = img_msg->channels[9].values[i];
                double start_u = img_msg->channels[10].values[i];
                double start_v = img_msg->channels[11].values[i];
                double end_u = img_msg->channels[12].values[i];
                double end_v = img_msg->channels[13].values[i];
                double start_velocity_x = img_msg->channels[14].values[i];
                double start_velocity_y = img_msg->channels[15].values[i];
                double end_velocity_x = img_msg->channels[16].values[i];
                double end_velocity_y = img_msg->channels[17].values[i];
                double vp_x = img_msg->channels[18].values[i];
                double vp_y = img_msg->channels[19].values[i];
                double vp_z = img_msg->channels[20].values[i];

                //ROS_ASSERT(z == 1);
                Eigen::Matrix<double, 15, 1> line_uv_velocity;
                line_uv_velocity << start_x, start_y, end_x, end_y, start_u, start_v, end_u, end_v, \
                start_velocity_x, start_velocity_y, end_velocity_x, end_velocity_y, vp_x, vp_y, vp_z;
                image_line[line_id].emplace_back(line_uv_velocity);
            }
            TicToc t_r;
            estimator.processImage(image, image_line, img_msg->header);
            double t_processImage = t_r.toc();

            double whole_t = t_s.toc();
//            std::string OUTPUT_PATH = "/home/hyunjun/time/backend.txt";
//            ofstream foutC(OUTPUT_PATH, ios::app);
//            foutC.setf(ios::fixed, ios::floatfield);
//            foutC.precision(3);
//            foutC << t_processImage << " " << whole_t << endl;
//            foutC.close();

            // printStatistics(estimator, whole_t);
            std_msgs::Header header = img_msg->header;
            header.frame_id = "world";

            pubOdometry(estimator, header);
            pubKeyPoses(estimator, header);
            pubCameraPose(estimator, header);
            pubPointCloud(estimator, header);
            pubLineCloud(estimator, header);
            pubTF(estimator, header);
            pubKeyframe(estimator);
            if (relo_msg != NULL)
                pubRelocalization(estimator);
            //ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());
        }
        m_estimator.unlock();
        m_buf.lock();
        m_state.lock();
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            update();
        m_state.unlock();
        m_buf.unlock();
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);
    estimator.setParameter();
#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
    ROS_WARN("waiting for image and imu...");

    registerPub(n);

    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
    ros::Subscriber sub_restart = n.subscribe("/feature_tracker/restart", 2000, restart_callback);
    ros::Subscriber sub_relo_points = n.subscribe("/pose_graph/match_points", 2000, relocalization_callback);
    ros::Subscriber sub_latest_img = n.subscribe("/feature_tracker/latest_img", 2000, latest_callback);


//    message_filters::Subscriber<nav_msgs::Path> GT_path(n, "/vins_estimator/GT_path", 1000);
//     //(LOOP_CLOSURE==1)
//    message_filters::Subscriber<nav_msgs::Path> VINS_path(n, "/pose_graph/pose_graph_path", 1000);
//    //(LOOP_CLOSURE==0)
//    //message_filters::Subscriber<nav_msgs::Path> VINS_path(n, "/pose_graph/no_loop_path", 1000);
//    message_filters::TimeSynchronizer<nav_msgs::Path, nav_msgs::Path> sync(GT_path, VINS_path, 1000);
//    //message_filters::TimeSynchronizer<nav_msgs::Path, nav_msgs::Path> sync(VINS_path, GT_path, 2000);
//    sync.registerCallback(boost::bind(&compare_callback, _1, _2));
//    pub_sync_gt_vins = n.advertise<geometry_msgs::PoseArray>("sync_data", 2000);

//    ros::Subscriber sub_only_no_loop_gt = n.subscribe("/vins_estimator/GT_path", 1000, no_loop_gt);
//    pub_only_no_loop_gt = n.advertise<geometry_msgs::PoseArray>("/vins_estimator/no_loop_gt", 1000);

//    ros::Subscriber sub_only_no_loop_pose = n.subscribe("/pose_graph/no_loop_path", 1000, no_loop_pose);
//    pub_only_no_loop_pose = n.advertise<geometry_msgs::PoseArray>("/vins_estimator/no_loop_pose", 1000);

    std::thread measurement_process{process};
    ros::spin();

    return 0;
}
