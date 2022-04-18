#include "visualization.h"

using namespace ros;
using namespace Eigen;
ros::Publisher pub_odometry, pub_latest_odometry;
ros::Publisher pub_path, pub_relo_path;
ros::Publisher pub_point_cloud, pub_margin_cloud;
ros::Publisher pub_key_poses;
ros::Publisher pub_relo_relative_pose;
ros::Publisher pub_camera_pose;
ros::Publisher pub_camera_pose_visual;
ros::Publisher pub_line_cloud;

nav_msgs::Path path, relo_path;

ros::Publisher pub_keyframe_pose;
ros::Publisher pub_keyframe_point;
ros::Publisher pub_extrinsic;

ros::Publisher pub_keyframe_line, pub_keyframe_line_stereo;
ros::Publisher pub_keyframe_line_2d;
ros::Publisher pub_keyframe_2d_3d;
ros::Publisher pub_line_margin;
ros::Publisher pub_line_deg;
ros::Publisher pub_line_array;
ros::Publisher pub_text;

CameraPoseVisualization cameraposevisual(0, 1, 0, 1);
CameraPoseVisualization keyframebasevisual(0.0, 0.0, 1.0, 1.0);
static double sum_of_path = 0;
static Vector3d last_path(0.0, 0.0, 0.0);

visualization_msgs::Marker key_lines;
visualization_msgs::Marker map_lines;
visualization_msgs::Marker margin_lines;
visualization_msgs::Marker margin_lines_deg;
visualization_msgs::MarkerArray text_array;

std::vector<visualization_msgs::Marker> margin_line_list;
std::vector<ros::Publisher> pub_margin_line_list;

Point3d prev_sp_3d_w_c0;
Point3d prev_ep_3d_w_c0;
Point3d prev_sp_3d_w_c1;
Point3d prev_ep_3d_w_c1;
int initial_flag=0;
double max_dist = 0;
int num = 4;
int i = 0;

void registerPub(ros::NodeHandle &n)
{
    pub_latest_odometry = n.advertise<nav_msgs::Odometry>("imu_propagate", 1000);
    pub_path = n.advertise<nav_msgs::Path>("path", 1000);
    pub_relo_path = n.advertise<nav_msgs::Path>("relocalization_path", 1000);
    pub_odometry = n.advertise<nav_msgs::Odometry>("odometry", 1000);
    pub_point_cloud = n.advertise<sensor_msgs::PointCloud>("point_cloud", 1000);
    pub_margin_cloud = n.advertise<sensor_msgs::PointCloud>("history_cloud", 1000);
    pub_key_poses = n.advertise<visualization_msgs::Marker>("key_poses", 1000);
    pub_camera_pose = n.advertise<nav_msgs::Odometry>("camera_pose", 1000);
    pub_camera_pose_visual = n.advertise<visualization_msgs::MarkerArray>("camera_pose_visual", 1000);
    pub_keyframe_pose = n.advertise<nav_msgs::Odometry>("keyframe_pose", 1000);
    pub_keyframe_point = n.advertise<sensor_msgs::PointCloud>("keyframe_point", 1000);
    pub_extrinsic = n.advertise<nav_msgs::Odometry>("extrinsic", 1000);
    pub_relo_relative_pose=  n.advertise<nav_msgs::Odometry>("relo_relative_pose", 1000);

    pub_keyframe_line = n.advertise<visualization_msgs::Marker>("keyframe_lines", 1000);
    pub_keyframe_line_stereo = n.advertise<visualization_msgs::Marker>("keyframe_lines_stereo", 1000);
    pub_keyframe_line_2d = n.advertise<visualization_msgs::Marker>("keyframe_lines_2d", 1000);
    pub_line_cloud = n.advertise<visualization_msgs::Marker>("line_cloud", 1000);
    pub_line_margin = n.advertise<visualization_msgs::Marker>("line_history_cloud", 1000);
    pub_line_deg = n.advertise<visualization_msgs::Marker>("degenerated_line", 1000);
    pub_line_array = n.advertise<visualization_msgs::MarkerArray>("line_history_clouds", 1000);
    pub_text = n.advertise<visualization_msgs::MarkerArray>("line_text",1000);
    declarePublisher(num, n, pub_margin_line_list);

    cameraposevisual.setScale(1);
    cameraposevisual.setLineWidth(0.05);
    keyframebasevisual.setScale(0.1);
    keyframebasevisual.setLineWidth(0.01);
}

void pubLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, const Eigen::Vector3d &V, const std_msgs::Header &header)
{
    Eigen::Quaterniond quadrotor_Q = Q ;

    nav_msgs::Odometry odometry;
    odometry.header = header;
    odometry.header.frame_id = "world";
    odometry.pose.pose.position.x = P.x();
    odometry.pose.pose.position.y = P.y();
    odometry.pose.pose.position.z = P.z();
    odometry.pose.pose.orientation.x = quadrotor_Q.x();
    odometry.pose.pose.orientation.y = quadrotor_Q.y();
    odometry.pose.pose.orientation.z = quadrotor_Q.z();
    odometry.pose.pose.orientation.w = quadrotor_Q.w();
    odometry.twist.twist.linear.x = V.x();
    odometry.twist.twist.linear.y = V.y();
    odometry.twist.twist.linear.z = V.z();
    pub_latest_odometry.publish(odometry);
}

void printStatistics(const Estimator &estimator, double t)
{
    if (estimator.solver_flag != Estimator::SolverFlag::NON_LINEAR)
        return;
    printf("position: %f, %f, %f\r", estimator.Ps[WINDOW_SIZE].x(), estimator.Ps[WINDOW_SIZE].y(), estimator.Ps[WINDOW_SIZE].z());
    ROS_DEBUG_STREAM("position: " << estimator.Ps[WINDOW_SIZE].transpose());
    ROS_DEBUG_STREAM("orientation: " << estimator.Vs[WINDOW_SIZE].transpose());
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        //ROS_DEBUG("calibration result for camera %d", i);
        ROS_DEBUG_STREAM("extirnsic tic: " << estimator.tic[i].transpose());
        ROS_DEBUG_STREAM("extrinsic ric: " << Utility::R2ypr(estimator.ric[i]).transpose());
        if (ESTIMATE_EXTRINSIC)
        {
            cv::FileStorage fs(EX_CALIB_RESULT_PATH, cv::FileStorage::WRITE);
            Eigen::Matrix3d eigen_R;
            Eigen::Vector3d eigen_T;
            eigen_R = estimator.ric[i];
            eigen_T = estimator.tic[i];
            cv::Mat cv_R, cv_T;
            cv::eigen2cv(eigen_R, cv_R);
            cv::eigen2cv(eigen_T, cv_T);
            fs << "extrinsicRotation" << cv_R << "extrinsicTranslation" << cv_T;
            fs.release();
        }
    }

    static double sum_of_time = 0;
    static int sum_of_calculation = 0;
    sum_of_time += t;
    sum_of_calculation++;
    ROS_DEBUG("vo solver costs: %f ms", t);
    ROS_DEBUG("average of time %f ms", sum_of_time / sum_of_calculation);

    sum_of_path += (estimator.Ps[WINDOW_SIZE] - last_path).norm();
    last_path = estimator.Ps[WINDOW_SIZE];
    ROS_DEBUG("sum of path %f", sum_of_path);
    if (ESTIMATE_TD)
        ROS_INFO("td %f", estimator.td);
}

void pubOdometry(const Estimator &estimator, const std_msgs::Header &header)
{
    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
    {
        nav_msgs::Odometry odometry;
        odometry.header = header;
        odometry.header.frame_id = "world";
        odometry.child_frame_id = "world";
        Quaterniond tmp_Q;
        tmp_Q = Quaterniond(estimator.Rs[WINDOW_SIZE]);
        odometry.pose.pose.position.x = estimator.Ps[WINDOW_SIZE].x();
        odometry.pose.pose.position.y = estimator.Ps[WINDOW_SIZE].y();
        odometry.pose.pose.position.z = estimator.Ps[WINDOW_SIZE].z();
        odometry.pose.pose.orientation.x = tmp_Q.x();
        odometry.pose.pose.orientation.y = tmp_Q.y();
        odometry.pose.pose.orientation.z = tmp_Q.z();
        odometry.pose.pose.orientation.w = tmp_Q.w();
        odometry.twist.twist.linear.x = estimator.Vs[WINDOW_SIZE].x();
        odometry.twist.twist.linear.y = estimator.Vs[WINDOW_SIZE].y();
        odometry.twist.twist.linear.z = estimator.Vs[WINDOW_SIZE].z();
        pub_odometry.publish(odometry);

        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header = header;
        pose_stamped.header.frame_id = "world";
        pose_stamped.pose = odometry.pose.pose;
        path.header = header;
        path.header.frame_id = "world";
        path.poses.push_back(pose_stamped);
        pub_path.publish(path);

        Vector3d correct_t;
        Vector3d correct_v;
        Quaterniond correct_q;
        correct_t = estimator.drift_correct_r * estimator.Ps[WINDOW_SIZE] + estimator.drift_correct_t;
        correct_q = estimator.drift_correct_r * estimator.Rs[WINDOW_SIZE];
        odometry.pose.pose.position.x = correct_t.x();
        odometry.pose.pose.position.y = correct_t.y();
        odometry.pose.pose.position.z = correct_t.z();
        odometry.pose.pose.orientation.x = correct_q.x();
        odometry.pose.pose.orientation.y = correct_q.y();
        odometry.pose.pose.orientation.z = correct_q.z();
        odometry.pose.pose.orientation.w = correct_q.w();

        pose_stamped.pose = odometry.pose.pose;
        relo_path.header = header;
        relo_path.header.frame_id = "world";
        relo_path.poses.push_back(pose_stamped);
        pub_relo_path.publish(relo_path);

        // write result to file
        ofstream foutC(VINS_RESULT_PATH, ios::app);
        foutC.setf(ios::fixed, ios::floatfield);
        foutC.precision(9);
        foutC << header.stamp.toSec()/* * 1e9 */<< " ";
        foutC.precision(6);
        foutC << estimator.Ps[WINDOW_SIZE].x() << " "
              << estimator.Ps[WINDOW_SIZE].y() << " "
              << estimator.Ps[WINDOW_SIZE].z() << " "
              << tmp_Q.x() << " "
              << tmp_Q.y() << " "
              << tmp_Q.z() << " "
              << tmp_Q.w() << endl;
        foutC.close();
    }
}

void pubKeyPoses(const Estimator &estimator, const std_msgs::Header &header)
{
    if (estimator.key_poses.size() == 0)
        return;
    visualization_msgs::Marker key_poses;
    key_poses.header = header;
    key_poses.header.frame_id = "world";
    key_poses.ns = "key_poses";
    key_poses.type = visualization_msgs::Marker::SPHERE_LIST;
    key_poses.action = visualization_msgs::Marker::ADD;
    key_poses.pose.orientation.w = 1.0;
    key_poses.lifetime = ros::Duration();

    //static int key_poses_id = 0;
    key_poses.id = 0; //key_poses_id++;
    key_poses.scale.x = 0.05;
    key_poses.scale.y = 0.05;
    key_poses.scale.z = 0.05;
    key_poses.color.b = 1.0;
    key_poses.color.a = 1.0;

    for (int i = 0; i <=   WINDOW_SIZE; i++)
    {
        geometry_msgs::Point pose_marker;
        Vector3d correct_pose;
        correct_pose = estimator.key_poses[i];
        pose_marker.x = correct_pose.x();
        pose_marker.y = correct_pose.y();
        pose_marker.z = correct_pose.z();
        key_poses.points.push_back(pose_marker);
    }
    pub_key_poses.publish(key_poses);
}

void pubCameraPose(const Estimator &estimator, const std_msgs::Header &header)
{
    int idx2 = WINDOW_SIZE - 1;

    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
    {
        int i = idx2;
        Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[0];
        Quaterniond R = Quaterniond(estimator.Rs[i] * estimator.ric[0]);

        nav_msgs::Odometry odometry;
        odometry.header = header;
        odometry.header.frame_id = "world";
        odometry.pose.pose.position.x = P.x();
        odometry.pose.pose.position.y = P.y();
        odometry.pose.pose.position.z = P.z();
        odometry.pose.pose.orientation.x = R.x();
        odometry.pose.pose.orientation.y = R.y();
        odometry.pose.pose.orientation.z = R.z();
        odometry.pose.pose.orientation.w = R.w();

        pub_camera_pose.publish(odometry);

        cameraposevisual.reset();
        cameraposevisual.add_pose(P, R);
        cameraposevisual.publish_by(pub_camera_pose_visual, odometry.header);
    }
}


void pubPointCloud(const Estimator &estimator, const std_msgs::Header &header)
{
    sensor_msgs::PointCloud point_cloud, loop_point_cloud;
    point_cloud.header = header;
    loop_point_cloud.header = header;


    for (auto &it_per_id : estimator.f_manager.feature)
    {
        int used_num;
        used_num = it_per_id.feature_per_frame.size();
        if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        if (it_per_id.start_frame > WINDOW_SIZE * 3.0 / 4.0 || it_per_id.solve_flag != 1)
            continue;
        int imu_i = it_per_id.start_frame;
        Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
        Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0]) + estimator.Ps[imu_i];

        geometry_msgs::Point32 p;
        p.x = w_pts_i(0);
        p.y = w_pts_i(1);
        p.z = w_pts_i(2);
        point_cloud.points.push_back(p);
    }
    pub_point_cloud.publish(point_cloud);


    // pub margined potin
    sensor_msgs::PointCloud margin_cloud;
    margin_cloud.header = header;

    for (auto &it_per_id : estimator.f_manager.feature)
    {
        int used_num;
        used_num = it_per_id.feature_per_frame.size();
        if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        //if (it_per_id->start_frame > WINDOW_SIZE * 3.0 / 4.0 || it_per_id->solve_flag != 1)
        //        continue;

        if (it_per_id.start_frame == 0 && it_per_id.feature_per_frame.size() <= 2
                && it_per_id.solve_flag == 1 )
        {
            int imu_i = it_per_id.start_frame;
            Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
            Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0]) + estimator.Ps[imu_i];

            geometry_msgs::Point32 p;
            p.x = w_pts_i(0);
            p.y = w_pts_i(1);
            p.z = w_pts_i(2);
            margin_cloud.points.push_back(p);
        }
    }
    pub_margin_cloud.publish(margin_cloud);
}

double scale = 1.0;
void pubLineCloud(const Estimator &estimator, std_msgs::Header &header)
{
    visualization_msgs::Marker key_lines;
    key_lines.header = header;
    key_lines.header.frame_id = "world";
    key_lines.ns = "key_lines";
    key_lines.type = visualization_msgs::Marker::LINE_LIST;
    key_lines.action = visualization_msgs::Marker::ADD;
    key_lines.pose.orientation.w = 1.0;
    key_lines.lifetime = ros::Duration(0);

    key_lines.id = 1; //key_poses_id++;
    key_lines.scale.x = 0.05;
    key_lines.color.r = 0.0;
    key_lines.color.g = 0.0;
    key_lines.color.b = 1.0;
    key_lines.color.a = 1.0;

    geometry_msgs::Point sp;
    geometry_msgs::Point ep;

    for(auto &it_per_id : estimator.f_manager.line_feature)
    {
        if(it_per_id.solve_flag == 1)
        {
            int imu_i = it_per_id.start_frame;
            Matrix3d R_wc = estimator.Rs[imu_i] * estimator.ric[0];
            Vector3d t_wc = estimator.Rs[imu_i] * estimator.tic[0] + estimator.Ps[imu_i] ;

            Vector3d sp_2d_c = it_per_id.line_feature_per_frame[0].start_point;
            Vector3d ep_2d_c = it_per_id.line_feature_per_frame[0].end_point;
            Vector3d sp_2d_p_c = Vector3d(sp_2d_c(0) + scale, -scale*(ep_2d_c(0) - sp_2d_c(0))/(ep_2d_c(1) - sp_2d_c(1)) + sp_2d_c(1), 1);
            Vector3d ep_2d_p_c = Vector3d(ep_2d_c(0) + scale, -scale*(ep_2d_c(0) - sp_2d_c(0))/(ep_2d_c(1) - sp_2d_c(1)) + ep_2d_c(1), 1);

            Vector3d pi_s = sp_2d_c.cross(sp_2d_p_c);
            Vector3d pi_e = ep_2d_c.cross(ep_2d_p_c);

            Vector4d pi_s_4d, pi_e_4d;
            pi_s_4d.head(3) = pi_s;
            pi_s_4d(3) = 1;
            pi_e_4d.head(3) = pi_e;
            pi_e_4d(3) = 1;

            AngleAxisd roll(it_per_id.orthonormal_vec(0), Vector3d::UnitX());
            AngleAxisd pitch(it_per_id.orthonormal_vec(1), Vector3d::UnitY());
            AngleAxisd yaw(it_per_id.orthonormal_vec(2), Vector3d::UnitZ());
            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> Rotation_psi;
            Rotation_psi = roll * pitch * yaw;
            double pi = it_per_id.orthonormal_vec(3);

            Vector3d n_w = cos(pi) * Rotation_psi.block<3,1>(0,0);
            Vector3d d_w = sin(pi) * Rotation_psi.block<3,1>(0,1);

            Matrix<double, 6, 1> line_w;
            line_w.block<3,1>(0,0) = n_w;
            line_w.block<3,1>(3,0) = d_w;

//            cout << "---------------" << endl;
//            cout << n_w(0) << ", " << n_w(1) << ", " << n_w(2) << endl;
//            cout << d_w(0) << ", " << d_w(1) << ", " << d_w(2) << endl;

            Matrix<double, 6, 6> T_cw;
            T_cw.setZero();
            T_cw.block<3,3>(0,0) = R_wc.transpose();
            T_cw.block<3,3>(0,3) = Utility::skewSymmetric(-R_wc.transpose()*t_wc) * R_wc.transpose();
            T_cw.block<3,3>(3,3) = R_wc.transpose();

            Matrix<double, 6, 1> line_c = T_cw * line_w;
            Vector3d n_c = line_c.block<3,1>(0,0);
            Vector3d d_c = line_c.block<3,1>(3,0);

            Matrix4d L_c;
            L_c.setZero();
            L_c.block<3,3>(0,0) = Utility::skewSymmetric(n_c);
            L_c.block<3,1>(0,3) = d_c;
            L_c.block<1,3>(3,0) = -d_c.transpose();

            Vector4d D_s = L_c * pi_s_4d;
            Vector4d D_e = L_c * pi_e_4d;
            Vector3d D_s_3d(D_s(0)/D_s(3), D_s(1)/D_s(3), D_s(2)/D_s(3));
            Vector3d D_e_3d(D_e(0)/D_e(3), D_e(1)/D_e(3), D_e(2)/D_e(3));

            Vector3d D_s_w = R_wc * D_s_3d + t_wc;
            Vector3d D_e_w = R_wc * D_e_3d + t_wc;

            if(
                    std::isnan(D_s_w(0)) || std::isnan(D_s_w(1)) || std::isnan(D_s_w(2)) ||
                    std::isnan(D_e_w(0)) || std::isnan(D_e_w(1)) || std::isnan(D_e_w(2))
                    || D_s_3d(2) < 0 || D_e_3d(2) < 0
//                    || (D_s_w - D_e_w).norm() > 10
              )
            {
                continue;
            }

            sp.x = D_s_w(0);
            sp.y = D_s_w(1);
            sp.z = D_s_w(2);

            ep.x = D_e_w(0);
            ep.y = D_e_w(1);
            ep.z = D_e_w(2);

//            double len_sp = sqrt(pow(sp.x, 2) + pow(sp.y, 2));
//            double len_ep = sqrt(pow(ep.x, 2) + pow(ep.y, 2));

//            if((len_sp + len_ep)/2 > max_dist)
//                max_dist = (len_sp + len_ep);

            key_lines.points.push_back(sp);
            key_lines.points.push_back(ep);
        }
    }
    pub_line_cloud.publish(key_lines);
//    cout << max_dist << endl;

    vector<geometry_msgs::Point> points;

    declareLineLists(header, num, margin_line_list);

    margin_lines.header = header;
    margin_lines.header.frame_id = "world";
    margin_lines.ns = "margin_lines";
    margin_lines.type = visualization_msgs::Marker::LINE_LIST;
    margin_lines.action = visualization_msgs::Marker::ADD;
    margin_lines.pose.orientation.w = 1.0;
    margin_lines.lifetime = ros::Duration(0);

    margin_lines.scale.x = 0.1;
    margin_lines.color.r = 0.0;
    margin_lines.color.g = 0.0;
    margin_lines.color.b = 0.0;
    margin_lines.color.a = 1.0;

    for(auto &it_per_id : estimator.f_manager.line_feature)
    {
        int used_num = it_per_id.line_feature_per_frame.size();
        if(it_per_id.solve_flag == 1 && used_num == 1 && it_per_id.start_frame == 0)
        {
//            if(it_per_id.line_feature_per_frame[0].vp(2) == 0)
//                continue;

            int imu_i = it_per_id.start_frame;
            Matrix3d R_wc = estimator.Rs[imu_i] * estimator.ric[0];
            Vector3d t_wc = estimator.Rs[imu_i] * estimator.tic[0] + estimator.Ps[imu_i] ;

            Vector3d sp_2d_c = it_per_id.line_feature_per_frame[0].start_point;
            Vector3d ep_2d_c = it_per_id.line_feature_per_frame[0].end_point;
            Vector3d sp_2d_p_c = Vector3d(sp_2d_c(0) + scale, -scale*(ep_2d_c(0) - sp_2d_c(0))/(ep_2d_c(1) - sp_2d_c(1)) + sp_2d_c(1), 1);
            Vector3d ep_2d_p_c = Vector3d(ep_2d_c(0) + scale, -scale*(ep_2d_c(0) - sp_2d_c(0))/(ep_2d_c(1) - sp_2d_c(1)) + ep_2d_c(1), 1);

            Vector3d pi_s = sp_2d_c.cross(sp_2d_p_c);
            Vector3d pi_e = ep_2d_c.cross(ep_2d_p_c);

            Vector4d pi_s_4d, pi_e_4d;
            pi_s_4d.head(3) = pi_s;
            pi_s_4d(3) = 1;
            pi_e_4d.head(3) = pi_e;
            pi_e_4d(3) = 1;

            AngleAxisd roll(it_per_id.orthonormal_vec(0), Vector3d::UnitX());
            AngleAxisd pitch(it_per_id.orthonormal_vec(1), Vector3d::UnitY());
            AngleAxisd yaw(it_per_id.orthonormal_vec(2), Vector3d::UnitZ());
            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> Rotation_psi;
            Rotation_psi = roll * pitch * yaw;
            double pi = it_per_id.orthonormal_vec(3);

            Vector3d n_w = cos(pi) * Rotation_psi.block<3,1>(0,0);
            Vector3d d_w = sin(pi) * Rotation_psi.block<3,1>(0,1);

            Matrix<double, 6, 1> line_w;
            line_w.block<3,1>(0,0) = n_w;
            line_w.block<3,1>(3,0) = d_w;

            Matrix<double, 6, 6> T_cw;
            T_cw.setZero();
            T_cw.block<3,3>(0,0) = R_wc.transpose();
            T_cw.block<3,3>(0,3) = Utility::skewSymmetric(-R_wc.transpose()*t_wc) * R_wc.transpose();
            T_cw.block<3,3>(3,3) = R_wc.transpose();

            Matrix<double, 6, 1> line_c = T_cw * line_w;
            Vector3d n_c = line_c.block<3,1>(0,0);
            Vector3d d_c = line_c.block<3,1>(3,0);

            double dist_w = n_w.norm()/d_w.norm();
            double dist_c = n_c.norm()/d_c.norm();
//            cout << dist_w << ", " <<  dist_c << endl;

//            if(dist_w < 1.0)
//                continue;

            Matrix4d L_c;
            L_c.setZero();
            L_c.block<3,3>(0,0) = Utility::skewSymmetric(n_c);
            L_c.block<3,1>(0,3) = d_c;
            L_c.block<1,3>(3,0) = -d_c.transpose();

            Vector4d D_s = L_c * pi_s_4d;
            Vector4d D_e = L_c * pi_e_4d;
            Vector3d D_s_3d(D_s(0)/D_s(3), D_s(1)/D_s(3), D_s(2)/D_s(3));
            Vector3d D_e_3d(D_e(0)/D_e(3), D_e(1)/D_e(3), D_e(2)/D_e(3));

            Vector3d D_s_w = R_wc * D_s_3d + t_wc;
            Vector3d D_e_w = R_wc * D_e_3d + t_wc;

            sp.x = D_s_w(0);
            sp.y = D_s_w(1);
            sp.z = D_s_w(2);

            ep.x = D_e_w(0);
            ep.y = D_e_w(1);
            ep.z = D_e_w(2);

            if(
                    isnan(D_s_w(0)) || isnan(D_s_w(1)) || isnan(D_s_w(2)) ||
                    isnan(D_e_w(0)) || isnan(D_e_w(1)) || isnan(D_e_w(2))
                    || D_s_3d(2) < 0 || D_e_3d(2) < 0
                    || (D_s_w - D_e_w).norm() > 10
               )
            {
//                it_per_id.solve_flag = 2;
                continue;
            }

            visualization_msgs::Marker text_marker;
            text_marker.header.frame_id = "/world";
            text_marker.header.stamp = ros::Time::now();
            text_marker.color.a = 1.0;
            text_marker.color.r = 1.0;
            text_marker.scale.z = 1.0;
            text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            text_marker.id = i++;
            text_marker.action = visualization_msgs::Marker::ADD;
            text_marker.pose.position.x = sp.x;
            text_marker.pose.position.y = sp.y;
            text_marker.pose.position.z = sp.z;
//            text_marker.text =
//                    std::to_string(it_per_id.orthonormal_vec(0)) + ", " +
//                    std::to_string(it_per_id.orthonormal_vec(1)) + ", " +
//                    std::to_string(it_per_id.orthonormal_vec(2)) + ", " +
//                    std::to_string(it_per_id.orthonormal_vec(3));
//            text_marker.text =
//                    std::to_string(d_c(0)) + ", " +
//                    std::to_string(d_c(1)) + ", " +
//                    std::to_string(d_c(2));
            text_marker.text =
                    std::to_string(dist_w) + ", " +
                    std::to_string(dist_c);

            text_array.markers.push_back(text_marker);

            margin_lines.points.push_back(sp);
            margin_lines.points.push_back(ep);
        }
    }
    pub_text.publish(text_array);

    float max_dist = 45; //Need tuning, MH:45 VR:10
    for(int i = 0; i < points.size()/2; i++)
    {
        float mid_dist = sqrt(pow(points.at(i*2).x, 2) +
                               pow(points.at(i*2).y, 2)) +
                          sqrt(pow(points.at(i*2 + 1).x, 2) +
                               pow(points.at(i*2 + 1).y, 2));
        float ratio = mid_dist/max_dist;
        int index = static_cast<int>(ratio * num);
        if(index < num){
            margin_line_list.at(index).points.push_back(points.at(i * 2));
            margin_line_list.at(index).points.push_back(points.at(i * 2 + 1));
        }
    }

    pub_line_margin.publish(margin_lines);
    pub_line_deg.publish(margin_lines_deg);
    publishAll(pub_margin_line_list, margin_line_list);
}


void pubTF(const Estimator &estimator, const std_msgs::Header &header)
{
    if( estimator.solver_flag != Estimator::SolverFlag::NON_LINEAR)
        return;
    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    // body frame
    Vector3d correct_t;
    Quaterniond correct_q;
    correct_t = estimator.Ps[WINDOW_SIZE];
    correct_q = estimator.Rs[WINDOW_SIZE];

    transform.setOrigin(tf::Vector3(correct_t(0),
                                    correct_t(1),
                                    correct_t(2)));
    q.setW(correct_q.w());
    q.setX(correct_q.x());
    q.setY(correct_q.y());
    q.setZ(correct_q.z());
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, header.stamp, "world", "body"));

    // camera frame
    transform.setOrigin(tf::Vector3(estimator.tic[0].x(),
                        estimator.tic[0].y(),
            estimator.tic[0].z()));
    q.setW(Quaterniond(estimator.ric[0]).w());
    q.setX(Quaterniond(estimator.ric[0]).x());
    q.setY(Quaterniond(estimator.ric[0]).y());
    q.setZ(Quaterniond(estimator.ric[0]).z());
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, header.stamp, "body", "camera"));

    nav_msgs::Odometry odometry;
    odometry.header = header;
    odometry.header.frame_id = "world";
    odometry.pose.pose.position.x = estimator.tic[0].x();
    odometry.pose.pose.position.y = estimator.tic[0].y();
    odometry.pose.pose.position.z = estimator.tic[0].z();
    Quaterniond tmp_q{estimator.ric[0]};
    odometry.pose.pose.orientation.x = tmp_q.x();
    odometry.pose.pose.orientation.y = tmp_q.y();
    odometry.pose.pose.orientation.z = tmp_q.z();
    odometry.pose.pose.orientation.w = tmp_q.w();
    pub_extrinsic.publish(odometry);

}

void pubKeyframe(const Estimator &estimator)
{
    // pub camera pose, 2D-3D points of keyframe
    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR && estimator.marginalization_flag == 0)
    {
        int i = WINDOW_SIZE - 2;
        //Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[0];
        Vector3d P = estimator.Ps[i];
        Quaterniond R = Quaterniond(estimator.Rs[i]);

        nav_msgs::Odometry odometry;
        odometry.header = estimator.Headers[WINDOW_SIZE - 2];
        odometry.header.frame_id = "world";
        odometry.pose.pose.position.x = P.x();
        odometry.pose.pose.position.y = P.y();
        odometry.pose.pose.position.z = P.z();
        odometry.pose.pose.orientation.x = R.x();
        odometry.pose.pose.orientation.y = R.y();
        odometry.pose.pose.orientation.z = R.z();
        odometry.pose.pose.orientation.w = R.w();
        //printf("time: %f t: %f %f %f r: %f %f %f %f\n", odometry.header.stamp.toSec(), P.x(), P.y(), P.z(), R.w(), R.x(), R.y(), R.z());

        pub_keyframe_pose.publish(odometry);


        sensor_msgs::PointCloud point_cloud;
        point_cloud.header = estimator.Headers[WINDOW_SIZE - 2];
        for (auto &it_per_id : estimator.f_manager.feature)
        {
            int frame_size = it_per_id.feature_per_frame.size();
            if(it_per_id.start_frame < WINDOW_SIZE - 2 && it_per_id.start_frame + frame_size - 1 >= WINDOW_SIZE - 2 && it_per_id.solve_flag == 1)
            {

                int imu_i = it_per_id.start_frame;
                Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
                Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0])
                        + estimator.Ps[imu_i];
                geometry_msgs::Point32 p;
                p.x = w_pts_i(0);
                p.y = w_pts_i(1);
                p.z = w_pts_i(2);
                point_cloud.points.push_back(p);

                int imu_j = WINDOW_SIZE - 2 - it_per_id.start_frame;
                sensor_msgs::ChannelFloat32 p_2d;
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].point.x());
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].point.y());
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].uv.x());
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].uv.y());
                p_2d.values.push_back(it_per_id.feature_id);
                point_cloud.channels.push_back(p_2d);
            }

        }
        pub_keyframe_point.publish(point_cloud);



    }
}

void pubRelocalization(const Estimator &estimator)
{
    nav_msgs::Odometry odometry;
    odometry.header.stamp = ros::Time(estimator.relo_frame_stamp);
    odometry.header.frame_id = "world";
    odometry.pose.pose.position.x = estimator.relo_relative_t.x();
    odometry.pose.pose.position.y = estimator.relo_relative_t.y();
    odometry.pose.pose.position.z = estimator.relo_relative_t.z();
    odometry.pose.pose.orientation.x = estimator.relo_relative_q.x();
    odometry.pose.pose.orientation.y = estimator.relo_relative_q.y();
    odometry.pose.pose.orientation.z = estimator.relo_relative_q.z();
    odometry.pose.pose.orientation.w = estimator.relo_relative_q.w();
    odometry.twist.twist.linear.x = estimator.relo_relative_yaw;
    odometry.twist.twist.linear.y = estimator.relo_frame_index;

    pub_relo_relative_pose.publish(odometry);
}

void declarePublisher(int num, ros::NodeHandle &n, std::vector<ros::Publisher> &MarkerPublisher)
{
    for(int i = 0; i < num; i++)
    {
        std::string s = std::to_string(i);
        std::string name = "line_marker" + s;
        ros::Publisher pub_color_line = n.advertise<visualization_msgs::Marker>(name, 10);
        MarkerPublisher.push_back(pub_color_line);
    }
}

void declareLineLists(std_msgs::Header &header, int num, std::vector<visualization_msgs::Marker> &lineList)
{
    for(int i = 0; i < num; i++)
    {
        visualization_msgs::Marker marker;
        marker.header = header;
        marker.header.frame_id = "world";
        marker.ns = "margin_lines";
        marker.type = visualization_msgs::Marker::LINE_LIST;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.lifetime = ros::Duration(0);
        marker.id = i;

        float r,g,b;
        getHSVColor(static_cast<float>(i)/num, r, g, b);
        marker.scale.x = 0.05;
//        marker.color.r = r;
//        marker.color.g = g;
//        marker.color.b = b;
        marker.color.a = 1.0;

        if(i == 0)
        {
            marker.color.r = 1.0;
            marker.color.g = 1.0;
            marker.color.b = 1.0;
        }
        else if(i == 1)
        {
            marker.color.r = 1.0;
            marker.color.g = 0.0;
            marker.color.b = 0.0;
        }
        else if(i == 2)
        {
            marker.color.r = 0.0;
            marker.color.g = 1.0;
            marker.color.b = 0.0;
        }
        else if(i == 3)
        {
            marker.color.r = 0.0;
            marker.color.g = 0.0;
            marker.color.b = 1.0;
        }
        lineList.push_back(marker);
    }
}

void getHSVColor(float h, float& red, float & green, float & blue)
{
    if(h > 1) h = 1;
    if(h < 0) h = 0;

    h = h * 0.667;


    double color_R;
    double color_G;
    double color_B;
    // blend over HSV-values (more colors)

    double s = 1.0;
    double v = 1.0;

    h -= floor(h);
    h *= 6;
    int i;
    double m, n, f;

    i = floor(h);
    f = h - i;
    if (!(i & 1))
        f = 1 - f; // if i is even
    m = v * (1 - s);
    n = v * (1 - s * f);

    switch (i) {
    case 6:
    case 0:
        color_R = v; color_G = n; color_B = m;
        break;
    case 1:
        color_R = n; color_G = v; color_B = m;
        break;
    case 2:
        color_R = m; color_G = v; color_B = n;
        break;
    case 3:
        color_R = m; color_G = n; color_B = v;
        break;
    case 4:
        color_R = n; color_G = m; color_B = v;
        break;
    case 5:
        color_R = v; color_G = m; color_B = n;
        break;
    default:
        color_R = 1; color_G = 0.5; color_B = 0.5;
        break;
    }
    red = color_R;
    green = color_G;
    blue = color_B;

}

void publishAll(std::vector<ros::Publisher>& MarkerPublishers, std::vector<visualization_msgs::Marker>& lineLists){
    for (int i = 0; i < MarkerPublishers.size(); ++i){
        MarkerPublishers.at(i).publish(lineLists.at(i));
    }
}
