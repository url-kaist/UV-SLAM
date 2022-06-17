#include "feature_manager.h"

int LineFeaturePerId::endFrame()
{
    return start_frame + line_feature_per_frame.size() - 1;
}

int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric[i].setIdentity();
}

void FeatureManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];
    }
}

void FeatureManager::clearState()
{
    feature.clear();
    line_feature.clear();
}

int FeatureManager::getLineFeatureCount()
{
    int cnt = 0;
    for (auto &it : line_feature)
    {
        it.used_num = it.line_feature_per_frame.size();

        //std::cout << "LINE/frame #: " << it.line_feature_per_frame.size() << std::endl;
        if(it.used_num < LINE_WINDOW)
            continue;
//        if(it.solve_flag == 0)
//            continue;

        cnt++;

    }
//    cout << "getLineFeatureCount: " << cnt << endl;

    //std::cout << "GETlinefeaturecount #: " << cnt << std::endl;

    return cnt;
}

int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &it : feature)
    {

        it.used_num = it.feature_per_frame.size();

        if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2)
        {
            cnt++;
        }
    }
    return cnt;
}

bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image,
                                                              const map<int, vector<Eigen::Matrix<double, 15, 1>>> &image_line, double td)
{

    //std::cout << "in addFeatureCheck" << std::endl;
    //std::cout << "frame_cout=start_frame: " << frame_count << std::endl;
    ROS_DEBUG("input feature: %d", (int)image.size());
    ROS_DEBUG("num of feature: %d", getFeatureCount());

    double parallax_sum = 0;
    int parallax_num = 0;
    last_track_num = 0;
    for (auto &id_pts : image)
    {
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td);

        int feature_id = id_pts.first;
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it)
        {
            return it.feature_id == feature_id;
        });

        if (it == feature.end())
        {
            feature.push_back(FeaturePerId(feature_id, frame_count));
            feature.back().feature_per_frame.push_back(f_per_fra);
        }
        else if (it->feature_id == feature_id)
        {
            it->feature_per_frame.push_back(f_per_fra);
            last_track_num++;
        }
    }

    unsigned int num_tracked_line = 0;
    for (auto &id_lines : image_line)
    {
        LineFeaturePerFrame l_per_fra(id_lines.second[0], td);

        int line_id = id_lines.first;

        auto it = find_if(line_feature.begin(), line_feature.end(), [line_id](const LineFeaturePerId &it)
        {
            return it.feature_id == line_id;
        });

        if (it == line_feature.end())
        {
            line_feature.push_back(LineFeaturePerId(line_id, frame_count));
            line_feature.back().line_feature_per_frame.push_back(l_per_fra);
//            cout << line_id << ", " << id_lines.second[0].first << endl;
        }
        else if (it->feature_id == line_id)
        {
            it->line_feature_per_frame.push_back(l_per_fra);
            num_tracked_line++;
            // ADDED CODE
            it->max_num = it->max_num+1;
        }
    }


    if (frame_count < 2 || last_track_num < 20)
        return true;

    for (auto &it_per_id : feature)
    {
        if (it_per_id.start_frame <= frame_count - 2 &&
                it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1)
        {
            parallax_sum += compensatedParallax2(it_per_id, frame_count);
            parallax_num++;
        }
    }

    if (parallax_num == 0)
    {
        return true;
    }
    else
    {
        ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
}

void FeatureManager::debugShow()
{
    ROS_DEBUG("debug show");
    for (auto &it : feature)
    {
        ROS_ASSERT(it.feature_per_frame.size() != 0);
        ROS_ASSERT(it.start_frame >= 0);
        ROS_ASSERT(it.used_num >= 0);

        ROS_DEBUG("%d,%d,%d ", it.feature_id, it.used_num, it.start_frame);
        int sum = 0;
        for (auto &j : it.feature_per_frame)
        {
            ROS_DEBUG("%d,", int(j.is_used));
            sum += j.is_used;
            printf("(%lf,%lf) ",j.point(0), j.point(1));
        }
        ROS_ASSERT(it.used_num == sum);
    }
}

vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &it : feature)
    {
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;

            a = it.feature_per_frame[idx_l].point;

            b = it.feature_per_frame[idx_r].point;

            corres.push_back(make_pair(a, b));
        }
    }
    return corres;
}


vector<pair<Vector3d, Vector3d>> FeatureManager::getLineCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;

//    std::cout << "[getCorr]line_feature size:" << line_feature.size() << std::endl;
    for (auto &it : line_feature)
    {

      //std::cout << "[getCorr]frame_count_l:" << frame_count_l << std::endl;
      //std::cout << "[getCorr]frame_count_r:" << frame_count_r << std::endl;

        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;

            a = it.line_feature_per_frame[idx_l].start_point;

            b = it.line_feature_per_frame[idx_r].start_point;

//            std::cout << "[getCorr]it.start_frame:" << it.start_frame << std::endl;
//            std::cout << "[getCorr]it.endFrame:" << it.endFrame() << std::endl;
//            std::cout << "[getCorr]point a: \n" << a << std::endl;
//            std::cout << "[getCorr]point b: \n" << b << std::endl;

            corres.push_back(make_pair(a, b));
        }
    }
    return corres;
}

void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        //ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2;
        }
        else
            it_per_id.solve_flag = 1;
    }
}

void FeatureManager::removeFailures()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2)
            feature.erase(it);
    }
}

void FeatureManager::removeLineFailures()
{
    for(auto it = line_feature.begin(), it_next = line_feature.begin();
        it != line_feature.end(); it = it_next)
    {
        it_next++;
        if(it->solve_flag == 2)
            line_feature.erase(it);
    }
}


void FeatureManager::clearDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth = 1.0 / x(++feature_index);
    }
}

VectorXd FeatureManager::getDepthVector()
{
    VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
#if 1
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
    }
    return dep_vec;
}

vector<Vector4d> FeatureManager::getLineOrthonormal()
{
//    cout << "getLineOrthonormal: " << getLineFeatureCount() << endl;
    vector<Vector4d> LineOrtho(getLineFeatureCount());
    int line_feature_index =-1;
    for(auto &it_line_per_id : line_feature)
    {
        it_line_per_id.used_num = it_line_per_id.line_feature_per_frame.size();
        if(it_line_per_id.used_num < LINE_WINDOW)
            continue;
//        if(it_line_per_id.solve_flag == 0)
//            continue;

        Vector4d tmp_vector;
        tmp_vector = it_line_per_id.orthonormal_vec;
        LineOrtho[++line_feature_index] = tmp_vector; //it_line_per_id.orthonormal_vec;

//        cout << tmp_vector[0] << ", " <<
//                tmp_vector[1] << ", " <<
//                tmp_vector[2] << ", " <<
//                tmp_vector[3] << endl;
    }
    return LineOrtho;
}

void FeatureManager::setLineOrtho(vector<Vector4d> &get_lineOrtho, Vector3d Ps[], Matrix3d Rs[],Vector3d tic, Matrix3d ric)
{
    int line_feature_index =-1;
    for (auto &it_per_id : line_feature)
    {
        it_per_id.used_num = it_per_id.line_feature_per_frame.size();
        if(it_per_id.used_num < LINE_WINDOW)
            continue;
//        if(it_line_per_id.solve_flag == 0)
//            continue;

        ++line_feature_index;

//        cout << "-------" << endl;
//        cout << it_line_per_id.orthonormal_vec[0] << " -> " << get_lineOrtho.at(line_feature_index)[0] << endl;
//        cout << it_line_per_id.orthonormal_vec[1] << " -> " << get_lineOrtho.at(line_feature_index)[1] << endl;
//        cout << it_line_per_id.orthonormal_vec[2] << " -> " << get_lineOrtho.at(line_feature_index)[2] << endl;
//        cout << it_line_per_id.orthonormal_vec[3] << " -> " << get_lineOrtho.at(line_feature_index)[3] << endl;

        AngleAxisd roll(it_per_id.orthonormal_vec(0), Vector3d::UnitX());
        AngleAxisd pitch(it_per_id.orthonormal_vec(1), Vector3d::UnitY());
        AngleAxisd yaw(it_per_id.orthonormal_vec(2), Vector3d::UnitZ());
        double pi = it_per_id.orthonormal_vec(3);

        Eigen::Matrix<double, 3, 3, Eigen::RowMajor> Rotation_psi;
        Rotation_psi = roll * pitch * yaw;

        Vector3d n_w = cos(pi) * Rotation_psi.block<3,1>(0,0);
        Vector3d d_w = sin(pi) * Rotation_psi.block<3,1>(0,1);

        int imu_i = it_per_id.start_frame;
        Matrix3d R_wc = Rs[imu_i] * ric;
        Vector3d t_wc = Rs[imu_i] * tic + Ps[imu_i];

        Matrix<double, 6, 1> l_w;
        l_w.block<3,1>(0,0) = n_w;
        l_w.block<3,1>(3,0) = d_w;

        Matrix<double, 6, 6> T_cw;
        T_cw.setZero();
        T_cw.block<3,3>(0,0) = R_wc.transpose();
        T_cw.block<3,3>(0,3) = Utility::skewSymmetric(-R_wc.transpose()*t_wc) * R_wc.transpose();
        T_cw.block<3,3>(3,3) = R_wc.transpose();

        Matrix<double, 6, 1> l_c = T_cw * l_w;
        Vector3d n_c = l_c.block<3,1>(0,0);
        Vector3d d_c = l_c.block<3,1>(3,0);

Matrix4d L_c;
        L_c.setZero();
        L_c.block<3,3>(0,0) = Utility::skewSymmetric(n_c);
        L_c.block<3,1>(0,3) = d_c;
        L_c.block<1,3>(3,0) = -d_c.transpose();

        double scale = 1.0;
        Vector3d sp_2d_c = it_per_id.line_feature_per_frame[0].start_point;
        Vector3d ep_2d_c = it_per_id.line_feature_per_frame[0].end_point;
        Vector3d sp_2d_p_c = Vector3d(sp_2d_c(0) + scale, -scale*(ep_2d_c(0) - sp_2d_c(0))/(ep_2d_c(1) - sp_2d_c(1)) + sp_2d_c(1), 1);
        Vector3d ep_2d_p_c = Vector3d(ep_2d_c(0) + scale, -scale*(ep_2d_c(0) - sp_2d_c(0))/(ep_2d_c(1) - sp_2d_c(1)) + ep_2d_c(1), 1);

        Vector3d pi_s = sp_2d_c.cross(sp_2d_p_c);
        Vector3d pi_e = ep_2d_c.cross(ep_2d_p_c);

        Vector4d pi_s_4d, pi_e_4d;
        pi_s_4d.head(3) = pi_s;
        pi_s_4d(3) = 0;
        pi_e_4d.head(3) = pi_e;
        pi_e_4d(3) = 0;

        Vector4d D_s = L_c * pi_s_4d;
        Vector4d D_e = L_c * pi_e_4d;
        Vector3d D_s_3d(D_s(0)/D_s(3), D_s(1)/D_s(3), D_s(2)/D_s(3));
        Vector3d D_e_3d(D_e(0)/D_e(3), D_e(1)/D_e(3), D_e(2)/D_e(3));

        Vector3d D_s_w = R_wc * D_s_3d + t_wc;
        Vector3d D_e_w = R_wc * D_e_3d + t_wc;

        if(D_s_3d(2) < 0 || D_e_3d(2) < 0)
        {
            it_per_id.solve_flag = 2;
            continue;
        }
        else
            it_per_id.solve_flag = 1;

        it_per_id.orthonormal_vec[0] = get_lineOrtho.at(line_feature_index)[0];
        it_per_id.orthonormal_vec[1] = get_lineOrtho.at(line_feature_index)[1];
        it_per_id.orthonormal_vec[2] = get_lineOrtho.at(line_feature_index)[2];
        it_per_id.orthonormal_vec[3] = get_lineOrtho.at(line_feature_index)[3];
    }
}



void FeatureManager::triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[])
{
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        if (it_per_id.estimated_depth > 0)
            continue;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        ROS_ASSERT(NUM_OF_CAM == 1);
        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
        int svd_idx = 0;

        Eigen::Matrix<double, 3, 4> P0;
        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d R = R0.transpose() * R1;
            Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            Eigen::Vector3d f = it_per_frame.point.normalized();
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            if (imu_i == imu_j)
                continue;
        }
        ROS_ASSERT(svd_idx == svd_A.rows());
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double svd_method = svd_V[2] / svd_V[3];
        //it_per_id->estimated_depth = -b / A;
        //it_per_id->estimated_depth = svd_V[2] / svd_V[3];

        it_per_id.estimated_depth = svd_method;
        //it_per_id->estimated_depth = INIT_DEPTH;

        if (it_per_id.estimated_depth < 0.1)
        {
            it_per_id.estimated_depth = INIT_DEPTH;
        }

    }
}

int sign(double x){
    if ( x > 0 ) return 1;
    else return -1;
}

double SafeAcos (double x)
{
    if (x < -1.0) x = -1.0 ;
    else if (x > 1.0) x = 1.0 ;
    return acos(x) ;
}

bool DoesFileExist (const std::string& name) {
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }
}

void FeatureManager::triangulateLine(Vector3d Ps[], Matrix3d Rs_estimate[], Vector3d tic[], Matrix3d ric[], Mat img)
{
    Vector4d left_plane, right_plane;
    Vector3d left_plane_3d, right_plane_3d;

    Vector3d direction_l;
    Vector3d normal_l;
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> Rotation_psi;

    Vector3d sp_3d;
    Vector3d ep_3d;

    int index = -1;

    cvtColor(img, img, CV_GRAY2BGR);
    for(auto &it_per_id : line_feature)
    {
        it_per_id.used_num = it_per_id.line_feature_per_frame.size();
        if(it_per_id.orthonormal_vec[3] != 0 || it_per_id.line_feature_per_frame.size() < 2)
            continue;

        int imu_i = it_per_id.start_frame;
        int imu_j = it_per_id.start_frame + it_per_id.used_num - 1;

        Matrix3d R_left = Rs[imu_i] * ric[0];
        Quaterniond q_left(R_left);
        Vector3d t_left = Rs[imu_i] * tic[0] + Ps[imu_i] ;

        Matrix3d R_right = Rs[imu_j] * ric[0];
        Quaterniond q_right(R_right);
        Vector3d t_right = Rs[imu_j] * tic[0] + Ps[imu_j] ;

        Quaterniond relative_q = q_left.inverse() * q_right;
        Vector3d relative_t = q_left.inverse().toRotationMatrix() * (t_right - t_left);

        Vector3d left_sp = it_per_id.line_feature_per_frame[0].start_point;
        Vector3d left_ep = it_per_id.line_feature_per_frame[0].end_point;

        Vector3d right_sp = it_per_id.line_feature_per_frame[it_per_id.used_num - 1].start_point;
        Vector3d right_ep = it_per_id.line_feature_per_frame[it_per_id.used_num - 1].end_point;

        Vector3d right_sp_l = relative_q * right_sp;
        Vector3d right_ep_l = relative_q * right_ep;

        Vector3d t_cj_ci = q_right.inverse().toRotationMatrix() * (t_left - t_right);
        t_cj_ci = t_cj_ci/t_cj_ci(2);

//        if(it_per_id.solve_flag == 0)
//        {

        calcPluckerLine(left_sp, left_ep, right_sp_l, right_ep_l,
                        Vector3d(0, 0, 0), relative_t, direction_l, normal_l,
                        left_plane, right_plane);

        Vector3d left_plane_3d(left_plane(0), left_plane(1), left_plane(2));
        Vector3d right_plane_3d(right_plane(0), right_plane(1), right_plane(2));

        Matrix<double, 6, 1> line_w, line_l;
        line_l.block<3,1>(0,0) = normal_l;
        line_l.block<3,1>(3,0) = direction_l;

        Matrix<double, 6, 6> T_wl;
        T_wl.setZero();
        T_wl.block<3,3>(0,0) = R_left;
        T_wl.block<3,3>(0,3) = Utility::skewSymmetric(t_left) * R_left;
        T_wl.block<3,3>(3,3) = R_left;

        line_w = T_wl * line_l;

        Vector3d n_w, d_w;
        n_w = line_w.block<3,1>(0,0);
        d_w = line_w.block<3,1>(3,0);

        Rotation_psi.block<3,1>(0,0) = n_w/(n_w.norm());
        Rotation_psi.block<3,1>(0,1) = d_w/(d_w.norm());
        Rotation_psi.block<3,1>(0,2) = n_w.cross(d_w)/(n_w.cross(d_w).norm());

//        if(n_w.norm()/d_w.norm() < 1.0)
//            continue;

        it_per_id.orthonormal_vec.head(3) = Rotation_psi.eulerAngles(0,1,2);
        it_per_id.orthonormal_vec(3) = atan2(n_w.norm(), d_w.norm());

        index++;
    }
}

void FeatureManager::removeOutlier()
{
    ROS_BREAK();
    int i = -1;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        i += it->used_num != 0;
        if (it->used_num != 0 && it->is_outlier == true)
        {
            feature.erase(it);
        }
    }
}

void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() < 2)
            {
                feature.erase(it);
                continue;
            }
            else
            {
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                double dep_j = pts_j(2);
                if (dep_j > 0)
                    it->estimated_depth = dep_j;
                else
                    it->estimated_depth = INIT_DEPTH;
            }
        }
        // remove tracking-lost feature after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }
}

void FeatureManager::removeBack()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

void FeatureManager::removeFront(int frame_count)
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame == frame_count)
        {
            it->start_frame--;
        }
        else
        {
            int j = WINDOW_SIZE - 1 - it->start_frame;
            if (it->endFrame() < frame_count - 1)
                continue;
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

void FeatureManager::removeLineBack()
{
    for (auto it = line_feature.begin(), it_next = line_feature.begin();
         it != line_feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            it->line_feature_per_frame.erase(it->line_feature_per_frame.begin());
            if (it->line_feature_per_frame.size() == 0)
                line_feature.erase(it);
        }
    }
}

void FeatureManager::removeLineFront(int frame_count)
{
    for (auto it = line_feature.begin(), it_next = line_feature.begin(); it != line_feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame == frame_count)
        {
            it->start_frame--;
        }
        else
        {
            int j = WINDOW_SIZE - 1 - it->start_frame;
            if (it->endFrame() < frame_count - 1)
                continue;
            it->line_feature_per_frame.erase(it->line_feature_per_frame.begin() + j);
            if (it->line_feature_per_frame.size() == 0)
                line_feature.erase(it);
        }
    }
}

double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
    //check the second last frame is keyframe or not
    //parallax betwwen seconde last frame and third last frame
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

    double ans = 0;
    Vector3d p_j = frame_j.point;

    double u_j = p_j(0);
    double v_j = p_j(1);

    Vector3d p_i = frame_i.point;
    Vector3d p_i_comp;

    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
    p_i_comp = p_i;
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;

    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}

void FeatureManager::calcOrthonormalRepresent(const Vector3d _direction_vec, const Vector3d _normal_vec, Vector3d &_out_psi, double &_out_phi)
{
    Matrix3d U_mat;

    double size_noraml_vec = sqrt(_normal_vec(0)*_normal_vec(0) + _normal_vec(1)*_normal_vec(1) + _normal_vec(2)*_normal_vec(2));
    Vector3d normalized_normal_vec = _normal_vec/size_noraml_vec;
    U_mat(0,0)=normalized_normal_vec(0);
    U_mat(1,0)=normalized_normal_vec(1);
    U_mat(2,0)=normalized_normal_vec(2);

    double size_direction_vec = sqrt(_direction_vec(0)*_direction_vec(0) + _direction_vec(1)*_direction_vec(1) + _direction_vec(2)*_direction_vec(2));
    Vector3d normalized_direction_vec = _direction_vec/size_direction_vec;
    U_mat(0,1)=normalized_direction_vec(0);
    U_mat(1,1)=normalized_direction_vec(1);
    U_mat(2,1)=normalized_direction_vec(2);

    Vector3d cross_btw_norm_dist;
    cross_btw_norm_dist = _normal_vec.cross(_direction_vec);
    double size_cross_btw_norm_dist = sqrt( cross_btw_norm_dist(0)*cross_btw_norm_dist(0) + cross_btw_norm_dist(1)*cross_btw_norm_dist(1) + cross_btw_norm_dist(2)*cross_btw_norm_dist(2));
    Vector3d normalized_cross_btw_norm_dist = cross_btw_norm_dist/size_cross_btw_norm_dist;
    U_mat(0,2)=normalized_cross_btw_norm_dist(0);
    U_mat(1,2)=normalized_cross_btw_norm_dist(1);
    U_mat(2,2)=normalized_cross_btw_norm_dist(2);

    // NEED TO CALCULATE pshi
    // https://stackoverflow.com/questions/15022630/how-to-calculate-the-angle-from-rotation-matrix
    //theta_x = atan2(r32,r33)
    //theta_y = atan2(-r31, sqrt(r32*r32 + r33*r33))
    //theta_z = atan2(r21, r11)

    double psi_x = atan2(U_mat(2,1), U_mat(2,2));
    double psi_y = atan2(-1.0*U_mat(2,0), sqrt(U_mat(2,1)*U_mat(2,1) + U_mat(2,2)*U_mat(2,2)));
    double psi_z = atan2(U_mat(1,0), U_mat(0,0));

    Vector3d out_psi = Vector3d(psi_x, psi_y, psi_z);
    _out_psi = out_psi;
    /*
  cout << "U_mat: \n" <<  U_mat << endl;
  cout << "theta_x: " << theta_x << endl;
  cout << "theta_y: " << theta_y << endl;
  cout << "theta_z: " << theta_z << endl;
  */

    Matrix2d W_mat;

    double size_norm_and_dis = sqrt( size_noraml_vec*size_noraml_vec + size_direction_vec*size_direction_vec );

    W_mat(0,0) = size_noraml_vec/size_norm_and_dis;
    W_mat(0,1) = -1.0* size_direction_vec/size_norm_and_dis;
    W_mat(1,0) = size_direction_vec/size_norm_and_dis;
    W_mat(1,1) = size_noraml_vec/size_norm_and_dis;

    /*
  cout << "W_mat: \n" << W_mat << endl;
  cout << "phi_1: " << acos(W_mat(0,0))<< endl;
  cout << "phi_2: " << asin(W_mat(1,0))<< endl;
  */

    //NEED TO CALCULATE phi

    double out_phi = acos(W_mat(0,0));
    _out_phi = out_phi;
}


void FeatureManager::calcPluckerLine(const Vector3d _prev_sp, const Vector3d _prev_ep,
                                     const Vector3d _curr_sp, const Vector3d _curr_ep,
                                     const Vector3d _origin_prev, const Vector3d _origin_curr,
                                     Vector3d &_out_direction_vec, Vector3d &_out_normal_vec,
                                     Vector4d &prev_plane, Vector4d &curr_plane)
{
    //  if ( good_match_vector.size() == 0 )
    //    return;
    //  //for(int i = 0; i < 1; i++) //_good_match_vector.size(); i++)
    //  int i = 0;

    //  Point2f prev_srt_pts = _prev_keyLine.at(_good_match_vector.at(i).queryIdx).getStartPoint();
    //  Point2f prev_end_pts = _prev_keyLine.at(_good_match_vector.at(i).queryIdx).getEndPoint();

    //  Point2f curr_srt_pts = _curr_keyLine.at(_good_match_vector.at(i).trainIdx).getStartPoint();
    //  Point2f curr_end_pts = _curr_keyLine.at(_good_match_vector.at(i).trainIdx).getEndPoint();


    Matrix3d prev_srt_skew;
    skewMatFromVector3d(_prev_sp, prev_srt_skew);
    Vector3d prev_end_vec = _prev_ep;
    Vector3d prev_mul_skew_vec = prev_srt_skew * prev_end_vec;
//    prev_mul_skew_vec.normalize();
    //    Vector4d prev_plane;
    prev_plane(0) = prev_mul_skew_vec(0);
    prev_plane(1) = prev_mul_skew_vec(1);
    prev_plane(2) = prev_mul_skew_vec(2);
    prev_plane(3) =
            prev_mul_skew_vec(0) * _origin_prev(0) +
            prev_mul_skew_vec(1) * _origin_prev(1) +
            prev_mul_skew_vec(2) * _origin_prev(2);

    Matrix3d curr_srt_skew;
    skewMatFromVector3d(_curr_sp, curr_srt_skew);
    //    Vector3d curr_end_vec = Vector3d(_curr_ep[0], _curr_ep[1], 1.0);
    Vector3d curr_end_vec = _curr_ep;
    Vector3d curr_mul_skew_vec = curr_srt_skew * curr_end_vec;
//    curr_mul_skew_vec.normalize();
    //    Vector4d curr_plane;
    curr_plane(0) = curr_mul_skew_vec(0);
    curr_plane(1) = curr_mul_skew_vec(1);
    curr_plane(2) = curr_mul_skew_vec(2);
    curr_plane(3) =
            curr_mul_skew_vec(0) * _origin_curr(0) +
            curr_mul_skew_vec(1) * _origin_curr(1) +
            curr_mul_skew_vec(2) * _origin_curr(2);

    prev_plane(3) = - prev_plane(3);
    curr_plane(3) = - curr_plane(3);
    /*
        cout <<"IN CALCPLUCK" <<endl;
        cout << prev_plane << endl;
        cout << curr_plane << endl;
    */
    //  cout <<"IN linefeaturetracker calcPlucker" << endl;
    //  cout <<"prev_srt skew: "<< curr_srt_skew << endl;
    //  cout <<"prev_end_vec: "<< curr_end_vec << endl;
    //  cout <<"prev_plane: "<< curr_plane << endl;
    //  cout <<"prev_plane^T: "<< curr_plane.transpose() << endl;


    Matrix4d dualPluckerMat = prev_plane*curr_plane.transpose() - curr_plane * prev_plane.transpose();
    /*
  cout <<"prev_plane*curr_plane.transpose(): "<< prev_plane*curr_plane.transpose() << endl;
  cout <<"curr_plane * prev_plane.transpose(): "<< curr_plane * prev_plane.transpose() << endl;
  cout <<"dualPluckerMat: "<< dualPluckerMat << endl;
  */

    _out_direction_vec(0) = dualPluckerMat(2,1);
    _out_direction_vec(1) = dualPluckerMat(0,2);
    _out_direction_vec(2) = dualPluckerMat(1,0);

    _out_normal_vec(0) = dualPluckerMat(0,3);
    _out_normal_vec(1) = dualPluckerMat(1,3);
    _out_normal_vec(2) = dualPluckerMat(2,3);
}

void FeatureManager::skewMatFromVector3d(const Vector3d &_in_pt, Matrix3d &_out_skew_mat)
{
    // INPUT
    //w1 w2 w3
    // OUTPUT
    // 0 -w3 w2
    // w3 0 -w1
    // -w2 w1 0
    _out_skew_mat(0,0) = 0.0;
    _out_skew_mat(0,1) = -1.0 * _in_pt[2];
    _out_skew_mat(0,2) = 1.0 * _in_pt[1];

    _out_skew_mat(1,0) = 1.0 * _in_pt[2];
    _out_skew_mat(1,1) = 0.0;
    _out_skew_mat(1,2) = -1.0 * _in_pt[0];

    _out_skew_mat(2,0) = -1.0 * _in_pt[1];
    _out_skew_mat(2,1) = 1.0 * _in_pt[0];
    _out_skew_mat(2,2) = 0.0;

}

Matrix<double, 6, 1> FeatureManager::plk_from_pose(  Matrix<double, 6, 1> plk_c, Eigen::Matrix3d Rcw, Eigen::Vector3d tcw ) {

    Eigen::Matrix3d Rwc = Rcw.transpose();
    Vector3d twc = -Rwc*tcw;
    return plk_to_pose( plk_c, Rwc, twc);
}

Matrix<double, 6, 1> FeatureManager::plk_to_pose( Matrix<double, 6, 1> plk_w, Eigen::Matrix3d Rcw, Eigen::Vector3d tcw ) {
    Vector3d nw = plk_w.head(3);
    Vector3d vw = plk_w.tail(3);
    Matrix3d tcw_skew;
    skewMatFromVector3d(tcw, tcw_skew);

    Vector3d nc = Rcw * nw + tcw_skew * Rcw * vw;
    Vector3d vc = Rcw * vw;

    Matrix<double, 6, 1> plk_c;
    plk_c.head(3) = nc;
    plk_c.tail(3) = vc;
    return plk_c;
}

Matrix<double, 6, 1> FeatureManager::orth_to_plk(Vector4d orth)
{
    Matrix<double, 6, 1> plk;

    Vector3d theta = orth.head(3);
    double phi = orth[3];

    double s1 = sin(theta[0]);
    double c1 = cos(theta[0]);
    double s2 = sin(theta[1]);
    double c2 = cos(theta[1]);
    double s3 = sin(theta[2]);
    double c3 = cos(theta[2]);

    Matrix3d R;
    R <<
      c2 * c3,   s1 * s2 * c3 - c1 * s3,   c1 * s2 * c3 + s1 * s3,
            c2 * s3,   s1 * s2 * s3 + c1 * c3,   c1 * s2 * s3 - s1 * c3,
            -s2,                  s1 * c2,                  c1 * c2;

    double w1 = cos(phi);
    double w2 = sin(phi);
    double d = w1/w2;      // 原点到直线的距离

    Vector3d u1 = R.col(0);
    Vector3d u2 = R.col(1);

    Vector3d n = w1 * u1;
    Vector3d v = w2 * u2;

    plk.head(3) = n;
    plk.tail(3) = v;

    //Vector3d Q = -R.col(2) * d;
    //plk.head(3) = Q.cross(v);
    //plk.tail(3) = v;

    return plk;
}

void FeatureManager::getHSVColor(float h, float& red, float & green, float & blue)
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
