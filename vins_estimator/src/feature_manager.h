#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
#include <random>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

#include <ros/console.h>
#include <ros/assert.h>

#include <opencv2/opencv.hpp>
#include <opencv2/line_descriptor.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>

#include <opencv2/viz/vizcore.hpp>

using namespace cv;
using namespace line_descriptor;


#include "parameters.h"

class LineFeaturePerFrame
{
  public:
    LineFeaturePerFrame(const Eigen::Matrix<double, 15, 1> &_point, double td)
    {
        start_point.x() = _point(0);
        start_point.y() = _point(1);
        start_point.z() = 1.0;
        end_point.x() = _point(2);
        end_point.y() = _point(3);
        end_point.z() = 1.0;
        start_uv.x() = _point(4);
        start_uv.y() = _point(5);
        end_uv.x() = _point(6);
        end_uv.y() = _point(7);
        start_velocity.x() = _point(8);
        start_velocity.y() = _point(9);
        end_velocity.x() = _point(10);
        end_velocity.y() = _point(11);
        vp.x() = _point(12);
        vp.y() = _point(13);
        vp.z() = _point(14);
        cur_td = td;
    }

    double cur_td;
    Vector3d start_point;
    Vector3d end_point;
    Vector2d start_uv;
    Vector2d end_uv;
    Vector2d start_velocity;
    Vector2d end_velocity;
    Vector3d vp;

    bool useful_Rt_flag; // FLAG to be used in residual (useful R|t)
    //int useful_Rt_count=0;

    Vector3d p_ls;
    Vector3d p_le;
};

class LineFeaturePerId
{
  public:
    const int feature_id;
    int start_frame;
    vector<LineFeaturePerFrame> line_feature_per_frame;

    int used_num;
    int max_num;
    int tri_calc_usable_line_count;
    //bool is_outlier;
    //bool is_margin;
    //minimal respresentation of a 3D spatial line (O)
    Vector3d psi;
    double pi;
    mutable int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;


    bool updated_pose;

    Vector3d direction_vec;
    Vector3d normal_vec;
    Vector4d orthonormal_vec;
    //Vector3d gt_p;

    vector<Vector3d> buf_p_ls;
    vector<Vector3d> buf_p_le;
    vector<int> buf_p_ls_index;

    Vector3d sp_3d;
    Vector3d ep_3d;

    Vector3d sp_3d_c0;
    Vector3d ep_3d_c0;
    Vector3d sp_3d_c1;
    Vector3d ep_3d_c1;
    Vector3d sp_2d_c0;
    Vector3d ep_2d_c0;
    Vector3d sp_2d_c1;
    Vector3d ep_2d_c1;
    Vector3d c0, c1;

    Matrix3d R;
    Vector3d t;

    Vector3d map_sp_3d_c0;
    Vector3d map_ep_3d_c0;
    Vector3d map_sp_3d_c1;
    Vector3d map_ep_3d_c1;

    int vp_id;

    LineFeaturePerId(int _feature_id, int _start_frame)
        : feature_id(_feature_id), start_frame(_start_frame), orthonormal_vec(0,0,0,0),
          used_num(0), psi(0, 0, 0),pi(0), solve_flag(0), updated_pose(0), direction_vec(0,0,0), normal_vec(0,0,0),
          sp_3d_c0(0,0,0), ep_3d_c0(0,0,0), sp_3d_c1(0,0,0), ep_3d_c1(0,0,0),
          sp_2d_c0(0,0,0), ep_2d_c0(0,0,0), sp_2d_c1(0,0,0), ep_2d_c1(0,0,0),
          c0(0,0,0), c1(0,0,0), max_num(0), sp_3d(0,0,0), ep_3d(0,0,0), vp_id(-1)
    {
        R = Matrix3d::Zero();
    }

    int endFrame();
};

class FeaturePerFrame
{
  public:
    FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point, double td)
    {
        point.x() = _point(0);
        point.y() = _point(1);
        point.z() = _point(2);
        uv.x() = _point(3);
        uv.y() = _point(4);
        velocity.x() = _point(5);
        velocity.y() = _point(6);
        cur_td = td;
    }
    double cur_td;
    Vector3d point;
    Vector2d uv;
    Vector2d velocity;
    double z;
    bool is_used;
    double parallax;
    MatrixXd A;
    VectorXd b;
    double dep_gradient;
};

class FeaturePerId
{
  public:
    const int feature_id;
    int start_frame;
    vector<FeaturePerFrame> feature_per_frame;

    int used_num;
    bool is_outlier;
    bool is_margin;
    double estimated_depth;
    int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;

    Vector3d gt_p;

    FeaturePerId(int _feature_id, int _start_frame)
        : feature_id(_feature_id), start_frame(_start_frame),
          used_num(0), estimated_depth(-1.0), solve_flag(0)
    {
    }

    int endFrame();
};

class FeatureManager
{
  public:
    FeatureManager(Matrix3d _Rs[]);

    void setRic(Matrix3d _ric[]);

    void clearState();

    int getFeatureCount();
    int getLineFeatureCount();

    bool addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image,
                                                  const map<int, vector<Eigen::Matrix<double, 15, 1>>> &image_line, double td);
    void debugShow();
    vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);

    vector<pair<Vector3d, Vector3d>> getLineCorresponding(int frame_count_l, int frame_count_r);


    //void updateDepth(const VectorXd &x);
    void setDepth(const VectorXd &x);
    void removeFailures();
    void removeLineFailures();
    void clearDepth(const VectorXd &x);
    VectorXd getDepthVector();
    void triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[]);
    void triangulateLine(Vector3d Ps[], Matrix3d Rs[],Vector3d tic[], Matrix3d ric[], Mat img);
    void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
    void removeBack();
    void removeFront(int frame_count);
    void removeLineBack();
    void removeLineFront(int frame_count);
    void removeOutlier();
    void calcPluckerLine(const Vector3d _prev_sp, const Vector3d _prev_ep,
                         const Vector3d _curr_sp, const Vector3d _curr_ep,
                         const Vector3d _origin_prev, const Vector3d _origin_curr,
                         Vector3d &_out_direction_vec, Vector3d &_out_normal_vec,
                         Vector4d &prev_plane, Vector4d &curr_plane);
    void skewMatFromVector3d(const Vector3d &_in_pt, Matrix3d &_out_skew_mat);
    void calcOrthonormalRepresent(const Vector3d _direction_vec, const Vector3d _normal_vec, Vector3d &_out_psi, double &_out_phi);

    Matrix<double, 6, 1> plk_from_pose( Matrix<double, 6, 1> plk_c, Eigen::Matrix3d Rcw, Eigen::Vector3d tcw );
    Matrix<double, 6, 1> plk_to_pose( Matrix<double, 6, 1> plk_w, Eigen::Matrix3d Rcw, Eigen::Vector3d tcw );
    Matrix<double, 6, 1> orth_to_plk(Vector4d orth);

    vector<Vector4d> getLineOrthonormal();
    void setOrthoPlucker(const vector<Vector4d> &get_lineOrtho);
    void setLineOrtho(vector<Vector4d> &get_lineOrtho, Vector3d Ps[], Matrix3d Rs[],Vector3d tic, Matrix3d ric);
    void getHSVColor(float h, float& red, float & green, float & blue);


    list<FeaturePerId> feature;
    list<LineFeaturePerId> line_feature;
    int last_track_num;
    viz::Viz3d myWindow;
    vector<viz::WLine> lines_prev_prev;

    int frame_diff_for_line = 4;

  private:
    double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
    const Matrix3d *Rs;
    Matrix3d ric[NUM_OF_CAM];

};

#endif
