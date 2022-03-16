#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>
#include <random>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "parameters.h"
#include "tic_toc.h"

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/line_descriptor.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>
#include <opencv/cv.h>

#include "math.h"
#include "utility.h"
#include "highgui.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;
using namespace cv;
using namespace line_descriptor;

typedef line_descriptor::BinaryDescriptor LineBD;
typedef line_descriptor::KeyLine LineKL;

class LineFeatureTracker
{
  public:
    LineFeatureTracker();

    void readImage4Line(const Mat &_img, double _cur_time);
    void imageUndistortion(Mat &_img, Mat &_out_undistort_img);
    void readIntrinsicParameter(const string &calib_file);
    void lineExtraction( Mat &cur_img, vector<LineKL> &_keyLine, Mat &_descriptor );
    void lineMergingTwoPhase( Mat &prev_img, Mat &cur_img, vector<LineKL> &prev_keyLine, vector<LineKL> &cur_keyLine, Mat &prev_descriptor, Mat &cur_descriptor, vector<DMatch> &good_match_vector );
    void lineMatching( vector<LineKL> &_prev_keyLine, vector<LineKL> &_curr_keyLine, Mat &_prev_descriptor, Mat &_curr_descriptor, vector<DMatch> &_good_match_vector);
    bool updateID(unsigned int i);
    void normalizePoints();

    void getVPHypVia2Lines(vector<KeyLine> cur_keyLine, vector<Vector3d> &para_vector, vector<double> &length_vector, vector<double> &orientation_vector,
                           std::vector<std::vector<Vector3d> > &vpHypo );
    void getSphereGrids(vector<KeyLine> cur_keyLine, vector<Vector3d> &para_vector, vector<double> &length_vector, vector<double> &orientation_vector,
                        std::vector<std::vector<double> > &sphereGrid );
    void getBestVpsHyp( std::vector<std::vector<double> > &sphereGrid, std::vector<std::vector<Vector3d> >  &vpHypo, std::vector<Vector3d> &vps  );
    void lines2Vps(vector<KeyLine> cur_keyLine, double thAngle, std::vector<Vector3d> &vps, std::vector<std::vector<int> > &clusters, vector<int> &vp_idx);
    void drawClusters( cv::Mat &img, std::vector<KeyLine> &lines, std::vector<std::vector<int> > &clusters );

    void lineRawResolution( Mat &cur_img, vector<LineKL> &predict_keyLines);

    camodocal::CameraPtr m_camera;
    camodocal::PinholeCameraPtr pinhole_camera;

    Mat Camera_Matrix = Mat(3,3,CV_32FC1, Scalar::all(0.0));
    Mat Discotrion_Coefficients = Mat(1, 4, CV_32FC1);
    Mat new_camera_matrix, undist_map1, undist_map2;

    Mat prev_img, curr_img, forw_img;
    Mat curr_descriptor, m_matched_descriptor;

    vector<Point2f> curr_start_pts, curr_end_pts;
    vector<line_descriptor::KeyLine> curr_keyLine, forw_keyLine, m_matched_keyLines;

    vector<int> ids, tmp_ids; // set every value of tmp_ids to -1 when the new lines are extracted
    vector<int> track_cnt, tmp_track_cnt; // not sure // initial value: 1
    vector<int> vp_ids, tmp_vp_ids;
    vector<Vector3d> vps;
    vector<Point2f> start_pts_velocity, end_pts_velocity;
    vector<Point2f> prev_start_un_pts, curr_start_un_pts, prev_end_un_pts, curr_end_un_pts; // not sure
    Vector3d vp;


    static int n_id;
    static int vp_id;
    int image_id = 0;

    Utility util;


    /// FOR KALMAN
    // States are position and velocity in X and Y directions; four states [X;Y;dX/dt;dY/dt]
    CvPoint pt_Prediction, pt_Correction;

    // Measurements are current position of the mouse [X;Y]
    CvMat* measurement = cvCreateMat(2, 1, CV_32FC1);

    // dynamic params (4), measurement params (2), control params (0)
    CvKalman* kalman  = cvCreateKalman(4, 2, 0);

    void CannyDetection(Mat &src, vector<line_descriptor::KeyLine> &keylines);
    bool getPointChain( const Mat & img, const Point2f pt, Point2f * chained_pt, int & direction, int step );
    void extractSegments( vector<Point2f> * points, vector<line_descriptor::KeyLine> * keylines);
    double distPointLine( const Mat & p, Mat & l );
    void additionalOperationsOnSegments(Mat & src, line_descriptor::KeyLine * kl);
    void pointInboardTest(Mat & src, Point2f * pt);
    bool mergeSegments(line_descriptor::KeyLine * kl1, line_descriptor::KeyLine * kl2, line_descriptor::KeyLine * kl_merged );
    void mergeLines(line_descriptor::KeyLine * kl1, line_descriptor::KeyLine * kl2, line_descriptor::KeyLine * kl_merged);
    void HoughDetection(Mat &src, vector<line_descriptor::KeyLine> &keylines);
    void OpticalFlowExtraction( Mat &prev_img, Mat &cur_img,
                               vector<LineKL> &prev_keyLine, vector<LineKL> &cur_keyLine,
                               Mat &prev_descriptor, Mat &cur_descriptor);
    double Union_dist(VectorXd a, VectorXd b);
    double Intersection_dist(VectorXd a, VectorXd b);
    VectorXd Union(VectorXd a, VectorXd b);
    VectorXd Intersection(VectorXd a, VectorXd b);
    double SafeAcos (double x);
    void removeRow(Eigen::MatrixXd& matrix, unsigned int rowToRemove) ;
    void removeColumn(Eigen::MatrixXd& matrix, unsigned int colToRemove);


    template<class tType>
    void incidentPoint( tType * pt, Mat & l );

    int threshold_length = 20;
    double threshold_dist = 1.5;
    int imagewidth, imageheight;
    int ROW_MARGIN = 15;
    int COL_MARGIN = 20;
};
