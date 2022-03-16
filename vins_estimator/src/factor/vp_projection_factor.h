#ifndef STRUCTURAL_FACTOR_H
#define STRUCTURAL_FACTOR_H

#include <ros/assert.h>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "../utility/utility.h"
#include "../parameters.h"
#include <malloc.h>
#include <ceres/rotation.h>

using namespace Eigen;

struct VPProjectionFactor
{
    VPProjectionFactor(Matrix3d _ric, Vector3d _tic, Vector3d _sp, Vector3d _ep, Vector3d _vp)
        : ric(_ric), tic(_tic), sp(_sp), ep(_ep), vp(_vp){}

    template <typename T>
    bool operator()(const T* const pose,
                    const T* const line,
                    T* residuals) const
    {
        const Eigen::Matrix<T, 3, 1> t_wb(pose[0], pose[1], pose[2]);
        const Eigen::Quaternion<T> q_wb(pose[6], pose[3], pose[4], pose[5]);
        const Eigen::AngleAxis<T> roll(line[0], Matrix<T,3,1>::UnitX());
        const Eigen::AngleAxis<T> pitch(line[1], Matrix<T,3,1>::UnitY());
        const Eigen::AngleAxis<T> yaw(line[2], Matrix<T,3,1>::UnitZ());
        const T pi = line[3];

        Eigen::Matrix<T, 3, 3> R_wc = q_wb * ric.template cast<T>();
        Eigen::Matrix<T, 3, 1> t_wc = q_wb * tic.template cast<T>() + t_wb;
        Eigen::Matrix<T, 3, 3, RowMajor> Rotation_psi;
        Rotation_psi = roll * pitch * yaw;

        Matrix<T, 3, 1> n_w = cos(pi) * Rotation_psi.template block<3,1>(0,0);
        Matrix<T, 3, 1> d_w = sin(pi) * Rotation_psi.template block<3,1>(0,1);
        Matrix<T, 6, 1> l_w;
        l_w.template block<3,1>(0,0) = n_w;
        l_w.template block<3,1>(3,0) = d_w;

        Matrix<T, 6, 6> T_cw;

        Matrix<T, 3, 1> t_cw = -R_wc.transpose() * t_wc;
        Matrix<T, 3, 3> t_cw_ss;
        t_cw_ss << T(0.0), -t_cw(2), t_cw(1),
                   t_cw(2), T(0.0), -t_cw(0),
                   -t_cw(1), t_cw(0), T(0.0);

        T_cw.setZero();
        T_cw.template block<3,3>(0,0) = R_wc.transpose();
        T_cw.template block<3,3>(0,3) = t_cw_ss * R_wc.transpose();
        T_cw.template block<3,3>(3,3) = R_wc.transpose();

        Matrix<T, 6, 1> l_c = T_cw * l_w;
        Matrix<T, 3, 1> n_c = l_c.template block<3,1>(0,0);
        Matrix<T, 3, 1> d_c = l_c.template block<3,1>(3,0);

        Matrix<T, 2, 1> d_c_2d(d_c(0)/d_c(2), d_c(1)/d_c(2));
        Matrix<T, 2, 1> vp_2d(vp(0), vp(1));

        residuals[0] = T(VP_FACTOR) * (d_c_2d(0) - vp_2d(0));
        residuals[1] = T(VP_FACTOR) * (d_c_2d(1) - vp_2d(1));
//        residuals[0] = T(VP_FACTOR) * (d_c_2d - vp_2d).norm();

        return true;
    }
private:
    Matrix3d ric;
    Vector3d tic;
    Vector3d sp;
    Vector3d ep;
    Vector3d vp;
};

#endif // STRUCTURAL_FACTOR_H
