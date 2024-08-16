#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>
#include "map_relocalization/common_lib.h"

class Optimizer{
public:
    static bool PoseOptimizationSingle(Eigen::Matrix3d& Rcw, Eigen::Vector3d& pcw, const unordered_map<int, int>& matches, const vec_vec4d& boxes, const vec_vec3d& lamp_world_pos, const Matrix6d cov, int nIterations = 10);
    static bool PoseOptimizationCenter(Eigen::Matrix3d& Rcw, Eigen::Vector3d& pcw, const deq_mat3d& deq_deltaR, const deq_vec3d& deq_deltap, const deque<unordered_map<int, int>>& matches, const deque<vec_vec4d, Eigen::aligned_allocator<vec_vec4d>> boxes, const vec_vec3d& lamp_world_pos, const int& frames, int nIterations = 10);
    static bool PoseOptimizationAllPt(Eigen::Matrix3d& Rcw, Eigen::Vector3d& pcw, const deq_mat3d& deq_deltaR, const deq_vec3d& deq_deltap, const deque<unordered_map<int, int>>& matches, const deque<vec_vec4d, Eigen::aligned_allocator<vec_vec4d>> boxes, const vector<vec_vec3d, Eigen::aligned_allocator<vec_vec3d>>& lamp_world_points, const int& frames, int nIterations = 10);
    static bool PoseOptimizationEpline(Eigen::Matrix3d& Rcw, Eigen::Vector3d& pcw, const deq_mat3d& deq_Rcw, const deq_vec3d& deq_pcw, const deque<unordered_map<int, int>>& matches, const deque<vec_vec4d, Eigen::aligned_allocator<vec_vec4d>> boxes, const vec_vec3d& lamp_world_pos, const int& frames, int nIterations = 10);
    static bool PoseOptimizationAngle(Eigen::Matrix3d& Rcw, Eigen::Vector3d& pcw, const deq_mat3d& deq_Rcw, const deq_vec3d& deq_pcw, const deque<unordered_map<int, int>>& matches, const deque<vec_vec4d, Eigen::aligned_allocator<vec_vec4d>> boxes, const deque<map_fid_vec5d>& features, const vec_vec3d& lamp_world_pos, const int& frames, const Matrix6d& cov, int nIterations = 10);
    static bool PoseOptimizationAngleBin(Eigen::Matrix3d& Rcw, Eigen::Vector3d& pcw, deq_mat3d& deq_Rcw, deq_vec3d& deq_pcw, const deque<unordered_map<int, int>>& matches, const deque<vec_vec4d, Eigen::aligned_allocator<vec_vec4d>> boxes, const deque<map_fid_vec5d>& features, const vec_vec3d& lamp_world_pos, const int& frames, const Matrix6d& cov, int nIterations = 10);
    static double avg_error_;
};

class EdgeSE3PriorPose: public g2o::BaseUnaryEdge<6, g2o::SE3Quat, g2o::VertexSE3Expmap>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeSE3PriorPose(){}

    virtual bool read(std::istream& is){}

    virtual bool write(std::ostream& os) const{}

    virtual void computeError() override{
        const g2o::VertexSE3Expmap* v1 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);

        g2o::SE3Quat error_ = _measurement.inverse() * v1->estimate();
        _error = error_.log();
    }

    virtual void linearizeOplus() override{
        g2o::VertexSE3Expmap* v1 = static_cast<g2o::VertexSE3Expmap*>(_vertices[0]);
        g2o::SE3Quat pose = v1->estimate();

        Matrix6d J = JRInv(pose);
        _jacobianOplusXi = J * pose.adj();
    }

    Matrix6d JRInv(const g2o::SE3Quat& e){
        Matrix6d J;
        J.block(0, 0, 3, 3) = vector_skew(e.log().head<3>());
        J.block(0, 3, 3, 3) = vector_skew(e.log().tail<3>());
        J.block(3, 0, 3, 3) = Eigen::Matrix3d::Zero();
        J.block(3, 3, 3, 3) = vector_skew(e.log().head<3>());
        J = 0.5 * J + Matrix6d::Identity();
        return J;
    }
};

class EdgeSE3CenterProjFixedHistPose: public g2o::BaseUnaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeSE3CenterProjFixedHistPose(const double& fx, const double& fy, const double& cx, const double& cy, const g2o::SE3Quat deltaPose): fx_(fx), fy_(fy), cx_(cx), cy_(cy), deltaPose_(deltaPose){

    }

    virtual bool read(std::istream& is){}

    virtual bool write(std::ostream& os) const{}

    virtual void computeError() override{
        const g2o::VertexSE3Expmap* v1 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        g2o::SE3Quat rePose = deltaPose_ * v1->estimate();
        _error = _measurement - cam_project(rePose.map(Xw_));
        cout << "mea: " << _measurement.transpose() << endl;
        cout << "proj: " << cam_project(rePose.map(Xw_)).transpose() << endl;
        // cout << "_error: " << endl << _error << endl;
    }

    virtual void linearizeOplus() override{
        g2o::VertexSE3Expmap* vi = static_cast<g2o::VertexSE3Expmap*>(_vertices[0]);
        Eigen::Vector3d tmp_xyz = vi->estimate().map(Xw_);
        Eigen::Vector3d xyz_trans = deltaPose_.map(tmp_xyz);

        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double invz = 1.0 / xyz_trans[2];
        double invz_2 = invz * invz;

        Eigen::MatrixXd jaco_projP(2, 3);
        jaco_projP << fx_ * invz, 0, - fx_ * x * invz_2,
                      0, fy_ * invz, - fy_ * y * invz_2;

        Eigen::Matrix3d delta_R = deltaPose_.rotation().toRotationMatrix();
        
        Eigen::MatrixXd jaco_pose(3, 6);
        jaco_pose.block(0, 0, 3, 3) = - vector_skew(tmp_xyz);
        jaco_pose.block(0, 3, 3, 3) = Eigen::Matrix3d::Identity();
        
        _jacobianOplusXi = - jaco_projP * delta_R * jaco_pose;
    }

    Eigen::Vector2d cam_project(const Eigen::Vector3d& trans_xyz) const{
        Eigen::Vector2d norm_z(trans_xyz(0) / trans_xyz(2), trans_xyz(1) / trans_xyz(2));
        Eigen::Vector2d proj;
        proj(0) = norm_z(0) * fx_ + cx_;
        proj(1) = norm_z(1) * fy_ + cy_;
        // cout << "_proj: " << endl << proj << endl;
        return proj;
    }

    Eigen::Vector3d Xw_;
    g2o::SE3Quat deltaPose_;
    double fx_, fy_, cx_, cy_;
};

class EdgeSE3AllProjFixedHistPose: public g2o::BaseUnaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeSE3AllProjFixedHistPose(const double& fx, const double& fy, const double& cx, const double& cy, const g2o::SE3Quat deltaPose): fx_(fx), fy_(fy), cx_(cx), cy_(cy), deltaPose_(deltaPose){
        alpha_ = 2;
    }

    virtual bool read(std::istream& is){}

    virtual bool write(std::ostream& os) const{}

    virtual void computeError() override{
        const g2o::VertexSE3Expmap* v1 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        g2o::SE3Quat rePose = deltaPose_ * v1->estimate();
        Eigen::Vector2d pt = cam_project(rePose.map(Xw_));
        if(pt.x() < x_min_){
            _error.x() = _measurement.x() - pt.x();
        }
        else if(pt.x() > x_max_){
            _error.x() = pt.x() - _measurement.x();
        }
        else{
            double d1 = pt.x() - x_min_;
            double d2 = pt.x() - x_max_;
            if(abs(d1) > abs(d2)){
                _error.x() = alpha_ * exp(1.0 / alpha_ * (pt.x() - x_max_)) + x_max_ - _measurement.x() - alpha_;
            }
            else{
                _error.x() = alpha_ * exp(1.0 / alpha_ * (x_min_ - pt.x())) + _measurement.x() - x_min_ - alpha_;
            }
        }
        
        if(pt.y() < y_min_){
            _error.y() = _measurement.y() - pt.y();
        }
        else if(pt.y() > y_max_){
            _error.y() = pt.y() - _measurement.y();
        }
        else{
            double d1 = pt.y() - y_min_;
            double d2 = pt.y() - y_max_;
            if(abs(d1) > abs(d2)){
                _error.y() = alpha_ * exp(1.0 / alpha_ * (pt.y() - y_max_)) + y_max_ - _measurement.y() - alpha_;
            }
            else{
                _error.y() = alpha_ * exp(1.0 / alpha_ * (y_min_ - pt.y())) + _measurement.y() - y_min_ - alpha_;
            }
        }
    }

    virtual void linearizeOplus() override{
        g2o::VertexSE3Expmap* vi = static_cast<g2o::VertexSE3Expmap*>(_vertices[0]);
        Eigen::Vector3d tmp_xyz = vi->estimate().map(Xw_);
        Eigen::Vector3d xyz_trans = deltaPose_.map(tmp_xyz);
        Eigen::Vector2d pt = cam_project(xyz_trans);

        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double invz = 1.0 / xyz_trans[2];
        double invz_2 = invz * invz;

        Eigen::MatrixXd jaco_projP(2, 3);
        jaco_projP << fx_ * invz, 0, - fx_ * x * invz_2,
                      0, fy_ * invz, - fy_ * y * invz_2;

        Eigen::Matrix3d delta_R = deltaPose_.rotation().toRotationMatrix();
        
        Eigen::MatrixXd jaco_pose(3, 6);
        jaco_pose.block(0, 0, 3, 3) << 0, tmp_xyz.z(), - tmp_xyz.y(),
                                       - tmp_xyz.z(), 0, tmp_xyz.x(),
                                       tmp_xyz.y(), - tmp_xyz.x(), 0;
        jaco_pose.block(0, 3, 3, 3) = Eigen::Matrix3d::Identity();
        
        _jacobianOplusXi = jaco_projP * delta_R * jaco_pose;
        if(pt.x() < x_min_){
            _jacobianOplusXi.row(0) *= -1;
        }
        else if(pt.x() > x_max_){
        }
        else{
            double d1 = pt.x() - x_min_;
            double d2 = pt.x() - x_max_;
            if(abs(d1) > abs(d2)){
                _jacobianOplusXi.row(0) *= exp(1.0 / alpha_ * (pt.x() - x_max_));
            }
            else{
                _jacobianOplusXi.row(0) *= -exp(1.0 / alpha_ * (x_min_ - pt.x()));
            }
        }

        if(pt.y() < y_min_){
            _jacobianOplusXi.row(1) *= -1;
        }
        else if(pt.y() > y_max_){
        }
        else{
            double d1 = pt.y() - y_min_;
            double d2 = pt.y() - y_max_;
            if(abs(d1) > abs(d2)){
                _jacobianOplusXi.row(1) *= exp(1.0 / alpha_ * (pt.y() - y_max_));
            }
            else{
                _jacobianOplusXi.row(1) *= -exp(1.0 / alpha_ * (y_min_ - pt.y()));
            }
        }
    }

    Eigen::Vector2d cam_project(const Eigen::Vector3d& trans_xyz) const{
        Eigen::Vector2d norm_z(trans_xyz(0) / trans_xyz(2), trans_xyz(1) / trans_xyz(2));
        Eigen::Vector2d proj;
        proj(0) = norm_z(0) * fx_ + cx_;
        proj(1) = norm_z(1) * fy_ + cy_;
        return proj;
    }

    Eigen::Vector3d Xw_;
    g2o::SE3Quat deltaPose_;
    double fx_, fy_, cx_, cy_;
    int x_min_, x_max_, y_min_, y_max_;
    double alpha_;
};

class VertexSO3Expt: public g2o::BaseVertex<6, Eigen::Matrix<double, 7, 1>>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexSO3Expt(){}

    virtual bool read(std::istream& is){}

    virtual bool write(std::ostream& os) const {}

    virtual void setToOriginImpl(){
        Eigen::Quaterniond origin_quat = Eigen::Quaterniond::Identity();
        _estimate << origin_quat.w(), origin_quat.x(), origin_quat.y(), origin_quat.z(), 0, 0, 0;
    }

    virtual void oplusImpl(const double* update){
        Eigen::Map<const Vector6d> tmp(update);
        Eigen::AngleAxisd rot(tmp.head<3>().norm(), tmp.head<3>().normalized());
        Eigen::Quaterniond ql(rot);
        ql.normalize();
        
        Eigen::Quaterniond q(_estimate(0), _estimate(1), _estimate(2), _estimate(3));
        q = ql * q;
        q.normalize();

        _estimate.head<4>() << q.w(), q.x(), q.y(), q.z();
        _estimate.tail<3>() << _estimate(4) + tmp(3), _estimate(5) + tmp(4), _estimate(6) + tmp(5);
    }
};

class EdgeSO3tPriorPose: public g2o::BaseUnaryEdge<6, g2o::SE3Quat, VertexSO3Expt>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeSO3tPriorPose(){}

    virtual bool read(std::istream& is){}

    virtual bool write(std::ostream& os) const {}

    virtual void computeError() override{
        const VertexSO3Expt* v1 = static_cast<const VertexSO3Expt*>(_vertices[0]);
        Eigen::Quaterniond quat(v1->estimate()(0), v1->estimate()(1), v1->estimate()(2), v1->estimate()(3));
        Eigen::Vector3d vec(v1->estimate().tail<3>());

        g2o::SE3Quat rePose(quat, vec);
        g2o::SE3Quat error_ = _measurement.inverse() * rePose;
        _error.head<3>() = error_.log().head<3>();
        _error.tail<3>() = error_.translation();
    }

    virtual void linearizeOplus() override{
        VertexSO3Expt* v1 = static_cast<VertexSO3Expt*>(_vertices[0]);
        Eigen::Quaterniond quat(v1->estimate()(0), v1->estimate()(1), v1->estimate()(2), v1->estimate()(3));
        Eigen::Vector3d vec(v1->estimate().tail<3>());

        g2o::SE3Quat rePose(quat, vec);
        g2o::SE3Quat error_ = _measurement.inverse() * rePose;
        
        Matrix6d J = Matrix6d::Identity();
        J.block<3, 3>(0, 0) = JRInv(error_.log().head<3>()) * rePose.rotation().toRotationMatrix().transpose();
        J.block<3, 3>(3, 3) = _measurement.rotation().toRotationMatrix().transpose();

        _jacobianOplusXi = J;
    }

    Eigen::Matrix3d JRInv(const Eigen::Vector3d& e){
        Eigen::Matrix3d J;
        Eigen::Vector3d normd_e = e.normalized();
        double half_theta = e.norm() * 0.5;
        double half_theta_cot = half_theta * cos(half_theta) / (sin(half_theta) + 1e-5);

        J = half_theta_cot * Eigen::Matrix3d::Identity() + (1 - half_theta_cot) * normd_e * normd_e.transpose() + half_theta * vector_skew(normd_e);
        return J;
    }
};


class EdgeSO3tCenterProj: public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexSO3Expt>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeSO3tCenterProj(const double& fx, const double& fy, const double& cx, const double& cy): fx_(fx), fy_(fy), cx_(cx), cy_(cy){

    }

    virtual bool read(std::istream& is){}

    virtual bool write(std::ostream& os) const{}

    virtual void computeError() override{
        const VertexSO3Expt* v1 = static_cast<const VertexSO3Expt*>(_vertices[0]);
        Eigen::Quaterniond quat(v1->estimate()(0), v1->estimate()(1), v1->estimate()(2), v1->estimate()(3));
        Eigen::Vector3d vec(v1->estimate().tail<3>());

        g2o::SE3Quat rePose(quat, vec);
        // cout << "rePose: " << endl << rePose << endl;
        _error = _measurement - cam_project(rePose.map(Xw_));
        cout << "error: " << endl << _error << endl;
    }

    virtual void linearizeOplus() override{
        VertexSO3Expt* vi = static_cast<VertexSO3Expt*>(_vertices[0]);
        Eigen::Quaterniond quat(vi->estimate()(0), vi->estimate()(1), vi->estimate()(2), vi->estimate()(3));
        Eigen::Vector3d vec(vi->estimate().tail<3>());

        g2o::SE3Quat rePose(quat, vec);
        Eigen::Vector3d xyz_trans = rePose.map(Xw_);

        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double invz = 1.0 / xyz_trans[2];
        double invz_2 = invz * invz;

        Eigen::MatrixXd jaco_projP(2, 3);
        jaco_projP << fx_ * invz, 0, - fx_ * x * invz_2,
                      0, fy_ * invz, - fy_ * y * invz_2;

        Eigen::MatrixXd jaco_pose(3, 6);        
        Eigen::Vector3d xyz_tmp = xyz_trans - rePose.translation();
        jaco_pose.block(0, 0, 3, 3) << 0, xyz_tmp.z(), - xyz_tmp.y(),
                                       - xyz_tmp.z(), 0, xyz_tmp.x(),
                                       xyz_tmp.y(), - xyz_tmp.x(), 0;

        jaco_pose.block(0, 3, 3, 3) = Eigen::Matrix3d::Identity();
        
        _jacobianOplusXi = - jaco_projP * jaco_pose;
    }

    Eigen::Vector2d cam_project(const Eigen::Vector3d& trans_xyz) const{
        Eigen::Vector2d norm_z(trans_xyz(0) / trans_xyz(2), trans_xyz(1) / trans_xyz(2));
        Eigen::Vector2d proj;
        proj(0) = norm_z(0) * fx_ + cx_;
        proj(1) = norm_z(1) * fy_ + cy_;
        return proj;
    }

    Eigen::Vector3d Xw_;
    double fx_, fy_, cx_, cy_;
};


class EdgeSO3tEplineFixedHistPose: public g2o::BaseUnaryEdge<1, double, VertexSO3Expt>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeSO3tEplineFixedHistPose(const Eigen::Matrix3d& hist_R, const Eigen::Vector3d& hist_t): hist_R_(hist_R), hist_t_(hist_t){

    }

    virtual bool read(std::istream& is){}

    virtual bool write(std::ostream& os) const{}

    void computeError(){
        const VertexSO3Expt* v1 = static_cast<const VertexSO3Expt*>(_vertices[0]);
        Eigen::Quaterniond quat(v1->estimate()(0), v1->estimate()(1), v1->estimate()(2), v1->estimate()(3));
        Eigen::Matrix3d R(quat);
        Eigen::Vector3d t(v1->estimate().tail<3>());

        Eigen::Matrix3d R_cur_hist = R * hist_R_.transpose();
        Eigen::Matrix3d R_hist_cur = hist_R_ * R.transpose();
        Eigen::Vector3d t_cur_hist = - R_cur_hist * hist_t_ + t;
        Eigen::Vector3d t_hist_cur = - R_hist_cur * t + hist_t_;

        _error(0, 0) = _measurement - liftup_cur_.transpose() * vector_skew(t_cur_hist) * R_cur_hist * liftup_hist_;
        _error(0, 0) += _measurement - liftup_hist_.transpose() * vector_skew(t_hist_cur) * R_hist_cur * liftup_cur_;
    }

    virtual void linearizeOplus(){
        const VertexSO3Expt* vi = static_cast<const VertexSO3Expt*>(_vertices[0]);
        Eigen::Quaterniond quat(vi->estimate()(0), vi->estimate()(1), vi->estimate()(2), vi->estimate()(3));
        Eigen::Matrix3d R(quat);
        Eigen::Vector3d t(vi->estimate().tail<3>());

        Eigen::Matrix3d R_cur_hist = R * hist_R_.transpose();
        Eigen::Matrix3d R_hist_cur = hist_R_ * R.transpose();
        Eigen::Vector3d t_cur_hist = - R_cur_hist * hist_t_ + t;
        Eigen::Vector3d t_hist_cur = - R_hist_cur * t + hist_t_;

        _jacobianOplusXi.block(0, 0, 1, 3) = liftup_cur_.transpose() * vector_skew(t_cur_hist) * vector_skew(R_cur_hist * liftup_hist_);
        _jacobianOplusXi.block(0, 3, 1, 3) = liftup_cur_.transpose() * vector_skew(R_cur_hist * liftup_hist_);
        _jacobianOplusXi.block(0, 0, 1, 3) -= liftup_hist_.transpose() * vector_skew(t_hist_cur) * R_hist_cur * vector_skew(liftup_cur_);
        _jacobianOplusXi.block(0, 3, 1, 3) -= liftup_hist_.transpose() * vector_skew(R_hist_cur * liftup_cur_) * R_hist_cur;
    }

    Eigen::Vector3d liftup_cur_, liftup_hist_;
    Eigen::Matrix3d hist_R_;
    Eigen::Vector3d hist_t_;
};

class EdgeSO3tForwardAngleFixedHistPose: public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexSO3Expt>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeSO3tForwardAngleFixedHistPose(const Eigen::Matrix3d& hist_R, const Eigen::Vector3d& hist_t): hist_R_(hist_R), hist_t_(hist_t){
    }

    virtual bool read(std::istream& is){}

    virtual bool write(std::ostream& os) const{}

    void computeError(){
        const VertexSO3Expt* v1 = static_cast<const VertexSO3Expt*>(_vertices[0]);
        Eigen::Quaterniond quat(v1->estimate()(0), v1->estimate()(1), v1->estimate()(2), v1->estimate()(3));
        Eigen::Matrix3d R(quat);
        Eigen::Vector3d t(v1->estimate().tail<3>());

        R_cur_hist_ = R * hist_R_.transpose();
        t_cur_hist_ = - R_cur_hist_ * hist_t_ + t;

        Eigen::Vector3d trans_hist = R_cur_hist_ * liftup_hist_ + t_cur_hist_;
        inv_norm_trans_hist_ = 1.0 / trans_hist.norm();
        normd_trans_hist_ = trans_hist.normalized();

        _error = _measurement - normd_trans_hist_;
        // cout << "_error: " << endl << _error << endl;
        cout << inv_norm_trans_hist_ << endl;
    }

    virtual void linearizeOplus(){
        const VertexSO3Expt* vi = static_cast<const VertexSO3Expt*>(_vertices[0]);
        Eigen::Quaterniond quat(vi->estimate()(0), vi->estimate()(1), vi->estimate()(2), vi->estimate()(3));
        Eigen::Matrix3d R(quat);
        Eigen::Vector3d t(vi->estimate().tail<3>());

        _jacobianOplusXi.block(0, 0, 3, 3) = (Eigen::Matrix3d::Identity() - normd_trans_hist_ * normd_trans_hist_.transpose()) * vector_skew(R_cur_hist_ * (liftup_hist_ - hist_t_)) * inv_norm_trans_hist_;
        _jacobianOplusXi.block(0, 3, 3, 3) = (- Eigen::Matrix3d::Identity() + normd_trans_hist_ * normd_trans_hist_.transpose()) * inv_norm_trans_hist_;
        cout << "_jacobianOplusXi: " << endl << _jacobianOplusXi << endl;
    }

    Eigen::Vector3d liftup_hist_;

    Eigen::Matrix3d hist_R_;
    Eigen::Vector3d hist_t_;

    Eigen::Matrix3d R_cur_hist_;
    Eigen::Vector3d t_cur_hist_;

    Eigen::Vector3d normd_trans_hist_;
    double inv_norm_trans_hist_;
};

class EdgeSO3tBackwardAngleFixedHistPose: public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexSO3Expt>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeSO3tBackwardAngleFixedHistPose(const Eigen::Matrix3d& hist_R, const Eigen::Vector3d& hist_t): hist_R_(hist_R), hist_t_(hist_t){
    }

    virtual bool read(std::istream& is){}

    virtual bool write(std::ostream& os) const{}

    void computeError(){
        const VertexSO3Expt* v1 = static_cast<const VertexSO3Expt*>(_vertices[0]);
        Eigen::Quaterniond quat(v1->estimate()(0), v1->estimate()(1), v1->estimate()(2), v1->estimate()(3));
        Eigen::Matrix3d R(quat);
        Eigen::Vector3d t(v1->estimate().tail<3>());

        R_hist_cur_ = hist_R_ * R.transpose();
        t_hist_cur_ = - R_hist_cur_ * t + hist_t_;

        Eigen::Vector3d trans_cur = R_hist_cur_ * liftup_cur_ + t_hist_cur_;
        inv_norm_trans_cur_ = 1.0 / trans_cur.norm();
        normd_trans_cur_ = trans_cur.normalized();

        _error = _measurement - normd_trans_cur_;
    }

    virtual void linearizeOplus(){
        const VertexSO3Expt* vi = static_cast<const VertexSO3Expt*>(_vertices[0]);
        Eigen::Quaterniond quat(vi->estimate()(0), vi->estimate()(1), vi->estimate()(2), vi->estimate()(3));
        Eigen::Matrix3d R(quat);
        Eigen::Vector3d t(vi->estimate().tail<3>());

        _jacobianOplusXi.block(0, 0, 3, 3) = (- Eigen::Matrix3d::Identity() + normd_trans_cur_ * normd_trans_cur_.transpose()) * inv_norm_trans_cur_ * R_hist_cur_ * vector_skew(liftup_cur_ - t);
        _jacobianOplusXi.block(0, 3, 3, 3) = (Eigen::Matrix3d::Identity() - normd_trans_cur_ * normd_trans_cur_.transpose()) * inv_norm_trans_cur_ * R_hist_cur_;
    }

    Eigen::Vector3d liftup_cur_;

    Eigen::Matrix3d hist_R_;
    Eigen::Vector3d hist_t_;

    Eigen::Matrix3d R_hist_cur_;
    Eigen::Vector3d t_hist_cur_;

    Eigen::Vector3d normd_trans_cur_;
    double inv_norm_trans_cur_;
};

class EdgeSO3tForwardAngleNotFixedHistPose: public g2o::BaseBinaryEdge<3, Eigen::Vector3d, VertexSO3Expt, VertexSO3Expt>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual bool read(std::istream& is){}

    virtual bool write(std::ostream& os) const{}

    void computeError(){
        const VertexSO3Expt* v1 = static_cast<const VertexSO3Expt*>(_vertices[0]);
        Eigen::Quaterniond quat(v1->estimate()(0), v1->estimate()(1), v1->estimate()(2), v1->estimate()(3));
        Eigen::Matrix3d R(quat);
        Eigen::Vector3d t(v1->estimate().tail<3>());

        const VertexSO3Expt* v2 = static_cast<const VertexSO3Expt*>(_vertices[1]);
        Eigen::Quaterniond hist_quat(v2->estimate()(0), v2->estimate()(1), v2->estimate()(2), v2->estimate()(3));
        Eigen::Matrix3d hist_R(hist_quat);
        Eigen::Vector3d hist_t(v2->estimate().tail<3>());

        R_cur_hist_ = R * hist_R.transpose();
        t_cur_hist_ = - R_cur_hist_ * hist_t + t;

        Eigen::Vector3d trans_hist = R_cur_hist_ * liftup_hist_ + t_cur_hist_;
        inv_norm_trans_hist_ = 1.0 / trans_hist.norm();
        normd_trans_hist_ = trans_hist.normalized();

        _error = _measurement - normd_trans_hist_;
        // cout << "_error: " << endl << _error << endl;
        cout << inv_norm_trans_hist_ << endl;
    }

    virtual void linearizeOplus(){
        const VertexSO3Expt* v1 = static_cast<const VertexSO3Expt*>(_vertices[0]);
        Eigen::Quaterniond quat(v1->estimate()(0), v1->estimate()(1), v1->estimate()(2), v1->estimate()(3));
        Eigen::Matrix3d R(quat);
        Eigen::Vector3d t(v1->estimate().tail<3>());

        const VertexSO3Expt* v2 = static_cast<const VertexSO3Expt*>(_vertices[1]);
        Eigen::Quaterniond hist_quat(v2->estimate()(0), v2->estimate()(1), v2->estimate()(2), v2->estimate()(3));
        Eigen::Matrix3d hist_R(hist_quat);
        Eigen::Vector3d hist_t(v2->estimate().tail<3>());

        _jacobianOplusXi.block(0, 0, 3, 3) = (Eigen::Matrix3d::Identity() - normd_trans_hist_ * normd_trans_hist_.transpose()) * vector_skew(R_cur_hist_ * (liftup_hist_ - hist_t)) * inv_norm_trans_hist_;
        _jacobianOplusXi.block(0, 3, 3, 3) = (- Eigen::Matrix3d::Identity() + normd_trans_hist_ * normd_trans_hist_.transpose()) * inv_norm_trans_hist_;
        // cout << "_jacobianOplusXi: " << endl << _jacobianOplusXi << endl;

        _jacobianOplusXj.block(0, 0, 3, 3) = (- Eigen::Matrix3d::Identity() + normd_trans_hist_ * normd_trans_hist_.transpose()) * inv_norm_trans_hist_ * R_cur_hist_ * vector_skew(liftup_hist_ - hist_t);
        _jacobianOplusXj.block(0, 3, 3, 3) = (Eigen::Matrix3d::Identity() - normd_trans_hist_ * normd_trans_hist_.transpose()) * inv_norm_trans_hist_ * R_cur_hist_;
    }

    Eigen::Vector3d liftup_hist_;

    Eigen::Matrix3d R_cur_hist_;
    Eigen::Vector3d t_cur_hist_;

    Eigen::Vector3d normd_trans_hist_;
    double inv_norm_trans_hist_;
};

#endif