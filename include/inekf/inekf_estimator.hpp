#ifndef INEKF_ESTIMATOR_HPP
#define INEKF_ESTIMATOR_HPP

#include <boost/thread/condition.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>
#include <boost/circular_buffer.hpp>
#include "map_relocalization/common_lib.h"
#include "inekf/InEKF.hpp"
#include "inekf/imu_odom.hpp"
#include "inekf/wins_state.hpp"

class BodyEstimator{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        BodyEstimator();
        ~BodyEstimator(){};
        void setImuParams(const ImuParam& imu_params);
        bool enabled(){return enabled_;}
        void enableFilter(){ enabled_=true;}
        void enableBiasInitialized(){ bias_initialized_ = true; }
        bool biasInitialized(){ return bias_initialized_; }
        bool initBias(const ImuMeasurement<double>& imu_packet_in);
        // void initState();
        void initState(const ImuMeasurement<double>& imu_packet_in,  const WheelEncodeMeasurement& wheel_state_packet_in, WinsState& state, const Eigen::Matrix3d& R = Eigen::Matrix3d::Identity(), const Eigen::Vector3d& p = Eigen::Vector3d::Zero());
        void ReinitPoseBiasState(const Eigen::Matrix3d& R, const Eigen::Vector3d& p, WinsState& state, const double& t, const Eigen::Vector3d& ba, const Eigen::Vector3d& bg);
        void ReinitPoseVState(const Eigen::Matrix3d& R, const Eigen::Vector3d& p, WinsState& state, const double& t, const Eigen::Vector3d& v);
        void ReinitPose(const Eigen::Matrix3d& R, const Eigen::Vector3d& p, WinsState& state, const double& t);
        // void initState(const ImuMeasurement<double>& imu_packet_in, const VelocityMeasurement& velocity_packet_in, WinsState& state);    ///TODO
        void propagateIMU(const ImuMeasurement<double>& imu_packet_in, WinsState& state);
        void propagateIMU(const double& t, WinsState& state);
        void correctVelocity(const WheelEncodeMeasurement& wheel_state_packet_in, WinsState& state, const Eigen::Matrix3d& velocity_cov);
        void correctPose(const double& t, WinsState& state, const unordered_map<int, int>& matches, const vec_vec3d& lamp_cur_cam_pos, const vec_vec4d& boxes, double high_lamp);
        void rollBack();
        // void correctVelocity(const VelocityMeasurement& velocity_packet_in, WinsState& state, const Eigen::Matrix<double,3,3>& velocity_cov);   ///TODO
        
        InEKF getFilter() const{ return filter_;}
        RobotState getState() const{ return filter_.getState();}

    private:
        // ROS related
        std::vector<geometry_msgs::PoseStamped> poses_;
        // inekf related
        InEKF filter_;
        bool enabled_ = false;
        bool bias_initialized_ = false;
        bool static_bias_initialization_ = false;
        bool estimator_debug_enabled_ = false;
        bool use_imu_ori_est_init_bias_ = false;
        vec_vec6d bias_init_vec_;
        Eigen::Vector3d bg0_ = Eigen::Vector3d::Zero();
        Eigen::Vector3d ba0_ = Eigen::Vector3d::Zero();

        double t_prev_;
        uint32_t seq_;
        double velocity_t_thres_ = 0.1;
        double pose_t_thres_ = 0.001;
        Vector6d imu_prev_;
};

class InekfEstimator{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    InekfEstimator(){
        Roi_ = Eigen::Matrix3d::Identity();
    }

    void setEstimator(const ImuParam& imu_params){
        estimator_.setImuParams(imu_params);
        normalized_ = imu_params.normalized;
    }
    void setWheelVelCov(const vector<double>& std){
        if(std.size() == 1){
            wheel_vel_cov_ = Eigen::Matrix3d::Identity() * std[0] * std[0];
        }
        else{
            wheel_vel_cov_ = Eigen::Matrix3d::Identity();
            wheel_vel_cov_(0, 0) = std[0] * std[0];
            wheel_vel_cov_(1, 1) = std[1] * std[1];
            wheel_vel_cov_(2, 2) = std[2] * std[2];
        }
    }

    void setBodyImuExt(const vector<double>& Roi){
        Roi_ << Roi[0], Roi[1], Roi[2],
                Roi[3], Roi[4], Roi[5],
                Roi[6], Roi[7], Roi[8];
    }

    bool Initialize(Measures& data, const Eigen::Matrix3d& R = Eigen::Matrix3d::Identity(), const Eigen::Vector3d& p = Eigen::Vector3d::Zero());
    void ReInitializePoseBias(const Eigen::Matrix3d& R, const Eigen::Vector3d& p, const double& t, const Eigen::Vector3d& ba, const Eigen::Vector3d& bg);
    void ReInitializePoseV(const Eigen::Matrix3d& R, const Eigen::Vector3d& p, const double& t, const Eigen::Vector3d& v);
    void ReInitializePose(const Eigen::Matrix3d& R, const Eigen::Vector3d& p, const double& t);
    void step(Measures& data);
    void stepCam(const double& t, const unordered_map<int, int>& matches, const vec_vec3d& lamp_world_pos, const vec_vec4d& boxes, double high_lamp);
    void stepCam(const double& t, const vector<int>& M, const vec_vec3d& lamp_world_pos, const vec_vec4d& boxes, double high_lamp);
    void rollBack();
    RobotState getState() { return estimator_.getState(); }
    Eigen::Matrix3d getRotation() { return estimator_.getState().getRotation(); }
    Eigen::Vector3d getPosition() { return estimator_.getState().getPosition(); }
    Eigen::Vector3d getGyroscopeBias() {return estimator_.getState().getGyroscopeBias(); }
    Eigen::Vector3d getAccelerometerBias() { return estimator_.getState().getAccelerometerBias(); }
    Eigen::Vector3d getVelocity() {return estimator_.getState().getVelocity(); }
    Matrix6d getPoseCovariance(){
        Matrix6d cov = Matrix6d::Identity();
        cov.block<3, 3>(0, 0) = estimator_.getState().getRotationCovariance();
        cov.block<3, 3>(3, 3) = estimator_.getState().getPositionCovariance();
        return cov;
    }

private:
    void updateNextIMU(Measures& data);
    void updateNextWheel(Measures& data);
    BodyEstimator estimator_;
    ImuMeasurementPtr imu_packet_;
    WheelEncodeMeasurementPtr wheel_velocity_packet_;
    WinsState state_, tmp_state_;
    Eigen::Matrix3d wheel_vel_cov_;
    Eigen::Matrix3d Roi_;
    bool normalized_;
    double g_ = 9.81;
};

#endif