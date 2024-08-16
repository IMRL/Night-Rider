#ifndef WINSTATE_HPP
#define WINSTATE_HPP

#include <stdint.h>
#include <iostream>
#include <memory>
#include <Eigen/Dense>
#include "ros/ros.h"

#include "inekf/imu_odom.hpp"
#include "inekf/utils.hpp"

class WinsState
{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        WinsState(){
            this->clear();
        };

        template <typename T>

        void setImu(const std::shared_ptr<ImuMeasurement<T>>& next_imu)
        {
            const auto imu_data = next_imu;
            
            Eigen::Vector3d euler = Rotation2Euler(this->getRotation());
            Eigen::Vector3d angularVelocity, eulerRates;
            angularVelocity <<  imu_data.get()->angular_velocity.x, 
                                imu_data.get()->angular_velocity.y, 
                                imu_data.get()->angular_velocity.z;
            eulerRates = AngularVelocity2EulerRates(euler, angularVelocity);
            dq_.block<3,1>(3,0) = eulerRates;

            return;
        }
        void setWheelEncoder(const std::shared_ptr<WheelEncodeMeasurement> next_wheel_state){
            const auto wheel_state_data = next_wheel_state;
            q_.block<3,1>(0,0) = wheel_state_data.get()->getWheelPosition();
            dq_.block<3,1>(0,0) = wheel_state_data.get()-> getLinearVelocity();
            return;
        };
        void setBaseRotation(const Eigen::Matrix3d& R) {
            q_.segment<3>(3) = Rotation2Euler(R);
        }

        void setBasePosition(const Eigen::Vector3d& p) {
            q_.segment<3>(0) = p;
        }

        void setBaseVelocity(const Eigen::Vector3d& v) {
            dq_.segment<3>(0) = v;
        }

        void setImuBias(const Eigen::VectorXd& bias){
            imu_bias_ = bias;
        }

        void clear(){
            q_ = Eigen::Matrix<double,10,1>::Zero();
            dq_ = Eigen::Matrix<double,10,1>::Zero();
            return;
        }

        Eigen::Matrix<double,10,1> q() const { return q_; }
        Eigen::Matrix<double,10,1> dq() const { return dq_; }

        // Get base position
        Eigen::Vector3d  getPosition() const { return q_.segment<3>(0); }

        // Get base quaternion
        Eigen::Quaternion<double>  getQuaternion() const { return Eigen::Quaternion<double>(this->getRotation()); }

        // Get rotation matrix
        Eigen::Matrix3d  getRotation() const { return Euler2Rotation(this->getEulerAngles()); }

        // Extract Euler angles and rates
        Eigen::Vector3d getEulerAngles() const { return q_.segment<3>(3); }
        Eigen::Vector3d getEulerRates() const { return dq_.segment<3>(3); }
        
        Eigen::Vector3d getAngularVelocity() const { 
            return EulerRates2AngularVelocity(this->getEulerAngles(), this->getEulerRates()); 
        }

        // Extract encoder positions
        Eigen::Matrix<double, 4, 1> getEncoderPositions() const{
            return q_.segment<4>(6); //<! take 4 elements start from idx = 6
        }

        Eigen::Matrix<double,4,1> getEncoderVelocities() const {
            return dq_.segment<4>(6); //<! take 4 elements start from idx = 6
        }
        
        Eigen::Vector3d getBodyVelocity() const { 
            Eigen::Vector3d v_world = dq_.segment<3>(0);
            Eigen::Matrix3d Rwb = this->getRotation();
            return Rwb.transpose() * v_world;
        }

        Eigen::Vector3d getWorldVelocity() const{
            return dq_.segment<3>(0);
        }

        Eigen::VectorXd getImuBias() const{
            return imu_bias_;
        }
        // Eigen::Matrix<double,15,15> getAllCov() const;

        // Extract each DOF position by name
        double x() const { return q_(0); }
        double y() const { return q_(1); }
        double z() const { return q_(2); }
        double yaw() const { return q_(3); }
        double pitch() const { return q_(4); }
        double roll() const { return q_(5); }
        // Extract each DOF velocity by name
        double dx() const { return dq_(0); }
        double dy() const { return dq_(1); }
        double dz() const { return dq_(2); }
        double dyaw() const { return dq_(3); }
        double dpitch() const { return dq_(4); }
        double droll() const { return dq_(5); }

        void setTime(double time_in){time_stamp_ = time_in;};
        double getTime() const{return time_stamp_;};


        friend std::ostream& operator<<(std::ostream& os,const WinsState& obj){
            os << "q: [";
            for (int i=0; i<obj.q_.rows()-1; ++i) {
                os << obj.q_(i) << ", ";
            } 
            os << obj.q_(obj.q_.rows()-1) << "]\n";

            os << "dq: [";
            for (int i=0; i<obj.dq_.rows()-1; ++i) {
                os << obj.dq_(i) << ", ";
            } 
            os << obj.dq_(obj.dq_.rows()-1) << "]";
            return os;
        }
    private:
        double time_stamp_;
        Eigen::Matrix<double, 10,1> q_;
        Eigen::Matrix<double, 10,1> dq_;
        Eigen::Matrix<double, 6,1> imu_bias_;
};


#endif