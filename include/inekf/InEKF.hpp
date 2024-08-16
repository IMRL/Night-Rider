#ifndef INEKF_HPP
#define INEKF_HPP
#include <Eigen/Dense>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <algorithm>
#include "map_relocalization/common_lib.h"
#include "inekf/RobotState.hpp"
#include "inekf/NoiseParams.hpp"
#include "inekf/LieGroup.h"
#include "inekf/Observation.hpp"

enum ErrorType {LeftInvariant,RightInvariant};

class InEKF 
{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        InEKF();
        InEKF(NoiseParams params);
        InEKF(RobotState state);
        InEKF(RobotState state,NoiseParams params);
        InEKF(RobotState state,NoiseParams params, ErrorType erro_type);
        ~InEKF();
        ErrorType getErroType() const;
        RobotState getState() const;
        NoiseParams getNoiseParams() const;
        std::map<int,bool> getContacts() const;
        std::map<int,int> getEistimatedContactPositions() const;
        mapIntVector3d getPriorLandmarks() const;
        std::map<int,int> getEistimatedLandmarks() const;
        Eigen::Vector3d getMagneticField() const;

        void setState(RobotState state);

        void setNoiseParams(NoiseParams params);

        void setExtrinsics(const Eigen::Matrix3d& Rcb, const Eigen::Vector3d& pcb);

        void setContacts(std::vector<std::pair<int,bool> > contacts);

        void setPriorLandmarks(const mapIntVector3d& prior_landmarks);

        void setMagneticField(Eigen::Vector3d& true_magnetic_field);

        void clear();

        void RemovePriorLandmarks(const int landmark_id);

        void RemovePriorLandmarks(const std::vector<int> landmark_ids);

        void RemoveLandmarks(const int landmark_id);

        void RemoveLandmarks(const std::vector<int> landmark_ids);

        void KeepLandmarks(const std::vector<int> landmark_ids);

        void Propagate(const Eigen::Matrix<double,6,1>& imu, double dt);

        void CorrectKinematics(const vectorKinematics& measured_kinematics);

        void CorrectLandmarks(const vectorLandmarks& measured_landmarks);

        void CorrectVelocity(const Eigen::Vector3d& measured_velocity, const Eigen::Matrix3d& covariance);

        void CorrectPose(const unordered_map<int, int>& matches, const vec_vec3d& lamp_cur_cam_pos, const vec_vec4d& boxes, double high_lamp);

        /** TODO: Untested magnetometer measurement*/
        void CorrectMagnetometer(const Eigen::Vector3d& measured_magnetic_field, const Eigen::Matrix3d& covariance);
        /** TODO: Untested GPS measurement*/
        void CorrectPosition(const Eigen::Vector3d& measured_position, const Eigen::Matrix3d& covariance, const Eigen::Vector3d& indices);
        /** TODO: Untested contact position measurement*/
        void CorrectContactPosition(const int id, const Eigen::Vector3d& measured_contact_position, const Eigen::Matrix3d& covariance, const Eigen::Vector3d& indices);

        void rollBack();
    
    private:
        ErrorType error_type_ = ErrorType::LeftInvariant; 
        bool estimate_bias_ = true; 
        RobotState state_, tmp_state_;
        NoiseParams noise_params_;
        const Eigen::Vector3d g_; // Gravity vector in world frame (z-up)
        std::map<int,bool> contacts_;
        std::map<int,int> estimated_contact_positions_;
        mapIntVector3d prior_landmarks_;
        std::map<int,int> estimated_landmarks_;
        Eigen::Vector3d magnetic_field_;
        Eigen::Matrix3d Rcb_;
        Eigen::Vector3d pcb_;

        Eigen::MatrixXd StateTransitionMatrix(Eigen::Vector3d& w, Eigen::Vector3d& a, double dt);
        Eigen::MatrixXd DiscreteNoiseMatrix(Eigen::MatrixXd& Phi, double dt);

        // Corrects state using invariant observation models
        void CorrectRightInvariant(const Observation& obs);
        void CorrectLeftInvariant(const Observation& obs);
        void CorrectRightInvariant(const Eigen::MatrixXd& Z, const Eigen::MatrixXd& H, const Eigen::MatrixXd& N, bool is_cam = false);
        void CorrectLeftInvariant(const Eigen::MatrixXd& Z, const Eigen::MatrixXd& H, const Eigen::MatrixXd& N);

};

#endif