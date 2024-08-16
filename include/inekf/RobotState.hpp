#ifndef ROBOTSTATE_HPP
#define ROBOTSTATE_HPP
#include <Eigen/Dense>
#include <iostream>

using namespace std;

enum StateType { WorldCentric,BodyCentric};

class RobotState{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        RobotState() :
            X_(Eigen::MatrixXd::Identity(5,5)), Theta_(Eigen::MatrixXd::Zero(6,1)), P_(Eigen::MatrixXd::Identity(15,15)) {}
        RobotState(const Eigen::MatrixXd& X) : 
            X_(X), Theta_(Eigen::MatrixXd::Zero(6,1)) {
            P_ = Eigen::MatrixXd::Identity(3*this->dimX()+this->dimTheta()-6, 3*this->dimX()+this->dimTheta()-6);
        }
        // Initialize with X and Theta
        RobotState(const Eigen::MatrixXd& X, const Eigen::VectorXd& Theta) : 
            X_(X), Theta_(Theta) {
            P_ = Eigen::MatrixXd::Identity(3*this->dimX()+this->dimTheta()-6, 3*this->dimX()+this->dimTheta()-6);
        }
        // Initialize with X, Theta and P
        RobotState(const Eigen::MatrixXd& X, const Eigen::VectorXd& Theta, const Eigen::MatrixXd& P) : 
            X_(X), Theta_(Theta), P_(P) {}

        const Eigen::MatrixXd getX() const { return X_; }
        const Eigen::VectorXd getTheta() const { return Theta_; }
        const Eigen::MatrixXd getP() const { return P_; }  //TODO 协方差
        const Eigen::Matrix3d getRotation() const { return X_.block<3,3>(0,0); }
        const Eigen::Vector3d getVelocity() const { return X_.block<3,1>(0,3); }
        const Eigen::Vector3d getPosition() const { return X_.block<3,1>(0,4); }
        const Eigen::Vector3d getVector(int index) const { return X_.block<3,1>(0,index); }

        const Eigen::Vector3d getGyroscopeBias() const { return Theta_.head(3); }
        const Eigen::Vector3d getAccelerometerBias() const { return Theta_.tail(3); }


        const Eigen::Matrix3d getRotationCovariance() const { return P_.block<3,3>(0,0); }
        const Eigen::Matrix3d getVelocityCovariance() const { return P_.block<3,3>(3,3); }
        const Eigen::Matrix3d getPositionCovariance() const { return P_.block<3,3>(6,6); }
        const Eigen::Matrix3d getGyroscopeBiasCovariance() const { return P_.block<3,3>(P_.rows()-6,P_.rows()-6); }
        const Eigen::Matrix3d getAccelerometerBiasCovariance() const { return P_.block<3,3>(P_.rows()-3,P_.rows()-3); }

        const int dimX() const { return X_.cols(); }
        const int dimTheta() const {return Theta_.rows();}
        const int dimP() const { return P_.cols(); }

        const StateType getStateType() const { return state_type_; }

        const Eigen::MatrixXd getWorldX() const {
            if (state_type_ == StateType::WorldCentric) {
                return this->getX();
            } else {
                return this->Xinv();
            }
        }
        const Eigen::Matrix3d getWorldRotation() const {
            if (state_type_ == StateType::WorldCentric) {
                return this->getRotation();
            } else {
                return this->getRotation().transpose();
            }
        }
        const Eigen::Vector3d getWorldVelocity() const {
            if (state_type_ == StateType::WorldCentric) {
                return this->getVelocity();
            } else {
                return -this->getRotation().transpose()*this->getVelocity();
            }
        }

        const Eigen::Vector3d getWorldPosition() const {
            if (state_type_ == StateType::WorldCentric) {
                return this->getPosition();
            } else {
                return -this->getRotation().transpose()*this->getPosition();
            }
        }

        const Eigen::MatrixXd getBodyX() const {
            if (state_type_ == StateType::BodyCentric) {
                return this->getX();
            } else {
                return this->Xinv();
            }
        }

        const Eigen::Matrix3d getBodyRotation() const {
            if (state_type_ == StateType::BodyCentric) {
                return this->getRotation();
            } else {
                return this->getRotation().transpose();
            }
        }

        const Eigen::Vector3d getBodyVelocity() const {
            if (state_type_ == StateType::BodyCentric) {
                return this->getVelocity();
            } else {
                return -this->getRotation().transpose()*this->getVelocity();
            }
        }

        const Eigen::Vector3d getBodyPosition() const {
            if (state_type_ == StateType::BodyCentric) {
                return this->getPosition();
            } else {
                return -this->getRotation().transpose()*this->getPosition();
            }
        }


        void setX(const Eigen::MatrixXd& X) { X_ = X; }
        void setTheta(const Eigen::VectorXd& Theta) { Theta_ = Theta; }
        void setP(const Eigen::MatrixXd& P) { P_ = P; }
        void setRotation(const Eigen::Matrix3d& R) { X_.block<3,3>(0,0) = R; }
        void setVelocity(const Eigen::Vector3d& v) { X_.block<3,1>(0,3) = v; }
        void setPosition(const Eigen::Vector3d& p) { X_.block<3,1>(0,4) = p; }
        void setGyroscopeBias(const Eigen::Vector3d& bg) { Theta_.head(3) = bg; }
        void setAccelerometerBias(const Eigen::Vector3d& ba) { Theta_.tail(3) = ba; }
        void setRotationCovariance(const Eigen::Matrix3d& cov) { P_.block<3,3>(0,0) = cov; }
        void setVelocityCovariance(const Eigen::Matrix3d& cov) { P_.block<3,3>(3,3) = cov; }
        void setPositionCovariance(const Eigen::Matrix3d& cov) { P_.block<3,3>(6,6) = cov; }
        void setGyroscopeBiasCovariance(const Eigen::Matrix3d& cov) { P_.block<3,3>(P_.rows()-6,P_.rows()-6) = cov; }
        void setAccelerometerBiasCovariance(const Eigen::Matrix3d& cov) { P_.block<3,3>(P_.rows()-3,P_.rows()-3) = cov; }
        
        void copyDiagX(int n, Eigen::MatrixXd& BigX) const {
            int dimX = this->dimX();
            for(int i=0; i<n; ++i) {
                int startIndex = BigX.rows();
                BigX.conservativeResize(startIndex + dimX, startIndex + dimX);
                BigX.block(startIndex,0,dimX,startIndex) = Eigen::MatrixXd::Zero(dimX,startIndex);
                BigX.block(0,startIndex,startIndex,dimX) = Eigen::MatrixXd::Zero(startIndex,dimX);
                BigX.block(startIndex,startIndex,dimX,dimX) = X_;
            }
            return;
        }

        void copyDiagXinv(int n, Eigen::MatrixXd& BigXinv) const {
            int dimX = this->dimX();
            Eigen::MatrixXd Xinv = this->Xinv();
            for(int i=0; i<n; ++i) {
                int startIndex = BigXinv.rows();
                BigXinv.conservativeResize(startIndex + dimX, startIndex + dimX);
                BigXinv.block(startIndex,0,dimX,startIndex) = Eigen::MatrixXd::Zero(dimX,startIndex);
                BigXinv.block(0,startIndex,startIndex,dimX) = Eigen::MatrixXd::Zero(startIndex,dimX);
                BigXinv.block(startIndex,startIndex,dimX,dimX) = Xinv;
            }
            return;
        }

        const Eigen::MatrixXd Xinv() const {
            int dimX = this->dimX();
            Eigen::MatrixXd Xinv = Eigen::MatrixXd::Identity(dimX,dimX);
            Eigen::Matrix3d RT = X_.block<3,3>(0,0).transpose();
            Xinv.block<3,3>(0,0) = RT;
            for(int i=3; i<dimX; ++i) {
                Xinv.block<3,1>(0,i) = -RT*X_.block<3,1>(0,i);
            }
            return Xinv;
        }

        friend std::ostream& operator<<(std::ostream& os, const RobotState& s){  
            os << "--------- Robot State -------------" << endl;
            os << "X:\n" << s.X_ << endl << endl;
            os << "Theta:\n" << s.Theta_ << endl << endl;
            // os << "P:\n" << s.P_ << endl;
            os << "-----------------------------------";
            return os;  
        } 

    private:
        StateType state_type_ = StateType::WorldCentric; 
        Eigen::MatrixXd X_;
        Eigen::VectorXd Theta_;
        Eigen::MatrixXd P_;

};


#endif