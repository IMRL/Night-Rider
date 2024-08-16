#ifndef NOISEPARAMS_HPP
#define NOISEPARAMS_HPP
#include <Eigen/Dense>
#include <iostream>

using namespace std;

class NoiseParams {

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        NoiseParams(){
            setGyroscopeNoise(0.01);
            setAccelerometerNoise(0.1);
            setGyroscopeBiasNoise(0.00001);
            setAccelerometerBiasNoise(0.0001);
            setContactNoise(0.1);
        }

        void setGyroscopeNoise(double std){ Qg_ = std*std*Eigen::Matrix3d::Identity(); }
        void setGyroscopeNoise(const Eigen::Vector3d& std){ Qg_ << std(0)*std(0),0,0, 0,std(1)*std(1),0, 0,0,std(2)*std(2); }
        void setGyroscopeNoise(const Eigen::Matrix3d& cov){ Qg_ = cov; }

        void setAccelerometerNoise(double std){ Qa_ = std*std*Eigen::Matrix3d::Identity(); }
        void setAccelerometerNoise(const Eigen::Vector3d& std){ Qa_ << std(0)*std(0),0,0, 0,std(1)*std(1),0, 0,0,std(2)*std(2); }
        void setAccelerometerNoise(const Eigen::Matrix3d& cov){ Qa_ = cov; }   

        void setGyroscopeBiasNoise(double std) { Qbg_ = std*std*Eigen::Matrix3d::Identity(); }
        void setGyroscopeBiasNoise(const Eigen::Vector3d& std) { Qbg_ << std(0)*std(0),0,0, 0,std(1)*std(1),0, 0,0,std(2)*std(2); }
        void setGyroscopeBiasNoise(const Eigen::Matrix3d& cov) { Qbg_ = cov; }

        void setAccelerometerBiasNoise(double std) { Qba_ = std*std*Eigen::Matrix3d::Identity(); }
        void setAccelerometerBiasNoise(const Eigen::Vector3d& std) { Qba_ << std(0)*std(0),0,0, 0,std(1)*std(1),0, 0,0,std(2)*std(2); }
        void setAccelerometerBiasNoise(const Eigen::Matrix3d& cov) { Qba_ = cov; }

        void setContactNoise(double std) { Qc_ = std*std*Eigen::Matrix3d::Identity(); }
        void setContactNoise(const Eigen::Vector3d& std) { Qc_ << std(0)*std(0),0,0, 0,std(1)*std(1),0, 0,0,std(2)*std(2); }
        void setContactNoise(const Eigen::Matrix3d& cov) { Qc_ = cov; }

        Eigen::Matrix3d getGyroscopeCov() { return Qg_; }
        Eigen::Matrix3d getAccelerometerCov() { return Qa_; }
        Eigen::Matrix3d getGyroscopeBiasCov() { return Qbg_; }
        Eigen::Matrix3d getAccelerometerBiasCov() { return Qba_; }
        Eigen::Matrix3d getContactCov() { return Qc_; }

        friend std::ostream& operator<<(std::ostream& os, const NoiseParams& p){
            os << "--------- Noise Params -------------" << endl;
            os << "Gyroscope Covariance:\n" << p.Qg_ << endl;
            os << "Accelerometer Covariance:\n" << p.Qa_ << endl;
            os << "Gyroscope Bias Covariance:\n" << p.Qbg_ << endl;
            os << "Accelerometer Bias Covariance:\n" << p.Qba_ << endl;
            os << "-----------------------------------" << endl;
            return os;
        }

    private:
        Eigen::Matrix3d Qg_;
        Eigen::Matrix3d Qa_;
        Eigen::Matrix3d Qbg_;
        Eigen::Matrix3d Qba_;
        Eigen::Matrix3d Ql_;
        Eigen::Matrix3d Qc_;
};
#endif 