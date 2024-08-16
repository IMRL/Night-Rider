#ifndef ODOM_INTEGRATOR_HPP
#define ODOM_INTEGRATOR_HPP

#include <iostream>
#include "map_relocalization/common_lib.h"

using namespace Eigen;

class OdomPreintegrator{
EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public:
    const double EPS = 1E-5;
    Vector6d mdelxEij;   // delta~Phiij(3*1),delta~pij(3*1) from Encoder PreIntegration, 6*1*float
    Matrix6d mSigmaEij;  // by Enc, 6*6*float
    Eigen::Vector2d eigdeltaPijM;  // deltaPii=0
    double deltaThetaijMz;
    double mdeltatij;

    OdomPreintegrator() : mdelxEij(Vector6d::Zero()), mSigmaEij(Matrix6d::Zero()), deltaThetaijMz(0), eigdeltaPijM(Eigen::Vector2d::Zero()), mdeltatij(0) {}
    OdomPreintegrator(const Matrix6d& SigmaEij){
        mdelxEij = Vector6d::Zero();
        deltaThetaijMz = 0;
        eigdeltaPijM = Eigen::Vector2d::Zero();
        mdeltatij = 0;
        mSigmaEij = SigmaEij;
    }
    OdomPreintegrator(const OdomPreintegrator &pre) :eigdeltaPijM(pre.eigdeltaPijM), deltaThetaijMz(pre.deltaThetaijMz), mdelxEij(pre.mdelxEij), mSigmaEij(pre.mSigmaEij) {
        mdeltatij = pre.mdeltatij;
    }
    OdomPreintegrator &operator=(const OdomPreintegrator &pre) {
        eigdeltaPijM = pre.eigdeltaPijM;
        deltaThetaijMz = pre.deltaThetaijMz;
        mdelxEij = pre.mdelxEij;
        mSigmaEij = pre.mSigmaEij;
        mdeltatij = pre.mdeltatij;  // don't copy list!
        return *this;
    }

    void PreIntegration(Measures& meas, const OdomParam& odom_param);
};

#endif