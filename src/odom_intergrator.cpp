#include "map_relocalization/odom_integrator.hpp"

void OdomPreintegrator::PreIntegration(Measures& meas, const OdomParam& odom_param){

    double start_time;
    if (meas.last_timestamp < 0)
        start_time = meas.odom_deq.front()->header.stamp.toSec();
    else
        start_time = meas.last_timestamp;
    
    // 保留了前一组数据中最后一帧Odom的数据
    auto odom_iter = meas.odom_deq.begin();
    int odom_count = 0;
    double last_odom_timestamp = 0.0;
    cout << "size: " << meas.odom_deq.size() << endl;
    while(odom_iter != meas.odom_deq.end()){
        double delta_t;
        if (odom_count == 0){
            delta_t = (*(odom_iter + 1))->header.stamp.toSec() - start_time;
        }
        else if (odom_count == meas.odom_deq.size() - 1){
            delta_t = meas.timestamp - (*odom_iter)->header.stamp.toSec();
        }
        else{
            delta_t = (*(odom_iter + 1))->header.stamp.toSec() - last_odom_timestamp;
        }
        double v_x = (*odom_iter)->twist.twist.linear.x;
        double w_z = (*odom_iter)->twist.twist.angular.z;
        // cout << "w_z: " << w_z << endl;
        double thetaj_1j = w_z * delta_t;

        Matrix6d A(Matrix6d::Identity()), C;
        Matrix<double,6,2> B;
        Matrix3d Rj_1j;
        Rj_1j << cos(thetaj_1j),-sin(thetaj_1j),0,
	             sin(thetaj_1j), cos(thetaj_1j),0,
	             0,              0,             1;
        
        Matrix3d Rij_1;
        Rij_1 << cos(deltaThetaijMz),-sin(deltaThetaijMz),0,
                 sin(deltaThetaijMz), cos(deltaThetaijMz),0,
                 0                  , 0                  ,1;

        A.block<3,3>(0,0) = Rj_1j.transpose();//delta~REj-1Ej.t()
        // B.setZero();
        // B(2,1) = delta_t/2/rc;
        // B(2,0) = - B(2,1);
        // Matrix<double,3,2> Bj_11;
        double dt2 = delta_t * delta_t;
        C.setZero();
        C(2,5) = delta_t;
        Matrix<double,3,6> Cj_11;
        double sinthdivw, one_costh_divw;
        double Bx, By, C0, C1;
        if (abs(thetaj_1j) < EPS){
            // 	A.block<3,3>(3,0)=Rij_1*g2o::skew(Vector3d(-vf*deltat,0,0));
            // 	Bj_11.block<2,2>(0,0)<<deltat/2,deltat/2,0,0;
            // 	Cj_11.block<3,3>(0,0)=deltat*Matrix3d::Identity();Cj_11.block<3,3>(0,3).setZero();
            sinthdivw = delta_t; one_costh_divw = w_z * dt2/2;
            Bx = -v_x * w_z * dt2 * delta_t/2; By = v_x * dt2/2;
            C0 = 0; C1 = -v_x * dt2/2;
        }
        else{
            sinthdivw = sin(thetaj_1j) / w_z; one_costh_divw = (1 - cos(thetaj_1j)) / w_z;
            Bx = v_x / w_z * (delta_t * cos(thetaj_1j) - sinthdivw); By = v_x / w_z * (delta_t * sin(thetaj_1j) - one_costh_divw);
            C0 = v_x / w_z * (delta_t - sinthdivw); C1 = -v_x / w_z * one_costh_divw;
        }
        A.block<3,3>(3,0) = Rij_1 * vector_skew(Vector3d(-v_x * sinthdivw, -v_x * one_costh_divw, 0));
        // Bj_11.block<3,2>(0,0) << sinthdivw/2 - Bx/2/rc, sinthdivw/2 + Bx/2/rc, one_costh_divw/2 - By/2/rc, one_costh_divw/2 + By/2/rc,0,0;
        C(0,3) = sinthdivw; C(0,4) = one_costh_divw; C(1,3) = -one_costh_divw; C(1,4) = sinthdivw;
        Cj_11 << sinthdivw,      -one_costh_divw, 0,       0,  0, Bx,
	             one_costh_divw, sinthdivw,       0,       0,  0, By,
	             0,              0,               delta_t, C0, C1, 0;
        // B.block<3,2>(3,0) = Rij_1 * Bj_11;
        C.block<3,6>(3,0) = Rij_1 * Cj_11;
        if (odom_param.mdt_cov_noise_fixed)//eta->etad
            // mSigmaEij = A * mSigmaEij * A.transpose() + B * odom_param.mSigma * B.transpose() + C * odom_param.mSigmam * delta_t * C.transpose();
            mSigmaEij = A * mSigmaEij * A.transpose() + C * odom_param.mSigmam * delta_t * C.transpose();
        else if (!odom_param.mFreqRef || delta_t < 1.5 / odom_param.mFreqRef)
            // mSigmaEij = A * mSigmaEij * A.transpose() + B * (odom_param.mSigma / delta_t) * B.transpose() + C * (odom_param.mSigmam * delta_t) * C.transpose();
            mSigmaEij = A * mSigmaEij * A.transpose() + C * (odom_param.mSigmam * delta_t) * C.transpose();
        else
            // mSigmaEij = A * mSigmaEij * A.transpose() + B * (odom_param.mSigma * odom_param.mFreqRef) * B.transpose() + C * (odom_param.mSigmam * delta_t) * C.transpose();
            mSigmaEij = A * mSigmaEij * A.transpose() + C * (odom_param.mSigmam * delta_t) * C.transpose();

        //update deltaPijM before update deltaThetaijM to use deltaThetaijMz as deltaTheta~ij-1z
        double thetaij = deltaThetaijMz + thetaj_1j; //Theta~eiej

        if (abs(thetaj_1j) < EPS){//or thetaj_1j==0
        //double arrdTmp[4]={cos(deltaThetaijMz),sin(deltaThetaijMz),-sin(deltaThetaijMz),cos(deltaThetaijMz)};//row-major:{cos(deltaTheij-1Mz),-sin(deltaTheij-1Mz),0,sin(deltaTheij-1Mz),cos(deltaThetaij-1Mz),0,0,0,1}; but Eigen defaultly uses col-major!
        //eigdeltaPijM+=Matrix2d(arrdTmp)*Vector2d(vf*deltat,0);//deltaPijM+Reiej-1*vej-1ej-1*deltat
            eigdeltaPijM += Rij_1.block<2,2>(0,0) * Vector2d(v_x * sinthdivw, v_x * one_costh_divw);//deltaPijM+Reiej-1*vej-1ej-1*deltat
        }
        else{
	        eigdeltaPijM += v_x/w_z * Vector2d(sin(thetaij) - sin(deltaThetaijMz), cos(deltaThetaijMz) - cos(thetaij));
        }
        //update deltaThetaijM
        deltaThetaijMz=thetaij;//deltaThetaij-1M + weiej-1 * deltatj-1j, notice we may use o in the code instead of e in the paper
        // cout << "eigdeltaPijM: " << eigdeltaPijM << endl << "deltaThetaijMz: " << deltaThetaijMz << endl;
        // mdeltatij += deltat;
        if (odom_count != meas.odom_deq.size() - 1)
            last_odom_timestamp = (*(odom_iter + 1))->header.stamp.toSec();
        ++odom_iter;
        ++odom_count;
    }
    mdelxEij[0] = mdelxEij[1] = mdelxEij[5] = 0;
    mdelxEij[2] = deltaThetaijMz;
    mdelxEij.segment<2>(3) = eigdeltaPijM;
    cout << "odom_count: " << odom_count << endl;
}   