
#include "inekf/InEKF.hpp"

using namespace std;

InEKF::InEKF() :
    g_((Eigen::VectorXd(3)<< 0,0,-9.81).finished()),
    magnetic_field_((Eigen::VectorXd(3)<< 0,0,0).finished()){
}


InEKF::InEKF(NoiseParams params) :
    g_((Eigen::VectorXd(3)<< 0,0,-9.81).finished()),
    magnetic_field_((Eigen::VectorXd(3)<< 0,0,0).finished()),
    noise_params_(params){
    }



InEKF::InEKF(RobotState state) : 
    g_((Eigen::VectorXd(3) << 0,0,-9.81).finished()), 
    magnetic_field_((Eigen::VectorXd(3) << std::cos(1.2049),0,std::sin(1.2049)).finished()), 
    state_(state) {
    }

InEKF::InEKF(RobotState state, NoiseParams params) : 
    g_((Eigen::VectorXd(3) << 0,0,-9.81).finished()), 
    magnetic_field_((Eigen::VectorXd(3) << std::cos(1.2049),0,std::sin(1.2049)).finished()), 
    state_(state), 
    noise_params_(params) {
    }

InEKF::InEKF(RobotState state, NoiseParams params, ErrorType error_type) : 
    g_((Eigen::VectorXd(3) << 0,0,-9.81).finished()), 
    magnetic_field_((Eigen::VectorXd(3) << std::cos(1.2049),0,std::sin(1.2049)).finished()), 
    state_(state), 
    noise_params_(params), 
    error_type_(error_type) {
    }

void InEKF::clear(){
    state_= RobotState();
    noise_params_=NoiseParams();
    prior_landmarks_.clear();
    estimated_landmarks_.clear();
    contacts_.clear();
    estimated_contact_positions_.clear();
}

InEKF::~InEKF(){}

ErrorType InEKF::getErroType() const {return error_type_ ;}

RobotState InEKF::getState() const {return state_ ;}

void InEKF::setState(RobotState state) { state_ = state;}

NoiseParams InEKF::getNoiseParams() const {return noise_params_;}

void InEKF::setNoiseParams(NoiseParams params ) { noise_params_ = params ;}

void InEKF::setExtrinsics(const Eigen::Matrix3d& Rcb, const Eigen::Vector3d& pcb) { Rcb_ = Rcb; pcb_ = pcb; }

mapIntVector3d InEKF::getPriorLandmarks() const {return prior_landmarks_;}

void InEKF::setPriorLandmarks(const mapIntVector3d& prior_landmarks) {prior_landmarks_ = prior_landmarks;}

map<int,int> InEKF::getEistimatedLandmarks() const {return estimated_landmarks_;}

map<int,int> InEKF::getEistimatedContactPositions() const {return estimated_contact_positions_;}

void InEKF::setContacts(vector<pair<int,bool>> contacts) {
        // Insert new measured contact states
    for (vector<pair<int,bool> >::iterator it=contacts.begin(); it!=contacts.end(); ++it) {
        pair<map<int,bool>::iterator,bool> ret = contacts_.insert(*it);
        // If contact is already in the map, replace with new value
        if (ret.second==false) {
            ret.first->second = it->second;
        }
    }
    return;
}

map<int,bool> InEKF::getContacts() const {return contacts_;}

// Set the true magnetic field   
void InEKF::setMagneticField(Eigen::Vector3d& true_magnetic_field) { magnetic_field_ = true_magnetic_field; }    ///TODO XSENSE have magnetic?

// Get the true magnetic field
Eigen::Vector3d InEKF::getMagneticField() const { return magnetic_field_; }

///***************************************************//

//InEKF计算实现//

//计算状态传递矩阵
Eigen::MatrixXd InEKF::StateTransitionMatrix(Eigen::Vector3d& w,Eigen::Vector3d& a,double dt) 
{
    Eigen::Vector3d phi = w*dt;
    Eigen::Matrix3d G0 = Gamma_SO3(phi,0);
    Eigen::Matrix3d G1 = Gamma_SO3(phi,1);
    Eigen::Matrix3d G2 = Gamma_SO3(phi,2);
    Eigen::Matrix3d G0t = G0.transpose();   //转置
    Eigen::Matrix3d G1t = G1.transpose();
    Eigen::Matrix3d G2t = G2.transpose();
    Eigen::Matrix3d G3t = Gamma_SO3(-phi,3);

    int dimX = state_.dimX();
    int dimTheta = state_.dimTheta();
    int dimP = state_.dimP();
    Eigen::MatrixXd Phi = Eigen::MatrixXd::Identity(dimP,dimP);
    //Compute the complicated bias terms (derived for the left invariant case)
    Eigen::Matrix3d ax = skew(a);
    Eigen::Matrix3d wx = skew(w);
    Eigen::Matrix3d wx2 = wx*wx;
    double dt2 = dt*dt;
    double dt3 = dt2*dt;
    double theta = w.norm();
    double theta2 = theta*theta;
    double theta3 = theta2*theta;
    double theta4 = theta3*theta;
    double theta5 = theta4*theta;
    double theta6 = theta5*theta;
    double theta7 = theta6*theta;
    double thetadt = theta*dt;
    double thetadt2 = thetadt*thetadt;
    double thetadt3 = thetadt2*thetadt;
    double sinthetadt = sin(thetadt);
    double costhetadt = cos(thetadt);
    double sin2thetadt = sin(2*thetadt);
    double cos2thetadt = cos(2*thetadt);
    double thetadtcosthetadt = thetadt*costhetadt;
    double thetadtsinthetadt = thetadt*sinthetadt;

    Eigen::Matrix3d Phi25L = G0t*(ax*G2t*dt2 
        + ((sinthetadt-thetadtcosthetadt)/(theta3))*(wx*ax)
        - ((cos2thetadt-4*costhetadt+3)/(4*theta4))*(wx*ax*wx)
        + ((4*sinthetadt+sin2thetadt-4*thetadtcosthetadt-2*thetadt)/(4*theta5))*(wx*ax*wx2)
        + ((thetadt2-2*thetadtsinthetadt-2*costhetadt+2)/(2*theta4))*(wx2*ax)
        - ((6*thetadt-8*sinthetadt+sin2thetadt)/(4*theta5))*(wx2*ax*wx)
        + ((2*thetadt2-4*thetadtsinthetadt-cos2thetadt+1)/(4*theta6))*(wx2*ax*wx2) );

    Eigen::Matrix3d Phi35L = G0t*(ax*G3t*dt3
        - ((thetadtsinthetadt+2*costhetadt-2)/(theta4))*(wx*ax)
        - ((6*thetadt-8*sinthetadt+sin2thetadt)/(8*theta5))*(wx*ax*wx)
        - ((2*thetadt2+8*thetadtsinthetadt+16*costhetadt+cos2thetadt-17)/(8*theta6))*(wx*ax*wx2)
        + ((thetadt3+6*thetadt-12*sinthetadt+6*thetadtcosthetadt)/(6*theta5))*(wx2*ax)
        - ((6*thetadt2+16*costhetadt-cos2thetadt-15)/(8*theta6))*(wx2*ax*wx)
        + ((4*thetadt3+6*thetadt-24*sinthetadt-3*sin2thetadt+24*thetadtcosthetadt)/(24*theta7))*(wx2*ax*wx2) );

    const double tolerance = 1e-6;
    if (theta< tolerance) {
        Phi25L = (1/2)*ax*dt2;
        Phi35L = (1/6)*ax*dt3;
    }

    if ((state_.getStateType() == StateType::WorldCentric && error_type_ == ErrorType::LeftInvariant) || (state_.getStateType() == StateType::BodyCentric && error_type_ ==ErrorType::RightInvariant))
    {   //左InEKF传递方程
        Phi.block<3,3>(0,0) = G0t; // Phi_11
        Phi.block<3,3>(3,0) = -G0t*skew(G1*a)*dt; // Phi_21
        Phi.block<3,3>(6,0) = -G0t*skew(G2*a)*dt2; // Phi_31
        Phi.block<3,3>(3,3) = G0t; // Phi_22
        Phi.block<3,3>(6,3) = G0t*dt; // Phi_32
        Phi.block<3,3>(6,6) = G0t; // Phi_33
        for (int i=5; i<dimX; ++i) {
            Phi.block<3,3>((i-2)*3,(i-2)*3) = G0t; // Phi_(3+i)(3+i)
        }
        Phi.block<3,3>(0,dimP-dimTheta) = -G1t*dt; // Phi_15
        Phi.block<3,3>(3,dimP-dimTheta) = Phi25L; // Phi_25
        Phi.block<3,3>(6,dimP-dimTheta) = Phi35L; // Phi_35
        Phi.block<3,3>(3,dimP-dimTheta+3) = -G1t*dt; // Phi_26
        Phi.block<3,3>(6,dimP-dimTheta+3) = -G0t*G2*dt2; // Phi_36
    }
    else //右InEKF传递方程
    {
        Eigen::Matrix3d gx = skew(g_);
        Eigen::Matrix3d R = state_.getRotation();
        Eigen::Vector3d v = state_.getVelocity();
        Eigen::Vector3d p = state_.getPosition();
        Eigen::Matrix3d RG0 = R*G0;
        Eigen::Matrix3d RG1dt = R*G1*dt;
        Eigen::Matrix3d RG2dt2 = R*G2*dt2;
        Phi.block<3,3>(3,0) = gx*dt; // Phi_21
        Phi.block<3,3>(6,0) = 0.5*gx*dt2; // Phi_31
        Phi.block<3,3>(6,3) = Eigen::Matrix3d::Identity()*dt; // Phi_32
        Phi.block<3,3>(0,dimP-dimTheta) = -RG1dt; // Phi_15
        Phi.block<3,3>(3,dimP-dimTheta) = -skew(v+RG1dt*a+g_*dt)*RG1dt + RG0*Phi25L; // Phi_25
        Phi.block<3,3>(6,dimP-dimTheta) = -skew(p+v*dt+RG2dt2*a+0.5*g_*dt2)*RG1dt + RG0*Phi35L; // Phi_35
        for (int i=5; i<dimX; ++i) {
            Phi.block<3,3>((i-2)*3,dimP-dimTheta) = -skew(state_.getVector(i))*RG1dt; // Phi_(3+i)5
        }
        Phi.block<3,3>(3,dimP-dimTheta+3) = -RG1dt; // Phi_26
        Phi.block<3,3>(6,dimP-dimTheta+3) = -RG2dt2; // Phi_36
    }
    return Phi;
}

//离散噪声矩阵计算
Eigen::MatrixXd InEKF::DiscreteNoiseMatrix(Eigen::MatrixXd& Phi,double dt)
{
    int dimX = state_.dimX();
    int dimTheta = state_.dimTheta();
    int dimP = state_.dimP();
    Eigen::MatrixXd G = Eigen::MatrixXd::Identity(dimP,dimP);

    if  ((state_.getStateType() == StateType::WorldCentric && error_type_ == ErrorType::RightInvariant) || (state_.getStateType() == StateType::BodyCentric && error_type_ == ErrorType::LeftInvariant))
    {
        G.block(0,0,dimP-dimTheta,dimP-dimTheta) = Adjoint_SEK3(state_.getWorldX()); 
    }

        // Continuous noise covariance 
    Eigen::MatrixXd Qc = Eigen::MatrixXd::Zero(dimP,dimP); // Landmark noise terms will remain zero
    Qc.block<3,3>(0,0) = noise_params_.getGyroscopeCov(); 
    Qc.block<3,3>(3,3) = noise_params_.getAccelerometerCov();
    for(map<int,int>::iterator it=estimated_contact_positions_.begin(); it!=estimated_contact_positions_.end(); ++it) {
        Qc.block<3,3>(3+3*(it->second-3),3+3*(it->second-3)) = noise_params_.getContactCov(); // Contact noise terms
    } // TODO: Use kinematic orientation to map noise from contact frame to body frame (not needed if noise is isotropic)
    Qc.block<3,3>(dimP-dimTheta,dimP-dimTheta) = noise_params_.getGyroscopeBiasCov();
    Qc.block<3,3>(dimP-dimTheta+3,dimP-dimTheta+3) = noise_params_.getAccelerometerBiasCov();

    // Noise Covariance Discretization
    Eigen::MatrixXd PhiG = Phi * G;
    Eigen::MatrixXd Qd = PhiG * Qc * PhiG.transpose() * dt; // Approximated discretized noise matrix (TODO: compute analytical)
    return Qd;

}

///*********///
// InEKF Propagation IMU
void InEKF::Propagate(const Eigen::Matrix<double,6,1>& imu,double dt)
{
    int dimX = state_.dimX();
    int dimTheta = state_.dimTheta();
    int dimP = state_.dimP();
    Eigen::Vector3d w = imu.head(3) - state_.getGyroscopeBias();
    Eigen::Vector3d a = imu.tail(3) - state_.getAccelerometerBias();

    Eigen::MatrixXd X = state_.getX();
    Eigen::MatrixXd Xinv = state_.Xinv();
    Eigen::MatrixXd P = state_.getP();

    //*****propagation covariance****//
    Eigen::MatrixXd Phi = this->StateTransitionMatrix(w,a,dt);
    Eigen::MatrixXd Qd = this->DiscreteNoiseMatrix(Phi,dt);
    Eigen::MatrixXd P_pred = Phi*P*Phi.transpose() + Qd;

    //如果estimate_bias_取false则去除相关性
    if (!estimate_bias_) {
        P_pred.block(0, dimP - dimTheta, dimP - dimTheta, dimTheta) = Eigen::MatrixXd::Zero(dimP - dimTheta, dimTheta);
        P_pred.block(dimP - dimTheta ,0, dimTheta, dimP - dimTheta) = Eigen::MatrixXd::Zero(dimTheta, dimP - dimTheta);
        P_pred.block(dimP - dimTheta,dimP - dimTheta, dimTheta, dimTheta) = Eigen::MatrixXd::Identity(dimTheta, dimTheta);
    }  

    //*****propagation mean****//
    Eigen::Matrix3d R = state_.getRotation();
    Eigen::Vector3d v = state_.getVelocity();
    Eigen::Vector3d p = state_.getPosition();
    Eigen::Vector3d phi = w*dt;
    Eigen::Matrix3d G0 = Gamma_SO3(phi,0);
    Eigen::Matrix3d G1 = Gamma_SO3(phi,1);
    Eigen::Matrix3d G2 = Gamma_SO3(phi,2);

    Eigen::MatrixXd X_pred = X;
    if (state_.getStateType() == StateType::WorldCentric) 
    {
        // Propagate world-centric state estimate
        X_pred.block<3,3>(0,0) = R * G0;
        X_pred.block<3,1>(0,3) = v + (R*G1*a + g_)*dt;
        X_pred.block<3,1>(0,4) = p + v*dt + (R*G2*a + 0.5*g_)*dt*dt;
    } 
    else 
    {
        // Propagate body-centric state estimate
        Eigen::MatrixXd X_pred = X;
        Eigen::Matrix3d G0t = G0.transpose();
        X_pred.block<3,3>(0,0) = G0t*R;
        X_pred.block<3,1>(0,3) = G0t*(v - (G1*a + R*g_)*dt);
        X_pred.block<3,1>(0,4) = G0t*(p + v*dt - (G2*a + 0.5*R*g_)*dt*dt);
        for (int i=5; i<dimX; ++i) {
            X_pred.block<3,1>(0,i) = G0t*X.block<3,1>(0,i);
        }
    } 

    //*****update state****//
    state_.setX(X_pred);
    state_.setP(P_pred); 

} 


//*****Correct state*******//

//*****for right invariant observation*****//
void InEKF::CorrectRightInvariant(const Eigen::MatrixXd& Z,const Eigen::MatrixXd& H,const Eigen::MatrixXd& N, bool is_cam)
{
    if(is_cam)
        tmp_state_ = state_;

    Eigen::MatrixXd X = state_.getX();
    Eigen::VectorXd Theta = state_.getTheta();
    Eigen::MatrixXd P = state_.getP();
    int dimX = state_.dimX();
    int dimTheta = state_.dimTheta();
    int dimP = state_.dimP();

    if (error_type_==ErrorType::LeftInvariant) 
    {
        Eigen::MatrixXd Adj = Eigen::MatrixXd::Identity(dimP,dimP);
        Adj.block(0,0,dimP-dimTheta,dimP-dimTheta) = Adjoint_SEK3(X); 
        P = (Adj * P * Adj.transpose()).eval(); 
    }

    //计算卡尔曼增益
    Eigen::MatrixXd PHT = P*H.transpose();
    Eigen::MatrixXd S = H*PHT +N;
    Eigen::MatrixXd K = PHT*S.inverse();

    // std::cout<< "Inekf running"<<std::endl;
    Eigen::VectorXd delta=K*Z;
    Eigen::MatrixXd dX = Exp_SEK3(delta.segment(0,delta.rows()-dimTheta));
    // if (is_cam){
    //     cout << "dX: " << dX << endl;
    //     cout << "delta: " << delta << endl;
    // }
    Eigen::MatrixXd dTheta = delta.segment(delta.rows()-dimTheta,dimTheta); 

    //***update state****//
    Eigen::MatrixXd X_new = dX*X;
    dTheta(2) = 0;
    Eigen::VectorXd Theta_new = Theta +dTheta;
    state_.setX(X_new);
    state_.setTheta(Theta_new);
    //update covariance
    Eigen::MatrixXd IKH = Eigen::MatrixXd::Identity(dimP,dimP) - K*H;
    Eigen::MatrixXd P_new = IKH*P*IKH.transpose() + K*N*K.transpose();

    P_new.row(dimP-dimTheta+2).setZero();
    P_new.col(dimP-dimTheta+2).setZero();
    P_new(dimP - dimTheta+2,dimP - dimTheta+2) = 1e-4;


    if (error_type_==ErrorType::LeftInvariant) {
        Eigen::MatrixXd AdjInv = Eigen::MatrixXd::Identity(dimP,dimP);
        AdjInv.block(0,0,dimP-dimTheta,dimP-dimTheta) = Adjoint_SEK3(state_.Xinv()); 
        P_new = (AdjInv * P_new * AdjInv.transpose()).eval();
    }

    state_.setP(P_new);
}

//*****for Left invariant observation*****//
void InEKF::CorrectLeftInvariant(const Eigen::MatrixXd& Z,const Eigen::MatrixXd& H,const Eigen::MatrixXd& N)
{
    Eigen::MatrixXd X = state_.getX();
    Eigen::VectorXd Theta = state_.getTheta();
    Eigen::MatrixXd P = state_.getP();
    int dimX = state_.dimX();
    int dimTheta = state_.dimTheta();
    int dimP = state_.dimP();

    if (error_type_==ErrorType::RightInvariant) 
    {
        Eigen::MatrixXd AdjInv = Eigen::MatrixXd::Identity(dimP,dimP);
        AdjInv.block(0,0,dimP-dimTheta,dimP-dimTheta) = Adjoint_SEK3(state_.Xinv()); 
        P = (AdjInv * P * AdjInv.transpose()).eval();
    }

    //计算卡尔曼增益
    Eigen::MatrixXd PHT = P*H.transpose();
    Eigen::MatrixXd S = H*PHT +N;
    Eigen::MatrixXd K = PHT*S.inverse();

    Eigen::VectorXd delta=K*Z;
    Eigen::MatrixXd dX = Exp_SEK3(delta.segment(0,delta.rows()-dimTheta));
    Eigen::MatrixXd dTheta = delta.segment(delta.rows()-dimTheta,dimTheta); 

    //***update state****//
    Eigen::MatrixXd X_new = X*dX; // Left-Invariant Update
    Eigen::VectorXd Theta_new = Theta + dTheta;

    state_.setX(X_new); 
    state_.setTheta(Theta_new);
    //update covariance
    Eigen::MatrixXd IKH = Eigen::MatrixXd::Identity(dimP,dimP) - K*H;
    Eigen::MatrixXd P_new = IKH*P*IKH.transpose() + K*N*K.transpose();

    if (error_type_==ErrorType::LeftInvariant) {
        Eigen::MatrixXd AdjInv = Eigen::MatrixXd::Identity(dimP,dimP);
        AdjInv.block(0,0,dimP-dimTheta,dimP-dimTheta) = Adjoint_SEK3(state_.Xinv()); 
        P_new = (AdjInv * P_new * AdjInv.transpose()).eval();
    }

    state_.setP(P_new);
}

//TODO   Create Observation from vector of landmark measurements
void InEKF::CorrectVelocity(const Eigen::Vector3d& measured_velocity,const Eigen::Matrix3d& covariance)
{
    Eigen::VectorXd Z, Y, b;
    Eigen::MatrixXd H, N, PI;

    // 填写观测数据
    int dimX = state_.dimX();
    int dimTheta = state_.dimTheta();
    int dimP = state_.dimP();

    H.conservativeResize(3, dimP);
    H.block(0,0,3,dimP) = Eigen::MatrixXd::Zero(3,dimP);
    H.block(0,3,3,3) = Eigen::Matrix3d::Identity(); 

    // Fill out N
    N.conservativeResize(3, 3);
    N = covariance;
    Eigen::Matrix3d R = state_.getRotation();
    Eigen::Vector3d v = state_.getVelocity();
    int startIndex = Z.rows();
    Z.conservativeResize(startIndex+3, Eigen::NoChange);
    Z.segment(0,3) = R*measured_velocity - v; 
    if (Z.rows()>0) 
    {
        this->CorrectRightInvariant(Z,H,N);
    }

}

void InEKF::CorrectPose(const unordered_map<int, int>& matches, const vec_vec3d& lamp_world_pos, const vec_vec4d& boxes, double high_lamp){
    Eigen::VectorXd Z, Y, b;
    Eigen::MatrixXd H, N, PI;

    // 填写观测数据
    int dimX = state_.dimX();
    int dimTheta = state_.dimTheta();
    int dimP = state_.dimP();

    Eigen::Matrix3d R = state_.getRotation();
    Eigen::Vector3d p = state_.getPosition();

    H.conservativeResize(2 * matches.size(), dimP);
    H = Eigen::MatrixXd::Zero(2 * matches.size(), dimP);

    N.conservativeResize(2 * matches.size(), 2 * matches.size());
    N = Eigen::MatrixXd::Zero(2 * matches.size(), 2 * matches.size());

    int startIndex = Z.rows();
    Z.conservativeResize(startIndex + 2 * matches.size(), Eigen::NoChange);

    // for(int i = 0; i < boxes.size(); i++){
    //     cout << "box: " << i << boxes[i] << endl;
    // }

    int i = 0;
    // bool left = false, right = false;
    for(auto iter = matches.begin(); iter != matches.end(); ++iter){
        cout << "matches: " << iter->first << " " << iter->second << endl;

        Eigen::Vector3d map_pt = lamp_world_pos[iter->second];
        // cout << "box: " << iter->first << " iter_index: " << iter->second << endl;
        Eigen::Vector3d map_pt_cam = Rcb_ * R.transpose() * (map_pt - p) + pcb_;
        double inv_z = 1.0 / map_pt_cam.z();

        Eigen::Matrix<double, 2, 3> H_part1;
        H_part1.row(0) = inv_z * (Rcb_.row(0) - inv_z * map_pt_cam.x() * Rcb_.row(2));
        H_part1.row(1) = inv_z * (Rcb_.row(1) - inv_z * map_pt_cam.y() * Rcb_.row(2));

        Eigen::Matrix3d H_part2 = R.transpose();

        Eigen::Matrix<double, 3, 9> H_part3 = Eigen::MatrixXd::Zero(3, 9);
        H_part3.block<3, 3>(0, 0) = skew(map_pt), H_part3.block<3, 3>(0, 6) = - Eigen::Matrix3d::Identity();

        H.block(2 * i, 0, 2, 9) = H_part1 * H_part2 * H_part3;

        //Covariance matrix
        double box_len = boxes[iter->first](2) - boxes[iter->first](0);
        double box_width = boxes[iter->first](3) - boxes[iter->first](1);

        Eigen::Matrix2d sqrt_noise = Eigen::Matrix2d::Identity();
        // sqrt_noise(0, 0) = (0.1 * box_len) / cam_fx;
        sqrt_noise(0, 0) = 0.5 / cam_fx;
        sqrt_noise(1, 1) = 0.5 / cam_fx;
        // sqrt_noise(1, 1) = (0.1 * box_width) / cam_fy;

        double h_dist = sqrt(map_pt_cam.x() * map_pt_cam.x() + map_pt_cam.z() + map_pt_cam.z());
        N.block(2 * i , 2 * i, 2, 2) = sqrt_noise * sqrt_noise;

        //Error matrix
        Eigen::Vector2d lift_pt;
        if(high_lamp < 0){
            if(map_pt_cam.y() < high_lamp){
                lift_pt << 0.5 * (boxes[iter->first](0) + boxes[iter->first](2)), 0.55 * boxes[iter->first](1) + 0.45 * boxes[iter->first](3);
                // cout << "find high lamp" << endl;
            }
            else
                lift_pt << 0.5 * (boxes[iter->first](0) + boxes[iter->first](2)), 0.5 * (boxes[iter->first](1) + boxes[iter->first](3));
        }
        else{
            lift_pt << 0.5 * (boxes[iter->first](0) + boxes[iter->first](2)), 0.5 * (boxes[iter->first](1) + boxes[iter->first](3));
        }
        // int off = 20;
        // if(lift_pt.x() <= 640 - off)
        //     left = true;
        // if(lift_pt.x() > 640 + off)
        //     right = true;

        // cout << "lift_pt: " << lift_pt << endl;
        // cout << "boxes: " << boxes[iter->first] << endl;
        lift_pt.x() = (lift_pt.x() - cam_cx) / cam_fx;
        lift_pt.y() = (lift_pt.y() - cam_cy) / cam_fy;

        Z.segment(2 * i, 2) = lift_pt - inv_z * map_pt_cam.segment(0, 2);
        // cout << "map_pt: " << map_pt.transpose() << endl;
        // cout << "map_pt_cam: " << map_pt_cam.transpose() << endl;
        // cout << "lift_pt: " << lift_pt << endl;
        // cout << "boxes: " << boxes[iter->first] << endl;
        // cout << "proj_pt: " << inv_z * map_pt_cam.segment(0, 2) << endl;

        ++i;
    }
    // cout << "H: " << endl << H << endl;
    // cout << "N: " << endl << N << endl;
    cout << "Z: " << endl << Z << endl;
    if (Z.rows()>0) 
    {
        // if (!left || !right){
        //     for(int i = 0; i < matches.size(); ++i){
        //         N.block(2 * i, 2 * i, 1, 1) = N.block(2 * i, 2 * i, 1, 1) * 2;
        //         N.block(2 * i + 1, 2 * i + 1, 1, 1) = N.block(2 * i + 1, 2 * i + 1, 1, 1) * 200;
        //     }
        // }
        // else{
        // if(matches.size() <= 3){
        //     for(int i = 0; i < matches.size(); ++i){
        //         N.block(2 * i, 2 * i, 1, 1) = N.block(2 * i, 2 * i, 1, 1) * 2.0 / matches.size();
        //         N.block(2 * i + 1, 2 * i + 1, 1, 1) = N.block(2 * i + 1, 2 * i + 1, 1, 1) * 50.0 / matches.size();
        //     }
        // }
        // }
        this->CorrectRightInvariant(Z, H, N, true);
    }

}

void InEKF::rollBack(){
    state_ = tmp_state_;
}

