#include "inekf/inekf_estimator.hpp"

BodyEstimator::BodyEstimator(): t_prev_(0), imu_prev_(Vector6d::Zero()) {
}

void BodyEstimator::setImuParams(const ImuParam& imu_params){
    NoiseParams params;
    if(imu_params.std_gy.size() == 1)
        params.setGyroscopeNoise(imu_params.std_gy[0]);
    else
        params.setGyroscopeNoise(Eigen::Vector3d(imu_params.std_gy[0], imu_params.std_gy[1], imu_params.std_gy[2]));

    if(imu_params.std_acc.size() == 1)
        params.setAccelerometerNoise(imu_params.std_acc[0]);
    else
        params.setAccelerometerNoise(Eigen::Vector3d(imu_params.std_acc[0], imu_params.std_acc[1], imu_params.std_acc[2]));
    
    if(imu_params.std_bg.size() == 1)
        params.setGyroscopeBiasNoise(imu_params.std_gy[0]);
    else
        params.setGyroscopeBiasNoise(Eigen::Vector3d(imu_params.std_bg[0], imu_params.std_bg[1], imu_params.std_bg[2]));
    
    if(imu_params.std_ba.size() == 1)
        params.setAccelerometerBiasNoise(imu_params.std_ba[0]);
    else
        params.setAccelerometerBiasNoise(Eigen::Vector3d(imu_params.std_ba[0], imu_params.std_ba[1], imu_params.std_ba[2]));

    filter_.setNoiseParams(params);

    Eigen::Matrix3d Rcb;
    Eigen::Vector3d pcb;

    // Rcb << imu_params.camera_odom_rot[0], imu_params.camera_odom_rot[1], imu_params.camera_odom_rot[2],
    //        imu_params.camera_odom_rot[3], imu_params.camera_odom_rot[4], imu_params.camera_odom_rot[5],
    //        imu_params.camera_odom_rot[6], imu_params.camera_odom_rot[7], imu_params.camera_odom_rot[8];
    // pcb << imu_params.camera_odom_pos[0], imu_params.camera_odom_pos[1], imu_params.camera_odom_pos[2];
    Rcb << imu_params.camera_imu_rot[0], imu_params.camera_imu_rot[1], imu_params.camera_imu_rot[2],
           imu_params.camera_imu_rot[3], imu_params.camera_imu_rot[4], imu_params.camera_imu_rot[5],
           imu_params.camera_imu_rot[6], imu_params.camera_imu_rot[7], imu_params.camera_imu_rot[8];
    pcb << imu_params.camera_imu_pos[0], imu_params.camera_imu_pos[1], imu_params.camera_imu_pos[2];
    filter_.setExtrinsics(Rcb, pcb);
    std::cout << "The NoiseParams are initialized to: \n" <<std::endl << filter_.getNoiseParams() <<std::endl;
}

void BodyEstimator::initState(const ImuMeasurement<double>& imu_packet_in, 
                        const WheelEncodeMeasurement& wheel_state_packet_in, WinsState& state, const Eigen::Matrix3d& R, const Eigen::Vector3d& p) {
    // Clear filter
    filter_.clear();

    // Initialize state mean
    Eigen::Quaternion<double> quat(imu_packet_in.orientation.w, 
                                   imu_packet_in.orientation.x,
                                   imu_packet_in.orientation.y,
                                   imu_packet_in.orientation.z); 
    // Eigen::Matrix3d R0 = quat.toRotationMatrix(); // Initialize based on VectorNav estimate
    // Eigen::Matrix3d R0 = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d R0 = R;

    Eigen::Vector3d v0_body = wheel_state_packet_in.getLinearVelocity();
    Eigen::Vector3d v0 = R0*v0_body; // initial velocity

    // Eigen::Vector3d v0 = {0.0,0.0,0.0};
    // Eigen::Vector3d p0 = {0.0, 0.0, 0.0}; // initial position, we set imu frame as world frame
    Eigen::Vector3d p0 = p;

    RobotState initial_state; 
    
    initial_state.setRotation(R0);
    initial_state.setVelocity(v0);
    initial_state.setPosition(p0);
    initial_state.setGyroscopeBias(bg0_);
    initial_state.setAccelerometerBias(ba0_);

    // Initialize state covariance
    initial_state.setRotationCovariance(0.0001*Eigen::Matrix3d::Identity()); //0.03
    initial_state.setVelocityCovariance(0.0001*Eigen::Matrix3d::Identity()); //0.01
    initial_state.setPositionCovariance(0.000001*Eigen::Matrix3d::Identity()); //0.00001
    initial_state.setGyroscopeBiasCovariance(0.0001*Eigen::Matrix3d::Identity()); //0.0001
    initial_state.setAccelerometerBiasCovariance(0.0025*Eigen::Matrix3d::Identity()); //0.0025

    filter_.setState(initial_state);
    std::cout << "Robot's state mean is initialized to: \n";
    std::cout << filter_.getState() << std::endl;
    std::cout << "Robot's state covariance is initialized to: \n";
    std::cout << filter_.getState().getP() << std::endl;

    // Set enabled flag
    t_prev_ = imu_packet_in.getTime();
    state.setTime(t_prev_);
    imu_prev_ << imu_packet_in.angular_velocity.x, 
                imu_packet_in.angular_velocity.y, 
                imu_packet_in.angular_velocity.z,
                imu_packet_in.linear_acceleration.x,
                imu_packet_in.linear_acceleration.y,
                imu_packet_in.linear_acceleration.z;
    enabled_ = true;
}

void BodyEstimator::ReinitPoseBiasState(const Eigen::Matrix3d& R, const Eigen::Vector3d& p, WinsState& state, const double& t, const Eigen::Vector3d& ba, const Eigen::Vector3d& bg){
    RobotState optimized_state;
    RobotState current_state = filter_.getState();

    optimized_state.setRotation(R);

    Eigen::Vector3d v_body_cur = current_state.getRotation().transpose() * current_state.getVelocity();
    optimized_state.setVelocity(R * v_body_cur);
    optimized_state.setPosition(p);
    optimized_state.setGyroscopeBias(bg);
    optimized_state.setAccelerometerBias(ba);

    // optimized_state.setRotationCovariance(0.03*Eigen::Matrix3d::Identity());
    optimized_state.setRotationCovariance(0.0001*Eigen::Matrix3d::Identity());
    optimized_state.setVelocityCovariance(0.0001*Eigen::Matrix3d::Identity());
    optimized_state.setPositionCovariance(0.000001*Eigen::Matrix3d::Identity());
    // optimized_state.setPositionCovariance(0.00001*Eigen::Matrix3d::Identity());
    optimized_state.setGyroscopeBiasCovariance(0.0001*Eigen::Matrix3d::Identity());
    optimized_state.setAccelerometerBiasCovariance(0.0025*Eigen::Matrix3d::Identity());

    filter_.clear();

    filter_.setState(optimized_state);
    std::cout << "Robot's state mean is set to: \n";
    std::cout << filter_.getState() << std::endl;
    std::cout << "Robot's state covariance is set to: \n";
    std::cout << filter_.getState().getP() << std::endl;

    state.setTime(t);
}

void BodyEstimator::ReinitPoseVState(const Eigen::Matrix3d& R, const Eigen::Vector3d& p, WinsState& state, const double& t, const Eigen::Vector3d& v){
    RobotState optimized_state;
    RobotState current_state = filter_.getState();

    optimized_state.setRotation(R);

    optimized_state.setVelocity(v);
    optimized_state.setPosition(p);
    optimized_state.setGyroscopeBias(current_state.getGyroscopeBias());
    optimized_state.setAccelerometerBias(current_state.getAccelerometerBias());

    // optimized_state.setRotationCovariance(0.03*Eigen::Matrix3d::Identity());
    optimized_state.setRotationCovariance(0.0001*Eigen::Matrix3d::Identity());
    optimized_state.setVelocityCovariance(0.0001*Eigen::Matrix3d::Identity());
    optimized_state.setPositionCovariance(0.000001*Eigen::Matrix3d::Identity());
    // optimized_state.setPositionCovariance(0.00001*Eigen::Matrix3d::Identity());
    optimized_state.setGyroscopeBiasCovariance(0.0001*Eigen::Matrix3d::Identity());
    optimized_state.setAccelerometerBiasCovariance(0.0025*Eigen::Matrix3d::Identity());

    filter_.clear();

    filter_.setState(optimized_state);
    std::cout << "Robot's state mean is set to: \n";
    std::cout << filter_.getState() << std::endl;
    std::cout << "Robot's state covariance is set to: \n";
    std::cout << filter_.getState().getP() << std::endl;

    state.setTime(t);
}

void BodyEstimator::ReinitPose(const Eigen::Matrix3d& R, const Eigen::Vector3d& p, WinsState& state, const double& t){
    RobotState optimized_state;
    RobotState current_state = filter_.getState();

    optimized_state.setRotation(R);

    optimized_state.setVelocity(Eigen::Vector3d::Zero());
    optimized_state.setPosition(p);
    optimized_state.setGyroscopeBias(current_state.getGyroscopeBias());
    optimized_state.setAccelerometerBias(current_state.getAccelerometerBias());

    // optimized_state.setRotationCovariance(0.03*Eigen::Matrix3d::Identity());
    optimized_state.setRotationCovariance(0.0001*Eigen::Matrix3d::Identity());
    optimized_state.setVelocityCovariance(0.0001*Eigen::Matrix3d::Identity());
    optimized_state.setPositionCovariance(0.000001*Eigen::Matrix3d::Identity());
    // optimized_state.setPositionCovariance(0.00001*Eigen::Matrix3d::Identity());
    optimized_state.setGyroscopeBiasCovariance(0.0001*Eigen::Matrix3d::Identity());
    optimized_state.setAccelerometerBiasCovariance(0.0025*Eigen::Matrix3d::Identity());

    filter_.clear();

    filter_.setState(optimized_state);
    std::cout << "Robot's state mean is set to: \n";
    std::cout << filter_.getState() << std::endl;
    std::cout << "Robot's state covariance is set to: \n";
    std::cout << filter_.getState().getP() << std::endl;

    state.setTime(t);
}

bool BodyEstimator::initBias(const ImuMeasurement<double>& imu_packet_in){
    // Initialize bias based on imu orientation and static assumption
    if (bias_init_vec_.size() < 250) {
        Eigen::Vector3d w, a;
        w << imu_packet_in.angular_velocity.x, 
             imu_packet_in.angular_velocity.y, 
             imu_packet_in.angular_velocity.z;
        a << imu_packet_in.linear_acceleration.x,
             imu_packet_in.linear_acceleration.y,
             imu_packet_in.linear_acceleration.z;
        Eigen::Matrix3d R;
        if(use_imu_ori_est_init_bias_){
            Eigen::Quaternion<double> quat(imu_packet_in.orientation.w, 
                                        imu_packet_in.orientation.x,
                                        imu_packet_in.orientation.y,
                                        imu_packet_in.orientation.z); 
            R = quat.toRotationMatrix();
        }
        else{
            R = Eigen::Matrix3d::Identity();
        }
        
        // std::cout<<"R: \n"<<R<<std::endl;
        std::cout<<"a: \n"<<a<<std::endl;
        // std::cout<<"a_world: \n"<<R.transpose()*a<<std::endl;
        
        Eigen::Vector3d g; g << 0,0,-9.81;
        a = (R.transpose()*(R*a + g)).eval();
        Eigen::Matrix<double,6,1> v; 
        v << w(0),w(1),w(2),a(0),a(1),a(2);
        bias_init_vec_.push_back(v); // Store imu data with gravity removed
        return false;
    } else {
        // Compute average bias of stored data
        Eigen::Matrix<double,6,1> avg = Eigen::Matrix<double,6,1>::Zero();
        for (int i = 0; i < bias_init_vec_.size(); ++i) {
            avg = (avg + bias_init_vec_[i]).eval();
        }
        avg = (avg / bias_init_vec_.size()).eval();
        std::cout << "IMU bias initialized to: " << avg.transpose() << std::endl;
        bg0_ = avg.head<3>();
        ba0_ = avg.tail<3>();
        return true;
    }
}

void BodyEstimator::propagateIMU(const ImuMeasurement<double>& imu_packet_in,WinsState& state)
{
    if (!bias_initialized_) 
    {
        initBias(imu_packet_in);
    }
    Eigen::Matrix<double,6,1> imu;
    imu <<  imu_packet_in.angular_velocity.x,
            imu_packet_in.angular_velocity.y, 
            imu_packet_in.angular_velocity.z,
            imu_packet_in.linear_acceleration.x, 
            imu_packet_in.linear_acceleration.y, 
            imu_packet_in.linear_acceleration.z;
    
    double t = imu_packet_in.getTime();

    // std::cout<<"imu value: "<< imu<<std::endl;
    // Propagate state based on IMU and contact data
    double dt = t - t_prev_;
    if(estimator_debug_enabled_){
        ROS_INFO("Tprev %0.6lf T %0.6lf dt %0.6lf \n", t_prev_, t, dt);
    }

    if (dt > 0)
        filter_.Propagate(imu_prev_, dt); 

    // correctKinematics(state);

    ///TODO: Check if imu strapdown model is correct
    RobotState estimate = filter_.getState();
    Eigen::Matrix3d R = estimate.getRotation();
    Eigen::Vector3d p = estimate.getPosition();
    Eigen::Vector3d v = estimate.getVelocity();
    Vector6d bias = estimate.getTheta();
    state.setBaseRotation(R);
    state.setBasePosition(p);
    state.setBaseVelocity(v); 
    state.setImuBias(bias);
    state.setTime(t);

    // Store previous imu data
    t_prev_ = t;
    imu_prev_ = imu;
    seq_ = imu_packet_in.header.seq;

    if (estimator_debug_enabled_) 
    {
        ROS_INFO("IMU Propagation Complete: linacceleation x: %0.6f y: %.06f z: %0.6f \n", 
            imu_packet_in.linear_acceleration.x,
            imu_packet_in.linear_acceleration.y,
            imu_packet_in.linear_acceleration.z);
    }
}

void BodyEstimator::propagateIMU(const double& t, WinsState& state){

    double dt = t - t_prev_;
    if (dt > 0)
        filter_.Propagate(imu_prev_, dt);
    
    RobotState estimate = filter_.getState();
    Eigen::Matrix3d R = estimate.getRotation();
    Eigen::Vector3d p = estimate.getPosition();
    Eigen::Vector3d v = estimate.getVelocity();
    Vector6d bias = estimate.getTheta();
    state.setBaseRotation(R);
    state.setBasePosition(p);
    state.setBaseVelocity(v); 
    state.setImuBias(bias);
    state.setTime(t);

    t_prev_ = t;
}

void BodyEstimator::correctVelocity(const WheelEncodeMeasurement& wheel_state_packet_in, WinsState& state, const Eigen::Matrix3d& velocity_cov)
{
    double t = wheel_state_packet_in.getTime();

    if(std::abs(t-state.getTime())<velocity_t_thres_)
    {
        Eigen::Vector3d measured_velocity = wheel_state_packet_in.getLinearVelocity();
        filter_.CorrectVelocity(measured_velocity, velocity_cov);

        RobotState estimate = filter_.getState();
        Eigen::Matrix3d R = estimate.getRotation(); 
        Eigen::Vector3d p = estimate.getPosition();
        Eigen::Vector3d v = estimate.getVelocity();
        Vector6d bias = estimate.getTheta();

        Eigen::Vector3d gyro_bias = estimate.getGyroscopeBias();
        Eigen::Vector3d acc_bias = estimate.getAccelerometerBias();  //TODO 这些偏制似乎没用到

        state.setBaseRotation(R);
        state.setBasePosition(p);
        state.setBaseVelocity(v); 
        state.setImuBias(bias);
        state.setTime(t);
    }
    else
    {
        ROS_INFO("Velocity not updated because huge time different.");
        std::cout << std::setprecision(20) << "t: " << t << std::endl;
        std::cout << std::setprecision(20) << "state t: "<< state.getTime() << std::endl;
        std::cout << std::setprecision(20) << "time diff: " << t-state.getTime() <<std::endl;
    }
}

void BodyEstimator::correctPose(const double& t, WinsState& state, const unordered_map<int, int>& matches, const vec_vec3d& lamp_world_pos, const vec_vec4d& boxes, double high_lamp){
    
    if(std::abs(t - state.getTime()) < pose_t_thres_){
        filter_.CorrectPose(matches, lamp_world_pos, boxes, high_lamp);

        RobotState estimate = filter_.getState();
        Eigen::Matrix3d R = estimate.getRotation(); 
        Eigen::Vector3d p = estimate.getPosition();
        Eigen::Vector3d v = estimate.getVelocity();
        Vector6d bias = estimate.getTheta();

        Eigen::Vector3d gyro_bias = estimate.getGyroscopeBias();
        Eigen::Vector3d acc_bias = estimate.getAccelerometerBias();  //TODO 这些偏制似乎没用到

        state.setBaseRotation(R);
        state.setBasePosition(p);
        state.setBaseVelocity(v); 
        state.setImuBias(bias);
        state.setTime(t);

        t_prev_ = t;
    }
    else
    {
        ROS_INFO("Pose not updated because huge time different.");
        std::cout << std::setprecision(20) << "t: " << t << std::endl;
        std::cout << std::setprecision(20) << "state t: "<< state.getTime() << std::endl;
        std::cout << std::setprecision(20) << "time diff: " << t-state.getTime() <<std::endl;
    }
}

void BodyEstimator::rollBack(){
    filter_.rollBack();
}

bool InekfEstimator::Initialize(Measures& data, const Eigen::Matrix3d& R, const Eigen::Vector3d& p){

    if(estimator_.biasInitialized()){
        updateNextIMU(data);
        updateNextWheel(data);
        estimator_.initState(*(imu_packet_.get()), *(wheel_velocity_packet_.get()), state_, R, p);
        estimator_.enableFilter();
        std::cout << "State initialized." << std::endl;
        return true; //TODO clear data
    }
    else{
        while(!data.imu_deq.empty()){
            updateNextIMU(data);
            bool success = estimator_.initBias(*(imu_packet_.get()));
            if(success){
                estimator_.enableBiasInitialized();
                break;
            }
        }
        return false; //TODO clear data
    }
}

void InekfEstimator::ReInitializePoseBias(const Eigen::Matrix3d& R, const Eigen::Vector3d& p, const double& t, const Eigen::Vector3d& ba, const Eigen::Vector3d& bg){
    estimator_.ReinitPoseBiasState(R, p, state_, t, ba, bg);
}

void InekfEstimator::ReInitializePoseV(const Eigen::Matrix3d& R, const Eigen::Vector3d& p, const double& t, const Eigen::Vector3d& v){
    estimator_.ReinitPoseVState(R, p, state_, t, v);
}

void InekfEstimator::ReInitializePose(const Eigen::Matrix3d& R, const Eigen::Vector3d& p, const double& t){
    estimator_.ReinitPose(R, p, state_, t);
}

void InekfEstimator::step(Measures& data){
    if (estimator_.enabled()){
        while(!(data.imu_deq.empty() && data.odom_deq.empty())){
            if(data.odom_deq.empty()){
                updateNextIMU(data);
                estimator_.propagateIMU(*(imu_packet_.get()), state_);
            }
            else{
                if(data.imu_deq.empty()){
                    updateNextWheel(data);
                    estimator_.correctVelocity(*(wheel_velocity_packet_.get()), state_, wheel_vel_cov_);//initialize wheel_vel_cov_
                }
                else{
                    auto imu0 = data.imu_deq.front();
                    auto odom0 = data.odom_deq.front();
                    if(imu0->header.stamp < odom0->header.stamp){
                        updateNextIMU(data);
                        estimator_.propagateIMU(*(imu_packet_.get()), state_);
                    }
                    else{
                        updateNextWheel(data);
                        estimator_.correctVelocity(*(wheel_velocity_packet_.get()), state_, wheel_vel_cov_);//initialize wheel_vel_cov_
                    }
                }
            }
        }
        assert(data.imu_deq.empty());
        assert(data.odom_deq.empty());
        estimator_.propagateIMU(data.timestamp, state_);
    }
}

void InekfEstimator::stepCam(const double& t, const unordered_map<int, int>& matches, const vec_vec3d& lamp_world_pos, const vec_vec4d& boxes, double high_lamp){

    if (estimator_.enabled()){
        estimator_.correctPose(t, state_, matches, lamp_world_pos, boxes, high_lamp);
        for(auto iter = matches.begin(); iter != matches.end(); ++iter){
            cout << "box: " << iter->first << "  lamp: " << iter->second << endl;
        }
        tmp_state_ = state_;
    }
}

void InekfEstimator::stepCam(const double& t, const vector<int>& M, const vec_vec3d& lamp_world_pos, const vec_vec4d& boxes, double high_lamp){

    if (estimator_.enabled()){
        unordered_map<int, int> matches;
        for(int i = 0; i < M.size(); i++){
            matches.insert(make_pair(i, M[i]));
        }
        estimator_.correctPose(t, state_, matches, lamp_world_pos, boxes, high_lamp);
        tmp_state_ = state_;
    }
}

void InekfEstimator::updateNextIMU(Measures& data){

    if(normalized_){
        sensor_msgs::ImuPtr new_imu(new sensor_msgs::Imu(*(data.imu_deq.front())));
        new_imu->linear_acceleration.x *= g_;
        new_imu->linear_acceleration.y *= g_;
        new_imu->linear_acceleration.z *= g_;
        data.imu_deq.pop_front();
        data.imu_deq.push_front(new_imu);
    }
    // imu_packet_ = std::make_shared<ImuMeasurement<double>>(*(data.imu_deq.front()), Roi_);
    imu_packet_ = std::make_shared<ImuMeasurement<double>>(*(data.imu_deq.front()));
    data.imu_deq.pop_front();
    state_.setImu(imu_packet_);

}

void InekfEstimator::updateNextWheel(Measures& data){
    
    // wheel_velocity_packet_ = std::make_shared<WheelEncodeMeasurement>(*(data.odom_deq.front()));
    wheel_velocity_packet_ = std::make_shared<WheelEncodeMeasurement>(*(data.odom_deq.front()), Roi_.transpose());
    // cout << "vel: " << wheel_velocity_packet_->getLinearVelocity() << endl;
    data.odom_deq.pop_front();

}

void InekfEstimator::rollBack(){
    state_ = tmp_state_;
    estimator_.rollBack();
}