#ifndef IMU_HPP
#define IMU_HPP

#include <ros/ros.h>
#include "std_msgs/Header.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "nav_msgs/Odometry.h"

enum MeasurementType 
{
    EMPTY,IMU,WHEEL_ENCODER,VELOCITY 
};

class Measurement {
    struct MeasurementHeader
    {
        uint64_t seq;
        double stamp;
        std::string frame_id;

        MeasurementHeader(){}

        MeasurementHeader(const std_msgs::Header& header_in) {
            seq = (uint64_t) header_in.seq;
            stamp = header_in.stamp.sec +header_in.stamp.nsec / 1000000000.0;
            frame_id = header_in.frame_id;
        }

    };

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        Measurement()
        {
            header.stamp=0;
            type_ = EMPTY;
        }
        virtual ~Measurement() = default;

        MeasurementHeader header;
        void setHeader(const std_msgs::Header& header_in)
        {
            header.seq = (uint64_t) header_in.seq;
            header.stamp = header_in.stamp.sec + header_in.stamp.nsec / 1000000000.0;
            // cout << "frame_id: " << header_in.frame_id;
            header.frame_id = header_in.frame_id;
        }


        double getTime() const
        {
            return header.stamp;
        }
        MeasurementType getType()
        {
            return type_;
        }

        friend std::ostream& operator<<(std::ostream& os, const Measurement& m);  //TODO 

    protected:
        MeasurementType type_;

};

struct MeasurementCompare {
    bool operator()(Measurement& lhs, Measurement& rhs) const {
        return lhs.getTime() > rhs.getTime();
    }
};


template <typename T>
struct ImuOrientation {
    T w, x, y, z;
};

template <typename T>
struct ImuAngularVelocity {
    T x, y, z;
};

template <typename T>
struct ImuLinearAcceleration {
    T x, y, z;
};

template <typename T>
class ImuMeasurement : public Measurement {
    public:
        ImuOrientation<T> orientation;
        ImuAngularVelocity<T> angular_velocity;
        ImuLinearAcceleration<T> linear_acceleration;

        Eigen::Matrix3d getRotation() { return R_; }
        
        void setRotation() {
            Eigen::Quaternion<double> q(orientation.w,orientation.x,orientation.y,orientation.z);
            R_ = q.toRotationMatrix();
        }

        // Construct IMU measurement
        ImuMeasurement() {
            type_ = IMU;
        }

        ImuMeasurement(const sensor_msgs::Imu& imu_msg){
            // default is (0, 0.7071, -0.7071, 0)
            // Eigen::Quaternion<double> rotation_imu2body(rotation_imu_body[0],rotation_imu_body[1],rotation_imu_body[2],rotation_imu_body[3]);
            // Eigen::Matrix3d rotation_imu_body_matrix;
            // rotation_imu_body_matrix = rotation_imu2body.toRotationMatrix();

            Eigen::Vector3d angular_velocity_imu;
            angular_velocity_imu << imu_msg.angular_velocity.x,
                                    imu_msg.angular_velocity.y,
                                    imu_msg.angular_velocity.z;

            Eigen::Vector3d linear_acceleration_imu;
            linear_acceleration_imu << imu_msg.linear_acceleration.x,
                                        imu_msg.linear_acceleration.y,
                                        imu_msg.linear_acceleration.z;

            linear_acceleration = {linear_acceleration_imu[0],
                                    linear_acceleration_imu[1],
                                    linear_acceleration_imu[2]};

            angular_velocity = {angular_velocity_imu[0],
                                angular_velocity_imu[1],
                                angular_velocity_imu[2]};

            
            // map original orientation estimation from imu to body
            Eigen::Quaternion<double> q(imu_msg.orientation.w, 
                                        imu_msg.orientation.x, 
                                        imu_msg.orientation.y, 
                                        imu_msg.orientation.z );
            // std::cout<<"q_original: \n"<<q<<std::endl;
            Eigen::Quaternion<double> q_body = q;
            // std::cout<<"q_body: \n"<<q_body<<std::endl;
            orientation = { 
                q_body.w(), 
                q_body.x(), 
                q_body.y(), 
                q_body.z() 
            };

            setRotation();

            setHeader(imu_msg.header);

            type_ = IMU;
        }

        // Overloaded constructor for construction using ros imu topic
        ImuMeasurement(const sensor_msgs::Imu& imu_msg, const Eigen::Matrix3d& rotation_imu_body_matrix) {

            // default is (0, 0.7071, -0.7071, 0)
            // Eigen::Quaternion<double> rotation_imu2body(rotation_imu_body[0],rotation_imu_body[1],rotation_imu_body[2],rotation_imu_body[3]);
            // Eigen::Matrix3d rotation_imu_body_matrix;
            // rotation_imu_body_matrix = rotation_imu2body.toRotationMatrix();
            Eigen::Quaterniond rotation_imu2body(rotation_imu_body_matrix);

            Eigen::Vector3d angular_velocity_imu;
            angular_velocity_imu << imu_msg.angular_velocity.x,
                                    imu_msg.angular_velocity.y,
                                    imu_msg.angular_velocity.z;

            Eigen::Vector3d linear_acceleration_imu;
            linear_acceleration_imu << imu_msg.linear_acceleration.x,
                                        imu_msg.linear_acceleration.y,
                                        imu_msg.linear_acceleration.z;

            // angular_velocity = (rotation_imu_body_matrix.transpose()*angular_velocity_imu).eval();
            
            angular_velocity = {
                (rotation_imu_body_matrix*angular_velocity_imu)[0],
                (rotation_imu_body_matrix*angular_velocity_imu)[1],
                (rotation_imu_body_matrix*angular_velocity_imu)[2]
            };

            // linear_acceleration = (rotation_body_imu_matrix.transpose()*linear_acceleration_imu).eval();
            linear_acceleration = {
                (rotation_imu_body_matrix*linear_acceleration_imu)[0],
                (rotation_imu_body_matrix*linear_acceleration_imu)[1],
                (rotation_imu_body_matrix*linear_acceleration_imu)[2]
            };
            
            // map original orientation estimation from imu to body
            Eigen::Quaternion<double> q(imu_msg.orientation.w, 
                                        imu_msg.orientation.x, 
                                        imu_msg.orientation.y, 
                                        imu_msg.orientation.z );
            // std::cout<<"q_original: \n"<<q<<std::endl;
            Eigen::Quaternion<double> q_body = q * rotation_imu2body;
            // std::cout<<"q_body: \n"<<q_body<<std::endl;
            orientation = { 
                q_body.w(), 
                q_body.x(), 
                q_body.y(), 
                q_body.z() 
            };

            setRotation();

            setHeader(imu_msg.header);

            type_ = IMU;
        }


    private:
        Eigen::Matrix3d R_;
};

class WheelEncodeMeasurement : public Measurement {
    public:
        WheelEncodeMeasurement()
        {
            type_ = WHEEL_ENCODER;
            translation_ = Eigen::Vector3d::Zero();
            rotation_ = Eigen::Matrix3d::Identity();
            transformation_ = Eigen::Matrix4d::Identity();
            wheel_linearV =  Eigen::Vector3d ::Zero();
            wheel_angularV = Eigen::Vector3d ::Zero();

        }

        WheelEncodeMeasurement(const nav_msgs::Odometry& odom_msg)
        {
            type_ = WHEEL_ENCODER;
            translation_ = Eigen::Vector3d::Zero();
            rotation_ = Eigen::Matrix3d::Identity();
            transformation_ = Eigen::Matrix4d::Identity();

            setTranslation(odom_msg.pose.pose.position);
            setRotation(odom_msg.pose.pose.orientation);
            setTransformation();
            setLinearVelocity(odom_msg.twist.twist.linear);
            setAngularVelocity(odom_msg.twist.twist.angular);
            setHeader(odom_msg.header);               
        }

        WheelEncodeMeasurement(const nav_msgs::Odometry& odom_msg, const Eigen::Matrix3d& rotation_odom_body_matrix){
            type_ = WHEEL_ENCODER;
            translation_ = Eigen::Vector3d::Zero();
            rotation_ = Eigen::Matrix3d::Identity();
            transformation_ = Eigen::Matrix4d::Identity();

            setTranslation(odom_msg.pose.pose.position);
            setRotation(odom_msg.pose.pose.orientation);
            setTransformation();
            setLinearVelocity(odom_msg.twist.twist.linear, rotation_odom_body_matrix);
            setAngularVelocity(odom_msg.twist.twist.angular, rotation_odom_body_matrix);
            setHeader(odom_msg.header);       
        }

        void setTranslation(const geometry_msgs::Point& position_in)
        {
            translation_(0) = position_in.x;
            translation_(1) = position_in.y;
            translation_(2) = position_in.z;
        }

        void setRotation(const geometry_msgs::Quaternion& orientation_in)
        {
            Eigen::Quaternion<double> orientation_quat(orientation_in.w,orientation_in.x,orientation_in.y,orientation_in.z);
            rotation_ = orientation_quat.toRotationMatrix();
        }
        
        void setTransformation()
        {
            transformation_.block<3,3>(0,0) = rotation_;    //旋转
            transformation_.block<3,1>(0,3) = translation_;   //平移
        }
        
        void setLinearVelocity(const geometry_msgs::Vector3& linearV_in)
        {
            wheel_linearV(0) = linearV_in.x;
            wheel_linearV(1) = linearV_in.y;
            wheel_linearV(2) = linearV_in.z;
                
        }

        void setLinearVelocity(const geometry_msgs::Vector3& linearV_in, const Eigen::Matrix3d& rotation_odom_body_matrix)
        {

            Eigen::Vector3d velocity(linearV_in.x, linearV_in.y, linearV_in.z);
            wheel_linearV = { (rotation_odom_body_matrix*velocity)[0],
                              (rotation_odom_body_matrix*velocity)[1],
                              (rotation_odom_body_matrix*velocity)[2]};
                
        }

        void setAngularVelocity(const geometry_msgs::Vector3& angularV_in)
        {
            wheel_angularV(0) = angularV_in.x;
            wheel_angularV(1) = angularV_in.y;
            wheel_angularV(2) = angularV_in.z;
                
        }

        void setAngularVelocity(const geometry_msgs::Vector3& angularV_in, const Eigen::Matrix3d& rotation_odom_body_matrix)
        {

            Eigen::Vector3d angular(angularV_in.x, angularV_in.y, angularV_in.z);
            wheel_angularV = { (rotation_odom_body_matrix*angular)[0],
                              (rotation_odom_body_matrix*angular)[1],
                              (rotation_odom_body_matrix*angular)[2]};
                
        }

        inline const Eigen::Matrix4d&  getTransformation() const {

            return transformation_;
        }

        inline const Eigen::Vector3d& getLinearVelocity() const{
            return wheel_linearV;
        }
        inline const Eigen::Vector3d& getAngularVelocity() const{
            return wheel_angularV;
        }
        inline const Eigen::Vector3d& getWheelPosition() const{
            return translation_;
        }
        
    private:
        Eigen::Vector3d translation_;
        Eigen::Matrix3d rotation_;
        Eigen::Matrix4d transformation_;
        Eigen::Vector3d wheel_linearV;
        Eigen::Vector3d wheel_angularV;


};


typedef std::shared_ptr<ImuMeasurement<double>> ImuMeasurementPtr;
typedef std::shared_ptr<WheelEncodeMeasurement> WheelEncodeMeasurementPtr;

#endif