#ifndef COMMON_LIB_H
#define COMMON_LIB_H

#include <deque>
#include <string>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/io.h>
#include <pcl_conversions/pcl_conversions.h>
#include "map_relocalization/BoundingBoxes.h"

using namespace std;

typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 4, 1> Vector4d;
typedef vector<Vector6d, Eigen::aligned_allocator<Vector6d>> vec_vec6d;
typedef vector<Vector4d, Eigen::aligned_allocator<Vector4d>> vec_vec4d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> vec_vec3d;
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> vec_vec2d;
typedef vector<Eigen::Matrix2d, Eigen::aligned_allocator<Eigen::Matrix2d>> vec_mat2d;
typedef vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> vec_mat3d;
typedef vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> vec_mat4d;
typedef deque<Vector4d, Eigen::aligned_allocator<Vector4d>> deq_vec4d;
typedef deque<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> deq_vec3d;
typedef deque<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> deq_mat3d;
typedef deque<Matrix6d, Eigen::aligned_allocator<Matrix6d>> deq_mat6d;
typedef map<int, Eigen::Vector3d, std::less<int>, Eigen::aligned_allocator<pair<const int, Eigen::Vector3d>>> map_vec3d;
typedef map<int, Eigen::Matrix<double, 5, 1>, std::less<int>, Eigen::aligned_allocator<std::pair<const int, Eigen::Matrix<double, 5, 1>>>> map_fid_vec5d;

static int WINDOW_SIZE = 13;
static int frame_count = 0;
static double cam_fx = 910.777, cam_fy = 910.656, cam_cx = 639.846, cam_cy = 355.401;

enum OptimizationMethod{
    CENTER_CUR,
    CENTER_HIST,
    POINTS_HIST,
    EPLINE_HIST,
    ANGLE_HIST,
    ANGLEBIN_HIST
};

enum WindowState{
    IMU_INITIAL,
    POSE_INITIAL,
    IMU_INITIAL_SEC,
    FILLING,
    SLIDING
};

struct IMG_MSG {
    double header;
    vector<Eigen::Vector3d> points;
    vector<int> id_of_point;
    vector<float> u_of_point;
    vector<float> v_of_point;
    vector<float> velocity_x_of_point;
    vector<float> velocity_y_of_point;
};

struct dict{
	string name;
	double value;
	dict(string a, double v):name(a),value(v){}
    dict(){
        name = "";
        value = 0.0;
    }
};

struct Match{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    vector<int> M;
    vector<int> hungary_result;
    Eigen::MatrixXd cost_matrix;
    int box_num, lamp_num; //包含no_lamp数量 = lamp_cur_pos.size() + 1
};

struct Measures{
    int deep_learned_boxes;
    double timestamp, last_timestamp;
    cv::Mat img;
    map_relocalization::BoundingBoxes::Ptr box;
    deque<sensor_msgs::Imu::ConstPtr> imu_deq;
    deque<nav_msgs::Odometry::ConstPtr> odom_deq;
    void clear(){last_timestamp = -1; odom_deq.clear(); imu_deq.clear();}
    Measures copy(){
        Measures data;
        data.timestamp = this->timestamp;
        data.last_timestamp = this->last_timestamp;
        data.img = this->img.clone();
        map_relocalization::BoundingBoxes::Ptr box(new map_relocalization::BoundingBoxes(*this->box));
        data.box = box;
        
        data.imu_deq.resize(this->imu_deq.size());
        for(int i = 0; i < this->imu_deq.size(); i++){
            sensor_msgs::ImuConstPtr imu_msg(new sensor_msgs::Imu(*this->imu_deq[i]));
            data.imu_deq[i] = imu_msg;
        }

        data.odom_deq.resize(this->odom_deq.size());
        for(int i = 0; i < this->odom_deq.size(); i++){
            nav_msgs::OdometryConstPtr odom_msg(new nav_msgs::Odometry(*this->odom_deq[i]));
            data.odom_deq[i] = odom_msg;
        }
        return data;
    }
};

struct OdomParam{
    bool mdt_cov_noise_fixed;
    double mvscale;
    double mrc;
    double mFreqRef;
    Eigen::Matrix2d mSigma;
    Matrix6d mSigmam;
    vector<double> camera_odom_pos, camera_odom_rot;
    vector<double> cov_vel;
};

struct ImuParam{
    vector<double> std_gy;
    vector<double> std_acc;
    vector<double> std_bg;
    vector<double> std_ba;
    vector<double> camera_imu_pos, camera_imu_rot;
    vector<double> odom_imu_pos, odom_imu_rot;
    vector<double> camera_odom_pos, camera_odom_rot;
    bool normalized;
};

inline bool comp_dict_min(const dict& l, const dict& r){
    return l.value < r.value;
}

inline bool comp_dict_max(const dict& l, const dict& r){
    return l.value < r.value;
}

inline bool comp_dvec(const pair<double, pair<int, int>>& l, const pair<double, pair<int, int>>& r){
    return l.first < r.first;
}

inline bool comp_vdp_down(const pair<double, int>& l, const pair<double, int>& r){
    return l.first > r.first;
}

inline Eigen::Matrix3d vector_skew(const Eigen::Vector3d& vec){
    Eigen::Matrix3d mtx;
    mtx << 0       ,  -vec.z(), vec.y(),
           vec.z() ,  0       , -vec.x(),
           -vec.y(),  vec.x() , 0;
    return mtx;
}

// inline double mahalanobis(const Eigen::VectorXd& err, const Eigen::MatrixXd& cov){
//     Eigen::MatrixXd cov_ld = cov.llt().matrixL();
//     Eigen::VectorXd vld = cov_ld.inverse() * err;
//     return vld.transpose() * vld;   
// }

// inline vector<int> find_compatible_idx(const Eigen::VectorXi& vec){
//     vector<int> idx;
//     for (int i = 0; i < vec.rows(); i++){
//         if (vec[i] >= 0){
//             idx.push_back(i);
//         }
//     }
//     return idx;
// }

inline vector<int> find_compatible_idx(const vector<int>& vec){
    vector<int> idx;
    for (int i = 0; i < vec.size(); i++){
        if (vec[i] >= 0){
            idx.push_back(i);
        }
    }
    return idx;
}

inline int parings(const vector<int>& M){
    return find_compatible_idx(M).size();
}

// inline int parings(const Eigen::VectorXi& M){
//     return find_compatible_idx(M).size();
// }

inline Eigen::Matrix3d left_jacobbian_inv(const Eigen::Vector3d& rotation_vector){
    double half_theta = rotation_vector.norm() * 0.5;
    double half_theta_cot = half_theta * cos(half_theta) / (sin(half_theta) + 1e-5);
    Eigen::Vector3d unit_rotation_vector = rotation_vector.normalized();
    return half_theta_cot * Eigen::Matrix3d::Identity() + (1 - half_theta_cot) * unit_rotation_vector * unit_rotation_vector.transpose() - half_theta * vector_skew(unit_rotation_vector);
}

inline Eigen::Matrix3d right_jacobbian_inv(const Eigen::Vector3d& rotation_vector){
    double half_theta = rotation_vector.norm() * 0.5;
    double half_theta_cot = half_theta * cos(half_theta) / (sin(half_theta) + 1e-5);
    Eigen::Vector3d unit_rotation_vector = rotation_vector.normalized();
    return half_theta_cot * Eigen::Matrix3d::Identity() + (1 - half_theta_cot) * unit_rotation_vector * unit_rotation_vector.transpose() + half_theta * vector_skew(unit_rotation_vector);
}

inline void c_recur(int k, int n, int m, vector<int> list, vector<vector<int>>& lists){
    list.push_back(k - 1);
    for (int i = k; i <= (m - n) && n > 0; ++i) {
        c_recur(i + 1, n - 1, m, list, lists);
    }
    if (n == 0) {
        lists.push_back(list);
    }
}

inline void Combo(const int& half_num, const int& turn, vector<vector<int>>& lists){
    vector<int> list;
    for (int i = 0; i <= half_num - turn; ++i){
        c_recur(i + 1, turn - 1, half_num, list, lists);
    }
}

inline void Permute(vector<int>& nums, int start, vector<vector<int>>& result){
    if(start >= nums.size()){
        result.push_back(nums);
        return;
    }
    for(int i = start; i < nums.size(); i++){
        swap(nums[start], nums[i]);
        Permute(nums, start + 1, result);
        swap(nums[start], nums[i]);
    }
}

inline vector<vector<int>> Permute(vector<int>& nums){
    vector<vector<int>> result;
    Permute(nums, 0, result);
    return result;
}

#endif