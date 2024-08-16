#include <mutex>
#include <thread>
#include <iostream>
#include <ros/ros.h>
#include <csignal>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sophus/so3.hpp>
#include <sophus/se3.hpp>
#include <condition_variable>
#include <string>
#include <vector>
#include <deque>
#include <map>
#include "map_relocalization/odom_integrator.hpp"
#include "map_relocalization/common_lib.h"
#include "map_relocalization/hungary_estimator.hpp"
#include "map_relocalization/optimizer.hpp"
#include "map_relocalization/feature_tracker.hpp"
#include "map_relocalization/CameraPoseVisualization.hpp"
#include "inekf/inekf_estimator.hpp"
#include "map_relocalization/particle_filter.hpp"
#include "map_relocalization/sampler.hpp"
#include <GeographicLib/LocalCartesian.hpp>
#include <tf/transform_broadcaster.h>
// #include "map_relocalization/feature_manager.hpp"

ofstream fp_pos, fp_match, fp_cov;
FILE *fp_log1, *fp_log2;

mutex mtx_buffer;
condition_variable sig_buffer;

Measures data, last_data;

OdomParam odom_params;
ImuParam imu_params;
bool need_init, use_elevation, use_gps;
double search_dist_scope, search_z_scope;
double init_x, init_y, init_z, init_alpha, init_beta, init_gamma, yaw_off, gps_init_x, gps_init_y, gps_init_z, gps_cov_threshold;
int num_particles;

string pointcloud_path, img_topic, imu_topic, odom_topic, box_topic, gps_topic;
double last_timestamp_img, last_timestamp_imu, last_timestamp_box, last_timestamp_odom;
double timediff_img_to_imu, timediff_img_to_odom, timediff_img_to_box, timediff_img_to_gps;
double high_lamp, delta_Rbb_th, delta_pbb_th, delta_Rbb_th2, delta_pbb_th2;
double z_th, dist_th, update_z_th, update_dist_th, reloc_z_th, reloc_dist_th, reloc_update_z_th, reloc_update_dist_th, alpha1, alpha2, beta2;

double min_parallex;

deque<double> time_buffer;
deque<cv::Mat> img_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
deque<nav_msgs::Odometry::ConstPtr> odom_buffer;
deque<map_relocalization::BoundingBoxes::Ptr> box_buffer;
deque<sensor_msgs::NavSatFix::ConstPtr> gps_buffer;

Eigen::Matrix3d K, K_inv;
int res_x, res_y;
int cluster_number;
bool image_pushed;

nav_msgs::Path global_init_path, global_odom_path, global_optimized_path, global_vins_wheel_path;

Eigen::Matrix3d Rwc, Rcw, Rcb, Rwb, Rbc, Rwb2, init_Rwb;
Eigen::Matrix3d tmp_Rcw;
Matrix6d cov_wb, cov_wb2;
Eigen::Vector3d thetacw;
Vector6d state;
Eigen::Vector3d pwc, pcw, pcb, pwb, pbc, pwb2, Ow, init_pwb;
Eigen::Vector3d tmp_pcw;
Matrix6d cov;
Eigen::Matrix3d cov_lamp; //各路灯3D位置的协方差矩阵

pcl::PointCloud<pcl::PointXYZI>::Ptr lamp_cloud(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cur_lamp_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
pcl::PointCloud<pcl::PointXYZRGB>::Ptr lamp_rgbcloud(new pcl::PointCloud<pcl::PointXYZRGB>());
vec_vec3d lamp_cur_pos, lamp_cur_cam_pos, lamp_cur_world_pos, lamp_world_pos;
vec_vec4d lamp_world_plane, lamp_cur_cam_plane, lamp_cur_world_plane;
vec_vec6d lamp_world_box, lamp_cur_world_box;
vector<vec_vec3d, Eigen::aligned_allocator<vec_vec3d>> lamp_world_points, lamp_cur_cam_box;
vector<int> lamp_cur_id;
int no_obs = 0, extend = 20, left_right_gap, grey_th;
bool last_no_ob = false, last_half_box = false, last_best_match = false, find_half_match = false;
bool reloc = false;
bool init_extend = false;
int reloc_id = -1, no_left_box = 0, no_right_box = 0, line = 0;
// vector<double> alphas, betas;

// map<int, Eigen::Vector3d, Eigen::aligned_allocator<pair<int, Eigen::Vector3d>>> lamp_cur_id_pos, lamp_cur_id_cam_pos;
// map<int, Eigen::Vector3d, Eigen::aligned_allocator<pair<int, Eigen::Vector3d>>> lamp_id_pos;
// map<int, vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>, Eigen::aligned_allocator<pair<int, vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>>>> lamp_id_points;

//用于存储bbox
map<int, cv::Point2d> box_id_center;
map<int, pair<cv::Point2d, cv::Point2d>> box_id_corner;
bool ablation_reloc = true, ablation_matext = true;

int iterate_max;
WindowState window_state;
string root_dir = ROOT_DIR;

//滑窗优化中的变量
deq_mat3d deq_Rcw;
deq_vec3d deq_pcw;
deq_mat6d deq_P;
deq_mat3d deq_deltaR;
deq_vec3d deq_deltap;
deque<unsigned long> deq_id;
deque<unordered_map<int, int>> deq_matches;
deque<vec_vec4d, Eigen::aligned_allocator<vec_vec4d>> deq_boxes;
deque<map_fid_vec5d> deq_features;

int global_id = 0, initialization_id = 0;

int num_extend = 0, num_hung = 0, area_extend = 0, area_hung = 0;

double Optimizer::avg_error_ = 0.0;
double max_time = -1, mean_time = 0.0;
ros::Publisher pub_vins_wheel;
ros::Publisher pub_vins_wheel_visual;
CameraPoseVisualization cam_vins_wheel(0, 0, 1, 1);

// struct VINS_POSE{
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
//     double timestamp;
//     Eigen::Vector3d translation;
//     Eigen::Quaterniond quat;
// };

// vector<VINS_POSE> vins_poses;

bool flg_exit = false;
void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
    fp_pos.close();
    fp_match.close();
    cout << num_hung << endl;
    cout << float(area_hung) / num_hung << endl;
    cout << num_extend << endl;
    cout << float(area_extend) / num_extend << endl;
    cout << "max_time: " << max_time << endl;
    cout << "mean_time: " << mean_time / global_id << endl;
    ros::shutdown();
}

void img_cbk(const sensor_msgs::CompressedImage::ConstPtr& msg){
    if (msg->header.stamp.toSec() < last_timestamp_img){
        ROS_ERROR("img loop back, clear buffer");
        img_buffer.clear();
    }
    last_timestamp_img = msg->header.stamp.toSec();

    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
    }
    catch (cv_bridge::Exception &e) {
        ROS_ERROR("fail to load img_msg");
        return;
    }
    cv::Mat img = cv_ptr->image;

    if (abs(last_timestamp_imu - last_timestamp_img) > 10.0 && !imu_buffer.empty() && !img_buffer.empty() )
    {
        ROS_ERROR("IMU and Image not Synced, IMU time: %lf, image header time: %lf \n",last_timestamp_imu, last_timestamp_img);
    }
    if (abs(last_timestamp_odom - last_timestamp_img) > 10.0 && !odom_buffer.empty() && !img_buffer.empty() )
    {
        ROS_ERROR("Odom and Image not Synced, Odom time: %lf, image header time: %lf \n",last_timestamp_odom, last_timestamp_img);
    }

    img_buffer.push_back(img);
    time_buffer.push_back(last_timestamp_img);
    sig_buffer.notify_all();
}

void box_cbk(const map_relocalization::BoundingBoxes::ConstPtr& msg){
    map_relocalization::BoundingBoxes::Ptr new_msg(new map_relocalization::BoundingBoxes(*msg));

    new_msg->header.stamp = ros::Time().fromSec(msg->header.stamp.toSec() - timediff_img_to_box);
    double timestamp = new_msg->header.stamp.toSec();
    if (timestamp < last_timestamp_box){
        ROS_WARN("box loop back, clear buffer");
        box_buffer.clear();
    }
    last_timestamp_box = timestamp;

    box_buffer.push_back(new_msg);
    sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr& msg){
    sensor_msgs::Imu::Ptr new_msg(new sensor_msgs::Imu(*msg));

    new_msg->header.stamp = ros::Time().fromSec(msg->header.stamp.toSec() - timediff_img_to_imu);
    double timestamp = new_msg->header.stamp.toSec();
    if (timestamp < last_timestamp_imu){
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
    }
    last_timestamp_imu = timestamp;

    imu_buffer.push_back(new_msg);
    sig_buffer.notify_all();
}

void odom_cbk(const nav_msgs::Odometry::ConstPtr& msg){
    nav_msgs::Odometry::Ptr new_msg(new nav_msgs::Odometry(*msg));

    new_msg->header.stamp = ros::Time().fromSec(msg->header.stamp.toSec() - timediff_img_to_odom);
    double timestamp = new_msg->header.stamp.toSec();
    if (timestamp < last_timestamp_odom){
        ROS_WARN("odom loop back, clear buffer");
        odom_buffer.clear();
    }
    last_timestamp_odom = timestamp;

    odom_buffer.push_back(new_msg);
    sig_buffer.notify_all();
}

void gps_cbk(const sensor_msgs::NavSatFix::ConstPtr& msg){
    sensor_msgs::NavSatFix::Ptr new_msg(new sensor_msgs::NavSatFix(*msg));
    double noise_x = new_msg->position_covariance[0];
    double noise_y = new_msg->position_covariance[4];
    double noise_z = new_msg->position_covariance[8];

    if(noise_x <= gps_cov_threshold && noise_y <= gps_cov_threshold){
        new_msg->header.stamp = ros::Time().fromSec(msg->header.stamp.toSec() - timediff_img_to_gps);
        gps_buffer[0] = new_msg;
        sig_buffer.notify_all();
    }
}

void path_cbk(const nav_msgs::Odometry::ConstPtr& msg){
    if(window_state == WindowState::FILLING || window_state == WindowState::SLIDING){
        geometry_msgs::PoseStamped pose;
        pose.header.stamp = ros::Time().fromSec(data.timestamp);
        pose.header.frame_id = "camera_init";

        Eigen::Vector3d vins_pwb(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
        Eigen::Quaterniond vins_qwb(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
        Eigen::Matrix3d vins_Rwb = vins_qwb.toRotationMatrix();

        Eigen::Vector3d correct_vins_pwb = init_Rwb * vins_pwb + init_pwb;
        Eigen::Matrix3d correct_vins_Rwb = init_Rwb * vins_Rwb;
        Eigen::Quaterniond correct_vins_qwb(correct_vins_Rwb);

        pose.pose.position.x = correct_vins_pwb.x();
        pose.pose.position.y = correct_vins_pwb.y();
        pose.pose.position.z = correct_vins_pwb.z();

        pose.pose.orientation.w = correct_vins_qwb.w();
        pose.pose.orientation.x = correct_vins_qwb.x();
        pose.pose.orientation.y = correct_vins_qwb.y();
        pose.pose.orientation.z = correct_vins_qwb.z();

        global_vins_wheel_path.poses.push_back(pose);
        pub_vins_wheel.publish(global_vins_wheel_path);
        static int vins_wheel_pose_num = 0;
        vins_wheel_pose_num++;
        if(vins_wheel_pose_num % 5 == 0){
            global_vins_wheel_path.poses.push_back(pose);
            pub_vins_wheel.publish(global_vins_wheel_path);
        }

        Eigen::Vector3d correct_vins_pwc = correct_vins_Rwb * pbc + correct_vins_pwb;
        Eigen::Matrix3d correct_vins_Rwc = correct_vins_Rwb * Rbc;
        cam_vins_wheel.reset();
        cam_vins_wheel.add_pose(correct_vins_pwc, Eigen::Quaterniond(correct_vins_Rwc));
        cam_vins_wheel.publish_by(pub_vins_wheel_visual, data.timestamp);
    }
}

bool readTxt(){
    ifstream pcdFile;
    pcdFile.open(pointcloud_path.c_str());
    if(!pcdFile.is_open()){
        ROS_ERROR("can not open pcd file %s", pointcloud_path);
        return false;
    }
    while(!pcdFile.eof()){
        string line;
        getline(pcdFile, line);
        // cout << line << endl;
        if(!line.empty()){
            stringstream ss;
            ss << line;

            pcl::PointXYZI point;

            ss >> point.intensity;

            Eigen::Vector3d pt_imu_world;
            ss >> pt_imu_world.x();
            ss >> pt_imu_world.y();
            ss >> pt_imu_world.z();

            point.x = pt_imu_world.x();
            point.y = pt_imu_world.y();
            point.z = pt_imu_world.z();
            lamp_cloud->push_back(point);

        }
    }
    return true;
}

// void readVINS(const string& path){
//     ifstream txtFile;
//     txtFile.open(path.c_str());
//     if(!txtFile.is_open()){
//         ROS_ERROR("can not open txt file %s", path);
//     }
//     while(!txtFile.eof()){
//         string line;
//         getline(txtFile, line);
//         if(!line.empty()){
//             stringstream ss;
//             ss << line;
//             VINS_POSE pose;
//             ss >> pose.timestamp;
//             ss >> pose.translation.x();
//             ss >> pose.translation.y();
//             ss >> pose.translation.z();
//             ss >> pose.quat.x();
//             ss >> pose.quat.y();
//             ss >> pose.quat.z();
//             ss >> pose.quat.w();

//             vins_poses.push_back(pose);
//         }
//     }
//     vector<VINS_POSE>(vins_poses).swap(vins_poses);
// }

inline void dump_state_to_log()  
{
    // V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    // Eigen::Quaterniond q = rot_ang(0)*rot_ang(1)*rot_ang(2);
    fprintf(fp_log1, "%lf ", data.timestamp);
    fprintf(fp_log2, "%lf ", data.timestamp);
    Eigen::Vector3d imu_pos1 = pwb, imu_pos2 = pwb2;
    Eigen::Matrix3d imu_rot1 = Rwb, imu_rot2 = Rwb2;
    Eigen::Quaterniond quat_imu_rot1 = Eigen::Quaterniond(imu_rot1);
    Eigen::Quaterniond quat_imu_rot2 = Eigen::Quaterniond(imu_rot2);

    fprintf(fp_log1, "%lf %lf %lf ", imu_pos1.x(), imu_pos1.y(), imu_pos1.z()); 
    fprintf(fp_log1, "%lf %lf %lf %lf", quat_imu_rot1.x(), quat_imu_rot1.y(), quat_imu_rot1.z(), quat_imu_rot1.w());
    fprintf(fp_log1, "\r\n"); 
    fflush(fp_log1);

    fprintf(fp_log2, "%lf %lf %lf ", imu_pos2.x(), imu_pos2.y(), imu_pos2.z()); 
    fprintf(fp_log2, "%lf %lf %lf %lf", quat_imu_rot2.x(), quat_imu_rot2.y(), quat_imu_rot2.z(), quat_imu_rot2.w());
    fprintf(fp_log2, "\r\n"); 
    fflush(fp_log2);
}

void readParameters(ros::NodeHandle& nh){
    nh.param<string>("load/pointcloud_path", pointcloud_path, "");
    nh.param<int>("load/cluster_num", cluster_number, 0);
    nh.param<string>("common/img_topic", img_topic, "/camera/color/image_raw/compressed");
    nh.param<string>("common/imu_topic", imu_topic, "/imu_data");
    nh.param<string>("common/odom_topic", odom_topic, "/odom");
    nh.param<string>("common/box_topic", box_topic, "/yolov7_bbox");
    nh.param<string>("common/gps_topic", gps_topic, "/gnss");
    nh.param<double>("common/timediff_img_to_box", timediff_img_to_box, 0.0);
    nh.param<double>("common/timediff_img_to_imu", timediff_img_to_imu, 0.0);
    nh.param<double>("common/timediff_img_to_odom", timediff_img_to_odom, 0.0);
    nh.param<double>("common/high_lamp", high_lamp, 1.0);
    nh.param<int>("common/window_size", WINDOW_SIZE, 13);
    nh.param<double>("common/keyframe_parallex", min_parallex, 10.0);
    nh.param<double>("common/z_th", z_th, 55);
    nh.param<double>("common/dist_th", dist_th, 75);
    nh.param<double>("common/reloc_z_th", reloc_z_th, 55);
    nh.param<double>("common/reloc_dist_th", reloc_dist_th, 75);
    nh.param<double>("common/update_z_th", update_z_th, 75);
    nh.param<double>("common/update_dist_th", update_dist_th, 100);
    nh.param<double>("common/reloc_update_z_th", reloc_update_z_th, 100);
    nh.param<double>("common/reloc_update_dist_th", reloc_update_dist_th, 125);
    nh.param<double>("common/alpha1", alpha1, 0.5);
    nh.param<double>("common/alpha2", alpha2, 0.35);
    nh.param<double>("common/beta2", beta2, 0.0);
    nh.param<int>("common/extend", extend, 20);
    nh.param<int>("common/grey_th", grey_th, 248);
    nh.param<int>("common/left_right_gap", left_right_gap, 20);
    nh.param<bool>("common/ablation_reloc", ablation_reloc, true);
    nh.param<bool>("common/ablation_matext", ablation_matext, true);
    if(!ablation_matext){
        z_th += 25;
        dist_th += 25;
        reloc_z_th += 25;
        reloc_dist_th += 25;
        update_z_th += 25;
        update_dist_th += 25;
        reloc_update_z_th += 25;
        reloc_update_dist_th += 25;
    }

    //重定位相关参数
    nh.param<double>("rematch/delta_Rbb_th", delta_Rbb_th, 0.03);
    nh.param<double>("rematch/delta_tbb_th", delta_pbb_th, 0.07);
    nh.param<double>("rematch/delta_Rbb_th2", delta_Rbb_th2, 0.015);
    nh.param<double>("rematch/delta_tbb_th2", delta_pbb_th2, 0.03);

    //相机内参及外参
    nh.param<double>("camera/cam_fx", cam_fx, 910.777);
    nh.param<double>("camera/cam_fy", cam_fy, 910.656);
    nh.param<double>("camera/cam_cx", cam_cx, 639.846);
    nh.param<double>("camera/cam_cy", cam_cy, 355.401);
    nh.param<int>("camera/res_x", res_x, 1280);
    nh.param<int>("camera/res_y", res_y, 720);

    //初始化参数
    nh.param<bool>("initialization/need_init", need_init, true);
    nh.param<bool>("gps/use_gps", use_gps, true);
    nh.param<double>("initialization/init_x", init_x, -10000);
    nh.param<double>("initialization/init_y", init_y, -10000);
    nh.param<double>("initialization/init_z", init_z, -10000);
    nh.param<double>("initialization/search_dist_scope", search_dist_scope, 45);
    nh.param<double>("initialization/search_z_scope", search_z_scope, 25);
    nh.param<bool>("initialization/init_extend", init_extend, false);
    if(use_gps){
        nh.param<double>("gps/init_alpha", init_alpha, -10000);
        nh.param<double>("gps/init_beta", init_beta, -10000);
        nh.param<double>("gps/init_gamma", init_gamma, -10000);
        nh.param<double>("gps/gps_cov_threshold", gps_cov_threshold, 3.0);
        nh.param<double>("gps/gps_init_x", gps_init_x, 0.0);
        nh.param<double>("gps/gps_init_y", gps_init_y, 0.0);
        nh.param<double>("gps/gps_init_z", gps_init_z, 0.0);
        nh.param<bool>("gps/use_elevation", use_elevation, false);
        nh.param<double>("gps/yaw_offset", yaw_off, 0.0);
    }
    else{
        nh.param<double>("initialization/init_alpha", init_alpha, -10000);
        nh.param<double>("initialization/init_beta", init_beta, -10000);
        nh.param<double>("initialization/init_gamma", init_gamma, -10000);
    }
    nh.param<int>("initialization/num_particles", num_particles, 8000);

    nh.param<vector<double>>("imu/gyroscope_std", imu_params.std_gy, vector<double>());
    nh.param<vector<double>>("imu/accelerometer_std", imu_params.std_acc, vector<double>());
    nh.param<vector<double>>("imu/gyroscope_bias_std", imu_params.std_bg, vector<double>());
    nh.param<vector<double>>("imu/accelerometer_bias_std", imu_params.std_ba, vector<double>());
    nh.param<vector<double>>("imu/Rci", imu_params.camera_imu_rot, vector<double>());
    nh.param<vector<double>>("imu/pci", imu_params.camera_imu_pos, vector<double>());
    nh.param<vector<double>>("imu/Roi", imu_params.odom_imu_rot, vector<double>());
    nh.param<vector<double>>("imu/poi", imu_params.odom_imu_pos, vector<double>());
    nh.param<vector<double>>("imu/Rco", imu_params.camera_odom_rot, vector<double>());
    nh.param<vector<double>>("imu/pco", imu_params.camera_odom_pos, vector<double>());
    nh.param<bool>("imu/normalized", imu_params.normalized, false);
    nh.param<vector<double>>("odometer/cov_vel", odom_params.cov_vel, vector<double>());

    if(cluster_number == 0)
        ROS_ERROR("Not identify the cluster numbers!");

    lamp_world_pos.resize(cluster_number);
    lamp_world_points.resize(cluster_number);
    lamp_world_box.resize(cluster_number);
    // if(need_init)
    //     lamp_world_2dpos.resize(cluster_number);
    // lamp_world_plane.resize(cluster_number);
    // alphas.resize(cluster_number);
    // betas.resize(cluster_number);
    if (pointcloud_path.find(".txt"))
        readTxt();
    else
        ROS_ERROR("only support txt file!");
    for (const auto& point: lamp_cloud->points){
        // cout << point.intensity;
        lamp_world_points[int(point.intensity)].push_back(Eigen::Vector3d(point.x, point.y, point.z));
    }

    // string vins_path = "/home/gtx/vins_wheel/output/scene3_seaside2/Vins_odometer_timestamp.txt";
    // readVINS(vins_path);

    cv::RNG rng(time(0));
    for (int i = 0; i < lamp_world_points.size(); i++){
        // cv::Scalar scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        cv::Scalar scalar(255, 255, 255);
        for (int j = 0; j < lamp_world_points[i].size(); j++){
            pcl::PointXYZRGB rgb_point;
            rgb_point.r = scalar[0];
            rgb_point.g = scalar[1];
            rgb_point.b = scalar[2];
            rgb_point.x = lamp_world_points[i][j].x();
            rgb_point.y = lamp_world_points[i][j].y();
            rgb_point.z = lamp_world_points[i][j].z();
            lamp_rgbcloud->points.push_back(rgb_point);
        }
    }

    for (int i = 0; i< cluster_number; i++){
        Eigen::Vector3d sum_points = Eigen::Vector3d::Zero();
        //计算中心点以及包围盒(采用轴对齐包围盒)
        double min_x = 10000.0, min_y = 10000.0, min_z = 10000.0, max_x = -10000.0, max_y = -10000.0, max_z = -10000.0;
        for (int j = 0; j < lamp_world_points[i].size(); j++){
            sum_points += lamp_world_points[i][j];
            min_x = min(lamp_world_points[i][j].x(), min_x);
            min_y = min(lamp_world_points[i][j].y(), min_y);
            min_z = min(lamp_world_points[i][j].z(), min_z);
            max_x = max(lamp_world_points[i][j].x(), max_x);
            max_y = max(lamp_world_points[i][j].y(), max_y);
            max_z = max(lamp_world_points[i][j].z(), max_z);
        }
        lamp_world_pos[i] = sum_points / lamp_world_points[i].size();
        // if(need_init)
        //     lamp_world_2dpos[i] = lamp_world_pos[i].head<2>();
        lamp_world_box[i] << min_x, min_y, min_z, max_x, max_y, max_z;
    }

    deq_Rcw.resize(WINDOW_SIZE + 1);
    deq_pcw.resize(WINDOW_SIZE + 1);
    deq_P.resize(WINDOW_SIZE + 1);
    deq_deltaR.resize(WINDOW_SIZE + 1);
    deq_deltap.resize(WINDOW_SIZE + 1);
    deq_matches.resize(WINDOW_SIZE + 1);
    deq_boxes.resize(WINDOW_SIZE + 1);
    deq_id.resize(WINDOW_SIZE + 1);
    // deq_features.resize(WINDOW_SIZE + 1);

    K << cam_fx, 0,      cam_cx,
         0,      cam_fy, cam_cy,
         0,      0,      1;
    K_inv = K.inverse();

    // cov_lamp << 0.005, 0,   0,
    //             0,   0.005, 0,
    //             0,   0,   0.001;
    cov_lamp = Eigen::Matrix3d::Zero();
}

bool first_process = true;
bool sync_packages(Measures& data){
    if (img_buffer.empty() || imu_buffer.empty() || odom_buffer.empty() || box_buffer.empty()){
        if (first_process){
            if (odom_buffer.empty() || imu_buffer.empty()){
                img_buffer.clear();
                box_buffer.clear();
                time_buffer.clear();
            }
        }
        return false;
    }

    data.timestamp = time_buffer.front();
    data.img = img_buffer.front();
    data.box = box_buffer.front();
    data.deep_learned_boxes = data.box->bounding_boxes.size();

    double imu_time = imu_buffer.front()->header.stamp.toSec();
    while((!imu_buffer.empty()) && (imu_time < time_buffer.front())){
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if(imu_time > time_buffer.front()) break;
        data.imu_deq.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    double odom_time = odom_buffer.front()->header.stamp.toSec();
    while((!odom_buffer.empty()) && (odom_time < time_buffer.front())){
        odom_time = odom_buffer.front()->header.stamp.toSec();
        if(odom_time > time_buffer.front()) break;
        data.odom_deq.push_back(odom_buffer.front());
        odom_buffer.pop_front();
    }

    if (first_process){
        data.last_timestamp = -1.0;
        first_process = false;
    }

    time_buffer.pop_front();
    img_buffer.pop_front();
    box_buffer.pop_front();

    return true;
}

void publish_samples_pose(const ros::Publisher& pub_particles, const Sampler& sample){
    geometry_msgs::PoseArray cloud_msg;
    cloud_msg.header.stamp = ros::Time().fromSec(data.timestamp);
    cloud_msg.header.frame_id = "camera_init";
    cloud_msg.poses.resize(sample.samples.size());

    for(int i = 0; i < sample.samples.size(); ++i){
        tf2::Quaternion q;
        q.setRPY(sample.samples[i].rot(2), sample.samples[i].rot(1), sample.samples[i].rot(0));
        tf2::convert(q, cloud_msg.poses[i].orientation);

        cloud_msg.poses[i].position.x = sample.samples[i].pos.x();
        cloud_msg.poses[i].position.y = sample.samples[i].pos.y();
        cloud_msg.poses[i].position.z = sample.samples[i].pos.z();
    }

    pub_particles.publish(cloud_msg);
}

void publish_selected_samples(const ros::Publisher& pub_particles, const Sampler& sample){
    geometry_msgs::PoseArray cloud_msg;
    cloud_msg.header.stamp = ros::Time().fromSec(data.timestamp);
    cloud_msg.header.frame_id = "camera_init";
    cloud_msg.poses.resize(sample.best_samples.size());

    for(int i = 0; i < sample.best_samples.size(); ++i){
        tf2::Quaternion q;
        q.setRPY(sample.best_samples[i].rot(2), sample.best_samples[i].rot(1), sample.best_samples[i].rot(0));
        tf2::convert(q, cloud_msg.poses[i].orientation);

        cloud_msg.poses[i].position.x = sample.best_samples[i].pos.x();
        cloud_msg.poses[i].position.y = sample.best_samples[i].pos.y();
        cloud_msg.poses[i].position.z = sample.best_samples[i].pos.z();
    }

    pub_particles.publish(cloud_msg);
}

void publish_samples_pose(const ros::Publisher& pub_particles, const ros::Publisher& pub_optimal_particle, const Eigen::Matrix3d& R1, const Eigen::Vector3d& p1, const Eigen::Matrix3d& R2, const Eigen::Vector3d& p2){
    geometry_msgs::PoseArray cloud_msg1, cloud_msg2;
    cloud_msg1.header.stamp = ros::Time().fromSec(data.timestamp);
    cloud_msg1.header.frame_id = "camera_init";
    cloud_msg1.poses.resize(1);

    cloud_msg2.header.stamp = ros::Time().fromSec(data.timestamp);
    cloud_msg2.header.frame_id = "camera_init";
    cloud_msg2.poses.resize(1);

    Eigen::AngleAxisd axis1(R1);
    Eigen::Quaterniond q1(axis1);
    cloud_msg1.poses[0].orientation.w = q1.w();
    cloud_msg1.poses[0].orientation.x = q1.x();
    cloud_msg1.poses[0].orientation.y = q1.y();
    cloud_msg1.poses[0].orientation.z = q1.z();

    cloud_msg1.poses[0].position.x = p1.x();
    cloud_msg1.poses[0].position.y = p1.y();
    cloud_msg1.poses[0].position.z = p1.z();

    Eigen::AngleAxisd axis2(R2);
    Eigen::Quaterniond q2(axis2);
    cloud_msg2.poses[0].orientation.w = q2.w();
    cloud_msg2.poses[0].orientation.x = q2.x();
    cloud_msg2.poses[0].orientation.y = q2.y();
    cloud_msg2.poses[0].orientation.z = q2.z();

    cloud_msg2.poses[0].position.x = p2.x();
    cloud_msg2.poses[0].position.y = p2.y();
    cloud_msg2.poses[0].position.z = p2.z();    

    pub_particles.publish(cloud_msg1);
    pub_optimal_particle.publish(cloud_msg2);
}

void publish_initial_path(const ros::Publisher& pub_init_path, Eigen::Matrix3d Rwb, Eigen::Vector3d pwb){
    geometry_msgs::PoseStamped pose;
    pose.header.stamp = ros::Time().fromSec(data.timestamp);
    pose.header.frame_id = "camera_init";

    pose.pose.position.x = pwb.x();
    pose.pose.position.y = pwb.y();
    pose.pose.position.z = pwb.z();

    Eigen::AngleAxisd axis(Rwb);
    Eigen::Quaterniond q(axis);
    pose.pose.orientation.w = q.w();
    pose.pose.orientation.x = q.x();
    pose.pose.orientation.y = q.y();
    pose.pose.orientation.z = q.z();

    static int init_pose_num = 0;
    init_pose_num++;
    if(init_pose_num % 3 == 0){
        global_init_path.poses.push_back(pose);
        pub_init_path.publish(global_init_path);
    }
}

void publish_odom_path(const ros::Publisher& pub_odom_path){
    geometry_msgs::PoseStamped pose;
    pose.header.stamp = ros::Time().fromSec(data.timestamp);
    pose.header.frame_id = "camera_init";

    pose.pose.position.x = pwb2.x();
    pose.pose.position.y = pwb2.y();
    pose.pose.position.z = pwb2.z();

    Eigen::AngleAxisd axis(Rwb2);
    Eigen::Quaterniond q(axis);
    pose.pose.orientation.w = q.w();
    pose.pose.orientation.x = q.x();
    pose.pose.orientation.y = q.y();
    pose.pose.orientation.z = q.z();

    static int odom_pose_num = 0;
    odom_pose_num++;
    if(odom_pose_num % 5 == 0){
        global_odom_path.poses.push_back(pose);
        pub_odom_path.publish(global_odom_path);
    }
}

void publish_optimized_path(const ros::Publisher& pub_optimized_path){
    geometry_msgs::PoseStamped pose;
    pose.header.stamp = ros::Time().fromSec(data.timestamp);
    pose.header.frame_id = "camera_init";

    pose.pose.position.x = pwb.x();
    pose.pose.position.y = pwb.y();
    pose.pose.position.z = pwb.z();

    Eigen::AngleAxisd axis(Rwb);
    Eigen::Quaterniond q(axis);
    pose.pose.orientation.w = q.w();
    pose.pose.orientation.x = q.x();
    pose.pose.orientation.y = q.y();
    pose.pose.orientation.z = q.z();

    static int optimized_pose_num = 0;
    optimized_pose_num++;
    if(optimized_pose_num % 5 == 0){
        global_optimized_path.poses.push_back(pose);
        pub_optimized_path.publish(global_optimized_path);
    }
}

void publish_image(const ros::Publisher& pub_image, const cv::Mat& image){
    std_msgs::Header head;
    head.stamp = ros::Time().fromSec(data.timestamp);
    head.frame_id = "camera_init";
    sensor_msgs::ImagePtr image_msg = cv_bridge::CvImage(head, "bgr8", image).toImageMsg();
    pub_image.publish(image_msg);
}

void publish_pointcloud(const ros::Publisher& pub_cloud, bool isRGB = true){
    sensor_msgs::PointCloud2 cloud_msg;
    if (isRGB)
        pcl::toROSMsg(*cur_lamp_cloud, cloud_msg);
    else
        pcl::toROSMsg(*lamp_rgbcloud, cloud_msg);
    cloud_msg.header.stamp = ros::Time().fromSec(data.timestamp);
    cloud_msg.header.frame_id = "camera_init";
    pub_cloud.publish(cloud_msg);
}

void Preprocess(Measures& data, bool pf_suc = true){

    cv::Mat grey_img(data.img.rows, data.img.cols, CV_8UC1);
    cv::cvtColor(data.img, grey_img, cv::COLOR_BGR2GRAY);
    cv::Mat bin_img;
    if(need_init && !pf_suc && init_extend){
        cv::threshold(grey_img, bin_img, 252, 255, cv::THRESH_BINARY);
    }
    else{
        cv::threshold(grey_img, bin_img, 250, 255, cv::THRESH_BINARY);
    }

    vector<cv::Rect> box1, box2;
    vector<vector<cv::Point>> contours;
    cv::findContours(bin_img, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    for(int j = 0; j < contours.size(); j++){
        cv::Rect rect = cv::boundingRect(contours[j]);
        // cout << rect.x << " " << rect.y << " " << rect.width << " " << rect.height << endl;

        int area = rect.width * rect.height;
        if(need_init && !pf_suc && init_extend){
            if(area < 40 || rect.width < 3 || rect.height < 3 || rect.height / rect.width > 3)
                continue;
        }
        else{
            if(area < 100 || rect.width < 10 || rect.height / rect.width > 5)
                continue;
        }

        if(rect.x != 0 && rect.y != 0 && rect.width != bin_img.cols && rect.height != bin_img.rows){
            box2.push_back(rect);
        }
    }

    for(int i = 0; i < data.box->bounding_boxes.size(); i++){
        cv::Point leftup(data.box->bounding_boxes[i].xmin, data.box->bounding_boxes[i].ymin);
        cv::Point rightdown(data.box->bounding_boxes[i].xmax, data.box->bounding_boxes[i].ymax);
        cv::Rect rect(leftup, rightdown);
        box1.push_back(rect);
    }

    vector<cv::Rect> box12;
    vector<bool> box2uniond(box2.size(), false);
    for(int i = 0; i < box1.size(); i++){
        cv::Rect rect1 = box1[i];
        bool has_union = false;

        cv::Rect rect3;
        for (int j = 0; j < box2.size(); j++){
            cv::Rect rect2 = box2[j];
            int bin_min_u = 10000, bin_min_v = 10000, bin_max_u = -10000, bin_max_v = -10000;
            if((rect1 & rect2).area()){
                has_union = true;
                box2uniond[j] = true;
                cv::Rect union_rect = rect1 | rect2;
                for (int u = union_rect.x; u < union_rect.x + union_rect.width; u++){
                    for (int v = union_rect.y; v < union_rect.y + union_rect.height; v++){
                        float intensity;
                        intensity = grey_img.ptr<uchar>(v)[u];
                
                        if (intensity > 252){
                            int light_nei = 0;
                            for(int du = -3; du < 4; du++)
                                for(int dv = -3; dv < 4; dv++){
                                    float light_intensity;
                                    light_intensity = grey_img.ptr<uchar>(v + dv)[u + du];
                                    if (light_intensity > 249) light_nei++;
                                }
                            if(light_nei < 15) continue;
                            bin_min_u = min(bin_min_u, u);
                            bin_min_v = min(bin_min_v, v);
                            bin_max_u = max(bin_max_u, u);
                            bin_max_v = max(bin_max_v, v);
                        }
                    }
                }
            }
            else
                continue;

            if(!need_init || pf_suc || !init_extend){
                int light_nei_v = 0;
                for(int v = bin_min_v; v < bin_max_v; ++v){
                    float light_intensity = grey_img.ptr<uchar>(v)[(bin_min_u + bin_max_u) / 2];
                    if(light_intensity > 245) light_nei_v++;
                }
                if(light_nei_v / float(bin_max_v - bin_min_v + 1) < 0.8)
                    continue;
            }

            if(bin_min_u < res_x && bin_min_v < res_y && bin_max_u >=0 && bin_max_v >= 0){
                rect3 = cv::Rect(cv::Point(bin_min_u, bin_min_v), cv::Point(bin_max_u, bin_max_v));
                box12.push_back(rect3);
            }
        }

        if(!has_union){
            int bin_min_u = 10000, bin_min_v = 10000, bin_max_u = -10000, bin_max_v = -10000;
            for (int u = rect1.x; u < rect1.x + rect1.width; u++){
                for (int v = rect1.y; v < rect1.y + rect1.height; v++){
                    float intensity;
                    intensity = grey_img.ptr<uchar>(v)[u];
            
                    if (intensity > 252){
                        int light_nei = 0;
                        for(int du = -3; du < 4; du++)
                            for(int dv = -3; dv < 4; dv++){
                                float light_intensity;
                                light_intensity = grey_img.ptr<uchar>(v + dv)[u + du];
                                if (light_intensity > 249) light_nei++;
                            }
                        if(light_nei < 15) continue;
                        bin_min_u = min(bin_min_u, u);
                        bin_min_v = min(bin_min_v, v);
                        bin_max_u = max(bin_max_u, u);
                        bin_max_v = max(bin_max_v, v);
                    }
                }
            }

            if(!need_init || pf_suc || !init_extend){
                int light_nei_v = 0;
                for(int v = bin_min_v; v < bin_max_v; ++v){
                    float light_intensity = grey_img.ptr<uchar>(v)[(bin_min_u + bin_max_u) / 2];
                    if(light_intensity > 245) light_nei_v++;
                }
                if(light_nei_v / float(bin_max_v - bin_min_v + 1) < 0.8)
                    continue;
            }
                
            if(bin_min_u < res_x && bin_min_v < res_y && bin_max_u >=0 && bin_max_v >= 0){
                rect3 = cv::Rect(cv::Point(bin_min_u, bin_min_v), cv::Point(bin_max_u, bin_max_v));
                box12.push_back(rect3);
            }
        }
    }
    if(need_init && !pf_suc && init_extend){
        int bin_min_u = 10000, bin_min_v = 10000, bin_max_u = -10000, bin_max_v = -10000;
        for(int i = 0; i < box2.size(); ++i){
            if(!box2uniond[i]){
                box12.push_back(box2[i]);
                // for (int u = box2[i].x; u < box2[i].x + box2[i].width; u++){
                //     for (int v = box2[i].y; v < box2[i].y + box2[i].height; v++){
                //         float intensity;
                //         intensity = grey_img.ptr<uchar>(v)[u];
                
                //         if (intensity > 245){
                //             int light_nei = 0;
                //             for(int du = -3; du < 4; du++)
                //                 for(int dv = -3; dv < 4; dv++){
                //                     float light_intensity;
                //                     light_intensity = grey_img.ptr<uchar>(v + dv)[u + du];
                //                     if (light_intensity > 240) light_nei++;
                //                 }
                //             if(light_nei < 15) continue;
                //             bin_min_u = min(bin_min_u, u);
                //             bin_min_v = min(bin_min_v, v);
                //             bin_max_u = max(bin_max_u, u);
                //             bin_max_v = max(bin_max_v, v);
                //         }
                //     }
                // }
                // if(bin_min_u < res_x && bin_min_v < res_y && bin_max_u >=0 && bin_max_v >= 0){
                //     cv::Rect rect3 = cv::Rect(cv::Point(bin_min_u, bin_min_v), cv::Point(bin_max_u, bin_max_v));
                //     box12.push_back(rect3);
                // }
            }
        }
    }

    auto iter_i = box12.begin();
    while(iter_i != box12.end()){
        auto iter_j = iter_i + 1;
        while(iter_j != box12.end()){
            if((*iter_i & *iter_j).area() > 0.25 * (*iter_i).area() || (*iter_i & *iter_j).area() > 0.25 * (*iter_j).area()){
                iter_j = box12.erase(iter_j);
                continue;
            }
            ++iter_j;
        }
        ++iter_i;
    }

    data.box->bounding_boxes.clear();
    for(int i = 0; i < box12.size(); i++){
        if(box12[i].x <= 4 || box12[i].y <= 4 || box12[i].x + box12[i].width >= res_x - 4 || box12[i].y + box12[i].height >= res_y - 4)
            continue;
        else{
            float avg_intensity = 0.0;
            int area = box12[i].height * box12[i].width;

            for (int u = box12[i].x; u < box12[i].x + box12[i].width; u++){
                for (int v = box12[i].y; v < box12[i].y + box12[i].height; v++){
                    float intensity = grey_img.ptr<uchar>(v)[u];
                    avg_intensity += intensity;
                }
            }
            if(avg_intensity / area < 175.0 || area < 5)
                continue;
            // if(area < 40)
            //     continue;

            map_relocalization::BoundingBox box;
            box.xmin = box12[i].x, box.xmax = box12[i].x + box12[i].width, box.ymin = box12[i].y, box.ymax = box12[i].y +box12[i].height;
            data.box->bounding_boxes.push_back(box);
        }
    }
    data.deep_learned_boxes = data.box->bounding_boxes.size();
    // for(int i = 0; i < data.box->bounding_boxes.size(); ++i){
    //     cout << "box " << i << endl;
    //     cout << data.box->bounding_boxes[i].xmin << " " << data.box->bounding_boxes[i].xmax << " " << data.box->bounding_boxes[i].ymin << " " << data.box->bounding_boxes[i].ymax << endl;
    // }

    // //去除误检测及二值化
    // auto iter = data.box->bounding_boxes.begin();
    // //二值化缩小面积
    // while(iter != data.box->bounding_boxes.end()){
    //     int bin_min_u = 10000, bin_min_v = 10000, bin_max_u = -10000, bin_max_v = -10000;
    //     for (int u = iter->xmin; u < iter->xmax; u++){
    //         for (int v = iter->ymin; v < iter->ymax; v++){
    //             float intensity;
    //             if (data.img.channels() > 1){
    //                 auto data_ptr = &data.img.ptr<uchar>(v)[u * data.img.channels()];
    //                 intensity = data_ptr[0] * 0.3 + data_ptr[1] * 0.59 + data_ptr[2] * 0.11;
    //             }
    //             else{
    //                 intensity = data.img.ptr<uchar>(v)[u];
    //             }
    //             if (intensity > 252){
    //                 int light_nei = 0;
    //                 for(int du = -3; du < 4; du++)
    //                     for(int dv = -3; dv < 4; dv++){
    //                         float light_intensity;
    //                         if (data.img.channels() > 1){
    //                             auto data_ptr = &data.img.ptr<uchar>(v)[u * data.img.channels()];
    //                             light_intensity = data_ptr[0] * 0.3 + data_ptr[1] * 0.59 + data_ptr[2] * 0.11;
    //                         }
    //                         else{
    //                             light_intensity = data.img.ptr<uchar>(v)[u];
    //                         }
    //                         if (light_intensity > 249) light_nei++;
    //                     }
    //                 if(light_nei < 15) continue;
    //                 bin_min_u = min(bin_min_u, u);
    //                 bin_min_v = min(bin_min_v, v);
    //                 bin_max_u = max(bin_max_u, u);
    //                 bin_max_v = max(bin_max_v, v);
    //             }
    //         }
    //     }
    //     int area = (bin_max_u - bin_min_u) * (bin_max_v - bin_min_v);
    //     //第一重检测判断面积是否太小，面积太小说明不是路灯
    //     if(area < 75)
    //         iter = data.box->bounding_boxes.erase(iter);
    //     else{
    //         //第二重检测判断光强是否足够，光强弱说明不是路灯
    //         float avg_intensity = 0.0;
    //         for (int u = bin_min_u; u < bin_max_u; u++){
    //             for (int v = bin_min_v; v < bin_max_v; v++){
    //                 float intensity;
    //                 if (data.img.channels() > 1){
    //                     auto data_ptr = &data.img.ptr<uchar>(v)[u * data.img.channels()];
    //                     intensity = data_ptr[0] * 0.3 + data_ptr[1] * 0.59 + data_ptr[2] * 0.11;
    //                     avg_intensity += intensity;
    //                 }
    //                 else{
    //                     intensity = data.img.ptr<uchar>(v)[u];
    //                     avg_intensity += intensity;
    //                 }
    //             }
    //         }
    //         if(avg_intensity / area < 175.0)
    //             iter = data.box->bounding_boxes.erase(iter);
    //         else{
    //             iter->xmin = bin_min_u;
    //             iter->xmax = bin_max_u;
    //             iter->ymin = bin_min_v;
    //             iter->ymax = bin_max_v;
    //             ++iter;
    //         }
    //     }
    // }
}

inline double ang_distance(const Eigen::Vector2d& box_center, const Eigen::Vector3d& lamp_center, const Eigen::Matrix3d& cov_lamp_center, const Eigen::Matrix2d& sigma2){
    Eigen::Vector3d liftup_center;
    liftup_center << (box_center.x() - cam_cx) / cam_fx, (box_center.y() - cam_cy) / cam_fy, 1;

    double inv_norm_liftup_center = 1.0 / liftup_center.norm();
    double inv_norm_lamp_center = 1.0 / lamp_center.norm();
    Eigen::Vector3d normd_liftup_center = liftup_center.normalized();
    Eigen::Vector3d normd_lamp_center = lamp_center.normalized();
    
    double cos_theta = normd_lamp_center.transpose() * normd_liftup_center;
    double err = 1 - cos_theta;
    // cout << err << endl;

    Eigen::Matrix3d cov_liftup_center = Eigen::Matrix3d::Zero();
    cov_liftup_center(0, 0) = sigma2(0, 0) / (cam_fx * cam_fx), cov_liftup_center(1, 1) = sigma2(1, 1) / (cam_fy * cam_fy);

    Eigen::Vector3d jaco_cos_liftup_center, jaco_cos_lamp_center;
    jaco_cos_liftup_center = inv_norm_liftup_center * (normd_lamp_center - normd_lamp_center.transpose() * normd_liftup_center * normd_liftup_center);
    jaco_cos_lamp_center = inv_norm_lamp_center * (normd_liftup_center - normd_liftup_center.transpose() * normd_lamp_center * normd_lamp_center);
    
    double cov_cos_theta = (jaco_cos_liftup_center.transpose() * cov_liftup_center * jaco_cos_liftup_center)(0) + 
                           (jaco_cos_lamp_center.transpose() * cov_lamp_center * jaco_cos_lamp_center)(0);
    // cout << "jaco_cos_lamp_center" << (jaco_cos_lamp_center.transpose() * cov_lamp_center * jaco_cos_lamp_center) << " jaco_cos_liftup_center" << (jaco_cos_liftup_center.transpose() * cov_liftup_center * jaco_cos_liftup_center) << endl;
    
    return exp(-0.5 * err * err / cov_cos_theta);
}

inline double md_distance(const Eigen::Vector2d& box_center, const Eigen::Vector2d& proj_center, const Eigen::Matrix2d& cov){
    Eigen::Vector2d err(box_center - proj_center);
    // cout << "err " << err.norm() << endl;
    Eigen::MatrixXd cov_ld = cov.llt().matrixL();
    // cout << "cov_ld: " << endl << cov_ld << endl;
    Eigen::VectorXd vld = cov_ld.inverse() * err;
    // cout << "vld " << endl << vld << endl;
    return exp(- 0.5 * vld.transpose() * vld);
}

Match ProbMatch(Measures& data, vec_vec2d& whole_box_centers, vec_vec4d& whole_box_corners){
    if(data.box->bounding_boxes.size() == 0){
        ROS_WARN("no boxes detected!");
        Match match_result;
        match_result.box_num = 0;
        match_result.lamp_num = 0;
        return match_result;
    }

    double sigma2_bd = 1.2;
    // 根据协方差矩阵确定搜索范围
    for (int i = 0; i < lamp_world_pos.size(); i++){
        if(isnan(lamp_world_pos[i].norm())) continue;
        if(lamp_world_points[i].size() <= 0) continue;
        Eigen::Vector3d Pc;
        Pc = Rcw * lamp_world_pos[i] + pcw;
        Eigen::Vector2d pt;
        double inv_z = 1.0 / Pc.z();
        pt << cam_fx * Pc.x() * inv_z + cam_cx, cam_fy * Pc.y() * inv_z + cam_cy;
        if ((sqrt(Pc.x() * Pc.x() + Pc.z() * Pc.z()) < dist_th) && Pc.z() < z_th && Pc.z() > -0.5 && pt.x() < res_x && pt.x() >= 0 && pt.y() < res_y && pt.y() >=0){ //55, 45
            // cout << "Pc: " << Pc << endl;
            bool find_rep = false;
            for (int j = 0; j < lamp_cur_cam_pos.size(); j++){
                Eigen::Vector3d l1 = lamp_cur_cam_pos[j].normalized();
                Eigen::Vector3d l2 = Pc.normalized();
                // cout << "cos_theta: " << l1.transpose() * l2 << endl;
                if((l1.transpose() * l2)(0) > 0.995){
                    double n1 = lamp_cur_cam_pos[j].norm();
                    double n2 = Pc.norm();
                    find_rep = true;
                    if(n1 > n2){
                        lamp_cur_cam_pos[j] = Pc;
                        lamp_cur_id[j] = i;
                        lamp_cur_world_pos[j] = lamp_world_pos[i];
                    }
                    break;
                }
            }
            if(find_rep) continue;
            lamp_cur_cam_pos.push_back(Pc);
            lamp_cur_id.push_back(i);
            lamp_cur_world_pos.push_back(lamp_world_pos[i]);
            // lamp_cur_world_box.push_back(lamp_world_box[i]);

            // Eigen::Matrix4d Hcw = Eigen::Matrix4d::Identity();
            // Hcw.block<3,3>(0, 0) = Rcw;
            // Hcw.block<3,1>(0, 3) = pcw;
            // lamp_cur_cam_plane.push_back(Hcw.inverse().transpose() * lamp_world_plane[i]);
            // lamp_cur_world_plane.push_back(lamp_world_plane[i]);

            // vec_vec3d single_box_endpoints;
            // for (size_t x = 0; x <= 1; x++){
            //     for (size_t y = 0; y <= 1; y++){
            //         for (size_t z = 0; z <= 1; z++){
            //             Eigen::Vector3d P_box_end(lamp_world_box[i](x * 3), lamp_world_box[i](y * 3 + 1), lamp_world_box[i](z * 3 + 2));
            //             P_box_end.noalias() = Rcw * P_box_end + pcw;
            //             single_box_endpoints.push_back(P_box_end);
            //         }
            //     }
            // }
            // lamp_cur_cam_box.push_back(single_box_endpoints);
        }
    }
    if(lamp_cur_cam_pos.size() == 0){
        ROS_WARN("no lamps around!");
        Match match_result;
        match_result.box_num = 0;
        match_result.lamp_num = 0;
        return match_result;
    }

    int box_nums = data.box->bounding_boxes.size();
    whole_box_centers.resize(box_nums);
    whole_box_corners.resize(box_nums);

    Eigen::MatrixXd md_matrix = Eigen::MatrixXd::Zero(box_nums, lamp_cur_cam_pos.size() + 1);
    Eigen::MatrixXd ad_matrix = Eigen::MatrixXd::Zero(box_nums, lamp_cur_cam_pos.size() + 1);

    vec_mat3d whole_liftup_HPH_R;
    vec_mat2d whole_HPH_R;
    // vec_mat4d whole_plane_HPH_R;
    vec_vec2d whole_pts;
    vec_vec4d whole_proj_box;
    for (int i = 0; i < lamp_cur_cam_pos.size(); i++){
        //计算中心点投影点
        Eigen::Vector2d pt;
        double inv_z = 1.0 / lamp_cur_cam_pos[i].z();
        pt << cam_fx * lamp_cur_cam_pos[i].x() * inv_z + cam_cx, cam_fy * lamp_cur_cam_pos[i].y() * inv_z + cam_cy;

        // cout << "id: " << lamp_cur_id[i] << " pt: " << pt.transpose() << endl;

        whole_pts.push_back(pt);

        // cout << "Pc: " << lamp_cur_cam_pos[i].transpose() << endl;

        // 计算雅可比矩阵用于传递协方差矩阵
        Eigen::Matrix<double, 2, 3> J_pt_Pc;
        J_pt_Pc << cam_fx * inv_z, 0,              - cam_fx * pt.x() * inv_z * inv_z,
                    0,             cam_fy * inv_z, - cam_fy * pt.y() * inv_z * inv_z;
                
        // cout << "J_pt_Pc: " << J_pt_Pc << endl;

        Eigen::Matrix<double, 3, 6> J_Pc_pose;
        // J_Pc_pose.block<3, 3>(0, 0) = - vector_skew(lamp_cur_cam_pos[i] - pcw);
        J_Pc_pose.block<3, 3>(0, 0) = vector_skew(lamp_cur_cam_pos[i]);
        J_Pc_pose.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();

        // cout << "J_Pc_pose: " << J_Pc_pose << endl;

        Eigen::Matrix<double, 2, 6> J_pt_pose;
        J_pt_pose = J_pt_Pc * J_Pc_pose;

        Eigen::Matrix<double, 2, 3> J_pt_Pw;
        J_pt_Pw = J_pt_Pc * Rcw;

        // cout << "J_pt_pose: " << J_pt_pose << endl;

        Eigen::Matrix2d HPH_R = J_pt_pose * cov * J_pt_pose.transpose() + J_pt_Pw * cov_lamp * J_pt_Pw.transpose();
        whole_HPH_R.push_back(HPH_R);

        // cout << "whole_HPH_R: " << HPH_R << endl;

        Eigen::Matrix3d liftup_HPH_R = J_Pc_pose * cov * J_Pc_pose.transpose() + Rcw * cov_lamp * Rcw.transpose();
        whole_liftup_HPH_R.push_back(liftup_HPH_R);
    }

    // 统计检测框中心点及边界点
    for (int i = 0; i < box_nums; i++){
        double center_x = (data.box->bounding_boxes[i].xmax + data.box->bounding_boxes[i].xmin) / 2;
        double center_y = (data.box->bounding_boxes[i].ymax + data.box->bounding_boxes[i].ymin) / 2;
        whole_box_centers[i] = Eigen::Vector2d(center_x, center_y);
    }

    //计算所有概率值
    for (int i = 0; i < box_nums; i++){
        double sigma2_x = data.box->bounding_boxes[i].xmax - data.box->bounding_boxes[i].xmin;
        double sigma2_y = data.box->bounding_boxes[i].ymax - data.box->bounding_boxes[i].ymin;

        if((reloc && global_id - reloc_id <= 150 && ablation_reloc) || !ablation_matext){
            sigma2_x = 144;
            sigma2_y = 144;
        }
        else{
            sigma2_x = 6.25;
            sigma2_y = 6.25;
        }

        Eigen::Matrix2d sigma2 = Eigen::Matrix2d::Identity();
        sigma2(0, 0) = sigma2_x, sigma2(1, 1) = sigma2_y;

        for (int j = 0; j < lamp_cur_cam_pos.size(); j++){
            //计算中心点距离对应的概率值
            md_matrix(i, j) = md_distance(whole_box_centers[i], whole_pts[j], whole_HPH_R[j] + sigma2);
            ad_matrix(i, j) = ang_distance(whole_box_centers[i], lamp_cur_cam_pos[j], whole_liftup_HPH_R[j], sigma2);

        }
    }


    //转换为指派问题    
    Eigen::MatrixXd weight_matrix_md = alpha1 * Eigen::MatrixXd::Ones(box_nums, int(lamp_cur_cam_pos.size() + 1));
    // TODO 动态权重分配
    // for(int i = 0; i < box_nums; i++){
    //     if(whole_box_corners[i][0] < 3 || whole_box_corners[i][2] >= res_x - 3 || whole_box_corners[i][1] < 3 || whole_box_corners[i][3] >= res_y - 3)
    //         weight_matrix_md.row(i) = Eigen::RowVectorXd::Ones(lamp_cur_cam_pos.size() + 1);
    // }
    int max_edge = box_nums + int(lamp_cur_cam_pos.size()); //-1可以分配给最多n个检测框
    // int max_edge = max(box_nums, int(lamp_cur_cam_pos.size() + 1));
    Eigen::MatrixXd cost = Eigen::MatrixXd::Zero(max_edge, max_edge);
    cost.block(0, 0, box_nums, lamp_cur_cam_pos.size() + 1) = weight_matrix_md.cwiseProduct(md_matrix) + (Eigen::MatrixXd::Ones(box_nums, int(lamp_cur_cam_pos.size() + 1)) - weight_matrix_md).cwiseProduct(ad_matrix);

    for(int i = 0; i < box_nums; i++){
        if(i == 0){
            cost.block(0, lamp_cur_cam_pos.size(), box_nums, 1) = Eigen::VectorXd::Ones(box_nums) - cost.block(0, 0, box_nums, lamp_cur_cam_pos.size()).rowwise().sum();
        }
        else
            cost.block(0, lamp_cur_cam_pos.size() + i, box_nums, 1) = cost.block(0, lamp_cur_cam_pos.size(), box_nums, 1);
    }

    fp_match << "md_matrix: " << endl << md_matrix.block(0, 0, box_nums, lamp_cur_cam_pos.size()) << endl;
    fp_match << "ad_matrix: " << endl << ad_matrix.block(0, 0, box_nums, lamp_cur_cam_pos.size()) << endl;
    // fp_match << "dIoU_matrix: " << endl << dIoU_matrix << endl;
    // fp_match << "bd_matrix: " << endl << bd_matrix << endl;
    fp_match << "org cost matrix: " << endl << cost.block(0, 0, box_nums, lamp_cur_cam_pos.size() + 1) << endl;

    double max_emt = cost.maxCoeff();
    cost = max_emt * Eigen::MatrixXd::Ones(cost.rows(), cost.cols()) - cost;
    fp_match << "final cost matrix: " << endl << cost.block(0, 0, box_nums, lamp_cur_cam_pos.size() + 1) << endl << endl << endl;

    Hungary hungary(max_edge, cost);
    vector<int> result = hungary.solve();

    fp_match << "matches: " << endl;
    vector<int> M(box_nums);
    for(int i = 0; i < result.size(); i++){
        if(i >= box_nums)
            break;
        //核对结果，若分配的cost值过大，该点不可能是合理的地图点
        if(result[i] < lamp_cur_cam_pos.size() && cost(i, result[i]) >= 0.9){
            result[i] = lamp_cur_cam_pos.size();
            double tmp = cost(i, lamp_cur_cam_pos.size());
            for(int j = lamp_cur_cam_pos.size(); j < cost.cols(); j++){
                cost(i, j) = cost(i, result[i]);
            }
            cost(i, result[i]) = tmp;
        }
        if(result[i] >= lamp_cur_cam_pos.size())
            M[i] = -1;
        else
            M[i] = lamp_cur_id[result[i]];
        fp_match << M[i] << " ";
    }

    Match match_result;
    match_result.M = M;
    match_result.hungary_result = result;
    match_result.cost_matrix = cost;
    match_result.box_num = box_nums;
    match_result.lamp_num = lamp_cur_cam_pos.size() + 1;

    return match_result;
}

void DetectAndProjectImage(const Measures& mea, cv::Mat& detect_image, cv::Mat& detect_project_image, vector<int> M){
    detect_image = mea.img.clone();
    cv::Mat detect_image_bef = mea.img.clone();
    // detect_project_image = mea.img.clone();
    // cout << detect_image.at<uchar>(100, 100);
    vector<vector<cv::Point>> boxes;
    int off = 0;
    cv::RNG rng(time(0));
    for (int i = 0; i < mea.deep_learned_boxes; i++){
        vector<cv::Point> box;
        box.push_back(cv::Point(mea.box->bounding_boxes[i].xmin-off,mea.box->bounding_boxes[i].ymin-off));
        box.push_back(cv::Point(mea.box->bounding_boxes[i].xmax+off,mea.box->bounding_boxes[i].ymin-off));
        box.push_back(cv::Point(mea.box->bounding_boxes[i].xmax+off,mea.box->bounding_boxes[i].ymax+off));
        box.push_back(cv::Point(mea.box->bounding_boxes[i].xmin-off,mea.box->bounding_boxes[i].ymax+off));
        boxes.push_back(box);
    }
    vector<cv::Scalar> vec_scalar;
    if (!mea.img.empty()){
        // for (int i = 0; i < mea.box->bounding_boxes.size(); i++){
        //     if (!M.empty()){
        //         if (M[i] >= 0){
        //             cv::Scalar scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        //             // cv::drawContours(detect_project_image, boxes, i, scalar, 6);
        //             // cv::putText(detect_image, to_string(M[i]), cv::Point(mea.box->bounding_boxes[i].xmin, mea.box->bounding_boxes[i].ymin), cv::FONT_HERSHEY_SIMPLEX, 1.0, CV_RGB(255,230,0), 3);
        //             vec_scalar.push_back(scalar);
        //         }
        //     }
        // }
        // cv::drawContours(detect_image, boxes, -1, cv::Scalar(0, 0, 255), 6);
        cv::drawContours(detect_project_image, boxes, -1, cv::Scalar(255, 0, 0), 6);
        cv::drawContours(detect_image_bef, boxes, -1, cv::Scalar(0, 0, 255), 6);
    }
    // cout << "Rcw: " << endl << Rcw << endl << "pcw: " << endl << pcw.transpose() << endl;
    for (int i = 0; i < M.size(); i++){
        if (M[i] >= 0){
            int lamp_id = M[i];
            // cout << lamp_id;
            for (int j = 0; j < lamp_world_points[lamp_id].size(); j++)
            {
                Eigen::Vector3d Pc = Rcw * lamp_world_points[lamp_id][j] + pcw;

                // cv::Point2d pt;
                // double inv_z = 1.0 / Pc.z();
                // pt.x = cam_fx * Pc.x() * inv_z + cam_cx;
                // pt.y = cam_fy * Pc.y() * inv_z + cam_cy;
                // cv::circle(detect_project_image, pt, 3, vec_scalar[i], -1);

                pcl::PointXYZRGB pcl_point;
                pcl_point.x = lamp_world_points[lamp_id][j].x();
                pcl_point.y = lamp_world_points[lamp_id][j].y();
                pcl_point.z = lamp_world_points[lamp_id][j].z();
                // pcl_point.r = vec_scalar[i][0];
                // pcl_point.g = vec_scalar[i][1];
                // pcl_point.b = vec_scalar[i][2];
                pcl_point.r = 255;
                pcl_point.g = 0;
                pcl_point.b = 0;
                cur_lamp_cloud->push_back(pcl_point);
            }

            Eigen::Vector3d Pc = Rcw * lamp_world_pos[lamp_id] + pcw;
            cv::Point2d pt;
            double inv_z = 1.0 / Pc.z();
            pt.x = cam_fx * Pc.x() * inv_z + cam_cx;
            pt.y = cam_fy * Pc.y() * inv_z + cam_cy;
            cv::circle(detect_image, pt, 10, cv::Scalar(0, 0, 255), -1);

        }
    }

    for (int id = 0; id < lamp_world_points.size(); id++){
        double min_u = 10000, min_v = 10000;
        for (int i = 0; i < lamp_world_points[id].size(); i++){
            Eigen::Vector3d Pc = Rcw * lamp_world_points[id][i] + pcw;
            if (Pc.norm() < update_z_th){
                if(Pc.z() < 0.05)
                    continue;
                cv::Point2d pt;
                double inv_z = 1.0 / Pc.z();
                pt.x = cam_fx * Pc.x() * inv_z + cam_cx;
                pt.y = cam_fy * Pc.y() * inv_z + cam_cy;

                min_u = min(pt.x, min_u);
                min_v = min(pt.y, min_v);

                cv::circle(detect_project_image, pt, 1, cv::Scalar(0, 0, 255), -1);
            }

            if(last_half_box && find_half_match){
                Eigen::Vector3d tmp_Pc = tmp_Rcw * lamp_world_points[id][i] + tmp_pcw;
                if (Pc.norm() < update_z_th){
                    if(tmp_Pc.z() < 0.05)
                        continue;
                    cv::Point2d pt;
                    double inv_z = 1.0 / tmp_Pc.z();
                    pt.x = cam_fx * tmp_Pc.x() * inv_z + cam_cx;
                    pt.y = cam_fy * tmp_Pc.y() * inv_z + cam_cy;

                    cv::circle(detect_image_bef, pt, 3, cv::Scalar(0, 255, 0), -1);
                }
            }
        }
        // if(min_u < 10000 && min_v < 10000){
        //     cv::putText(detect_project_image, to_string(id), cv::Point(min_u - 5, min_v - 5), cv::FONT_HERSHEY_SIMPLEX, 1.0, CV_RGB(255, 230, 0), 3);
        // }
    }
}

void BodyPoseToCameraPose(const Eigen::Matrix3d& Rij, const Eigen::Vector3d& pij, const Matrix6d& sigmaij){
    Rwc = Rij * Rbc;
    pwc = Rij * pbc + pij;
    Rcw = Rwc.transpose();
    pcw = - Rcw * pwc;
    Ow = - Rwc * pcw;

    Matrix6d trans_jacobbian = Matrix6d::Identity();
    trans_jacobbian.block<3, 3>(0, 0) = Rcb;
    trans_jacobbian.block<3, 3>(3, 3) = - Rcw;
    cov = trans_jacobbian * sigmaij * trans_jacobbian.transpose();
    cout << "cov: " << cov << endl;
}

void CameraPoseToBodyPose(){
    Matrix6d trans_jacobbian;
    Rwc = Rcw.transpose();
    pwc = -Rwc * pcw;
    Rwb = Rwc * Rcb;
    pwb = Rwc * pcb + pwc;
    trans_jacobbian.block<3, 3>(0, 0) = Rbc;
    trans_jacobbian.block<3, 3>(3, 3) = - Rwc;
    trans_jacobbian.block<3, 3>(3, 0) = Eigen::Matrix3d::Zero(); //- Rwc * vector_skew(pcb - pwc) = 0;
    trans_jacobbian.block<3, 3>(0, 3) = Eigen::Matrix3d::Zero();

    // cout << "Rwc: " << Rwc << endl << "pwc: " << pwc << endl;

    cov_wb = trans_jacobbian * cov * trans_jacobbian.transpose();
}

bool DetectLine(const vector<int>& tmp_matches, const unordered_map<int, int>& matches){
    vector<int> all_ids = tmp_matches;
    auto match_iter = matches.begin();
    while(match_iter != matches.end()){
        all_ids.push_back(match_iter->second);
        match_iter++;
    }
    if(all_ids.size() <= 2){
        return true;
    }
    else{
        Eigen::Vector3d p1 = lamp_world_pos[all_ids[0]];
        Eigen::Vector3d p2 = lamp_world_pos[all_ids[1]];
        Eigen::Vector3d d = (p2 - p1).normalized();
        Eigen::Vector3d m = vector_skew(p1) * d;
        for (int i = 2; i < all_ids.size(); i++)
        {
            Eigen::Vector3d err = m - vector_skew(lamp_world_pos[all_ids[i]]) * d;
            double dist = err.norm();
            if(dist > 0.8){
                return false;
            }
        }
        return true;
    }
}

void CorrectDegenerationDeep(const Measures& data, unordered_map<int, int>& matches, const vec_vec4d& boxes, unordered_map<int, int>& new_matches, cv::Mat& dp_img){

    vec_vec3d tmp_old_cam_points, tmp_new_cam_points;
    vec_vec2d tmp_pixs;
    vector<int> tmp_idxs;
    for (int i = 0; i < lamp_world_pos.size(); i++){
        if(isnan(lamp_world_pos[i].norm())) continue;
        if(lamp_world_points[i].size() <= 0) continue;
        bool find_rep = false;
        for (auto iter = matches.begin(); iter != matches.end(); ++iter){
            if(iter->second == i){
                find_rep = true;
                break;
            }
        }
        if(find_rep){
            Eigen::Vector3d Pc;
            Pc = Rcw * lamp_world_pos[i] + pcw;
            tmp_old_cam_points.push_back(Pc);
            continue;
        }

        Eigen::Vector3d Pc;
        Pc = Rcw * lamp_world_pos[i] + pcw;
        Eigen::Vector2d pt;
        double inv_z = 1.0 / Pc.z();
        pt << cam_fx * Pc.x() * inv_z + cam_cx, cam_fy * Pc.y() * inv_z + cam_cy;
        
        int max_norm_th, max_z_th, min_norm_th, min_z_th;
        max_norm_th = update_dist_th;
        max_z_th = update_z_th;
        min_norm_th = 0;
        min_z_th = 0;
        if (Pc.norm() < max_norm_th && Pc.norm() > min_norm_th && Pc.z() < max_z_th && Pc.z() > min_z_th && pt.x() < res_x - 15 && pt.x() >= 15 && pt.y() - 15 < res_y && pt.y() >= 15){ //70, 55
            bool find_occ = false;
            //查看当前点是否被遮挡，首先和已经匹配上的点比较
            for (int j = 0; j < tmp_old_cam_points.size(); j++){
                Eigen::Vector3d l1 = tmp_old_cam_points[j].normalized();
                Eigen::Vector3d l2 = Pc.normalized();
                // cout << "cos_theta: " << l1.transpose() * l2 << endl;
                //由于已经匹配上的点已经在之前判断，因此直接舍弃当前点
                if((l1.transpose() * l2)(0) > 0.9998){
                    find_occ = true;
                }
            }
            if(find_occ) continue;
            //随后和新加入的点比较
            for (int j = 0; j < tmp_new_cam_points.size(); j++){
                Eigen::Vector3d l1 = tmp_new_cam_points[j].normalized();
                Eigen::Vector3d l2 = Pc.normalized();
                // cout << "cos_theta: " << l1.transpose() * l2 << endl;
                if((l1.transpose() * l2)(0) > 0.9998){
                    double n1 = l1.norm();
                    double n2 = l2.norm();
                    find_occ = true;
                    //若存在遮挡现象，选取近的那一点
                    if(n1 > n2){
                        tmp_new_cam_points[j] = Pc;
                        tmp_idxs[j] = i;
                        tmp_pixs[j] = pt;
                    }
                    break;
                }
            }
            if(find_occ) continue;

            tmp_new_cam_points.push_back(Pc);
            tmp_idxs.push_back(i);
            tmp_pixs.push_back(pt);
        }
    }
    if(tmp_new_cam_points.size() < 1){
        //扩大范围
        for (int i = 0; i < lamp_world_pos.size(); i++){
            if(isnan(lamp_world_pos[i].norm())) continue;
            if(lamp_world_points[i].size() <= 0) continue;
            bool find_rep = false;
            for (auto iter = matches.begin(); iter != matches.end(); ++iter){
                if(iter->second == i){
                    find_rep = true;
                    break;
                }
            }
            if(find_rep){
                Eigen::Vector3d Pc;
                Pc = Rcw * lamp_world_pos[i] + pcw;
                tmp_old_cam_points.push_back(Pc);
                continue;
            }

            Eigen::Vector3d Pc;
            Pc = Rcw * lamp_world_pos[i] + pcw;
            Eigen::Vector2d pt;
            double inv_z = 1.0 / Pc.z();
            pt << cam_fx * Pc.x() * inv_z + cam_cx, cam_fy * Pc.y() * inv_z + cam_cy;
            
            int max_norm_th, max_z_th, min_norm_th, min_z_th;
            max_norm_th = update_dist_th + 50;
            max_z_th = update_z_th + 50;
            min_norm_th = update_dist_th;
            min_z_th = update_z_th;  min_z_th = update_z_th;
            if (Pc.norm() < max_norm_th && Pc.norm() > min_norm_th && Pc.z() < max_z_th && Pc.z() > min_z_th && pt.x() < res_x && pt.x() >= 0 && pt.y() < res_y && pt.y() >=0){ //70, 55
                bool find_occ = false;
                //查看当前点是否被遮挡，首先和已经匹配上的点比较
                for (int j = 0; j < tmp_old_cam_points.size(); j++){
                    Eigen::Vector3d l1 = tmp_old_cam_points[j].normalized();
                    Eigen::Vector3d l2 = Pc.normalized();
                    // cout << "cos_theta: " << l1.transpose() * l2 << endl;
                    //由于已经匹配上的点已经在之前判断，因此直接舍弃当前点
                    if((l1.transpose() * l2)(0) > 0.9998){
                        find_occ = true;
                    }
                }
                if(find_occ) continue;
                //随后和新加入的点比较
                for (int j = 0; j < tmp_new_cam_points.size(); j++){
                    Eigen::Vector3d l1 = tmp_new_cam_points[j].normalized();
                    Eigen::Vector3d l2 = Pc.normalized();
                    // cout << "cos_theta: " << l1.transpose() * l2 << endl;
                    if((l1.transpose() * l2)(0) > 0.9998){
                        double n1 = l1.norm();
                        double n2 = l2.norm();
                        find_occ = true;
                        //若存在遮挡现象，选取近的那一点
                        if(n1 > n2){
                            tmp_new_cam_points[j] = Pc;
                            tmp_idxs[j] = i;
                            tmp_pixs[j] = pt;
                        }
                        break;
                    }
                }
                if(find_occ) continue;

                tmp_new_cam_points.push_back(Pc);
                tmp_idxs.push_back(i);
                tmp_pixs.push_back(pt);
            }
        }
    }

    vector<int> matched_boxes, unmatched_boxes;
    for(int i = 0; i < boxes.size(); ++i){
        bool has_match = false;
        for(auto iter = matches.begin(); iter != matches.end(); ++iter){
            if(iter->first == i){
                has_match = true;
                break;
            }
        }
        if(has_match){
            matched_boxes.push_back(i);
        }
        else{
            unmatched_boxes.push_back(i);
        }
    }

    vector<int> tmp_matches;
    bool is_line = false;
    if (DetectLine(tmp_matches, matches)){
        is_line = true;
        ++line;
    }
    else{
        is_line = false;
        line = 0;
    }

    // if(!reloc){
    bool left_box = false, right_box = false;
    for(auto iter = matches.begin(); iter != matches.end(); ++iter){
        if(boxes[iter->first][2] < res_x / 2 - left_right_gap && boxes[iter->first][0] >= 0)
            left_box = true;
        if(boxes[iter->first][0] >= res_x / 2 + left_right_gap && boxes[iter->first][2] < res_x)
            right_box = true;
    }

    if(!left_box){
        fp_match << "left half image not has matches" << endl;
        cout << "left half image not has matches" << endl;
        ++no_left_box;
    }
    if(!right_box){
        fp_match << "right half image not has matches" << endl;
        cout << "right half image not has matches" << endl;
        ++no_right_box;
    }
    if(right_box && left_box){
        no_left_box = 0;
        no_right_box = 0;
    }

    if(right_box && left_box && !is_line){
        last_half_box = false;
        find_half_match = false;
    }
    else{
        find_half_match = false;
        last_half_box = true;
    }
    if(((!left_box && no_left_box >= 3) || (!right_box && no_right_box >= 3) || (is_line && line >= 3))){
        bool find_new_match = false;
        for (int i = 0; i < tmp_new_cam_points.size(); i++){
            bool is_matched = false;
            for (int j = 0; j < tmp_matches.size(); j++){
                if(tmp_matches[j] == tmp_idxs[i])
                    is_matched = true;
            }
            for(auto iter = matches.begin(); iter != matches.end(); ++iter){
                if(iter->second == tmp_idxs[i])
                    is_matched = true;
            }
            if(is_matched)
                continue;

            Eigen::Vector2d pt = tmp_pixs[i];
            if ((!left_box && pt.x() >= res_x / 2 && !is_line) || (!right_box && pt.x() < res_x / 2 && !is_line))
                continue;

            Eigen::Vector3d Pc = tmp_new_cam_points[i];
            int id = tmp_idxs[i];

            int extend = 7;
            cv::Rect rect(cv::Point2i(pt.x() - extend, pt.y() - 60), cv::Point2i(pt.x() + extend, pt.y() + 60));
            if(pt.x() - extend <= 0 || pt.x() + extend >= res_x)
                continue;

            // cv::rectangle(dp_img, rect, cv::Scalar(255, 255, 0), 2);

            cout << "find new points, try to match" << endl;
            
            if ((!left_box && pt.x() < res_x / 2) || (!right_box && pt.x() >= res_x / 2) || is_line){
                cv::Rect min_rect;
                double min_dist = -1;
                int min_box_idx = -1;
                for (int j = 0; j < unmatched_boxes.size(); j++){
                    //box to rect
                    Vector4d box = boxes[unmatched_boxes[j]];
                    cv::Rect box_rect = cv::Rect(cv::Point2i(box[0], box[1]), cv::Point2i(box[2], box[3]));
                    if((rect & box_rect).area() > 0){
                        Eigen::Vector2d center_box;
                        center_box << 0.5 * (box_rect.tl().x + box_rect.br().x), 0.5 * (box_rect.tl().y + box_rect.br().y);
                        
                        double dist = (pt - center_box).norm();
                        if(min_dist > 0){
                            if(min_dist > dist){
                                dist = min_dist;
                                min_rect = box_rect;
                                min_box_idx = j;
                            }
                        }
                        else{
                            min_dist = dist;
                            min_rect = box_rect;
                            min_box_idx = j;
                        }
                    }
                }
                if(min_dist > 0){
                    find_new_match = true;
                    new_matches.insert(make_pair(unmatched_boxes[min_box_idx], id));
                }
            }
        }
        if(find_new_match){
            find_new_match = false;
            find_half_match = true;
            no_left_box = 0;
            no_right_box = 0;
            line = 0;
            last_half_box = true;
        }
    }

}

void UpdateBox(const Measures& data, const unordered_map<int, int>& matches, unordered_map<int, int>& new_matches, const vec_vec4d& boxes, vec_vec4d& new_boxes, cv::Mat& img, cv::Mat& dp_img){
    cv::Mat grey_img(data.img.rows, data.img.cols, CV_8UC1);
    cv::cvtColor(data.img, grey_img, cv::COLOR_BGR2GRAY);
    cv::Mat bin_img;
    cv::threshold(grey_img, bin_img, grey_th, 255, cv::THRESH_BINARY);

    vector<cv::Rect> tmp_boxes;
    vector<vector<cv::Point>> contours;
    cv::findContours(bin_img, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    for(int i = 0; i < contours.size(); i++){
        cv::Rect rect = cv::boundingRect(contours[i]);
        // cout << rect.x << " " << rect.y << " " << rect.width << " " << rect.height << endl;

        int area = rect.width * rect.height;
        if(reloc && global_id - reloc_id <= 150 && global_id - reloc_id >= 30){
            if(area < 15 || rect.width < 4 || rect.height < 4 || rect.height / rect.width > 6)
                continue;
        }
        else{
            if(area < 20 || rect.width < 4 || rect.height < 4 || rect.height / rect.width > 6)
                continue;
        }

        int light_nei_v = 0;
        for(int v = rect.y; v < rect.y + rect.height; ++v){
            float light_intensity = grey_img.ptr<uchar>(v)[rect.x + rect.width / 2];
            if(light_intensity > 245) light_nei_v++;
        }
        if(light_nei_v / float(rect.width + 1) < 0.8 && area > 400)
            continue;

        if(rect.x >= 0 && rect.y >= 0 && rect.width < bin_img.cols && rect.height < bin_img.rows){
            tmp_boxes.push_back(rect);
            // cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 3);
            cv::rectangle(dp_img, rect, cv::Scalar(0, 255, 0), 6);
        }
    }

    //去除重复的检测框
    auto iter_box1 = tmp_boxes.begin();
    while(iter_box1 != tmp_boxes.end()){
        bool find_rep = false;
        for(auto iter_box2 = boxes.begin(); iter_box2 != boxes.end(); ++iter_box2){
            cv::Rect old_box(cv::Point2i(int((*iter_box2)(0)), int((*iter_box2)(1))), cv::Point2i(int((*iter_box2)(2)), int((*iter_box2)(3))));
            if(((*iter_box1) & old_box).area() > 0.25 * iter_box1->area() || ((*iter_box1) & old_box).area() > 0.25 * old_box.area()){
                auto iter = matches.begin();
                while(iter != matches.end()){
                    if(iter->first == iter_box2 - boxes.begin()){
                        find_rep = true;
                        break;
                    }
                    ++iter;
                }
                // find_rep = true;
                break;
            }
        }
        if(find_rep){
            iter_box1 = tmp_boxes.erase(iter_box1);
        }
        else{
            ++iter_box1;
        }
    }

    vector<double> org_dist;
    vector<int> tmp_matches;
    vector<cv::Rect> tmp_new_boxes;

    vec_vec3d tmp_old_cam_points, tmp_new_cam_points;
    vec_vec2d tmp_pixs;
    vector<int> tmp_idxs;
    for (int i = 0; i < lamp_world_pos.size(); i++){
        if(isnan(lamp_world_pos[i].norm())) continue;
        if(lamp_world_points[i].size() <= 0) continue;
        bool find_rep = false;
        for (auto iter = matches.begin(); iter != matches.end(); ++iter){
            if(iter->second == i){
                find_rep = true;
                break;
            }
        }
        if(find_rep){
            Eigen::Vector3d Pc;
            Pc = Rcw * lamp_world_pos[i] + pcw;
            tmp_old_cam_points.push_back(Pc);
            continue;
        }

        Eigen::Vector3d Pc;
        Pc = Rcw * lamp_world_pos[i] + pcw;
        Eigen::Vector2d pt;
        double inv_z = 1.0 / Pc.z();
        pt << cam_fx * Pc.x() * inv_z + cam_cx, cam_fy * Pc.y() * inv_z + cam_cy;
        
        int max_norm_th, max_z_th, min_norm_th, min_z_th;
        if (reloc && global_id - reloc_id <= 150 && global_id - reloc_id >= 30){
            cout << "reloc_id: " << reloc_id << endl;
            max_norm_th = reloc_update_dist_th;
            max_z_th = reloc_update_z_th;
            min_norm_th = reloc_dist_th - 10;
            min_z_th = reloc_z_th - 10;
        }
        else{
            max_norm_th = update_dist_th;
            max_z_th = update_z_th;
            min_norm_th = 0;
            min_z_th = 0;
        }
        if (Pc.norm() < max_norm_th && Pc.norm() > min_norm_th && Pc.z() < max_z_th && Pc.z() > min_z_th && pt.x() < res_x - 15 && pt.x() >= 15 && pt.y() - 15 < res_y && pt.y() >= 15){ //70, 55
            bool find_occ = false;
            //查看当前点是否被遮挡，首先和已经匹配上的点比较
            for (int j = 0; j < tmp_old_cam_points.size(); j++){
                Eigen::Vector3d l1 = tmp_old_cam_points[j].normalized();
                Eigen::Vector3d l2 = Pc.normalized();
                // cout << "cos_theta: " << l1.transpose() * l2 << endl;
                //由于已经匹配上的点已经在之前判断，因此直接舍弃当前点
                if((l1.transpose() * l2)(0) > 0.9998){
                    find_occ = true;
                }
            }
            if(find_occ) continue;
            //随后和新加入的点比较
            for (int j = 0; j < tmp_new_cam_points.size(); j++){
                Eigen::Vector3d l1 = tmp_new_cam_points[j].normalized();
                Eigen::Vector3d l2 = Pc.normalized();
                // cout << "cos_theta: " << l1.transpose() * l2 << endl;
                if((l1.transpose() * l2)(0) > 0.9998){
                    double n1 = l1.norm();
                    double n2 = l2.norm();
                    find_occ = true;
                    //若存在遮挡现象，选取近的那一点
                    if(n1 > n2){
                        tmp_new_cam_points[j] = Pc;
                        tmp_idxs[j] = i;
                        tmp_pixs[j] = pt;
                    }
                    break;
                }
            }
            if(find_occ) continue;

            tmp_new_cam_points.push_back(Pc);
            tmp_idxs.push_back(i);
            tmp_pixs.push_back(pt);
        }
    }
    if(tmp_new_cam_points.size() < 1){
        //扩大范围
        for (int i = 0; i < lamp_world_pos.size(); i++){
            if(isnan(lamp_world_pos[i].norm())) continue;
            if(lamp_world_points[i].size() <= 0) continue;
            bool find_rep = false;
            for (auto iter = matches.begin(); iter != matches.end(); ++iter){
                if(iter->second == i){
                    find_rep = true;
                    break;
                }
            }
            if(find_rep){
                Eigen::Vector3d Pc;
                Pc = Rcw * lamp_world_pos[i] + pcw;
                tmp_old_cam_points.push_back(Pc);
                continue;
            }

            Eigen::Vector3d Pc;
            Pc = Rcw * lamp_world_pos[i] + pcw;
            Eigen::Vector2d pt;
            double inv_z = 1.0 / Pc.z();
            pt << cam_fx * Pc.x() * inv_z + cam_cx, cam_fy * Pc.y() * inv_z + cam_cy;
            
            int max_norm_th, max_z_th, min_norm_th, min_z_th;
            if (reloc && global_id - reloc_id <= 150 && global_id - reloc_id >= 30){
                cout << "reloc_id: " << reloc_id << endl;
                max_norm_th = reloc_update_dist_th;
                max_z_th = reloc_update_z_th;
                min_norm_th = reloc_dist_th - 10;
                min_z_th = reloc_z_th - 10;
            }
            else{
                max_norm_th = update_dist_th + 50;
                max_z_th = update_z_th + 50;
                min_norm_th = update_dist_th;
                min_z_th = update_z_th;
            }
            if (Pc.norm() < max_norm_th && Pc.norm() > min_norm_th && Pc.z() < max_z_th && Pc.z() > min_z_th && pt.x() < res_x && pt.x() >= 0 && pt.y() < res_y && pt.y() >=0){ //70, 55
                bool find_occ = false;
                //查看当前点是否被遮挡，首先和已经匹配上的点比较
                for (int j = 0; j < tmp_old_cam_points.size(); j++){
                    Eigen::Vector3d l1 = tmp_old_cam_points[j].normalized();
                    Eigen::Vector3d l2 = Pc.normalized();
                    // cout << "cos_theta: " << l1.transpose() * l2 << endl;
                    //由于已经匹配上的点已经在之前判断，因此直接舍弃当前点
                    if((l1.transpose() * l2)(0) > 0.9998){
                        find_occ = true;
                    }
                }
                if(find_occ) continue;
                //随后和新加入的点比较
                for (int j = 0; j < tmp_new_cam_points.size(); j++){
                    Eigen::Vector3d l1 = tmp_new_cam_points[j].normalized();
                    Eigen::Vector3d l2 = Pc.normalized();
                    // cout << "cos_theta: " << l1.transpose() * l2 << endl;
                    if((l1.transpose() * l2)(0) > 0.9998){
                        double n1 = l1.norm();
                        double n2 = l2.norm();
                        find_occ = true;
                        //若存在遮挡现象，选取近的那一点
                        if(n1 > n2){
                            tmp_new_cam_points[j] = Pc;
                            tmp_idxs[j] = i;
                            tmp_pixs[j] = pt;
                        }
                        break;
                    }
                }
                if(find_occ) continue;

                tmp_new_cam_points.push_back(Pc);
                tmp_idxs.push_back(i);
                tmp_pixs.push_back(pt);
            }
        }
    }

    //寻找最合适的检测框
    // int extend;
    // extend = 20;
    for (int i = 0; i < tmp_new_cam_points.size(); i++){
        
        Eigen::Vector2d pt = tmp_pixs[i];
        Eigen::Vector3d Pc = tmp_new_cam_points[i];
        int id = tmp_idxs[i];

        cv::Rect rect(cv::Point2i(pt.x() - extend, pt.y() - extend), cv::Point2i(pt.x() + extend, pt.y() + extend));

        // cv::rectangle(img, rect, cv::Scalar(255, 0, 0), 2);
        
        if (reloc && global_id - reloc_id <= 150 && global_id - reloc_id >= 30 && 0){
        //如果存在重叠的检测框但却没有任何点位于这些检测框内
        // if(max_num == -1 && exist_box){
        //     //根据距离判断最优检测框
            cv::Rect min_rect;
            double min_dist = -1;
            for (int j = 0; j < tmp_boxes.size(); j++){
                if((rect & tmp_boxes[j]).area() > 0){
                    Eigen::Vector2d center_box;
                    center_box << 0.5 * (tmp_boxes[j].tl().x + tmp_boxes[j].br().x), 0.5 * (tmp_boxes[j].tl().y + tmp_boxes[j].br().y);
                    
                    double dist = (pt - center_box).norm();
                    if(min_dist > 0){
                        if(min_dist > dist){
                            dist = min_dist;
                            min_rect = tmp_boxes[j];
                        }
                    }
                    else{
                        min_dist = dist;
                        min_rect = tmp_boxes[j];
                    }
                }
            }

            if(min_dist > 0){
                tmp_new_boxes.push_back(min_rect);
                org_dist.push_back(Pc.norm());
                tmp_matches.push_back(id);
            }
        // }
        }
        else{
            int max_num = -1, sec_max_num = -1;
            // bool exist_box = false;
            cv::Rect min_rect, sec_min_rect;
            for (int j = 0; j < tmp_boxes.size(); j++){
                if((rect & tmp_boxes[j]).area() > 0){
                    // exist_box = true;

                    int num = 0;
                    int off = 4;
                    for (int k = 0; k < lamp_world_points[tmp_idxs[i]].size(); k++){
                        Eigen::Vector3d Pci = Rcw * lamp_world_points[tmp_idxs[i]][k] + pcw;
                        
                        Eigen::Vector2d pti;
                        double inv_z = 1.0 / Pci.z();
                        pti << cam_fx * Pci.x() * inv_z + cam_cx, cam_fy * Pci.y() * inv_z + cam_cy;

                        if(pti.x() >= tmp_boxes[j].tl().x - off && pti.y() >= tmp_boxes[j].tl().y - off && pti.x() <= tmp_boxes[j].br().x + off && pti.y() <= tmp_boxes[j].br().y + off){
                            ++num;
                        }
                    }
                    if(num != 0){
                        if(max_num < num){
                            sec_max_num = max_num;
                            sec_min_rect = min_rect;

                            max_num = num;
                            min_rect = tmp_boxes[j];
                        }
                        else if(max_num >= num && sec_max_num < num){
                            sec_max_num = num;
                            sec_min_rect = tmp_boxes[j];
                        }
                    }
                }
            }
            
            if(max_num > 0 && max_num >= 2 * sec_max_num){
                tmp_new_boxes.push_back(min_rect);
                org_dist.push_back(Pc.norm());
                tmp_matches.push_back(id);
            }
            else if(max_num > 0 && max_num < 2 * sec_max_num){
                min_rect.area() > sec_min_rect.area() ? tmp_new_boxes.push_back(min_rect) : tmp_new_boxes.push_back(sec_min_rect);
                org_dist.push_back(Pc.norm());
                tmp_matches.push_back(id);
            }
        }

    }

    for(int i = 0; i < tmp_matches.size(); i++){
        cout << "org matches: " << tmp_matches[i] << endl;
        cout << "org boxes: " << endl << tmp_new_boxes[i] << endl;
        cout << "org dist: " << org_dist[i] << endl;
    }

    bool is_line = false;
    if (DetectLine(tmp_matches, matches)){
        is_line = true;
        ++line;
    }
    else{
        is_line = false;
        line = 0;
    }

    // if(!reloc){
    bool left_box = false, right_box = false;
    vector<cv::Rect> half_tmp_boxes;
    for(int i = 0; i < tmp_new_boxes.size(); i++){
        cout << "rect: " << tmp_new_boxes[i].tl().x << " " << tmp_new_boxes[i].br().x;
        if(tmp_new_boxes[i].br().x < res_x / 2 - left_right_gap && tmp_new_boxes[i].tl().x >= 0)
            left_box = true;
        if(tmp_new_boxes[i].tl().x >= res_x / 2 + left_right_gap && tmp_new_boxes[i].br().x < res_x)
            right_box = true;
    }
    for(auto iter = matches.begin(); iter != matches.end(); ++iter){
        cout << "boxes" << endl << boxes[iter->first] << endl;
        if(boxes[iter->first](2) < res_x / 2 - left_right_gap && boxes[iter->first](0) >= 0)
            left_box = true;
        if(boxes[iter->first](0) >= res_x / 2 + left_right_gap && boxes[iter->first](2) < res_x)
            right_box = true;
    }
    if(!left_box){
        fp_match << "left half image not has matches" << endl;
        cout << "left half image not has matches" << endl;
        cv::threshold(grey_img, bin_img, 240, 255, cv::THRESH_BINARY);
        cv::findContours(bin_img, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
        for(int i = 0; i < contours.size(); i++){
            cv::Rect rect = cv::boundingRect(contours[i]);
            // cout << rect.x << " " << rect.y << " " << rect.width << " " << rect.height << endl;

            int area = rect.width * rect.height;
            if(area < 12 || rect.width < 3 || rect.height < 3 || rect.br().x >= res_x / 2 - left_right_gap)
                continue;

            if(rect.x >= 0 && rect.y >= 0 && rect.width < bin_img.cols && rect.height < bin_img.rows){
                half_tmp_boxes.push_back(rect);
                // cv::rectangle(img, rect, cv::Scalar(0, 255, 255), 3);
            }
        }
        ++no_left_box;
    }
    if(!right_box){
        fp_match << "right half image not has matches" << endl;
        cout << "right half image not has matches" << endl;
        cv::threshold(grey_img, bin_img, 240, 255, cv::THRESH_BINARY);
        cv::findContours(bin_img, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
        for(int i = 0; i < contours.size(); i++){
            cv::Rect rect = cv::boundingRect(contours[i]);
            // cout << rect.x << " " << rect.y << " " << rect.width << " " << rect.height << endl;

            int area = rect.width * rect.height;
            if(area < 12 || rect.width < 3 || rect.height < 3 || rect.tl().x < res_x / 2 + left_right_gap)
                continue;

            if(rect.x >= 0 && rect.y >= 0 && rect.width < bin_img.cols && rect.height < bin_img.rows){
                half_tmp_boxes.push_back(rect);
                // cv::rectangle(img, rect, cv::Scalar(0, 255, 255), 3);
            }
        }
        ++no_right_box;
    }
    if(right_box && left_box){
        no_left_box = 0;
        no_right_box = 0;
    }

    if(right_box && left_box && !is_line){
        last_half_box = false;
        find_half_match = false;
    }
    else{
        find_half_match = false;
        last_half_box = true;
    }
    if(((!left_box && !half_tmp_boxes.empty() && no_left_box >= 3) || (!right_box && !half_tmp_boxes.empty() && no_right_box >= 3) || (is_line && line >= 3))&&(ablation_reloc || no_obs <= 15)){
        bool find_new_match = false;
        for (int i = 0; i < tmp_new_cam_points.size(); i++){
            bool is_matched = false;
            for (int j = 0; j < tmp_matches.size(); j++){
                if(tmp_matches[j] == tmp_idxs[i])
                    is_matched = true;
            }
            for(auto iter = matches.begin(); iter != matches.end(); ++iter){
                if(iter->second == tmp_idxs[i])
                    is_matched = true;
            }
            if(is_matched)
                continue;
            Eigen::Vector2d pt = tmp_pixs[i];
            if ((!left_box && pt.x() >= res_x / 2 && !is_line) || (!right_box && pt.x() < res_x / 2 && !is_line))
                continue;

            Eigen::Vector3d Pc = tmp_new_cam_points[i];
            int id = tmp_idxs[i];

            int extend = 7;
            cv::Rect rect(cv::Point2i(pt.x() - extend, pt.y() - 60), cv::Point2i(pt.x() + extend, pt.y() + 60));
            if(pt.x() - extend <= 0 || pt.x() + extend >= res_x)
                continue;

            // if (is_line && line >= 3)
            //     cv::rectangle(dp_img, rect, cv::Scalar(255, 255, 0), 2);

            cout << "find new points, try to match" << endl;
            
            if ((!left_box && pt.x() < res_x / 2) || (!right_box && pt.x() >= res_x / 2) || is_line){
                cv::Rect min_rect;
                double min_dist = -1;
                for (int j = 0; j < tmp_boxes.size(); j++){
                    if((rect & tmp_boxes[j]).area() > 0){
                        Eigen::Vector2d center_box;
                        center_box << 0.5 * (tmp_boxes[j].tl().x + tmp_boxes[j].br().x), 0.5 * (tmp_boxes[j].tl().y + tmp_boxes[j].br().y);
                        
                        double dist = (pt - center_box).norm();
                        if(min_dist > 0){
                            if(min_dist > dist){
                                dist = min_dist;
                                min_rect = tmp_boxes[j];
                            }
                        }
                        else{
                            min_dist = dist;
                            min_rect = tmp_boxes[j];
                        }
                    }
                }
                if(min_dist > 0){
                    find_new_match = true;
                    tmp_new_boxes.push_back(min_rect);
                    org_dist.push_back(Pc.norm());
                    tmp_matches.push_back(id);
                }
            }
        }
        if(find_new_match){
            find_new_match = false;
            find_half_match = true;
            no_left_box = 0;
            no_right_box = 0;
            line = 0;
            last_half_box = true;
        }
    }
    // }

    //检测是否重复，一个框对应多个路灯，若存在直接去掉
    auto iter_box_i = tmp_new_boxes.begin();
    while(iter_box_i != tmp_new_boxes.end()){
        
        auto iter_box_j = iter_box_i + 1;
        double dist_i = org_dist[iter_box_i - tmp_new_boxes.begin()];
        bool erase_i = false;

        while(iter_box_j != tmp_new_boxes.end()){
            if(((*iter_box_i) & (*iter_box_j)).area() > 0.9 * (*iter_box_i).area() || ((*iter_box_i) & (*iter_box_j)).area() > 0.9 * (*iter_box_j).area()){
                
                // double dist_j = org_dist[iter_box_j - tmp_new_boxes.begin()];
                // if(dist_i > dist_j){
                //     erase_i = true;
                //     ++iter_box_j;
                // }
                // else{
                //     iter_box_j = tmp_new_boxes.erase(iter_box_j);
                //     org_dist.erase(org_dist.begin() + size_t(iter_box_j - tmp_new_boxes.begin()));
                //     tmp_matches.erase(tmp_matches.begin() + size_t(iter_box_j - tmp_new_boxes.begin()));
                // }
                iter_box_j = tmp_new_boxes.erase(iter_box_j);
                org_dist.erase(org_dist.begin() + size_t(iter_box_j - tmp_new_boxes.begin()));
                tmp_matches.erase(tmp_matches.begin() + size_t(iter_box_j - tmp_new_boxes.begin()));
                erase_i = true;
                continue;

            }
            ++iter_box_j;
        }
        if(erase_i){
            iter_box_i = tmp_new_boxes.erase(iter_box_i);
            org_dist.erase(org_dist.begin() + size_t(iter_box_i - tmp_new_boxes.begin()));
            tmp_matches.erase(tmp_matches.begin() + size_t(iter_box_i - tmp_new_boxes.begin()));
            continue;
        }
        ++iter_box_i;
    }

    // for(int i = 0; i < tmp_matches.size(); i++){
    //     cout << "new matches: " << tmp_matches[i] << endl;
    //     cout << "new boxes: " << endl << tmp_new_boxes[i] << endl;
    //     cout << "new dist: " << org_dist[i] << endl;
    // }

    for(int i = 0; i < tmp_new_boxes.size(); i++){
        Vector4d box;
        box(0) = tmp_new_boxes[i].tl().x;
        box(1) = tmp_new_boxes[i].tl().y;
        box(2) = tmp_new_boxes[i].br().x;
        box(3) = tmp_new_boxes[i].br().y;

        new_boxes.push_back(box);
    }
    for(int i = 0; i < tmp_matches.size(); i++){
        new_matches.insert(make_pair(i, tmp_matches[i]));
        // matches.insert(make_pair(boxes.size() + i, tmp_matches[i]));
        cv::Rect rect = tmp_new_boxes[i];
        // cv::rectangle(img, rect, cv::Scalar(0, 255, 0), 3);
        // cv::putText(img, to_string(tmp_matches[i]), rect.tl(), cv::FONT_HERSHEY_SIMPLEX, 1.5, CV_RGB(255, 230, 0), 3);
    }
    // boxes.insert(boxes.end(), new_boxes.begin(), new_boxes.end());
}

bool CheckPose(const Eigen::Matrix3d& opt_Rwb, const Eigen::Vector3d& opt_pwb, const unordered_map<int, int>& matches, const vec_vec4d& boxes, double& avg_error){
    
    bool success;
    
    Eigen::Matrix3d delta_Rbb = Rwb.transpose() * opt_Rwb;
    Eigen::Vector3d delta_euler = delta_Rbb.eulerAngles(2, 1, 0);
    Eigen::Vector3d delta_pbb = Rwb.transpose() * (opt_pwb - pwb);
    Eigen::AngleAxisd delta_rbb(delta_Rbb);

    if(delta_rbb.angle() > delta_Rbb_th || delta_pbb.norm() > delta_pbb_th){
        success = false;
        // cout << "optimization failed, delta_Rbb: " << endl << delta_Rbb << endl << "delta_pbb: " << endl << delta_pbb << endl;
        fp_pos << "optimization failed, delta_Rbb: " << endl << delta_Rbb << endl << "delta_pbb: " << endl << delta_pbb << endl;
        fp_pos << "delta_angle: " << delta_rbb.angle() << " delta_dist: " << delta_pbb.norm() << endl;
    }
    else
        success = true;
    
    avg_error = 0.0;
    Eigen::Vector2d ori_n_error = Eigen::Vector2d::Zero();
    for(auto iter = matches.begin(); iter != matches.end(); ++iter){
        
        Eigen::Vector3d pt_3d_world = lamp_world_pos[iter->second];
        Eigen::Vector3d pt_3d_cam = Rcb * opt_Rwb.transpose() * (pt_3d_world - opt_pwb) + pcb;
        double inv_z = 1.0 / pt_3d_cam.z();
        pt_3d_cam = pt_3d_cam * inv_z;

        Eigen::Vector2d proj_pt;
        proj_pt.x() = cam_fx * pt_3d_cam.x() + cam_cx;
        proj_pt.y() = cam_fy * pt_3d_cam.y() + cam_cy;

        Vector4d box = boxes[iter->first];
        double off_x = 20;
        double off_y = 20;
        if(proj_pt.x() < box(0) - off_x || proj_pt.x() > box(2) + off_x || proj_pt.y() < box(1) - off_y || proj_pt.y() > box(3) + off_y){
            success = false;
            // cout << "optimization failed, proj_pt: " << proj_pt.x() << " " << proj_pt.y() << " box_pt: " << box << endl;
            fp_pos << "optimization failed, proj_pt: " << proj_pt.x() << " " << proj_pt.y() << " box_pt: " << box << endl;
        }
        
        Eigen::Vector3d pt_3d_ori_cam = Rcb * Rwb.transpose() * (pt_3d_world - pwb) + pcb;
        double inv_ori_z = 1.0 / pt_3d_ori_cam.z();
        pt_3d_ori_cam = pt_3d_ori_cam * inv_z;

        Eigen::Vector2d proj_ori_pt;
        proj_ori_pt.x() = cam_fx * pt_3d_ori_cam.x() + cam_cx;
        proj_ori_pt.y() = cam_fy * pt_3d_ori_cam.y() + cam_cy;

        ori_n_error += proj_ori_pt - proj_pt;

        Eigen::Vector2d center_pt;
        center_pt << 0.5 * (box(0) + box(2)), 0.5 * (box(1) + box(3));
        avg_error += (center_pt - proj_pt).norm();
        fp_pos << "proj_error: " << (center_pt - proj_pt).norm() << endl;
    }

    fp_match << "total_ori_n_error: " << ori_n_error.transpose() << endl;
    ori_n_error /= matches.size();
    fp_match << "avg_ori_n_error: " << ori_n_error.transpose() << endl;
    fp_match << "total_proj_error: " << avg_error << endl;
    avg_error /= matches.size();
    fp_match << "avg_proj_error: " << avg_error << endl;
    // avg_error = avg_error + 10 * delta_rbb.angle() + 3 * delta_pbb.norm();
    avg_error = avg_error + 100 * delta_rbb.angle() + 2 * delta_pbb.head<2>().norm();
    fp_match << "rotation_error: " << delta_rbb.angle() << endl;
    fp_match << "euler_error: " << delta_euler.transpose() << endl;
    fp_match << "translation_error_no_z: " << delta_pbb.head<2>().norm() << endl;
    fp_match << "translation_error: " << delta_pbb.norm() << endl;

    // if(ori_n_error.norm() > 150){
    //     avg_error += 15;
    // }
    if(ori_n_error.x() > 100 || ori_n_error.y() > 260){
        avg_error += 15;
    }

    return success;
}

bool CheckPose(const Eigen::Matrix3d& opt_Rwb, const Eigen::Vector3d& opt_pwb, const unordered_map<int, int>& matches, const vec_vec4d& boxes){
    
    bool success;
    
    Eigen::Matrix3d delta_Rbb = Rwb.transpose() * opt_Rwb;
    Eigen::Vector3d delta_pbb = Rwb.transpose() * (opt_pwb - pwb);
    Eigen::AngleAxisd delta_rbb(delta_Rbb);

    if(delta_rbb.angle() > delta_Rbb_th2 || delta_pbb.norm() > delta_pbb_th2){
        success = false;
        // cout << "optimization failed, delta_Rbb: " << endl << delta_Rbb << endl << "delta_pbb: " << endl << delta_pbb << endl;
        fp_pos << "update optimization failed, delta_Rbb: " << endl << delta_Rbb << endl << "delta_pbb: " << endl << delta_pbb << endl;
        fp_pos << "delta_angle: " << delta_rbb.angle() << " delta_dist: " << delta_pbb.norm() << endl;
    }
    else
        success = true;
    
    for(auto iter = matches.begin(); iter != matches.end(); ++iter){
        
        Eigen::Vector3d pt_3d_world = lamp_world_pos[iter->second];
        Eigen::Vector3d pt_3d_cam = Rcb * opt_Rwb.transpose() * (pt_3d_world - opt_pwb) + pcb;
        double inv_z = 1.0 / pt_3d_cam.z();
        pt_3d_cam = pt_3d_cam * inv_z;

        Eigen::Vector2d proj_pt;
        proj_pt.x() = cam_fx * pt_3d_cam.x() + cam_cx;
        proj_pt.y() = cam_fy * pt_3d_cam.y() + cam_cy;

        Vector4d box = boxes[iter->first];
        double off_x = 3;
        double off_y = 3;
        if(proj_pt.x() < box(0) - off_x || proj_pt.x() > box(2) + off_x || proj_pt.y() < box(1) - off_y || proj_pt.y() > box(3) + off_y){
            success = false;
            // cout << "optimization failed, proj_pt: " << proj_pt.x() << " " << proj_pt.y() << " box_pt: " << box << endl;
            fp_pos << "update optimization failed, proj_pt: " << proj_pt.x() << " " << proj_pt.y() << " box_pt: " << box << endl;
        }
        
        Eigen::Vector2d center_pt;
        center_pt << 0.5 * (box(0) + box(2)), 0.5 * (box(1) + box(3));
        fp_pos << "proj_error: " << (center_pt - proj_pt).norm() << endl;
    }

    return success;
}

void ReMatch(Match& match_result, InekfEstimator& estimator, const double timestamp, unordered_map<int, int>& matches, const vec_vec4d& boxes, double& avg_error){

    fp_match << "Match1 failed!!! Begin to rematch!!!" << endl; 
    vector<int> M = match_result.M;
    vector<int> result = match_result.hungary_result;
    Eigen::MatrixXd cost = match_result.cost_matrix;

    //统计每一行最小值和次小值的差异，并记录最小值的行、列id
    vector<pair<double, pair<int, int>>> diffs(M.size());
    for(int i = 0; i < cost.rows(); i++){
        //skip virtual rows
        if(i >= M.size())
            break;
        
        double min_cost = 10.0, sec_min_cost = 10.0;
        int min_colid;
        for(int j = 0; j < cost.cols(); j++){
            if(j > match_result.lamp_num)
                break;
            
            if(cost(i, j) < sec_min_cost && cost(i, j) > min_cost){
                sec_min_cost = cost(i, j);
            }
            else if(cost(i, j) < min_cost){
                sec_min_cost = min_cost;
                min_cost = cost(i, j);
                min_colid = j;
            }
        }

        diffs[i] = make_pair(sec_min_cost - min_cost , make_pair(i, min_colid));
    }
    vector<pair<double, pair<int, int>>> sorted_diffs = diffs;
    sort(sorted_diffs.begin(), sorted_diffs.end(), comp_dvec);


    int half_num = (match_result.box_num + 1) / 2;
    int turn = 1;
    int optimization_num = 1;
    bool success = false;
    unordered_map<int, int> tmp_matches = matches;
    double max_value = cost.maxCoeff();

    vector<vector<int>> combos;
    Combo(half_num, turn, combos);
    while(turn < half_num){
        for(int i = 0; i < combos.size(); i++){
            for(int j = 0; j < combos[i].size(); j++){
                int select_row = sorted_diffs[combos[i][j]].second.first;
                if(M[select_row] == -1)
                    continue;

                for(int col = 0; col < match_result.lamp_num; col++){
                    cost(select_row, col) = max_value;
                }
                for(int col = match_result.lamp_num; col < cost.cols(); col++){
                    cost(select_row, col) = 0.0;
                }
            }

            Hungary solver(cost.rows(), cost);
            result = solver.solve();

            tmp_matches.clear();
            for(int i = 0; i < match_result.box_num; i++){
                if(result[i] >= match_result.lamp_num)
                    M[i] = -1;
                else
                    M[i] = lamp_cur_id[result[i]];
                if(M[i] >= 0)
                    tmp_matches.insert(make_pair(i, M[i]));
            }

            estimator.stepCam(timestamp, tmp_matches, lamp_world_pos, boxes, high_lamp);
            Eigen::Matrix3d tmp_Rwb = estimator.getRotation();
            Eigen::Vector3d tmp_pwb = estimator.getPosition();
            success = CheckPose(tmp_Rwb, tmp_pwb, tmp_matches, boxes, avg_error);
            if(!success){
                ROS_WARN("optimization failed!");
                estimator.rollBack();
                fp_match << "----------------------------------------------------" << endl;
                fp_match << "the " << optimization_num << "th optimzation failed!" << endl;
                fp_match << "current turn is: " << turn << endl;
                fp_match << "current match_result is: " << endl;
                for(int i = 0; i < M.size(); i++){
                    fp_match << "the " << i << "th bounding box corresponds to lamp " << M[i] << endl;
                }
                fp_match << "current cost is: " << endl << cost << endl;
                ++optimization_num;
                cost = match_result.cost_matrix;
            }
            else{
                ROS_WARN("optimization success!");
                fp_match << "++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
                fp_match << "the " << optimization_num << "th optimization succeded!" << endl;

                match_result.M = M;
                match_result.hungary_result = result;
                match_result.cost_matrix = cost;
                matches.clear();
                matches = tmp_matches;
                fp_match << "current match_result is: " << endl;
                for(int i = 0; i < M.size(); i++){
                    fp_match << "the " << i << "th bounding box corresponds to lamp " << M[i] << endl;
                }
                break;
            }
        }
        if(success){
            break;
        }
        else{
            ++turn;
            combos.clear();
            Combo(half_num, turn, combos);
        }
    }
    if(success){
        if(parings(M) < 2){

        }
    }
    else{
        ROS_WARN("Rematch1 failed! clear all matches");
        fp_match << "Rematch1 failed! clear all matches!!" << endl;
        for(int i = 0; i < M.size(); i++){
            M[i] = -1;
        }
        matches.clear();
    }

}

void ReMatch(InekfEstimator& estimator, const double timestamp, unordered_map<int, int>& matches, vec_vec4d& boxes){

    fp_match << "Match2 failed!!! Begin to rematch!!!" << endl; 

    //统计两次位姿投影差异最大的点，依次排序并将其去掉
    vector<pair<double, int>> errors;
    for(auto iter = matches.begin(); iter != matches.end(); ++iter){
        
        Eigen::Vector3d pt_3d_world = lamp_world_pos[iter->second];
        Eigen::Vector3d pt_3d_cam = Rcb * Rwb.transpose() * (pt_3d_world - pwb) + pcb;
        double inv_z = 1.0 / pt_3d_cam.z();
        pt_3d_cam = pt_3d_cam * inv_z;

        Eigen::Vector2d proj_pt;
        proj_pt.x() = cam_fx * pt_3d_cam.x() + cam_cx;
        proj_pt.y() = cam_fy * pt_3d_cam.y() + cam_cy;

        Vector4d box = boxes[iter->first];
        
        Eigen::Vector2d center_pt;
        center_pt << 0.5 * (box(0) + box(2)), 0.5 * (box(1) + box(3));
        errors.push_back(make_pair((center_pt - proj_pt).norm(), iter->first));
    }

    sort(errors.begin(), errors.end(), comp_vdp_down);
    vector<bool> delete_box(boxes.size(), false);
    bool rematch_suc = false;
    for(int i = 0; i < errors.size() - 1; ++i){
        delete_box[errors[i].second] = true;
        vec_vec4d tmp_boxes;
        unordered_map<int, int> tmp_matches;
        for(int j = 0; j < boxes.size(); j++){
            if(!delete_box[j]){
                tmp_boxes.push_back(boxes[errors[i].second]);
                tmp_matches.insert(make_pair(tmp_boxes.size() - 1, matches[errors[i].second]));
            }
        }
        estimator.stepCam(data.timestamp, tmp_matches, lamp_world_pos, tmp_boxes, high_lamp);
        Eigen::Matrix3d tmp_Rwb = estimator.getRotation();
        Eigen::Vector3d tmp_pwb = estimator.getPosition();
        if(CheckPose(tmp_Rwb, tmp_pwb, tmp_matches, tmp_boxes)){
            fp_match << "Rematch2 success!!!" << endl; 
            matches = tmp_matches;
            boxes = tmp_boxes;
            rematch_suc = true;
            break;
        }
        else{
            estimator.rollBack();
        }
    }
    if(!rematch_suc){
        fp_match << "Rematch2 failed!!!" << endl;
        matches.clear();
        boxes.clear();
    }
}

void MatchFilter(unordered_map<int, int>& matches, vec_vec4d& boxes, double& th1, double& th2){
    if(matches.empty()) return;
    vec_vec4d new_boxes;
    unordered_map<int, int> new_matches;
    for(int i = 0; i < boxes.size(); ++i){
        int index = matches[i];
        Eigen::Vector3d cur_Pc = Rcw * lamp_world_pos[index] + pcw;
        bool multi_points = false;
        for(int j = 0; j < lamp_world_pos.size(); j++){
            if(isnan(lamp_world_pos[j].norm())) continue;
            if(lamp_world_points[j].size() <= 0) continue;
            if(j == index) continue;

            Eigen::Vector3d Pc;
            Pc = Rcw * lamp_world_pos[j] + pcw;
            Eigen::Vector2d pt;
            double inv_z = 1.0 / Pc.z();
            pt << cam_fx * Pc.x() * inv_z + cam_cx, cam_fy * Pc.y() * inv_z + cam_cy;
            
            int max_norm_th, max_z_th, min_norm_th, min_z_th;
            max_norm_th = th1;
            max_z_th = th2;
            min_norm_th = 0;
            min_z_th = 0;
            if (Pc.norm() < max_norm_th + 25 && Pc.norm() > min_norm_th && Pc.z() < max_z_th + 25 && Pc.z() > min_z_th && pt.x() < boxes[i][2]+1 && pt.x() >= boxes[i][0]-1 && pt.y() < boxes[i][3]+1 && pt.y() >= boxes[i][1] - 1){ //70, 55
                bool find_occ = false;
                //查看当前点是否被遮挡，首先和已经匹配上的点比较
                Eigen::Vector3d l1 = cur_Pc.normalized();
                Eigen::Vector3d l2 = Pc.normalized();
                if((l1.transpose() * l2)(0) > 0.9999)
                    find_occ = true;

                if(find_occ){
                    continue;
                }
                else{
                //存在检测框被扩大了
                    multi_points = true;
                }
            }            
        }
        if(!multi_points){
            new_boxes.push_back(boxes[i]);
            new_matches.insert(make_pair(new_boxes.size() - 1, index));
        }
    }
    boxes = new_boxes;
    matches = new_matches;
}

vector<int> Relocalization(Measures& data, unordered_map<int, int>& matches, vec_vec4d& boxes, InekfEstimator& estimator){
    if(data.box->bounding_boxes.size() == 0){
        ROS_WARN("no boxes detected!");
        fp_match << "no boxes detected!" << endl;
        return vector<int>();
    }

    for(int i = 0; i < data.box->bounding_boxes.size(); i++){
        Vector4d box;
        box << data.box->bounding_boxes[i].xmin, data.box->bounding_boxes[i].ymin, data.box->bounding_boxes[i].xmax, data.box->bounding_boxes[i].ymax;
        fp_match << "boxes: " << box[0] << " " << box[1] << " " << box[2] << " " << box[3] << endl;
        boxes.push_back(box);
    }

    //add new boxes
    cv::Mat grey_img(data.img.rows, data.img.cols, CV_8UC1);
    cv::cvtColor(data.img, grey_img, cv::COLOR_BGR2GRAY);
    cv::Mat bin_img;
    cv::threshold(grey_img, bin_img, 252, 255, cv::THRESH_BINARY);

    vector<cv::Rect> tmp_boxes;
    vector<vector<cv::Point>> contours;
    cv::findContours(bin_img, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    for(int i = 0; i < contours.size(); i++){
        cv::Rect rect = cv::boundingRect(contours[i]);
        // cout << rect.x << " " << rect.y << " " << rect.width << " " << rect.height << endl;

        int area = rect.width * rect.height;
        if(area < 60 || rect.width < 6 || rect.height < 6) // 80, 10, 10
            continue;

        int light_nei_v = 0;
        for(int v = rect.y; v < rect.y + rect.height; ++v){
            float light_intensity = grey_img.ptr<uchar>(v)[rect.x + rect.width / 2];
            if(light_intensity > 245) light_nei_v++;
        }
        // if(light_nei_v / float(rect.width + 1) < 0.8 && area > 400)
        //     continue;

        if(rect.x != 0 && rect.y != 0 && rect.width != bin_img.cols && rect.height != bin_img.rows){
            tmp_boxes.push_back(rect);
            // cv::rectangle(img, rect, cv::Scalar(255, 255, 255), 3);
            fp_match << "tmp_boxes: " << rect.x << " " << rect.y << " " << rect.width << " " << rect.height << endl;
        }
    }

    //去除重复的检测框
    auto iter_box1 = tmp_boxes.begin();
    while(iter_box1 != tmp_boxes.end()){
        bool find_rep = false;
        for(auto iter_box2 = boxes.begin(); iter_box2 != boxes.end(); ++iter_box2){
            cv::Rect old_box(cv::Point2i(int((*iter_box2)(0)), int((*iter_box2)(1))), cv::Point2i(int((*iter_box2)(2)), int((*iter_box2)(3))));
            if(((*iter_box1) & old_box).area() > 0.25 * iter_box1->area() || ((*iter_box1) & old_box).area() > 0.25 * old_box.area()){
                find_rep = true;
                break;
            }
        }
        if(find_rep){
            iter_box1 = tmp_boxes.erase(iter_box1);
        }
        else{
            ++iter_box1;
        }
    }
    for(int i = 0; i < tmp_boxes.size(); ++i){
        cv::Rect rect = tmp_boxes[i];
        fp_match << "tmp_boxes after filtering: " << rect.x << " " << rect.y << " " << rect.width << " " << rect.height << endl;
        Vector4d box;
        box << rect.tl().x, rect.tl().y, rect.br().x, rect.br().y;
        boxes.push_back(box);
    }

    for (int i = 0; i < lamp_world_pos.size(); i++){
        if(isnan(lamp_world_pos[i].norm())) continue;
        if(lamp_world_points[i].size() <= 0) continue;
        Eigen::Vector3d Pc;
        Pc = Rcw * lamp_world_pos[i] + pcw;
        Eigen::Vector2d pt;
        double inv_z = 1.0 / Pc.z();
        pt << cam_fx * Pc.x() * inv_z + cam_cx, cam_fy * Pc.y() * inv_z + cam_cy;
        if ((sqrt(Pc.x() * Pc.x() + Pc.z() * Pc.z()) < reloc_dist_th) && Pc.z() < reloc_z_th && Pc.z() > -0.5 && pt.x() < res_x && pt.x() >= 0 && pt.y() < res_y && pt.y() >=0){
            bool find_rep = false;
            for (int j = 0; j < lamp_cur_cam_pos.size(); j++){
                Eigen::Vector3d l1 = lamp_cur_cam_pos[j].normalized();
                Eigen::Vector3d l2 = Pc.normalized();
                // cout << "cos_theta: " << l1.transpose() * l2 << endl;
                if((l1.transpose() * l2)(0) > 0.995){
                    double n1 = lamp_cur_cam_pos[j].norm();
                    double n2 = Pc.norm();
                    find_rep = true;
                    if(n1 > n2){
                        lamp_cur_cam_pos[j] = Pc;
                        lamp_cur_id[j] = i;
                        lamp_cur_world_pos[j] = lamp_world_pos[i];
                    }
                    break;
                }
            }
            if(find_rep) continue;
            lamp_cur_cam_pos.push_back(Pc);
            lamp_cur_id.push_back(i);
            lamp_cur_world_pos.push_back(lamp_world_pos[i]);
        }
    }

    if(lamp_cur_cam_pos.size() == 0){
        ROS_WARN("no lamps around!");
        fp_match << "no lamps around!" << endl;
        return vector<int>();
    }

    // int box_num = data.box->bounding_boxes.size();
    int box_num = boxes.size(), lamp_num = lamp_cur_cam_pos.size();
    vector<int> M, miM;
    M.resize(box_num);
    unordered_map<int, int> mimatches;
    double mierror = -1;

    fp_match << "Begin to BF match! " << endl;

    int num = min(lamp_num, box_num);
    for(int i1 = 1; i1 <= num; ++i1){
        vector<vector<int>> combos;
        Combo(lamp_num, i1, combos);
        vector<vector<int>> combos_box;
        Combo(box_num, i1, combos_box);
        for(int j1 = 0; j1 < combos.size(); j1++){
            vector<vector<int>> perms = Permute(combos[j1]);
            for(int k1 = 0; k1 < perms.size(); k1++){
                for(int j2 = 0; j2 < combos_box.size(); j2++){
                    assert(perms[k1].size() == combos_box[j2].size());
                    int l1 = 0, l2 = 0;//begin to make pairs(lamp_idx, box_idx)
                    while(l1 != perms[k1].size() || l2 != box_num){
                        if(std::find(combos_box[j2].begin(), combos_box[j2].end(), l2) == combos_box[j2].end()){
                            M[l2] = -1;
                            fp_match << "box: " << l2 << endl << " corresponds to lamp " << "-1" << endl;
                            ++l2;    
                        }
                        else{
                            matches.insert(make_pair(l2, lamp_cur_id[perms[k1][l1]]));
                            M[l2] = lamp_cur_id[perms[k1][l1]];
                            fp_match << "box: " << l2 << endl << boxes[l2] << endl << " corresponds to lamp " << lamp_cur_id[perms[k1][l1]] << endl;
                            ++l2;
                            ++l1;
                        }
                    }
                    estimator.stepCam(data.timestamp, matches, lamp_world_pos, boxes, high_lamp);
                    Eigen::Matrix3d tmp_Rwb = estimator.getRotation();
                    Eigen::Vector3d tmp_pwb = estimator.getPosition();
                    double avg_error;
                    CheckPose(tmp_Rwb, tmp_pwb, matches, boxes, avg_error);
                    // avg_error = avg_error + 4 * (lamp_num - matches.size());
                    if(last_best_match){
                        avg_error = avg_error + 15 * (lamp_num - matches.size());
                    }
                    else{
                        avg_error = avg_error + 8 * (lamp_num - matches.size());
                    }
                    fp_match << "avg_error: " << avg_error << endl;
                    if(avg_error < 35){
                        if(mierror < 0){
                            mierror = avg_error;
                            miM = M;
                            mimatches = matches;
                        }
                        else{
                            if(mierror > avg_error){
                                mierror = avg_error;
                                miM = M;
                                mimatches = matches;
                            }
                        }
                    }
                    matches.clear();
                    estimator.rollBack();
                }
            }
        }
    }
    if(mierror > 0){
        fp_match << "Best match found!" << endl;
        for(int i = 0; i < miM.size(); ++i){
            fp_match << "box: " << i << endl << boxes[i] << endl << " corresponds to lamp " << miM[i] << endl;
        }
        matches = mimatches;
        M = miM;
        for(int i = data.box->bounding_boxes.size(); i < M.size(); i++){
            map_relocalization::BoundingBox bbox;
            bbox.xmin = boxes[i](0), bbox.xmax = boxes[i](2), bbox.ymin = boxes[i](1), bbox.ymax = boxes[i](3);
            data.box->bounding_boxes.push_back(bbox);
        }
        last_best_match = true;
        return M;
    }
    else{
        fp_match << "No match found! Reloc failed!" << endl;
        matches.clear();
        M.clear();
        last_best_match = false;
        return M;
    }

    // fp_match << "Begin to BF match! " << endl;


    // for(int i = 0; i < combos.size(); i++){
    //     fp_match << "combo: " << i << endl;
    //     for(int j = 0; j < combos[i].size(); j++){
    //         fp_match << combos[i][j] << " ";
    //     }
    //     fp_match << endl;
    //     for(int j = 0; j < combos[i].size(); j++){
    //         if(combos[i][j] < lamp_cur_cam_pos.size()){
    //             matches.insert(make_pair(j, lamp_cur_id[combos[i][j]]));
    //             M[j] = lamp_cur_id[combos[i][j]];
    //             fp_match << "box: " << j << endl << boxes[j] << endl << " corresponds to lamp " << lamp_cur_id[combos[i][j]] << endl;
    //         }
    //         else{
    //             M[j] = -1;
    //             fp_match << "box: " << j << endl << boxes[j] << endl << " corresponds to -1" << endl;
    //         }
    //     }
    //     estimator.stepCam(data.timestamp, matches, lamp_world_pos, boxes, high_lamp);
    //     Eigen::Matrix3d tmp_Rwb = estimator.getRotation();
    //     Eigen::Vector3d tmp_pwb = estimator.getPosition();
        
    //     double avg_error;
    //     CheckPose(tmp_Rwb, tmp_pwb, matches, boxes, avg_error);
    //     avg_error = avg_error + 4 * (M.size() - parings(M));
    //     fp_match << "avg_error: " << avg_error << endl;
    //     if(avg_error < 25){
    //         if(mierror < 0){
    //             mierror = avg_error;
    //             miM = M;
    //             mimatches = matches;
    //         }
    //         else{
    //             if(mierror > avg_error){
    //                 mierror = avg_error;
    //                 miM = M;
    //                 mimatches = matches;
    //             }
    //         }
    //     }
    //     matches.clear();
    //     estimator.rollBack();
    // }
    // if(mierror > 0){
    //     fp_match << "Best match found!" << endl;
    //     for(int i = 0; i < miM.size(); ++i){
    //         fp_match << "box: " << i << endl << boxes[i] << endl << " corresponds to lamp " << miM[i] << endl;
    //     }
    //     matches = mimatches;
    //     M = miM;
    //     return M;
    // }
    // else{
    //     fp_match << "No match found! Reloc failed!" << endl;
    //     matches.clear();
    //     return M;
    // }
}

void SlideWindow(){
    for(int i = 0; i < WINDOW_SIZE; i++){
        deq_Rcw[i] = deq_Rcw[i + 1];
        deq_pcw[i] = deq_pcw[i + 1];
        deq_P[i] = deq_P[i + 1];
        deq_deltap[i] = deq_deltap[i + 1];
        deq_deltaR[i] = deq_deltaR[i + 1];
        deq_matches[i] = deq_matches[i + 1];
        deq_boxes[i] = deq_boxes[i + 1];
        deq_id[i] = deq_id[i + 1];
        // deq_features[i] = deq_features[i + 1];
    }
    vec_vec4d().swap(deq_boxes[WINDOW_SIZE]);
    deq_matches[WINDOW_SIZE].clear();
    // deq_features[WINDOW_SIZE].clear();
}

int main(int argc, char** argv){
    ros::init(argc, argv, "map_relocalization");
    ros::NodeHandle nh;

    //读取点云数据以及相关参数
    readParameters(nh);

    string log_state_txt = root_dir + "/log/pos.txt";
    string log_matches_txt = root_dir + "/log/matches.txt";
    string log_imu_pos1_txt = root_dir + "/log/imu_pos_post.txt";
    string log_imu_pos2_txt = root_dir + "/log/imu_pos_pre.txt";
    string log_imu_cov_txt = root_dir + "/log/imu_cov_post.txt";
    fp_pos = ofstream(log_state_txt, ios::trunc);
    fp_match = ofstream(log_matches_txt, ios::trunc);
    fp_cov = ofstream(log_imu_cov_txt, ios::trunc);
    fp_log1 = fopen(log_imu_pos1_txt.c_str(), "w");
    fp_log2 = fopen(log_imu_pos2_txt.c_str(), "w");

    ros::Subscriber sub_img = nh.subscribe(img_topic, 200000, img_cbk);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
    ros::Subscriber sub_box = nh.subscribe(box_topic, 200000, box_cbk);
    ros::Subscriber sub_odom = nh.subscribe(odom_topic, 200000, odom_cbk);
    ros::Subscriber sub_gps = nh.subscribe(gps_topic, 200000, gps_cbk);
    ros::Subscriber sub_vins_path = nh.subscribe("/vins_estimator/odometry", 200000, path_cbk);
    pub_vins_wheel = nh.advertise<nav_msgs::Path>("/aligned_vins_path", 100000);
    gps_buffer.resize(1);

    ros::Publisher pub_detect_img = nh.advertise<sensor_msgs::Image>("/detect_img", 100000);
    ros::Publisher pub_detect_project_img = nh.advertise<sensor_msgs::Image>("/detect_project_img", 100000);
    ros::Publisher pub_flow_img = nh.advertise<sensor_msgs::Image>("/flow_img", 100000);
    ros::Publisher pub_epipline_img = nh.advertise<sensor_msgs::Image>("/epipline_img", 100000);
    ros::Publisher pub_pointcloud = nh.advertise<sensor_msgs::PointCloud2>("/lamp", 100000);
    ros::Publisher pub_odom_path = nh.advertise<nav_msgs::Path>("/odom_path", 100000);
    ros::Publisher pub_optimized_path = nh.advertise<nav_msgs::Path>("/optimized_path", 100000);
    ros::Publisher pub_initial_path = nh.advertise<nav_msgs::Path>("/initial_path", 100000);
    ros::Publisher pub_cur_pointcloud = nh.advertise<sensor_msgs::PointCloud2>("/cur_lamp", 100000);
    ros::Publisher pub_particles = nh.advertise<geometry_msgs::PoseArray>("/particle_cloud", 100000);
    ros::Publisher pub_optimal_particle = nh.advertise<geometry_msgs::PoseArray>("/optimal_particle", 100000);
    ros::Publisher pub_selected_particles = nh.advertise<geometry_msgs::PoseArray>("/selected_particles", 100000);
    ros::Publisher pub_optimized_camera_visual = nh.advertise<visualization_msgs::MarkerArray>("/optimized_pose_visual", 100000);
    pub_vins_wheel_visual = nh.advertise<visualization_msgs::MarkerArray>("/vins_wheel_visual", 100000);

    global_odom_path.header.stamp = ros::Time::now();
    global_odom_path.header.frame_id = "camera_init";
    global_optimized_path.header.stamp = ros::Time::now();
    global_optimized_path.header.frame_id = "camera_init";
    global_vins_wheel_path.header.stamp = ros::Time::now();
    global_vins_wheel_path.header.frame_id = "camera_init";

    Rwc = Eigen::Matrix3d::Identity();
    pwc = Eigen::Vector3d::Zero();
    cov = Matrix6d::Zero();

    CameraPoseVisualization cam(1, 0, 0, 1);

    InekfEstimator estimator1, estimator2;
    cout << "Use Imu as body coordinate!" << endl;
    // Rcb <<  0, -1,  0,
    //         0,  0,  -1,
    //         1,  0,  0;
    // pcb = Eigen::Vector3d::Zero();
    // Rcb << imu_params.camera_odom_rot[0], imu_params.camera_odom_rot[1], imu_params.camera_odom_rot[2],
    //     imu_params.camera_odom_rot[3], imu_params.camera_odom_rot[4], imu_params.camera_odom_rot[5],
    //     imu_params.camera_odom_rot[6], imu_params.camera_odom_rot[7], imu_params.camera_odom_rot[8];
    // pcb << imu_params.camera_odom_pos[0], imu_params.camera_odom_pos[1], imu_params.camera_odom_pos[2];
    Rcb << imu_params.camera_imu_rot[0], imu_params.camera_imu_rot[1], imu_params.camera_imu_rot[2],
            imu_params.camera_imu_rot[3], imu_params.camera_imu_rot[4], imu_params.camera_imu_rot[5],
            imu_params.camera_imu_rot[6], imu_params.camera_imu_rot[7], imu_params.camera_imu_rot[8];
    pcb << imu_params.camera_imu_pos[0], imu_params.camera_imu_pos[1], imu_params.camera_imu_pos[2];
    estimator1.setEstimator(imu_params);
    estimator1.setWheelVelCov(odom_params.cov_vel);
    estimator1.setBodyImuExt(imu_params.odom_imu_rot);
    estimator2.setEstimator(imu_params);
    estimator2.setWheelVelCov(odom_params.cov_vel);
    estimator2.setBodyImuExt(imu_params.odom_imu_rot);

    Rbc = Rcb.transpose();
    pbc = - Rbc * pcb;

    Rwb = Eigen::Matrix3d::Identity();
    pwb = Eigen::Vector3d::Zero();
    Rwb2 = Eigen::Matrix3d::Identity();
    pwb2 = Eigen::Vector3d::Zero();
    cov_wb = Matrix6d::Zero();
    cov_wb2 = Matrix6d::Zero();

    tf::TransformBroadcaster br;
    tf::Transform transform;
    // cout << "Rcb: " << Rcb << endl << "pcb: " << pcb << endl;

    if(need_init)
        window_state = WindowState::POSE_INITIAL;
    else
        window_state = WindowState::IMU_INITIAL;

    // FeatureTracker tracker(res_x, res_y, 30.0, 50, K_inv, K);
    cv::Mat op_img, detect_project_img;
    // FeatureManager f_manager(min_parallex / max(cam_fx, cam_fy));

    Sampler sampler(30, search_dist_scope, search_z_scope);

    signal(SIGINT, SigHandle);
    bool status = ros::ok();
    ros::Rate rate(5000);
    while(status){
        //位姿初始化
        // Initialization();
        if (flg_exit) break;
        ros::spinOnce();
        
        if(use_gps){
            if (!sampler.initFinish && gps_buffer[0] && need_init){
                GeographicLib::LocalCartesian local_cartesian;
                local_cartesian.Reset(gps_init_x, gps_init_y, gps_init_z);
                local_cartesian.Forward(gps_buffer[0]->latitude, gps_buffer[0]->longitude, gps_buffer[0]->altitude, init_x, init_y, init_z);
                // cout << "gps_init_x: " << gps_init_x << " gps_init_y: " << gps_init_y << endl;
                cout << "gps_x: " << gps_buffer[0]->latitude << " gps_y: " << gps_buffer[0]->longitude << endl;
                // cout << "init_x: " << init_x << " init_y: " << init_y << endl;

                double noise_x = gps_buffer[0]->position_covariance[0];
                double noise_y = gps_buffer[0]->position_covariance[4];
                double noise_z = gps_buffer[0]->position_covariance[8];

                // Eigen::Matrix3d Rigps;
                // Rigps << cos(yaw_off), sin(yaw_off), 0,
                //          -sin(yaw_off), cos(yaw_off),0,
                //          0,            0,            1;
                
                // Eigen::Matrix3d Roi;   
                // Roi << imu_params.odom_imu_rot[0], imu_params.odom_imu_rot[1], imu_params.odom_imu_rot[2],
                //         imu_params.odom_imu_rot[3], imu_params.odom_imu_rot[4], imu_params.odom_imu_rot[5],
                //         imu_params.odom_imu_rot[6], imu_params.odom_imu_rot[7], imu_params.odom_imu_rot[8];

                // Eigen::Matrix3d Rogps = Roi * Rigps;
                
                // Eigen::Vector3d poi;
                // poi << imu_params.odom_imu_pos[0], imu_params.odom_imu_pos[1], imu_params.odom_imu_pos[2];

                // Eigen::Vector3d init_pos_odom = Rogps * Eigen::Vector3d(init_x, init_y, init_z);

                // Eigen::Matrix3d gps_noise = Eigen::Vector3d(noise_x, noise_y, noise_z).asDiagonal();
                // Eigen::Matrix3d odom_noise = Rogps * gps_noise * Rogps.transpose();
                sampler.init(lamp_world_pos, num_particles, yaw_off, init_x, noise_x, init_y, noise_y, init_alpha, init_beta, init_gamma);
                // sampler.init(lamp_world_pos, num_particles, init_pos_odom, odom_noise, init_alpha, init_beta, init_gamma);
            }
            else if(!gps_buffer[0]){
                goto nogps;
            }
        }
        else{
            if (!sampler.initFinish && need_init){
                sampler.init(lamp_world_pos, num_particles, init_x, init_y, init_z, init_alpha, init_beta, init_gamma);
            }
        }

        if (sync_packages(data)){
            fp_pos << "timestamp: " << setprecision(20) << data.timestamp << "  " << "frame_id: " << global_id << " frame_count: " << frame_count << endl;
            fp_match << "timestamp: " << setprecision(20) << data.timestamp << "  " << "frame_id: " << global_id << " frame_count: " << frame_count << endl;

            if(need_init)
                Preprocess(data, sampler.initializationSuc()); //预先滤除不合格的bounding box
            else
                Preprocess(data);

            Match match_result;
            vector<int> M;
            vec_vec2d whole_box_centers;
            vec_vec4d whole_box_corners;
            if (window_state == WindowState::POSE_INITIAL){
                if(!sampler.initializationSuc()){
                    if(use_gps){
                        for(int i = 0; i <=6; ++i){
                            sampler.updateWeights(data, lamp_world_pos, Rcb, pcb, res_x, res_y);
                            sampler.sortWeight();

                            vector<Sample> samples = sampler.best_samples;
                            Sample best_sample = samples[0];

                            Eigen::Matrix3d best_Rwb;
                            tf2::Quaternion q;
                            q.setRPY(best_sample.rot(2), best_sample.rot(1), best_sample.rot(0));
                            geometry_msgs::Quaternion ori;
                            tf2::convert(q, ori);
                            Eigen::Quaterniond q_eig;
                            q_eig.w() = ori.w;
                            q_eig.x() = ori.x;
                            q_eig.y() = ori.y;
                            q_eig.z() = ori.z;
                            best_Rwb = q_eig.toRotationMatrix();
                            // best_Rwb = Eigen::AngleAxisd(best_sample.rot[0], Eigen::Vector3d::UnitZ()) * 
                            //             Eigen::AngleAxisd(best_sample.rot[1], Eigen::Vector3d::UnitY()) * 
                            //             Eigen::AngleAxisd(best_sample.rot[2], Eigen::Vector3d::UnitX());
                            
                            Eigen::Vector3d best_pwb = best_sample.pos;
                            cout << "best Rwb: " << best_Rwb << endl;
                            cout << "best pwb: " << best_pwb.transpose() << endl;

                            Rwb = best_Rwb;
                            pwb = best_pwb;

                            ++initialization_id;

                            if(initialization_id >= 6){
                                cout << "Initialization completed!" << endl;
                                unordered_map<int, int> matches;
                                vector<int> M = best_sample.associations;
                                for(int i = 0; i < M.size(); i++){
                                    if(M[i] >= 0){
                                        matches.insert(make_pair(i, M[i]));
                                    }
                                }
                                vec_vec4d boxes;
                                boxes.resize(data.box->bounding_boxes.size());
                                for(int i = 0; i < boxes.size(); i++){
                                    Vector4d box;
                                    box(0) = data.box->bounding_boxes[i].xmin;
                                    box(1) = data.box->bounding_boxes[i].ymin;
                                    box(2) = data.box->bounding_boxes[i].xmax;
                                    box(3) = data.box->bounding_boxes[i].ymax;
                                    boxes[i] = box;
                                }

                                BodyPoseToCameraPose(Rwb, pwb, Matrix6d::Zero());
                                Optimizer::PoseOptimizationSingle(Rcw, pcw, matches, boxes, lamp_world_pos, Matrix6d::Identity());
                                CameraPoseToBodyPose();

                                // cout << "init Rwb: " << Rwb << endl;
                                // cout << "init pwb: " << pwb.transpose() << endl;

                                // estimator1.ReInitializePose(Rwb, pwb, data.timestamp);
                                // estimator2.ReInitializePose(Rwb, pwb, data.timestamp);

                                sampler.setIntializedPose();
                                Rwb2 = Rwb;
                                pwb2 = pwb;

                                init_Rwb = Rwb;
                                init_pwb = pwb;

                                fp_pos << "Rotation matrix: " << endl << Rwb << endl;
                                fp_pos << "Translation vector: " << endl << pwb << endl;
                                deq_Rcw[frame_count] = Rcw;
                                deq_pcw[frame_count] = pcw;
                                deq_deltaR[frame_count] = Eigen::Matrix3d::Identity();
                                deq_deltap[frame_count] = Eigen::Vector3d::Zero();
                                deq_matches[frame_count] = matches;
                                deq_boxes[frame_count] = boxes;
                                deq_id[frame_count] = global_id;

                                transform.setOrigin(tf::Vector3(init_pwb.x(), init_pwb.y(), init_pwb.z()));
                                Eigen::Quaterniond init_qwb(init_Rwb);
                                transform.setRotation(tf::Quaternion(init_qwb.x(), init_qwb.y(), init_qwb.z(), init_qwb.w()));
                                br.sendTransform(tf::StampedTransform(transform, ros::Time(data.timestamp), "camera_init", "move_camera"));

                                ++frame_count;
                                window_state = WindowState::IMU_INITIAL;
                                publish_samples_pose(pub_particles, pub_optimal_particle, best_Rwb, best_pwb, Rwb, pwb);
                                publish_selected_samples(pub_selected_particles, sampler);
                                // cout << "rot: " << best_sample.rot.transpose() << endl;
                                // cout << "pos: " << best_sample.pos.transpose() << endl;
                                // for(int i = 0; i < sampler.best_samples.size(); ++i){
                                //     cout << "roti: " << sampler.best_samples[i].rot.transpose() << endl;
                                //     cout << "posi: " << sampler.best_samples[i].pos.transpose() << endl;
                                // }
                            }
                            else{
                                BodyPoseToCameraPose(Rwb, pwb, Matrix6d::Zero());
                                sampler.resample(initialization_id);
                                publish_samples_pose(pub_particles, sampler);
                                publish_selected_samples(pub_selected_particles, sampler);
                            }

                            // 查看投影结果
                            vector<int> M = best_sample.associations;
                            cv::Mat detect_image, detect_project_image;
                            DetectAndProjectImage(data, detect_image, detect_project_img, M);

                            // publish_optimized_path(pub_optimized_path);
                            publish_image(pub_detect_img, detect_image);
                            publish_image(pub_detect_project_img, detect_project_image);
                            publish_image(pub_flow_img, op_img);
                            publish_pointcloud(pub_cur_pointcloud, true);

                            ++global_id;
                            data.clear();
                            data.last_timestamp = data.timestamp;                        
                        }
                    }
                    else{
                        sampler.updateWeights(data, lamp_world_pos, Rcb, pcb, res_x, res_y);
                        sampler.sortWeight();

                        vector<Sample> samples = sampler.best_samples;
                        Sample best_sample = samples[0];

                        Eigen::Matrix3d best_Rwb;
                        tf2::Quaternion q;
                        q.setRPY(best_sample.rot(2), best_sample.rot(1), best_sample.rot(0));
                        geometry_msgs::Quaternion ori;
                        tf2::convert(q, ori);
                        Eigen::Quaterniond q_eig;
                        q_eig.w() = ori.w;
                        q_eig.x() = ori.x;
                        q_eig.y() = ori.y;
                        q_eig.z() = ori.z;
                        best_Rwb = q_eig.toRotationMatrix();

                        Eigen::Vector3d best_pwb = best_sample.pos;
                        cout << "best Rwb: " << best_Rwb << endl;
                        cout << "best pwb: " << best_pwb.transpose() << endl;

                        Rwb = best_Rwb;
                        pwb = best_pwb;

                        cout << "Initialization completed!" << endl;
                        unordered_map<int, int> matches;
                        vector<int> M = best_sample.associations;
                        for(int i = 0; i < M.size(); i++){
                            if(M[i] >= 0){
                                matches.insert(make_pair(i, M[i]));
                            }
                        }
                        vec_vec4d boxes;
                        boxes.resize(data.box->bounding_boxes.size());
                        for(int i = 0; i < boxes.size(); i++){
                            Vector4d box;
                            box(0) = data.box->bounding_boxes[i].xmin;
                            box(1) = data.box->bounding_boxes[i].ymin;
                            box(2) = data.box->bounding_boxes[i].xmax;
                            box(3) = data.box->bounding_boxes[i].ymax;
                            boxes[i] = box;
                        }

                        BodyPoseToCameraPose(Rwb, pwb, Matrix6d::Zero());
                        Optimizer::PoseOptimizationSingle(Rcw, pcw, matches, boxes, lamp_world_pos, Matrix6d::Identity());
                        CameraPoseToBodyPose();

                        sampler.setIntializedPose();
                        Rwb2 = Rwb;
                        pwb2 = pwb;

                        init_Rwb = Rwb;
                        init_pwb = pwb;

                        fp_pos << "Rotation matrix: " << endl << Rwb << endl;
                        fp_pos << "Translation vector: " << endl << pwb << endl;
                        deq_Rcw[frame_count] = Rcw;
                        deq_pcw[frame_count] = pcw;
                        deq_deltaR[frame_count] = Eigen::Matrix3d::Identity();
                        deq_deltap[frame_count] = Eigen::Vector3d::Zero();
                        deq_matches[frame_count] = matches;
                        deq_boxes[frame_count] = boxes;
                        deq_id[frame_count] = global_id;

                        ++frame_count;
                        window_state = WindowState::IMU_INITIAL;
                        publish_samples_pose(pub_particles, pub_optimal_particle, best_Rwb, best_pwb, Rwb, pwb);
                        publish_selected_samples(pub_selected_particles, sampler);

                        // vector<int> M = best_sample.associations;
                        cv::Mat detect_image, detect_project_image;
                        DetectAndProjectImage(data, detect_image, detect_project_img, M);

                        // publish_optimized_path(pub_optimized_path);
                        publish_image(pub_detect_img, detect_image);
                        publish_image(pub_detect_project_img, detect_project_img);
                        publish_image(pub_flow_img, op_img);
                        publish_pointcloud(pub_cur_pointcloud, true);

                        ++global_id;
                        data.clear();
                        data.last_timestamp = data.timestamp;
                        transform.setOrigin(tf::Vector3(init_pwb.x(), init_pwb.y(), init_pwb.z()));
                        Eigen::Quaterniond init_qwb(init_Rwb);
                        transform.setRotation(tf::Quaternion(init_qwb.x(), init_qwb.y(), init_qwb.z(), init_qwb.w()));
                        br.sendTransform(tf::StampedTransform(transform, ros::Time(data.timestamp), "camera_init", "move_camera"));
                    }
                }
                continue;
            }
            else if(window_state == WindowState::IMU_INITIAL){
                Measures data_copy = data.copy();
                // static initialization
                if(estimator1.Initialize(data, Rwb, pwb) && estimator2.Initialize(data_copy, Rwb, pwb)){
                    cout << "IMU Initialization success!" << endl;
                    window_state = WindowState::FILLING;
                    // BodyPoseToCameraPose(Rwb, pwb, Matrix6d::Zero());
                    fp_pos << "Rotation matrix: " << endl << Rwb << endl;
                    fp_pos << "Translation vector: " << endl << pwb << endl;
                    if(!need_init){
                        deq_Rcw[frame_count] = Rcw;
                        deq_pcw[frame_count] = pcw;
                        deq_deltaR[frame_count] = Eigen::Matrix3d::Identity();
                        deq_deltap[frame_count] = Eigen::Vector3d::Zero();
                        deq_id[frame_count] = global_id;

                        ++frame_count;

                        match_result = ProbMatch(data, whole_box_centers, whole_box_corners);
                        M = match_result.M;

                        for(int i = 0; i < M.size(); i++){
                            if(M[i] >= 0)
                                deq_matches[frame_count].insert(make_pair(i, M[i]));
                        }
                        for(int i = 0; i < data.box->bounding_boxes.size(); i++){
                            Vector4d box;
                            box << data.box->bounding_boxes[i].xmin, data.box->bounding_boxes[i].ymin, data.box->bounding_boxes[i].xmax, data.box->bounding_boxes[i].ymax;
                            deq_boxes[frame_count].push_back(box);
                        }
                    }
                }
                else{
                    data.clear();
                    data_copy.clear();
                    ++global_id;
                    continue;
                }

                data.clear();
                data_copy.clear();
                ++global_id;
            }
            else{
                Eigen::Vector3d rij, pij;
                Eigen::Matrix3d Rij;

                Measures data_copy = data.copy();
                estimator1.step(data);
                estimator2.step(data_copy);
                // Rij = Rwb.transpose() * estimator1.getRotation();
                // pij = Rwb.transpose() * (estimator1.getPosition() - pwb);
                // Eigen::AngleAxisd rotij(Rij);
                // rij = rotij.angle() * rotij.axis();

                Rwb = estimator1.getRotation();
                pwb = estimator1.getPosition();
                cov_wb = estimator1.getPoseCovariance();
                cout << "cov_wb_prior: " << cov_wb << endl;
                BodyPoseToCameraPose(Rwb, pwb, cov_wb);
                fp_pos << "Rotation matrix before optimization: " << endl << Rwb << endl;
                fp_pos << "Translation vector before optimization: " << endl << pwb << endl;

                Rwb2 = estimator2.getRotation();
                pwb2 = estimator2.getPosition();
                cout << "cov_wb_imu_odom: " << estimator2.getPoseCovariance() << endl;

                //定位丢失，采用暴力匹配方式重新定位
                if(reloc && global_id - reloc_id > 150){
                    reloc = false;
                }
                if(no_obs > 15 && frame_count == WINDOW_SIZE && ablation_reloc){
                    fp_match << "Triger relocalization module!" << endl;
                    reloc = true;
                    reloc_id = global_id;
                    M = Relocalization(data, deq_matches[frame_count], deq_boxes[frame_count], estimator1);
                    if(deq_matches[frame_count].size() == 0){
                        op_img = data.img.clone();
                        detect_project_img = data.img.clone();
                        ++no_obs;
                    }
                    else{
                        estimator1.stepCam(data.timestamp, deq_matches[frame_count], lamp_world_pos, deq_boxes[frame_count], high_lamp);
                        Rwb = estimator1.getRotation();
                        pwb = estimator1.getPosition();
                        cov_wb = estimator1.getPoseCovariance();
                        cout << "cov_wb_post1: " << cov_wb << endl;
                        BodyPoseToCameraPose(Rwb, pwb, cov_wb);

                        fp_pos << "Rotation matrix after optimization: " << endl << Rwb << endl;
                        fp_pos << "Translation vector after optimization: " << endl << pwb << endl;

                        vec_vec4d new_boxes;
                        unordered_map<int, int> new_matches;
                        op_img = data.img.clone();
                        detect_project_img = data.img.clone();
                        UpdateBox(data, deq_matches[frame_count], new_matches, deq_boxes[frame_count], new_boxes, op_img, detect_project_img);
                        num_extend += new_matches.size();
                        for(int i = 0; i < new_boxes.size(); ++i){
                            Vector4d box = new_boxes[i];
                            area_extend += (box[2] - box[0]) * (box[3] - box[1]);
                        }
                        if(!new_boxes.empty() && !new_matches.empty()){
                            if(last_half_box){
                                tmp_Rcw = Rcw;
                                tmp_pcw = pcw;
                            }
                            estimator1.stepCam(data.timestamp, new_matches, lamp_world_pos, new_boxes, high_lamp);

                            Rwb = estimator1.getRotation();
                            pwb = estimator1.getPosition();
                            cov_wb = estimator1.getPoseCovariance();
                            cout << "cov_wb_post2: " << cov_wb << endl;
                            BodyPoseToCameraPose(Rwb, pwb, cov_wb);

                            fp_pos << "Rotation matrix after optimization: " << endl << Rwb << endl;
                            fp_pos << "Translation vector after optimization: " << endl << pwb << endl;

                            for(int i = 0; i < new_boxes.size(); i++){
                                Vector4d box = new_boxes[i];
                                auto iter = new_matches.find(i);
                                
                                if(iter != new_matches.end()){
                                    map_relocalization::BoundingBox bbox;
                                    bbox.xmin = box(0), bbox.xmax = box(2), bbox.ymin = box(1), bbox.ymax = box(3);
                                    data.box->bounding_boxes.push_back(bbox);
                                    deq_boxes[frame_count].push_back(box);
                                    cout << "find: " << iter->first << " " << iter->second << endl;
                                    M.push_back(iter->second);
                                    deq_matches[frame_count].insert(make_pair(M.size() - 1, iter->second));
                                }
                            }
                            fp_match << "Update M list: " << endl;
                            for(int i = 0; i < M.size(); i++){
                                fp_match << "box " << i << " corresponds to lamp " << M[i] << endl;
                                fp_match << "box " << i << " contour points are " << data.box->bounding_boxes[i] << endl;
                            }
                        }
                        if(deq_matches[frame_count].size() < 2){
                            ROS_WARN("Not enough matches!");
                            fp_match << "Not enough matches!" << endl;
                            no_obs++;
                            last_no_ob = true;
                        }
                        else{
                            fp_match << "Find enough matches, exit relocalization module." << endl;
                            last_no_ob = false;
                            last_best_match = false;
                            no_obs = 0;
                        }
                    }
                }

                else{
                    vec_vec2d whole_box_centers;
                    vec_vec4d whole_box_corners;

                    auto start = chrono::steady_clock::now();
                    match_result = ProbMatch(data, whole_box_centers, whole_box_corners);
                    auto end = chrono::steady_clock::now();

                    auto tt = chrono::duration_cast<chrono::microseconds>(end - start);
                    if(max_time > 0 && max_time < tt.count()){
                        max_time = tt.count();
                    }
                    else if(max_time < 0){
                        max_time = tt.count();
                    }
                    mean_time += tt.count();

                    M = match_result.M;

                    fp_match << "M list: " << endl;
                    for(int i = 0; i < M.size(); i++)
                        fp_match << "box " << i << " corresponds to lamp " << M[i] << endl;

                    if(frame_count > 0){
                        // deq_features[frame_count] = image;
                        deq_deltaR[frame_count] = deq_Rcw[frame_count - 1] * Rcw.transpose(); //Rct-1ct
                        // cout << "Rij: " << Rij << endl;
                        deq_deltap[frame_count] = deq_Rcw[frame_count - 1] * (- Rcw.transpose() * pcw) + deq_pcw[frame_count - 1];
                        for(int i = 0; i < data.box->bounding_boxes.size(); i++){
                            Vector4d box;
                            box << data.box->bounding_boxes[i].xmin, data.box->bounding_boxes[i].ymin, data.box->bounding_boxes[i].xmax, data.box->bounding_boxes[i].ymax;
                            deq_boxes[frame_count].push_back(box);
                            // cout << box << endl;
                        }
                        for(int i = 0; i < M.size(); i++)
                            if(M[i] >= 0)
                                deq_matches[frame_count].insert(make_pair(i, M[i]));
                        cout << "deq_matches0: " << deq_matches[frame_count].size();
                        num_hung += deq_matches[frame_count].size();
                        for(int i = 0; i < deq_boxes[frame_count].size(); ++i){
                            if(deq_matches[frame_count].find(i) != deq_matches[frame_count].end()){
                                Vector4d box = deq_boxes[frame_count][i];
                                area_hung += (box[2] - box[0]) * (box[3] - box[1]);
                            }
                        }
                        //TODO if number of matches less than 2    2种方式，暴力匹配 / 扩大方差
                        // MatchFilter(deq_matches[frame_count], deq_boxes[frame_count], z_th, dist_th);
                        estimator1.stepCam(data.timestamp, deq_matches[frame_count], lamp_world_pos, deq_boxes[frame_count], high_lamp);

                        Eigen::Matrix3d tmp_Rwb = estimator1.getRotation();
                        Eigen::Vector3d tmp_pwb = estimator1.getPosition();

                        fp_pos << "Rotation matrix after optimization: " << endl << tmp_Rwb << endl;
                        fp_pos << "Translation vector after optimization: " << endl << tmp_pwb << endl;

                        double avg_error = 0.0;
                        if(!CheckPose(tmp_Rwb, tmp_pwb, deq_matches[frame_count], deq_boxes[frame_count], avg_error) && frame_count == WINDOW_SIZE && 0){ //!CheckPose(tmp_Rwb, tmp_pwb, deq_matches[frame_count], deq_boxes[frame_count], avg_error) && frame_count == WINDOW_SIZE
                            ROS_WARN("optimization failed!");
                            fp_match << "optimization failed!" << endl;
                            //回滚操作
                            estimator1.rollBack();
                            //重新计算匈牙利算法
                            ReMatch(match_result, estimator1, data.timestamp, deq_matches[frame_count], deq_boxes[frame_count], avg_error);
                            M = match_result.M;

                            Rwb = estimator1.getRotation();
                            pwb = estimator1.getPosition();
                            cov_wb = estimator1.getPoseCovariance();
                            cout << "cov_wb_post1: " << cov_wb << endl;
                            BodyPoseToCameraPose(Rwb, pwb, cov_wb);

                            if(ablation_matext){
                                vec_vec4d new_boxes;
                                unordered_map<int, int> new_matches;
                                op_img = data.img.clone();
                                detect_project_img = data.img.clone();
                                UpdateBox(data, deq_matches[frame_count], new_matches, deq_boxes[frame_count], new_boxes, op_img, detect_project_img);
                                cout << "deq_matches2: " << deq_matches[frame_count].size() << endl;
                                MatchFilter(deq_matches[frame_count], deq_boxes[frame_count], update_z_th, update_dist_th);
                                cout << "deq_matches3: " << deq_matches[frame_count].size() << endl;
                                if(!new_boxes.empty() && !new_matches.empty()){
                                    estimator1.stepCam(data.timestamp, new_matches, lamp_world_pos, new_boxes, high_lamp);

                                    // Eigen::Matrix3d tmp_Rwb = estimator1.getRotation();
                                    // Eigen::Vector3d tmp_pwb = estimator1.getPosition();
                                    // if(!CheckPose(tmp_Rwb, tmp_pwb, new_matches, new_boxes) && frame_count == WINDOW_SIZE && !last_half_box){
                                    //     estimator1.rollBack();
                                    //     ReMatch(estimator1, data.timestamp, new_matches, new_boxes);
                                    // }
                                    Rwb = estimator1.getRotation();
                                    pwb = estimator1.getPosition();
                                    cov_wb = estimator1.getPoseCovariance();
                                    cout << "cov_wb_post2: " << cov_wb << endl;
                                    BodyPoseToCameraPose(Rwb, pwb, cov_wb);

                                    for(int i = 0; i < new_boxes.size(); i++){
                                        Vector4d box = new_boxes[i];
                                        auto iter = new_matches.find(i);
                                        
                                        if(iter != new_matches.end()){
                                            map_relocalization::BoundingBox bbox;
                                            bbox.xmin = box(0), bbox.xmax = box(2), bbox.ymin = box(1), bbox.ymax = box(3);
                                            data.box->bounding_boxes.push_back(bbox);
                                            deq_boxes[frame_count].push_back(box);
                                            cout << "find: " << iter->first << " " << iter->second << endl;
                                            M.push_back(iter->second);
                                            deq_matches[frame_count].insert(make_pair(M.size() - 1, iter->second));
                                        }
                                    }
                                    fp_match << "Update M list: " << endl;
                                    for(int i = 0; i < M.size(); i++){
                                        fp_match << "box " << i << " corresponds to lamp " << M[i] << endl;
                                        fp_match << "box " << i << " contour points are " << data.box->bounding_boxes[i] << endl;
                                    }
                                }
                            }
                            if(deq_matches[frame_count].size() < 1){
                                ROS_WARN("Not enough matches!");
                                fp_match << "Not enough matches!" << endl;
                                no_obs++;
                                last_no_ob = true;
                            }
                            else{
                                last_no_ob = false;
                                no_obs = 0;
                            }
                        }
                        else{                            
                            Rwb = tmp_Rwb;
                            pwb = tmp_pwb;
                            cov_wb = estimator1.getPoseCovariance();
                            cout << "cov_wb_post1: " << cov_wb << endl;
                            BodyPoseToCameraPose(Rwb, pwb, cov_wb);

                            if(ablation_matext){
                                vec_vec4d new_boxes;
                                unordered_map<int, int> new_matches;
                                op_img = data.img.clone();
                                detect_project_img = data.img.clone();
                                UpdateBox(data, deq_matches[frame_count], new_matches, deq_boxes[frame_count], new_boxes, op_img, detect_project_img);
                                cout << "deq_matches2: " << deq_matches[frame_count].size() << endl;
                                MatchFilter(deq_matches[frame_count], deq_boxes[frame_count], update_z_th, update_dist_th);
                                cout << "deq_matches3: " << deq_matches[frame_count].size() << endl;
                                num_extend += new_matches.size();
                                for(int i = 0; i < new_boxes.size(); ++i){
                                    Vector4d box = new_boxes[i];
                                    area_extend += (box[2] - box[0]) * (box[3] - box[1]);
                                }
                                if(!new_boxes.empty() && !new_matches.empty()){
                                    if(last_half_box){
                                        tmp_Rcw = Rcw;
                                        tmp_pcw = pcw;
                                    }
                                    estimator1.stepCam(data.timestamp, new_matches, lamp_world_pos, new_boxes, high_lamp);

                                    // Eigen::Matrix3d tmp_Rwb = estimator1.getRotation();
                                    // Eigen::Vector3d tmp_pwb = estimator1.getPosition();
                                    // if(!CheckPose(tmp_Rwb, tmp_pwb, new_matches, new_boxes) && frame_count == WINDOW_SIZE && !last_half_box){
                                    //     estimator1.rollBack();
                                    //     ReMatch(estimator1, data.timestamp, new_matches, new_boxes);
                                    // }
                                    Rwb = estimator1.getRotation();
                                    pwb = estimator1.getPosition();
                                    cov_wb = estimator1.getPoseCovariance();
                                    cout << "cov_wb_post2: " << cov_wb << endl;
                                    BodyPoseToCameraPose(Rwb, pwb, cov_wb);

                                    for(int i = 0; i < new_boxes.size(); i++){
                                        Vector4d box = new_boxes[i];
                                        auto iter = new_matches.find(i);
                                        
                                        if(iter != new_matches.end()){
                                            map_relocalization::BoundingBox bbox;
                                            bbox.xmin = box(0), bbox.xmax = box(2), bbox.ymin = box(1), bbox.ymax = box(3);
                                            data.box->bounding_boxes.push_back(bbox);
                                            deq_boxes[frame_count].push_back(box);
                                            cout << "find: " << iter->first << " " << iter->second << endl;
                                            M.push_back(iter->second);
                                            deq_matches[frame_count].insert(make_pair(M.size() - 1, iter->second));
                                        }
                                    }
                                    fp_match << "Update M list: " << endl;
                                    for(int i = 0; i < M.size(); i++){
                                        fp_match << "box " << i << " corresponds to lamp " << M[i] << endl;
                                        fp_match << "box " << i << " contour points are " << data.box->bounding_boxes[i] << endl;
                                    }
                                }
                            }
                            else{
                                op_img = data.img.clone();
                                detect_project_img = data.img.clone();
                                unordered_map<int, int> new_matches;
                                CorrectDegenerationDeep(data, deq_matches[frame_count], deq_boxes[frame_count], new_matches, detect_project_img);
                                if(!new_matches.empty()){
                                    estimator1.stepCam(data.timestamp, new_matches, lamp_world_pos, deq_boxes[frame_count], high_lamp);

                                    Rwb = estimator1.getRotation();
                                    pwb = estimator1.getPosition();
                                    cov_wb = estimator1.getPoseCovariance();
                                    BodyPoseToCameraPose(Rwb, pwb, cov_wb);

                                    for(auto iter = new_matches.begin(); iter != new_matches.end(); ++iter){
                                        if(M[iter->first] < 0){
                                            M[iter->first] = iter->second;
                                        }
                                        else{
                                            ROS_ERROR("Wrong matches added! %d, %d", iter->first, iter->second);
                                        }
                                            
                                        deq_matches[frame_count].insert(make_pair(iter->first, iter->second));
                                    }
                                }
                            }

                            cout << "deq_matches4: " << deq_matches[frame_count].size() << endl;
                            if(deq_matches[frame_count].size() < 1){
                                ROS_WARN("Not enough matches!");
                                fp_match << "Not enough matches!" << endl;
                                no_obs++;
                                last_no_ob = true;
                            }
                            else{
                                last_no_ob = false;
                                no_obs = 0;
                            }
                        }

                        fp_pos << "Rotation matrix after optimization: " << endl << Rwb << endl;
                        fp_pos << "Translation vector after optimization: " << endl << pwb << endl;
                    }
                }
                //滑窗
                if(window_state == WindowState::FILLING){
                    Eigen::Matrix3d delta_R = deq_Rcw[frame_count - 1] * Rcw.transpose();
                    Eigen::Vector3d delta_T = deq_Rcw[frame_count - 1] * (- Rcw.transpose() * pcw) + deq_pcw[frame_count - 1];
                    Eigen::AngleAxisd delta_r(delta_R);
                    if (delta_r.angle() > 0.06 || delta_T.norm() > 0.15){
                        deq_Rcw[frame_count] = Rcw;
                        deq_pcw[frame_count] = pcw;
                        deq_P[frame_count] = cov;
                        deq_deltaR[frame_count] = delta_R;
                        deq_deltap[frame_count] = delta_T;
                        deq_id[frame_count] = global_id;
                        // for(int i = 0; i < M.size(); i++)
                        //     deq_matches[frame_count].insert(make_pair(i, M[i]));
                        // for(int i = 0; i < data.box->bounding_boxes.size(); i++){
                        //     Vector4d box;
                        //     box << data.box->bounding_boxes[i].xmin, data.box->bounding_boxes[i].ymin, data.box->bounding_boxes[i].xmax, data.box->bounding_boxes[i].ymax;
                        //     deq_boxes[frame_count].push_back(box);
                        // }
                        cout << "keyframe insert! frame count is " << frame_count << endl;
                        if(delta_r.angle() > 0.06)
                            fp_pos << "keframe inserted due to rotation, rotation angle is " << delta_r.angle() << endl;
                        if(delta_T.norm() > 0.15)
                            fp_pos << "keframe inserted due to translation, translation is: " << endl << delta_T.norm() << endl;
                        // if(!mp_unchange)
                        //     fp_pos << "keframe inserted due to lamp change." << endl;
                        fp_pos << "delta Rcw is " << endl << deq_deltaR[frame_count] << endl;
                        fp_pos << "delta pcw is " << endl << deq_deltap[frame_count] << endl;
                        ++frame_count;
                    }
                    else{
                        // deq_deltaR.pop_back();
                        // deq_deltap.pop_back();
                        vec_vec4d().swap(deq_boxes[frame_count]);
                        deq_matches[frame_count].clear();
                        // deq_features[frame_count].clear();
                    }
                    if(frame_count == WINDOW_SIZE)
                        window_state = WindowState::SLIDING;
                }
                else{
                    Eigen::Matrix3d delta_R = deq_Rcw[frame_count - 1] * Rcw.transpose();
                    Eigen::Vector3d delta_T = deq_Rcw[frame_count - 1] * (- Rcw.transpose() * pcw) + deq_pcw[frame_count - 1];
                    Eigen::AngleAxisd delta_r(delta_R);
                    if (delta_r.angle() > 0.06 || delta_T.norm() > 0.15){
                        deq_Rcw[frame_count] = Rcw;
                        deq_pcw[frame_count] = pcw;
                        deq_P[frame_count] = cov;
                        deq_deltaR[frame_count] = delta_R;
                        deq_deltap[frame_count] = delta_T;
                        deq_id[frame_count] = global_id;
                        // for(int i = 0; i < M.size(); i++)
                        //     deq_matches[frame_count].insert(make_pair(i, M[i]));
                        // for(int i = 0; i < data.box->bounding_boxes.size(); i++){
                        //     deq_boxes[frame_count] = whole_box_corners;
                        // }
                        SlideWindow();
                    }
                    else{
                        // deq_deltaR.pop_back();
                        // deq_deltap.pop_back();
                        vec_vec4d().swap(deq_boxes[frame_count]);
                        deq_matches[frame_count].clear();
                        // deq_features[frame_count].clear();
                    }
                }
            }
            dump_state_to_log();
            fp_cov << "timestamp: " << endl;
            fp_cov << setprecision(20) << data.timestamp << endl;
            fp_cov << "cov_wb: " << endl << cov_wb << endl;
            Eigen::Vector3d euler_angles = Rwb.eulerAngles(2, 1, 0);
            fp_cov << "rwb: " << endl << euler_angles.transpose() << endl;
            fp_cov << "pwb: " << endl << pwb.transpose() << endl;
            publish_odom_path(pub_odom_path);
            publish_optimized_path(pub_optimized_path);

            transform.setOrigin(tf::Vector3(pwb.x(), pwb.y(), pwb.z()));
            Eigen::Quaterniond qwb(Rwb);
            transform.setRotation(tf::Quaternion(qwb.x(), qwb.y(), qwb.z(), qwb.w()));
            br.sendTransform(tf::StampedTransform(transform, ros::Time(data.timestamp), "camera_init", "move_camera"));

            cam.reset();
            cam.add_pose(pwc, Eigen::Quaterniond(Rwc));
            cam.publish_by(pub_optimized_camera_visual, data.timestamp);
            // 查看投影结果
            cv::Mat detect_image, detect_project_image;
            detect_image = data.img;
            detect_project_image = data.img;
            DetectAndProjectImage(data, detect_image, detect_project_img, M);

            // publish_optimized_path(pub_optimized_path);
            publish_image(pub_detect_img, detect_image);
            publish_image(pub_detect_project_img, detect_project_img);
            publish_image(pub_flow_img, op_img);
            publish_pointcloud(pub_cur_pointcloud, true);
            // pub_pointcloud.publish(*lamp_cloud);
            lamp_cur_id.clear();
            lamp_cur_cam_pos.clear();
            lamp_cur_world_pos.clear();
            cur_lamp_cloud->clear();
            // lamp_cur_cam_plane.clear();
            // lamp_cur_world_plane.clear();
            // if(global_id > 0){
            //     static int vins_pose_num = 0;
            //     if(vins_poses[vins_pose_num].timestamp <= data.timestamp){

            //         Eigen::Vector3d correct_vins_pwb = init_Rwb * vins_poses[vins_pose_num].translation + init_pwb;
            //         Eigen::Quaterniond correct_vins_qwb = Eigen::Quaterniond(init_Rwb) * vins_poses[vins_pose_num].quat;
            //         correct_vins_qwb.normalize();

            //         Eigen::Matrix3d correct_vins_Rwb = correct_vins_qwb.toRotationMatrix();

            //         geometry_msgs::PoseStamped pose;
            //         pose.pose.position.x = correct_vins_pwb.x();
            //         pose.pose.position.y = correct_vins_pwb.y();
            //         pose.pose.position.z = correct_vins_pwb.z();

            //         pose.pose.orientation.w = correct_vins_qwb.w();
            //         pose.pose.orientation.x = correct_vins_qwb.x();
            //         pose.pose.orientation.y = correct_vins_qwb.y();
            //         pose.pose.orientation.z = correct_vins_qwb.z();

            //         vins_pose_num++;
            //         if(vins_pose_num % 5 == 0){
            //             global_vins_wheel_path.poses.push_back(pose);
            //             pub_vins_wheel.publish(global_vins_wheel_path);
            //         }

            //         Eigen::Vector3d correct_vins_pwc = correct_vins_Rwb * pbc + correct_vins_pwb;
            //         Eigen::Matrix3d correct_vins_Rwc = correct_vins_Rwb * Rbc;
            //         cam_vins_wheel.reset();
            //         cam_vins_wheel.add_pose(correct_vins_pwc, Eigen::Quaterniond(correct_vins_Rwc));
            //         cam_vins_wheel.publish_by(pub_vins_wheel_visual, data.timestamp);

            //     }
            // }

            ++global_id;

            last_data = data;
            data.last_timestamp = data.timestamp;
        }
        nogps:
        publish_pointcloud(pub_pointcloud, false);
        if(global_id > 0){
            publish_odom_path(pub_odom_path);
            publish_optimized_path(pub_optimized_path);
            pub_vins_wheel.publish(global_vins_wheel_path);
            cam.publish_by(pub_optimized_camera_visual, data.timestamp);
            cam_vins_wheel.publish_by(pub_vins_wheel_visual, data.timestamp);
        }
        status = ros::ok();
        rate.sleep();
    }
}