#ifndef FEATURE_TRACKER_HPP
#define FEATURE_TRACKER_HPP

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

class FeatureTracker
{
  public:
    FeatureTracker(const int& col, const int& row, const double& min_dist, const int& max_cnt, const Eigen::Matrix3d K_inv, const Eigen::Matrix3d& K, const Eigen::Vector2d un_k = Eigen::Vector2d::Zero(), const Eigen::Vector2d un_p = Eigen::Vector2d::Zero()): 
    col_(col), row_(row), min_dist_(min_dist), max_cnt_(max_cnt), K_inv_(K_inv), K_(K), un_k_(un_k), un_p_(un_p){
    }

    bool inBorder(const cv::Point2f &pt);

    void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);

    void reduceVector(vector<int> &v, vector<uchar> status);

    void readImage(const cv::Mat &_img,double _cur_time, cv::Mat &op_img);

    void setMask();

    void addPoints();

    bool updateID(unsigned int i);

    void readIntrinsicParameter(const string &calib_file);

    void distortion(const Eigen::Vector2d& p_u, Eigen::Vector2d& d_u);

    void liftProjective(const Eigen::Vector2d& p, Eigen::Vector3d& P);

    void showUndistortion(const string &name);

    void rejectWithF();

    void undistortedPoints();

    cv::Mat mask;
    cv::Mat fisheye_mask;
    cv::Mat prev_img, cur_img, forw_img;
    vector<cv::Point2f> n_pts;
    vector<cv::Point2f> prev_pts, cur_pts, forw_pts;
    vector<cv::Point2f> prev_un_pts, cur_un_pts;
    vector<cv::Point2f> pts_velocity;
    vector<int> ids;
    vector<int> track_cnt;
    map<int, cv::Point2f> cur_un_pts_map;
    map<int, cv::Point2f> prev_un_pts_map;
    double cur_time;
    double prev_time;

    static int n_id;
    int col_, row_;
    double min_dist_;
    int max_cnt_;

    Eigen::Matrix3d K_inv_, K_;
    Eigen::Vector2d un_k_, un_p_;
};

#endif