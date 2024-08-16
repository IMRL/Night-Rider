#ifndef SAMPLER_HPP_
#define SAMPLER_HPP_

#include "map_relocalization/common_lib.h"

#define PI 3.1415926

struct Sample{

    Eigen::Vector3d pos;
    Eigen::Vector3d rot;
    double weight;
    vector<int> associations;
};

class Sampler{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    vector<Sample> samples;
    vector<Sample> best_samples;
    bool initFinish;
    Sampler(int k, double search_dist_scope, double search_z_scope): initialized_pose_(false), initFinish(false), k_(k), search_dist_scope_(search_dist_scope), search_z_scope_(search_z_scope){ 
        best_samples.resize(k_);
    }
    ~Sampler() {}

    void init(const vec_vec3d& points, int num_samples, double init_x, double init_y, double init_z, double init_alpha , double init_beta, double init_gamma);

    void init(const vec_vec3d& points, int num_samples, double yaw, double init_x_gps, double cov_init_x_gps, double init_y_gps, double cov_init_y_gps, double init_alpha, double init_beta, double init_gamma);

    void init(const vec_vec3d& points, int num_samples, const Eigen::Vector3d& init_pos, const Eigen::Matrix3d& cov, double init_alpha, double init_beta, double init_gamma);

    void updateWeights(const Measures& data, const vec_vec3d& points, const Eigen::Matrix3d& Rcb, const Eigen::Vector3d& pcb, const int& res_x, const int& res_y);

    double md_distance(const Eigen::Vector2d& box_center, const Eigen::Vector2d& proj_center, const Eigen::Matrix2d& cov);

    double ang_distance(const Eigen::Vector2d& box_center, const Eigen::Vector3d& lamp_center, const Eigen::Matrix2d& sigma2);

    void resample(const int iter);

    void sortWeight();

    bool initializationSuc(){
        return initialized_pose_;
    }

    void setIntializedPose(){
        initialized_pose_ = true;
    }

private:
    int num_samples_;
    vector<double> weights_;
    bool initialized_pose_;
    int k_;
    double init_x_, init_y_, local_min_, local_max_;
    double region_x_, region_y_, region_z_;
    Eigen::Matrix2d cov_;
    double search_dist_scope_, search_z_scope_;
};

#endif
