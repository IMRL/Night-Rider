#ifndef PARTICLE_FILTER_HPP_
#define PARTICLE_FILTER_HPP_

#include "map_relocalization/common_lib.h"

#define PI 3.1415926

struct Particle{

    int id;
    Eigen::Vector3d pos;
    Eigen::Vector3d rot;
    double weight;
    vector<int> associations;
};

class ParticleFilter{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    vector<Particle> particles;
    ParticleFilter(): initialized_pose_(false){}
    ~ParticleFilter() {}

    void init(const vec_vec2d& points, int num_particles, double init_x = -10000, double init_y = -10000, double init_theta = -10000);

    void init(const vec_vec3d& points, int num_particles, double init_x = -10000, double init_y = -10000, double init_z = -10000, double init_alpha = -10000, double init_beta = -10000, double init_gamma = -10000);

    void prediction(const Eigen::Matrix3d& Rij, const Eigen::Vector3d& pij, Matrix6d cov);

    void updateWeights(const Measures& data, const vec_vec3d& points, const Eigen::Matrix3d& Rcb, const Eigen::Vector3d& pcb, const int& res_x, const int& res_y);

    double md_distance(const Eigen::Vector2d& box_center, const Eigen::Vector2d& proj_center, const Eigen::Matrix2d& cov);

    double ang_distance(const Eigen::Vector2d& box_center, const Eigen::Vector3d& lamp_center, const Eigen::Matrix2d& sigma2);

    void resample();

    bool initializationSuc(){
        return initialized_pose_;
    }

    void setIntializedPose(){
        initialized_pose_ = true;
    }

private:
    int num_particles_;
    vector<double> weights_;
    bool initialized_pose_;
};

#endif