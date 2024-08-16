#include "map_relocalization/particle_filter.hpp"
#include <armadillo>
#include <random>
#include "map_relocalization/hungary_estimator.hpp"

// void ParticleFilter::init(const vec_vec3d& points, int num_particles, double init_x, double init_y, double init_theta){
//     if(init_x > -10000 && init_y > -10000)
//         num_particles = num_particles;
//     else
//         num_particles = 3 * num_particles;
//     num_particles_ = num_particles;
//     double min_x = 2000, min_y = 2000, max_x = -2000, max_y = -2000;
//     for(int i = 0; i < points.size(); ++i){
//         Eigen::Vector2d point = points[i];
//         min_x = min(min_x, point.x()) - 1;
//         min_y = min(min_y, point.y()) - 1;
//         max_x = max(max_x, point.x()) + 1;
//         max_y = max(max_y, point.y()) + 1;
//     }

//     cv::RNG rng(cv::getTickCount());
//     int N = 0;
//     double weight = 1.0 / num_particles;
//     while( N < num_particles){
//         double x, y, theta;
//         if(init_x > -10000 && init_y > -10000 && init_theta > -10000){
//             x = rng.uniform(init_x - 30, init_x + 30);
//             y = rng.uniform(init_y - 30, init_y + 30);
//             theta = rng.uniform(init_theta - PI / 2, init_theta + PI / 2);
//         }
//         else if(init_x > -10000 && init_y > -10000){
//             x = rng.uniform(init_x - 30, init_x + 30);
//             y = rng.uniform(init_y - 30, init_y + 30);
//             theta = rng.uniform(-PI, PI);
//         }
//         else{
//             x = rng.uniform(min_x, max_x);
//             y = rng.uniform(min_y, max_y);
//             theta = rng.uniform(-PI, PI);
//         }

//         bool exist_left_lamp = false, exist_right_lamp = false;
//         for(int i = 0; i < points.size(); ++i){
//             Eigen::Vector2d point = points[i];
//             if(point.y() - y < 13 && point.y() - y >= 0)
//                 exist_right_lamp = true;
//             if(point.y() - y < 0 && point.y() - y >= -13)
//                 exist_left_lamp = true;
//         }
//         if(!exist_left_lamp || !exist_right_lamp)
//             continue;

//         Particle particle;
//         particle.id = N;
//         particle.x = x;
//         particle.y = y;
//         particle.theta = theta;
//         particle.weight = weight;
        
//         particles.push_back(particle);
//         weights_.push_back(weight);
//         ++N;
//     }
// }

void ParticleFilter::init(const vec_vec3d& points, int num_particles, double init_x, double init_y, double init_z, double init_alpha, double init_beta, double init_gamma){
    if(init_x > -10000 && init_y > -10000 && init_z > -10000)
        num_particles = num_particles;
    else
        num_particles = 4 * num_particles;
    num_particles_ = num_particles;
    weights_.resize(num_particles_);
    particles.resize(num_particles_);

    double min_x = 2000, min_y = 2000, min_z = 2000, max_x = -2000, max_y = -2000, max_z = -2000;
    for(int i = 0; i < points.size(); ++i){
        Eigen::Vector3d point = points[i];
        min_x = min(min_x, point.x()) - 1;
        min_y = min(min_y, point.y()) - 1;
        min_z = min(min_z, point.z());
        max_x = max(max_x, point.x()) + 1;
        max_y = max(max_y, point.y()) + 1;
        max_z = max(max_z, point.z());
    }

    cv::RNG rng(cv::getTickCount());
    int N = 0;
    double weight = 1.0 / num_particles;
    while( N < num_particles){
        double x, y, z, alpha, beta, gamma;
        if(init_x > -10000 && init_y > -10000 && init_z > -10000 && init_alpha > -10000 && init_beta > -10000 && init_gamma > -10000){
            x = init_x + rng.gaussian(2);
            y = init_y + rng.gaussian(2);
            z = init_z + rng.gaussian(0.2);
            // alpha = init_alpha + rng.gaussian(PI / 100);
            // beta = init_beta + rng.gaussian(PI / 100);
            alpha = init_alpha;
            beta = init_beta;
            gamma = init_gamma + rng.gaussian(PI / 10);
            // x = rng.uniform(init_x - 15, init_x + 15);
            // y = rng.uniform(init_y - 15, init_y + 15);
            // z = rng.uniform(init_z - 1, init_z + 1);
            // alpha = rng.uniform(init_alpha - PI / 100, init_alpha + PI / 100);
            // beta = rng.uniform(init_beta - PI / 10, init_beta + PI / 10);
            // gamma = rng.uniform(init_gamma - PI / 4, init_gamma + PI / 4);
        }
        else if(init_x > -10000 && init_y > -10000 && init_z > -10000){
            // x = rng.uniform(init_x - 15, init_x + 15);
            // y = rng.uniform(init_y - 15, init_y + 15);
            // z = rng.uniform(init_z - 1, init_z + 1);
            x = init_x + rng.gaussian(5);
            y = init_y + rng.gaussian(5);
            z = init_z + rng.gaussian(0.5);
            // alpha = rng.uniform(-PI / 50, PI / 50);
            // beta = rng.uniform(-PI / 10, PI / 10);
            alpha = 0.0;
            beta = 0.0;
            gamma = rng.uniform(-PI, PI);
        }
        else{
            x = rng.uniform(min_x, max_x);
            y = rng.uniform(min_y, max_y);
            z = rng.uniform(min_z, max_z);
            alpha = 0.0;
            beta = 0.0;
            // alpha = rng.uniform(-PI / 50, PI / 50);
            // beta = rng.uniform(-PI / 10, PI / 10);
            gamma = rng.uniform(-PI, PI);
        }
        
        Eigen::Vector3d pos_particle = Eigen::Vector3d(x, y, z);
        Eigen::Vector3d rot_particle = Eigen::Vector3d(gamma, beta, alpha);
        cout << "init_euler: " << rot_particle.transpose() << endl; 
        bool exist_left_lamp = false, exist_right_lamp = false, exist_top_lamp = false, exist_down_lamp = false;
        for(int i = 0; i < points.size(); ++i){
            Eigen::Vector3d point = points[i];
            if((point - pos_particle).norm() < 40){
                if(point.y() - y < 13 && point.y() - y >= 0)
                    exist_right_lamp = true;
                if(point.y() - y < 0 && point.y() - y >= -13)
                    exist_left_lamp = true;
                if(point.z() - z < 2.7 && point.z() - z >= 0)
                    exist_top_lamp = true;
                if(point.z() - z < 0 && point.z() - z >= -0.5)
                    exist_down_lamp = true;
            }
        }
        if(!exist_left_lamp || !exist_right_lamp || !exist_top_lamp || !exist_down_lamp)
            continue;
        
        Particle particle;
        particle.id = N;
        particle.pos = pos_particle;
        particle.rot = rot_particle;
        particle.weight = weight;
        
        particles[N] = particle;
        weights_[N] = weight;
        ++N;
    }
}

void ParticleFilter::prediction(const Eigen::Matrix3d& Rij, const Eigen::Vector3d& pij, Matrix6d cov){

    cv::RNG rng(cv::getTickCount());
    Eigen::AngleAxisd rij(Rij);
    Eigen::Vector3d vrij(rij.angle() * rij.axis());
    cout << "vrij: " << vrij.transpose() << endl;
    cout << "cov: " << cov << endl;
    Vector6d Tij;
    Tij.head<3>() = vrij;
    Tij.tail<3>() = pij;

    arma::vec arma_Tij = arma::vec(Tij.data(), Tij.rows(), false, false);
    arma::mat arma_Pij = arma::mat(cov.data(), cov.rows(), cov.cols(), false, false);
    for(int i = 0; i < particles.size(); ++i){
        Eigen::Vector3d particle_pos = particles[i].pos;
        Eigen::Vector3d particle_rot = particles[i].rot;

        arma::mat delta_T_arma = arma::mvnrnd(arma_Tij, arma_Pij, 1);
        Eigen::MatrixXd delta_T_eig = Eigen::Map<Eigen::MatrixXd>(delta_T_arma.memptr(), delta_T_arma.n_rows, delta_T_arma.n_cols);
        assert(delta_T_eig.rows() == 6);

        Eigen::AngleAxisd delta_rot(delta_T_eig.block<3, 1>(0, 0).norm(), delta_T_eig.block<3, 1>(0, 0).normalized());
        Eigen::Vector3d delta_euler = delta_rot.matrix().eulerAngles(2, 1, 0);
        delta_euler(1) = 0;
        delta_euler(2) = 0;
        if(delta_euler(0) > 3)
            delta_euler(0) = delta_euler(0) - PI;
        if(delta_euler(0) < -3)
            delta_euler(0) = delta_euler(0) + PI;

        Eigen::Vector3d delta_pos = delta_T_eig.block<3, 1>(3, 0);
        // cout << "delta_pos: " << delta_pos.transpose() << endl;
        // cout << "delta_euler: " << delta_euler.transpose() << endl;

        particles[i].pos = particle_pos + delta_pos;
        particles[i].rot = particle_rot + delta_euler;
        if(particles[i].rot(0) > PI){
            particles[i].rot(0) = particles[i].rot(0) - 2 * PI;
        }
        if(particles[i].rot(0) < -PI){
            particles[i].rot(0) = particles[i].rot(0) + 2 * PI;
        }
        // cout << "pos: " << particles[i].pos.transpose() << endl;
        // cout << "rot: " << particles[i].rot.transpose() << endl;
    }

}

double ParticleFilter::md_distance(const Eigen::Vector2d& box_center, const Eigen::Vector2d& proj_center, const Eigen::Matrix2d& cov){
    Eigen::Vector2d err(box_center - proj_center);
    // cout << "err " << err.norm() << endl;
    Eigen::Matrix2d cov_ld = cov.llt().matrixL();
    // cout << "cov_ld: " << endl << cov_ld << endl;
    Eigen::Vector2d vld = cov_ld.inverse() * err;
    // cout << "vld " << endl << vld << endl;
    return exp(- 0.5 * vld.transpose() * vld);
}

double ParticleFilter::ang_distance(const Eigen::Vector2d& box_center, const Eigen::Vector3d& lamp_center, const Eigen::Matrix2d& sigma2){
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
    
    double cov_cos_theta = (jaco_cos_liftup_center.transpose() * cov_liftup_center * jaco_cos_liftup_center)(0);
    // cout << " jaco_cos_liftup_center" << (jaco_cos_liftup_center.transpose() * cov_liftup_center * jaco_cos_liftup_center) << endl;
    
    return exp(-0.5 * err * err / cov_cos_theta);
}

void ParticleFilter::updateWeights(const Measures& data, const vec_vec3d& points, const Eigen::Matrix3d& Rcb, const Eigen::Vector3d& pcb, const int& res_x, const int& res_y){

    if(data.box->bounding_boxes.size() == 0){
        cout << "no boxes exist! initialization failed!" << endl;
        return;
    }
    double weight_normalizer = 0.0;
    double best_weight = 0;
    Eigen::MatrixXd best_prob;
    for(int i = 0; i < particles.size(); ++i){
        Eigen::Matrix3d particle_Rwb;
        particle_Rwb = Eigen::AngleAxisd(particles[i].rot[0], Eigen::Vector3d::UnitZ()) * 
                        Eigen::AngleAxisd(particles[i].rot[1], Eigen::Vector3d::UnitY()) * 
                        Eigen::AngleAxisd(particles[i].rot[2], Eigen::Vector3d::UnitX());
        Eigen::Matrix3d particle_Rcw = Rcb * particle_Rwb.transpose();
        Eigen::Vector3d particle_pwb = particles[i].pos;
        Eigen::Vector3d particle_pcw = - Rcb * particle_Rwb.transpose() * particle_pwb + pcb;

        vec_vec3d particle_Pc;
        vec_vec2d particle_pt;
        vector<int> particle_id;
        for (int j = 0; j < points.size(); j++){
            if(isnan(points[j].norm())) continue;
            Eigen::Vector3d Pc;
            Pc = particle_Rcw * points[j] + particle_pcw;
            double inv_z = 1.0 / Pc.z();
            // Pc = Pc * inv_z;

            Eigen::Vector2d pt;
            pt << cam_fx * inv_z * Pc.x() + cam_cx, cam_fy * inv_z * Pc.y() + cam_cy;

            if ((sqrt(Pc.x() * Pc.x() + Pc.z() * Pc.z()) < 75) && Pc.z() < 55 && Pc.z() > 1.0 && pt.x() < res_x - 5 && pt.x() >= 5 && pt.y() < res_y - 5 && pt.y() >= 5){ //55, 45
                // cout << "Pc: " << Pc << endl;
                // bool find_rep = false;
                // for (int k = 0; k < particle_Pc.size(); k++){
                //     Eigen::Vector3d l1 = particle_Pc[k].normalized();
                //     Eigen::Vector3d l2 = Pc.normalized();
                //     // cout << "cos_theta: " << l1.transpose() * l2 << endl;
                //     if((l1.transpose() * l2)(0) > 0.998){
                //         double n1 = particle_Pc[k].norm();
                //         double n2 = Pc.norm();
                //         find_rep = true;
                //         if(n1 > n2){
                //             particle_Pc[k] = Pc;
                //             particle_pt[k] = pt;
                //             particle_id[k] = j;
                //         }
                //         break;
                //     }
                // }
                // if(find_rep) continue;
                particle_Pc.push_back(Pc);
                particle_pt.push_back(pt);
                particle_id.push_back(j);
            }
        }

        if(particle_Pc.size() == 0){
            cout << "no lamp exists around this particle, assign weight to 0." << endl;
            particles[i].weight = 0;
            continue;
        }

        int box_nums = data.box->bounding_boxes.size();
        Eigen::MatrixXd particle_md_matrix = Eigen::MatrixXd::Zero(box_nums, particle_Pc.size() + 1);
        Eigen::MatrixXd particle_ad_matrix = Eigen::MatrixXd::Zero(box_nums, particle_Pc.size() + 1);
        
        for (int j = 0; j < box_nums; j++){
            double center_x = (data.box->bounding_boxes[j].xmax + data.box->bounding_boxes[j].xmin) / 2;
            double center_y = (data.box->bounding_boxes[j].ymax + data.box->bounding_boxes[j].ymin) / 2;

            double sigma2_x = data.box->bounding_boxes[j].xmax - data.box->bounding_boxes[j].xmin;
            double sigma2_y = data.box->bounding_boxes[j].ymax - data.box->bounding_boxes[j].ymin;
            // double sigma2_x = 5;
            // double sigma2_y = 5;

            sigma2_x = 1 * sigma2_x * sigma2_x;
            sigma2_y = 1 * sigma2_y * sigma2_y;

            Eigen::Matrix2d sigma2 = Eigen::Matrix2d::Identity();
            sigma2(0, 0) = sigma2_x;
            sigma2(1, 1) = sigma2_y;

            // cout << data.box->bounding_boxes[j].xmax << " " << data.box->bounding_boxes[j].xmin << " " << data.box->bounding_boxes[j].ymax << " " << data.box->bounding_boxes[j].ymin << endl;

            for (int k = 0; k < particle_pt.size(); k++){
                particle_md_matrix(j, k) = md_distance(Eigen::Vector2d(center_x, center_y), particle_pt[k], sigma2);
                // particle_ad_matrix(j, k) = ang_distance(Eigen::Vector2d(center_x, center_y), particle_Pc[k], sigma2);
            }
        }

        // cout << "md_distance: " << endl << particle_md_matrix << endl;
        // cout << "ang_distance: " << endl << particle_ad_matrix << endl;

        int max_edge = box_nums + int(particle_Pc.size());
        Eigen::MatrixXd cost = Eigen::MatrixXd::Zero(max_edge, max_edge), prob = Eigen::MatrixXd::Zero(max_edge, max_edge);
        prob.block(0, 0, box_nums, particle_Pc.size() + 1) = particle_md_matrix;

        for(int j = 0; j < box_nums; j++){
            if(j == 0)
                prob.block(0, particle_Pc.size(), box_nums, 1) = Eigen::VectorXd::Ones(box_nums) - prob.block(0, 0, box_nums, particle_Pc.size()).rowwise().sum();
            else
                prob.block(0, particle_Pc.size() + j, box_nums, 1) = prob.block(0, particle_Pc.size(), box_nums, 1);
        }

        double max_emt = prob.maxCoeff();
        cost = max_emt * Eigen::MatrixXd::Ones(cost.rows(), cost.cols()) - prob;

        // cout << "prob: " << endl << prob << endl;

        Hungary hungary(max_edge, cost);
        vector<int> result = hungary.solve();

        particles[i].associations.clear();
        particles[i].associations.resize(box_nums);
        for(int j = 0; j < result.size(); ++j){
            if(j >= box_nums)
                break;
            if(result[j] < particle_Pc.size()){
                particles[i].weight *= prob(j, result[j]);
                particles[i].associations[j] = particle_id[result[j]];
            }
            else{
                particles[i].associations[j] = -1;
                if(prob(j, result[j]) <= 0.0){
                    particles[i].weight *= 0.0001;
                }
                else{
                    particles[i].weight *= 0.0001;
                }
            }
        }

        if(particles[i].weight > best_weight){
            best_weight = particles[i].weight;
            best_prob = prob;
        }
        weight_normalizer += particles[i].weight;
    }

    for(int i = 0; i < particles.size(); ++i){
        particles[i].weight /= weight_normalizer;
        weights_[i] = particles[i].weight;
    }

    cout << "best_prob: " << endl << best_prob << endl;

}

void ParticleFilter::resample(){

    double sum_weight2 = 0.0;
    for (int i = 0; i < particles.size(); ++i){
        sum_weight2 += weights_[i] * weights_[i];
    }
    double N_eff = 1.0 / sum_weight2;
    cout << "N_eff:  " << N_eff << endl;
    cout << "particles: " << particles.size() << endl;
    if(N_eff / particles.size() >= 0.4)
        return;
    
    vector<Particle> resampled_particles;
    default_random_engine gen;

	uniform_int_distribution<int> particle_index(0, num_particles_ - 1);
	
	int current_index = particle_index(gen);
	
	double beta = 0.0;
	
	double max_weight_2 = 2.0 * *max_element(weights_.begin(), weights_.end());
	
	for (int i = 0; i < particles.size(); ++i) {
		uniform_real_distribution<double> random_weight(0.0, max_weight_2);
		beta += random_weight(gen);

        while (beta > weights_[current_index]) {
            beta -= weights_[current_index];
            current_index = (current_index + 1) % num_particles_;
        }
	    resampled_particles.push_back(particles[current_index]);
	}
	particles = resampled_particles;
}