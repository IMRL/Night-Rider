#include "map_relocalization/sampler.hpp"
#include "map_relocalization/hungary_estimator.hpp"
#include <armadillo>
#include <random>

// void Sampler::init(const vec_vec3d& points, int num_samples, double init_x, double init_y, double init_z, double init_alpha, double init_beta, double init_gamma){
//     if(init_x > -10000 && init_y > -10000 && init_z > -10000)
//         num_samples = num_samples;
//     else
//         num_samples = 4 * num_samples;
//     num_samples_ = num_samples;
//     weights_.resize(num_samples_);
//     samples.resize(num_samples_);

//     double min_x = 2000, min_y = 2000, min_z = 2000, max_x = -2000, max_y = -2000, max_z = -2000;
//     for(int i = 0; i < points.size(); ++i){
//         Eigen::Vector3d point = points[i];
//         min_x = min(min_x, point.x()) - 1;
//         min_y = min(min_y, point.y()) - 1;
//         min_z = min(min_z, point.z());
//         max_x = max(max_x, point.x()) + 1;
//         max_y = max(max_y, point.y()) + 1;
//         max_z = max(max_z, point.z());
//     }

//     cov_ = Eigen::Vector2d(2, 2).asDiagonal();

//     cv::RNG rng(cv::getTickCount());
//     int N = 0;
//     double weight = 1.0 / num_samples;
//     while( N < num_samples){
//         double x, y, z, alpha, beta, gamma;
//         if(init_x > -10000 && init_y > -10000 && init_z > -10000 && init_alpha > -10000 && init_beta > -10000 && init_gamma > -10000){
//             x = init_x + rng.gaussian(2);
//             y = init_y + rng.gaussian(2);
//             z = init_z + rng.gaussian(0.2);
//             alpha = init_alpha;
//             beta = init_beta;
//             gamma = init_gamma + rng.gaussian(PI / 10);
//         }
//         else if(init_x > -10000 && init_y > -10000 && init_z > -10000){
//             x = init_x + rng.gaussian(5);
//             y = init_y + rng.gaussian(5);
//             z = init_z + rng.gaussian(0.5);
//             alpha = 0.0;
//             beta = 0.0;
//             gamma = rng.uniform(-PI, PI);
//         }
//         else{
//             x = rng.uniform(min_x, max_x);
//             y = rng.uniform(min_y, max_y);
//             z = rng.uniform(min_z, max_z);
//             alpha = 0.0;
//             beta = 0.0;
//             gamma = rng.uniform(-PI, PI);
//         }
        
//         Eigen::Vector3d pos_sample = Eigen::Vector3d(x, y, z);
//         Eigen::Vector3d rot_sample = Eigen::Vector3d(gamma, beta, alpha);
//         cout << "init_euler: " << rot_sample.transpose() << endl; 
//         bool exist_left_lamp = false, exist_right_lamp = false, exist_top_lamp = false, exist_down_lamp = false;
//         for(int i = 0; i < points.size(); ++i){
//             Eigen::Vector3d point = points[i];
//             if((point - pos_sample).norm() < 40){
//                 if(point.y() - y < 13 && point.y() - y >= 0)
//                     exist_right_lamp = true;
//                 if(point.y() - y < 0 && point.y() - y >= -13)
//                     exist_left_lamp = true;
//                 if(point.z() - z < 2.7 && point.z() - z >= 0)
//                     exist_top_lamp = true;
//                 if(point.z() - z < 0 && point.z() - z >= -0.5)
//                     exist_down_lamp = true;
//             }
//         }
//         if(!exist_left_lamp || !exist_right_lamp || !exist_top_lamp || !exist_down_lamp)
//             continue;
        
//         Sample sample;
//         sample.pos = pos_sample;
//         sample.rot = rot_sample;
//         sample.weight = weight;
        
//         samples[N] = sample;
//         weights_[N] = weight;
//         ++N;
//     }
//     initFinish = true;
// }

void Sampler::init(const vec_vec3d& points, int num_samples, double yaw, double init_x_gps, double cov_init_x_gps, double init_y_gps, double cov_init_y_gps, double init_alpha, double init_beta, double init_gamma){
    num_samples_ = num_samples;
    weights_.resize(num_samples_);
    samples.resize(num_samples_);

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

    double init_x, init_y;
    init_x = cos(yaw) * init_x_gps + sin(yaw) * init_y_gps;
    init_y = -sin(yaw) * init_x_gps + cos(yaw) * init_y_gps;
    init_x_ = init_x, init_y_ = init_y;
    cov_ = Eigen::Vector2d(cov_init_x_gps + 0.5, cov_init_y_gps + 0.5).asDiagonal();
    cout << "init_x: " << init_x << " init_y: " << init_y << endl;

    double local_min_z = 2000, local_max_z = -2000;
    for(int i = 0; i < points.size(); ++i){
        Eigen::Vector3d point = points[i];
        // cout << point.z() << endl;
        if((point.x() - init_x) * (point.x() - init_x) + (point.y() - init_y) * (point.y() - init_y) < 3600){
            local_min_z = min(local_min_z, point.z());
            local_max_z = max(local_max_z, point.z());
        }
    }
    local_min_ = local_min_z;
    if(local_max_z - 2 > local_min_z)
        local_max_ = local_max_z - 2.0;
    else
        local_max_ = local_max_z;
    cout << "local_min: " << local_min_ << " local_max: " << local_max_ << endl;
    
    cv::RNG rng(cv::getTickCount());
    int N = 0;
    double weight = 1.0 / num_samples;
    while( N < num_samples){
        double x, y, z, alpha, beta, gamma;
        // x = init_x + rng.gaussian(sqrt(cov_init_x_gps) + 0.5);
        // y = init_y + rng.gaussian(sqrt(cov_init_y_gps) + 0.5);
        x = rng.uniform(init_x - 3 * cov_init_x_gps - 1, init_x + 3 * cov_init_x_gps + 1);
        y = rng.uniform(init_y - 3 * cov_init_y_gps - 1, init_y + 3 * cov_init_y_gps + 1);
        z = rng.uniform(local_min_ - 0.5, local_max_);
        // cout << "z: " << z << endl;

        if(init_alpha > -10000 && init_beta > -10000 && init_gamma > -10000){
            alpha = init_alpha;
            beta = init_beta;
            gamma = init_gamma + rng.gaussian(PI / 12);
        }
        else{
            alpha = 0;
            beta = 0;
            gamma = rng.uniform(-PI, PI);
        }
        
        Eigen::Vector3d pos_sample = Eigen::Vector3d(x, y, z);
        Eigen::Vector3d rot_sample = Eigen::Vector3d(gamma, beta, alpha);
        bool exist_left_lamp = false, exist_right_lamp = false, exist_top_lamp = false, exist_down_lamp = false;
        for(int i = 0; i < points.size(); ++i){
            Eigen::Vector3d point = points[i];
            if((point - pos_sample).norm() < 40){
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
        
        Sample sample;
        sample.pos = pos_sample;
        sample.rot = rot_sample;
        sample.weight = weight;
        
        samples[N] = sample;
        weights_[N] = weight;
        ++N;
    }
    initFinish = true;
}

void Sampler::init(const vec_vec3d& points, int num_samples, double init_x, double init_y, double init_z, double init_alpha, double init_beta, double init_gamma){
    num_samples_ = num_samples;
    weights_.resize(num_samples_);
    samples.resize(num_samples_);

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
    double weight = 1.0 / num_samples;
    while( N < num_samples){
        double x, y, z, alpha, beta, gamma;
        // x = init_x + rng.gaussian(sqrt(cov_init_x_gps) + 0.5);
        // y = init_y + rng.gaussian(sqrt(cov_init_y_gps) + 0.5);
        x = init_x + rng.gaussian(0.2);
        y = init_y + rng.gaussian(0.2);
        z = init_z + rng.gaussian(0.1);
        // cout << "z: " << z << endl;

        alpha = 0;
        beta = 0;
        gamma = rng.gaussian(0.1);
        
        Eigen::Vector3d pos_sample = Eigen::Vector3d(x, y, z);
        Eigen::Vector3d rot_sample = Eigen::Vector3d(gamma, beta, alpha);
        bool exist_left_lamp = false, exist_right_lamp = false, exist_top_lamp = false, exist_down_lamp = false;
        for(int i = 0; i < points.size(); ++i){
            Eigen::Vector3d point = points[i];
            if((point - pos_sample).norm() < 40){
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
        
        Sample sample;
        sample.pos = pos_sample;
        sample.rot = rot_sample;
        sample.weight = weight;
        
        samples[N] = sample;
        weights_[N] = weight;
        ++N;
    }
    initFinish = true;
}

void Sampler::init(const vec_vec3d& points, int num_samples, const Eigen::Vector3d& init_pos, const Eigen::Matrix3d& cov, double init_alpha, double init_beta, double init_gamma){
    num_samples_ = num_samples;
    weights_.resize(num_samples_);
    samples.resize(num_samples_);

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

    cout << "init_x: " << init_pos.x() << " init_y: " << init_pos.y() << endl;

    double local_min_z = 2000, local_max_z = -2000;
    for(int i = 0; i < points.size(); ++i){
        Eigen::Vector3d point = points[i];
        if((point.x() - init_pos.x()) * (point.x() - init_pos.x()) + (point.y() - init_pos.y()) * (point.y() - init_pos.y()) < 3600){
            // cout << point.z() << endl;
            local_min_z = min(local_min_z, point.z());
            local_max_z = max(local_max_z, point.z());
        }
    }
    local_min_ = local_min_z;
    if(local_max_z - 2 > local_min_z)
        local_max_ = local_max_z - 2.0;
    else
        local_max_ = local_max_z;
    cout << "local_min: " << local_min_ << " local_max: " << local_max_ << endl;

    cov_ = cov.block<2, 2>(0, 0);
    Eigen::Matrix2d L = cov_.llt().matrixL();
    Eigen::Vector2d sigma_normal = L * Eigen::Vector2d(1, 1);
    cout << "sigma_normal: " << sigma_normal.transpose() << endl;
    
    cv::RNG rng(cv::getTickCount());
    int N = 0;
    double weight = 1.0 / num_samples;
    while( N < num_samples){
        double x, y, z, alpha, beta, gamma;
        // x = init_x + rng.gaussian(sqrt(cov_init_x_gps) + 0.5);
        // y = init_y + rng.gaussian(sqrt(cov_init_y_gps) + 0.5);
        x = rng.uniform(init_pos.x() - 2 * sigma_normal.x() - 1, init_pos.x() + 2 * sigma_normal.x() + 1);
        y = rng.uniform(init_pos.y() - 2 * sigma_normal.y() - 1, init_pos.y() + 2 * sigma_normal.y() + 1);
        z = rng.uniform(local_min_, local_max_);
        // cout << "z: " << z << endl;

        if(init_alpha > -10000 && init_beta > -10000 && init_gamma > -10000){
            alpha = init_alpha;
            beta = init_beta;
            gamma = init_gamma + rng.gaussian(PI / 6);
        }
        else{
            alpha = 0;
            beta = 0;
            gamma = rng.uniform(-PI, PI);
        }
        
        Eigen::Vector3d pos_sample = Eigen::Vector3d(x, y, z);
        Eigen::Vector3d rot_sample = Eigen::Vector3d(gamma, beta, alpha);
        bool exist_left_lamp = false, exist_right_lamp = false, exist_top_lamp = false, exist_down_lamp = false;
        for(int i = 0; i < points.size(); ++i){
            Eigen::Vector3d point = points[i];
            if((point - pos_sample).norm() < 40){
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
        
        Sample sample;
        sample.pos = pos_sample;
        sample.rot = rot_sample;
        sample.weight = weight;
        
        samples[N] = sample;
        weights_[N] = weight;
        ++N;
    }
    initFinish = true;
}

void Sampler::updateWeights(const Measures& data, const vec_vec3d& points, const Eigen::Matrix3d& Rcb, const Eigen::Vector3d& pcb, const int& res_x, const int& res_y){

    if(data.box->bounding_boxes.size() == 0){
        cout << "no boxes exist! initialization failed!" << endl;
        return;
    }
    double weight_normalizer = 0.0;
    double best_weight = 0;
    Eigen::MatrixXd best_prob;

    // vector<vector<double>> samples_dist(num_samples_, vector<double>(data.box->bounding_boxes.size(), -1));
    // vector<double> min_dist(data.box->bounding_boxes.size(), 10000);
    for(int i = 0; i < samples.size(); ++i){
        Eigen::Matrix3d sample_Rwb;
        sample_Rwb = Eigen::AngleAxisd(samples[i].rot[0], Eigen::Vector3d::UnitZ()) * 
                        Eigen::AngleAxisd(samples[i].rot[1], Eigen::Vector3d::UnitY()) * 
                        Eigen::AngleAxisd(samples[i].rot[2], Eigen::Vector3d::UnitX());
        Eigen::Matrix3d sample_Rcw = Rcb * sample_Rwb.transpose();
        Eigen::Vector3d sample_pwb = samples[i].pos;
        Eigen::Vector3d sample_pcw = - Rcb * sample_Rwb.transpose() * sample_pwb + pcb;

        vec_vec3d sample_Pc;
        vec_vec2d sample_pt;
        vector<int> sample_id;
        for (int j = 0; j < points.size(); j++){
            if(isnan(points[j].norm())) continue;
            Eigen::Vector3d Pc;
            Pc = sample_Rcw * points[j] + sample_pcw;
            double inv_z = 1.0 / Pc.z();
            // Pc = Pc * inv_z;

            Eigen::Vector2d pt;
            pt << cam_fx * inv_z * Pc.x() + cam_cx, cam_fy * inv_z * Pc.y() + cam_cy;

            if ((sqrt(Pc.x() * Pc.x() + Pc.z() * Pc.z()) < search_dist_scope_) && Pc.z() < search_z_scope_ && Pc.z() > 1.0 && pt.x() < res_x - 5 && pt.x() >= 5 && pt.y() < res_y - 5 && pt.y() >= 5){ //55, 45
                // cout << "Pc: " << Pc << endl;
                // bool find_rep = false;
                // for (int k = 0; k < sample_Pc.size(); k++){
                //     Eigen::Vector3d l1 = sample_Pc[k].normalized();
                //     Eigen::Vector3d l2 = Pc.normalized();
                //     // cout << "cos_theta: " << l1.transpose() * l2 << endl;
                //     if((l1.transpose() * l2)(0) > 0.998){
                //         double n1 = sample_Pc[k].norm();
                //         double n2 = Pc.norm();
                //         find_rep = true;
                //         if(n1 > n2){
                //             sample_Pc[k] = Pc;
                //             sample_pt[k] = pt;
                //             sample_id[k] = j;
                //         }
                //         break;
                //     }
                // }
                // if(find_rep) continue;
                sample_Pc.push_back(Pc);
                sample_pt.push_back(pt);
                sample_id.push_back(j);
            }
        }

        if(sample_Pc.size() == 0){
            // cout << "no lamp exists around this sample, assign weight to 0." << endl;
            samples[i].weight = 0;
            continue;
        }

        int box_nums = data.box->bounding_boxes.size();
        Eigen::MatrixXd sample_md_matrix = Eigen::MatrixXd::Zero(box_nums, sample_Pc.size() + 1);
        Eigen::MatrixXd sample_ad_matrix = Eigen::MatrixXd::Zero(box_nums, sample_Pc.size() + 1);
        
        for (int j = 0; j < box_nums; j++){
            double center_x = (data.box->bounding_boxes[j].xmax + data.box->bounding_boxes[j].xmin) / 2;
            double center_y = (data.box->bounding_boxes[j].ymax + data.box->bounding_boxes[j].ymin) / 2;

            double sigma2_x = data.box->bounding_boxes[j].xmax - data.box->bounding_boxes[j].xmin;
            double sigma2_y = data.box->bounding_boxes[j].ymax - data.box->bounding_boxes[j].ymin;
            // double sigma2_x = 400;
            // double sigma2_y = 400;

            sigma2_x = 1 * sigma2_x * sigma2_x + 20 / sigma2_x;
            sigma2_y = 1 * sigma2_y * sigma2_y + 20 / sigma2_y;

            Eigen::Matrix2d sigma2 = Eigen::Matrix2d::Identity();
            sigma2(0, 0) = sigma2_x;
            sigma2(1, 1) = sigma2_y;

            // cout << data.box->bounding_boxes[j].xmax << " " << data.box->bounding_boxes[j].xmin << " " << data.box->bounding_boxes[j].ymax << " " << data.box->bounding_boxes[j].ymin << endl;

            for (int k = 0; k < sample_pt.size(); k++){
                sample_md_matrix(j, k) = md_distance(Eigen::Vector2d(center_x, center_y), sample_pt[k], sigma2);
                // sample_ad_matrix(j, k) = ang_distance(Eigen::Vector2d(center_x, center_y), sample_Pc[k], sigma2);
            }
        }

        // cout << "md_distance: " << endl << sample_md_matrix << endl;
        // cout << "ang_distance: " << endl << sample_ad_matrix << endl;

        int max_edge = box_nums + int(sample_Pc.size());
        Eigen::MatrixXd cost = Eigen::MatrixXd::Zero(max_edge, max_edge), prob = Eigen::MatrixXd::Zero(max_edge, max_edge);
        prob.block(0, 0, box_nums, sample_Pc.size() + 1) = sample_md_matrix;

        for(int j = 0; j < box_nums; j++){
            if(j == 0)
                prob.block(0, sample_Pc.size(), box_nums, 1) = Eigen::VectorXd::Ones(box_nums) - prob.block(0, 0, box_nums, sample_Pc.size()).rowwise().sum();
            else
                prob.block(0, sample_Pc.size() + j, box_nums, 1) = prob.block(0, sample_Pc.size(), box_nums, 1);
        }

        double max_emt = prob.maxCoeff();
        cost = max_emt * Eigen::MatrixXd::Ones(cost.rows(), cost.cols()) - prob;

        // cout << "prob: " << endl << prob << endl;

        Hungary hungary(max_edge, cost);
        vector<int> result = hungary.solve();

        // vector<double> dists(box_nums, -1);

        samples[i].associations.clear();
        samples[i].associations.resize(box_nums);
        for(int j = 0; j < result.size(); ++j){
            if(j >= box_nums)
                break;
            if(result[j] < sample_Pc.size()){
                samples[i].weight *= prob(j, result[j]) * (1.0 / sample_Pc[result[j]].norm());
                samples[i].associations[j] = sample_id[result[j]];
                // dists[j] = sample_Pc[result[j]].norm();
                // min_dist[j] = min(min_dist[j], dists[j]);
            }
            else{
                samples[i].associations[j] = -1;
                if(prob(j, result[j]) <= 0.0){
                    samples[i].weight *= 0.0001;
                }
                else{
                    samples[i].weight *= 0.0001;
                }
            }
        }
        // samples_dist[i] = dists;

        if(samples[i].weight > best_weight){
            best_weight = samples[i].weight;
            best_prob = prob;
        }
        weight_normalizer += samples[i].weight;
    }

    // for(int i = 0; i < samples.size(); ++i){
    //     for(int j = 0; j < data.box->bounding_boxes.size(); ++j){
    //         if(min_dist[j] < 10000 && samples_dist[i][j] > 0){
    //             samples[i].weight *= pow(min_dist[j] / samples_dist[i][j], 2);
    //         }
    //     }
    //     weight_normalizer += samples[i].weight;
    //     if(samples[i].weight > best_weight){
    //         best_weight = samples[i].weight;
    //         best_prob = prob;
    //     }
    // }

    for(int i = 0; i < samples.size(); ++i){
        samples[i].weight /= weight_normalizer;
        weights_[i] = samples[i].weight;
    }

    cout << "best_prob: " << endl << best_prob << endl;
}

double Sampler::md_distance(const Eigen::Vector2d& box_center, const Eigen::Vector2d& proj_center, const Eigen::Matrix2d& cov){
    Eigen::Vector2d err(box_center - proj_center);
    // cout << "err " << err.norm() << endl;
    Eigen::Matrix2d cov_ld = cov.llt().matrixL();
    // cout << "cov_ld: " << endl << cov_ld << endl;
    Eigen::Vector2d vld = cov_ld.inverse() * err;
    // cout << "vld " << endl << vld << endl;
    return exp(- 0.5 * vld.transpose() * vld);
}

double Sampler::ang_distance(const Eigen::Vector2d& box_center, const Eigen::Vector3d& lamp_center, const Eigen::Matrix2d& sigma2){
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

void Sampler::sortWeight(){

    vector<pair<double, int>> weight_id;
    weight_id.resize(weights_.size());
    for(int i = 0; i < weights_.size(); ++i){
        weight_id[i] = make_pair(weights_[i], i);
    }
    sort(weight_id.begin(), weight_id.end(), comp_vdp_down);

    vector<Sample> resamples;
    resamples.resize(num_samples_);
    for(int i = 0; i < weight_id.size(); ++i){
        resamples[i] = samples[weight_id[i].second];
        weights_[i] = weight_id[i].first;
        if(i < k_){
            best_samples[i] = resamples[i];
        }
    }
    samples = resamples;
}

void Sampler::resample(const int iter){

    cout << "----------------iter " << iter << "-------------------" << endl;
    if(iter < 3 && 0){
        // vector<Sample> resamples;
        // vector<double> reweights;

        // bool occ[10][10][8];
        // for(int i = 0; i < 10; ++i){
        //     for(int j = 0; j < 10; ++j){
        //         for(int k = 0; k < 8; ++k){
        //             occ[i][j][k] = false;
        //         }
        //     }
        // }

        // int n = 0;
        // for(int i = 0; i < num_samples_; i++){
        //     Eigen::Vector3d pos = samples[i].pos;
        //     if(pos.x() - init_x_ - 2 * cov_x_ >= 0 || pos.x() - init_x_ + 2 * cov_x_ <= 0 || pos.y() - init_y_ - 2 * cov_y_ >= 0 || pos.y() - init_y_ + 2 * cov_y_ <= 0 || pos.z() <= local_min_ || pos.z() >= local_max_){
        //         continue;
        //     }

        //     default_random_engine gen;
        //     int index_x = int((pos.x() - init_x_ + 2 * cov_x_) / region_x_);
        //     int index_y = int((pos.y() - init_y_ + 2 * cov_y_) / region_y_);
        //     int index_z = int((pos.z() - local_min_) / region_z_);
        //     resamples.push_back(samples[i]);
        //     reweights.push_back(weights_[i]);
        //     ++n;

        //     if(!occ[index_x][index_y][index_z]){
        //         occ[index_x][index_y][index_z] = true;
        //         for(int j = 0; j < 19; ++j){
        //             Sample sample;
        //             normal_distribution<double> dist_x(samples[j].pos[0], region_x_ / 4);
        //             normal_distribution<double> dist_y(samples[j].pos[1], region_y_ / 4);
        //             normal_distribution<double> dist_z(samples[j].pos[2], region_z_ / 4);
        //             normal_distribution<double> dist_roll(samples[j].rot[2], 0.0001);
        //             normal_distribution<double> dist_pitch(samples[j].rot[1], 0.0001);
        //             normal_distribution<double> dist_yaw(samples[j].rot[0], 0.1 * PI);

        //             Eigen::Vector3d pos, rot;
        //             pos.x() = dist_x(gen);
        //             pos.y() = dist_y(gen);
        //             pos.z() = dist_z(gen);
        //             rot.x() = dist_yaw(gen);
        //             rot.y() = dist_pitch(gen);
        //             rot.z() = dist_roll(gen);

        //             sample.rot = rot;
        //             sample.pos = pos;
        //             sample.associations = samples[i].associations;
        //             sample.weight = samples[i].weight;
        //             resamples.push_back(sample);
        //             reweights.push_back(weights_[i]);
        //             ++n;
        //         }
        //     }
        // }

        // //补充采样数量至num_samples_
        // double weight_normalizer = 0.0;
        // for(int i = 0; i < k_; ++i){
        //     weight_normalizer += weights_[i];
        // }
        // vector<double> sum_weight(k_, 0);
        // for(int i = 0; i < k_; ++i){
        //     cout << "pos: " << samples[i].pos.transpose() << endl;
        //     cout << "rot: " << samples[i].rot.transpose() << endl;
        //     weights_[i] = weights_[i] / weight_normalizer;
        //     if(i == 0)
        //         sum_weight[i] = weights_[i];
        //     else
        //         sum_weight[i] = sum_weight[i - 1] + weights_[i];
        // }
        // cout << "sum_weight: " << endl;
        // for(int i = 0; i < k_; ++i){
        //     cout << sum_weight[i] << " ";
        // }

        // default_random_engine gen;
        // double decay = 1.0 / iter;
        // for(int i = n; i < num_samples_; ++i){
        //     uniform_real_distribution<double> random_weight(0.0, 1.0);
        //     double beta = random_weight(gen);
        //     for (int j = 0; j < k_; ++j){
        //         if (sum_weight[j] > beta){
        //             Sample sample;
        //             normal_distribution<double> dist_x(samples[j].pos[0], cov_x_ * decay);
        //             normal_distribution<double> dist_y(samples[j].pos[1], cov_y_ * decay);
        //             normal_distribution<double> dist_z(samples[j].pos[2], 0.2 * decay);
        //             normal_distribution<double> dist_roll(samples[j].rot[2], 0.0001 * decay);
        //             normal_distribution<double> dist_pitch(samples[j].rot[1], 0.0001 * decay);
        //             normal_distribution<double> dist_yaw(samples[j].rot[0], 0.15 * PI * decay);

        //             Eigen::Vector3d pos, rot;
        //             pos.x() = dist_x(gen);
        //             pos.y() = dist_y(gen);
        //             pos.z() = dist_z(gen);
        //             rot.x() = dist_yaw(gen);
        //             rot.y() = dist_pitch(gen);
        //             rot.z() = dist_roll(gen);

        //             sample.rot = rot;
        //             sample.pos = pos;
        //             sample.associations = samples[j].associations;
        //             sample.weight = samples[j].weight;
        //             resamples.push_back(sample);
        //             reweights.push_back(weights_[i]);
        //             break;
        //         }
        //     }
        // }
        // vector<Sample>(resamples).swap(resamples);
        // vector<double>(reweights).swap(reweights);
    }
    else{
        double weight_normalizer = 0.0;
        for(int i = 0; i < k_; ++i){
            weight_normalizer += weights_[i];
        }

        vector<double> sum_weight(k_, 0);
        for(int i = 0; i < k_; ++i){
            // cout << "pos: " << samples[i].pos.transpose() << endl;
            // cout << "rot: " << samples[i].rot.transpose() << endl;
            weights_[i] = weights_[i] / weight_normalizer;
            if(i == 0)
                sum_weight[i] = weights_[i];
            else
                sum_weight[i] = sum_weight[i - 1] + weights_[i];
        }

        default_random_engine gen;
        
        // double decay = 1.0 / (pow(2, iter));
        double decay = 1.0 / iter;

        Eigen::Matrix2d decay_cov = decay * cov_;
        arma::mat arma_Pij = arma::mat(decay_cov.data(), cov_.rows(), cov_.cols(), false, false);

        for (int i = k_; i < num_samples_; ++i){
            uniform_real_distribution<double> random_weight(0.0, 1.0);
            double beta = random_weight(gen);
            for (int j = 0; j < k_; ++j){
                if (sum_weight[j] > beta){
                    Sample sample;
                    arma::vec arma_Tij = arma::vec(samples[j].pos.head<2>().data(), 2, false, false);
                    arma::mat pos_xy_arma = arma::mvnrnd(arma_Tij, arma_Pij, 1);

                    // normal_distribution<double> dist_x(samples[j].pos[0], cov_x_ * decay);
                    // normal_distribution<double> dist_y(samples[j].pos[1], cov_y_ * decay);
                    normal_distribution<double> dist_z(samples[j].pos[2], 0.2 * decay);
                    normal_distribution<double> dist_roll(samples[j].rot[2], 0.0001 * decay);
                    normal_distribution<double> dist_pitch(samples[j].rot[1], 0.0001 * decay);
                    normal_distribution<double> dist_yaw(samples[j].rot[0], 0.05 * PI * decay);

                    Eigen::Vector3d pos, rot;
                    Eigen::MatrixXd pos_xy_eig = Eigen::Map<Eigen::MatrixXd>(pos_xy_arma.memptr(), pos_xy_arma.n_rows, pos_xy_arma.n_cols);
                    assert(pos_xy_eig.rows() == 2);
                    pos.x() = pos_xy_eig(0);
                    pos.y() = pos_xy_eig(1);

                    // pos.x() = dist_x(gen);
                    // pos.y() = dist_y(gen);
                    pos.z() = dist_z(gen);
                    rot.x() = dist_yaw(gen);
                    rot.y() = 0.0;
                    rot.z() = 0.0;

                    sample.rot = rot;
                    sample.pos = pos;
                    sample.associations = samples[j].associations;
                    sample.weight = samples[j].weight;
                    samples[i] = sample;
                    weights_[i] = weights_[j];
                    break;
                }
            }
        }
    }
}