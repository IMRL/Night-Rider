#include "map_relocalization/optimizer.hpp"
#include "map_relocalization/tic_toc.h"

bool Optimizer::PoseOptimizationSingle(Eigen::Matrix3d& Rcw, Eigen::Vector3d& pcw, const unordered_map<int, int>& matches, const vec_vec4d& boxes, const vec_vec3d& lamp_world_pos, const Matrix6d cov, int nIterations){
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
    BlockSolverType::LinearSolverType* linearSolver = new g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>();

    BlockSolverType* solver_ptr = new BlockSolverType(unique_ptr<BlockSolverType::LinearSolverType>(linearSolver));

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(unique_ptr<BlockSolverType>(solver_ptr));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    g2o::VertexSE3Expmap* poseSE3 = new g2o::VertexSE3Expmap();
    poseSE3->setEstimate(g2o::SE3Quat(Rcw, pcw));
    poseSE3->setId(0);
    poseSE3->setFixed(false);
    optimizer.addVertex(poseSE3);

    EdgeSE3PriorPose* edge_pr = new EdgeSE3PriorPose();
    edge_pr->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
    edge_pr->setMeasurement(g2o::SE3Quat(Rcw, pcw));
    edge_pr->setInformation(Matrix6d::Identity());
    // cout << "cov: " << endl << cov << endl;
    
    optimizer.addEdge(edge_pr);

    // vector<g2o::EdgeSE3ProjectXYZOnlyPose*> edges;
    // for(auto iter = matches.begin(); iter != matches.end(); ++iter){

    //     Eigen::Vector2d box_center;
    //     box_center << 0.5 * (boxes[iter->first](0) + boxes[iter->first](2)), 0.5 * (boxes[iter->first](1) + boxes[iter->first](3));
        
    //     g2o::EdgeSE3ProjectXYZOnlyPose* edge = new g2o::EdgeSE3ProjectXYZOnlyPose();
    //     edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
    //     edge->setMeasurement(box_center);
    //     edge->setInformation(40 * Eigen::Matrix2d::Identity());
    //     edge->fx = cam_fx;
    //     edge->fy = cam_fy;
    //     edge->cx = cam_cx;
    //     edge->cy = cam_cy;
    //     edge->Xw = lamp_world_pos[iter->second];
    //     optimizer.addEdge(edge);
    //     edges.push_back(edge);
    // }

    vector<EdgeSE3CenterProjFixedHistPose*> edges;
    for(auto iter = matches.begin(); iter != matches.end(); ++iter){
        EdgeSE3CenterProjFixedHistPose* edge = new EdgeSE3CenterProjFixedHistPose(cam_fx, cam_fy, cam_cx, cam_cy, g2o::SE3Quat(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero()));
        edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));

        Eigen::Vector2d box_center;
        box_center << 0.5 * (boxes[iter->first](0) + boxes[iter->first](2)), 0.5 * (boxes[iter->first](1) + boxes[iter->first](3));
        double half_box_len, half_box_width;
        half_box_len = 0.5 * (boxes[iter->first](2) - boxes[iter->first](0));
        half_box_width = 0.5 * (boxes[iter->first](3) - boxes[iter->first](1));
        // Eigen::Matrix2d sigma2 = Eigen::Matrix2d::Identity();
        // sigma2(0, 0) = half_box_len * half_box_len, sigma2(1, 1) = half_box_width * half_box_width;
        // cout << "sigma2: " << endl << sigma2 << endl;
        // double dist = lamp_world_pos[iter->second].norm();

        edge->setMeasurement(box_center);
        edge->setInformation(Eigen::Matrix2d::Identity() / matches.size());
        edge->Xw_ = lamp_world_pos[iter->second];

        optimizer.addEdge(edge);

        edges.push_back(edge);

        // cout << "lamp_pos: " << lamp_world_pos[iter->second].transpose() << endl;
    }

    TicToc t_solve;
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);
    // cout << "time cost for solution is: " << t_solve.toc() << endl;

    // for(int i = 0; i < edges.size(); i++){
    //     cout << "measurement: " << endl << edges[i]->measurement() << endl;
    //     cout << "error: " << endl << edges[i]->error() << endl;
    // }

    g2o::SE3Quat update_pose = poseSE3->estimate();
    Rcw = update_pose.rotation().toRotationMatrix();
    pcw = update_pose.translation();

    // cout << "Rcw: " << endl << Rcw << endl << "pcw: " << endl << pcw.transpose() << endl;

    return true;

    // g2o::SE3Quat update_pose = poseSE3->estimate();
    // Eigen::Matrix3d delta_RR = update_pose.rotation().toRotationMatrix() * Rcw.transpose();
    // Eigen::Vector3d delta_pp = update_pose.translation() - delta_RR * pcw;
    // Eigen::AngleAxisd delta_rr(delta_RR);

    // Rcw = update_pose.rotation().toRotationMatrix();
    // pcw = update_pose.translation();
    // return true;

    // auto iter = matches.begin();
    // bool success = true;
    // avg_error_ = 0.0;
    // for(int i = 0; i < edges.size(); i++){
    //     double box_length, box_width;
    //     box_length = boxes[iter->first](2) - boxes[iter->first](0);
    //     box_width = boxes[iter->first](3) - boxes[iter->first](1);

    //     EdgeSE3CenterProjFixedHistPose* e = edges[i];
    //     e->computeError();
    //     Eigen::Vector2d error = e->error();

    //     avg_error_ += error.squaredNorm();

    //     if(error.x() > 0.5 * box_length || error.y() > 0.5 * box_width){
    //         success = false;
    //     }
    //     ++iter;
    // }

    // avg_error_ /= edges.size();

    // g2o::SE3Quat update_pose = poseSE3->estimate();
    // Eigen::Matrix3d delta_RR = update_pose.rotation().toRotationMatrix() * Rcw.transpose();
    // Eigen::Vector3d delta_pp = update_pose.translation() - delta_RR * pcw;
    // Eigen::AngleAxisd delta_rr(delta_RR);

    // if(delta_rr.angle() > 0.1 || delta_pp.norm() > 0.16)
    //     success = false;

    // if(success){
    //     Rcw = update_pose.rotation().toRotationMatrix();
    //     pcw = update_pose.translation();
    //     return true;
    // }
    // else{
    //     return false;
    // }
}

bool Optimizer::PoseOptimizationCenter(Eigen::Matrix3d& Rcw, Eigen::Vector3d& pcw, const deq_mat3d& deq_deltaR, const deq_vec3d& deq_deltap, const deque<unordered_map<int, int>>& matches, const deque<vec_vec4d, Eigen::aligned_allocator<vec_vec4d>> boxes, const vec_vec3d& lamp_world_pos, const int& frames, int nIterations){
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
    BlockSolverType::LinearSolverType* linearSolver = new g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>();

    BlockSolverType* solver_ptr = new BlockSolverType(unique_ptr<BlockSolverType::LinearSolverType>(linearSolver));

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(unique_ptr<BlockSolverType>(solver_ptr));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    g2o::VertexSE3Expmap* poseSE3 = new g2o::VertexSE3Expmap();
    poseSE3->setEstimate(g2o::SE3Quat(Rcw, pcw));
    poseSE3->setId(0);
    poseSE3->setFixed(false);
    optimizer.addVertex(poseSE3);

    Eigen::Matrix3d delta_R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d delta_p = Eigen::Vector3d::Zero();
    vector<vector<EdgeSE3CenterProjFixedHistPose*>> frame_edges;
    // cout << "frames: " << frames << endl;
    for(int i = frames; i > 0; i--){
        delta_R = deq_deltaR[i] * delta_R;
        delta_p = deq_deltaR[i] * delta_p + deq_deltap[i];

        vector<EdgeSE3CenterProjFixedHistPose*> edges;
        // cout << "matches: " << matches[frames].begin()->first << " " << matches[frames].begin()->second << endl;
        for(auto iter = matches[i].begin(); iter != matches[i].end(); ++iter){
            EdgeSE3CenterProjFixedHistPose* edge = new EdgeSE3CenterProjFixedHistPose(cam_fx, cam_fy, cam_cx, cam_cy, g2o::SE3Quat(delta_R, delta_p));
            edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));

            Eigen::Vector2d box_center;
            box_center << 0.5 * (boxes[i][iter->first](0) + boxes[i][iter->first](2)), 0.5 * (boxes[i][iter->first](1) + boxes[i][iter->first](3));
            // cout << "box_center: " << endl << box_center << endl;
            edge->setMeasurement(box_center);
            edge->setInformation(2 * Eigen::Matrix2d::Identity());
            edge->Xw_ = lamp_world_pos[iter->second];

            optimizer.addEdge(edge);

            edges.push_back(edge);
        }
        frame_edges.push_back(edges);
    }

    TicToc t_solve;
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);
    // cout << "time cost for solution is: " << t_solve.toc() << endl;

    bool success = true;
    auto iter = matches[frames].begin();
    avg_error_ = 0.0;
    for(int j = 0; j < frame_edges[0].size(); j++){
        double box_length, box_width;
        box_length = boxes[frames][iter->first](2) - boxes[frames][iter->first](0);
        box_width = boxes[frames][iter->first](3) - boxes[frames][iter->first](1);

        EdgeSE3CenterProjFixedHistPose* e = frame_edges[0][j];
        e->computeError();
        Eigen::Vector2d error = e->error();

        avg_error_ += error.squaredNorm();
        if(error.x() > 0.85 * box_length || error.y() > 0.85 * box_width){
            success = false;
        }
        ++iter;
    }

    avg_error_ /= frame_edges[0].size();

    g2o::SE3Quat update_pose = poseSE3->estimate();
    Eigen::Matrix3d delta_RR = update_pose.rotation().toRotationMatrix() * Rcw.transpose();
    Eigen::Vector3d delta_pp = update_pose.translation() - delta_RR * pcw;
    Eigen::AngleAxisd delta_rr(delta_RR);

    if(delta_rr.angle() > 0.1 || delta_pp.norm() > 0.16)
        success = false;

    if(success){
        Rcw = update_pose.rotation().toRotationMatrix();
        pcw = update_pose.translation();
        return true;
    }
    else{
        return false;
    }
}

bool Optimizer::PoseOptimizationAllPt(Eigen::Matrix3d& Rcw, Eigen::Vector3d& pcw, const deq_mat3d& deq_deltaR, const deq_vec3d& deq_deltap, const deque<unordered_map<int, int>>& matches, const deque<vec_vec4d, Eigen::aligned_allocator<vec_vec4d>> boxes, const vector<vec_vec3d, Eigen::aligned_allocator<vec_vec3d>>& lamp_world_points, const int& frames, int nIterations){
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
    BlockSolverType::LinearSolverType* linearSolver = new g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>();

    BlockSolverType* solver_ptr = new BlockSolverType(unique_ptr<BlockSolverType::LinearSolverType>(linearSolver));

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(unique_ptr<BlockSolverType>(solver_ptr));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    g2o::VertexSE3Expmap* poseSE3 = new g2o::VertexSE3Expmap();
    poseSE3->setEstimate(g2o::SE3Quat(Rcw, pcw));
    poseSE3->setId(0);
    poseSE3->setFixed(false);
    optimizer.addVertex(poseSE3);

    Eigen::Matrix3d delta_R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d delta_p = Eigen::Vector3d::Zero();
    vector<vector<EdgeSE3AllProjFixedHistPose*>> frame_edges;
    for(int i = frames; i > 0; i--){
        delta_R = deq_deltaR[i] * delta_R;
        delta_p = deq_deltaR[i] * delta_p + deq_deltap[i];

        vector<EdgeSE3AllProjFixedHistPose*> edges;
        for(auto iter = matches[i].begin(); iter != matches[i].end(); ++iter){
            for(int j = 0; j < lamp_world_points[iter->second].size(); j++){
                EdgeSE3AllProjFixedHistPose* edge = new EdgeSE3AllProjFixedHistPose(cam_fx, cam_fy, cam_cx, cam_cy, g2o::SE3Quat(delta_R, delta_p));
                edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));

                Eigen::Vector2d box_center;
                box_center << 0.5 * (boxes[i][iter->first](0) + boxes[i][iter->first](2)), 0.5 * (boxes[i][iter->first](1) + boxes[i][iter->first](3));
                edge->setMeasurement(box_center);
                edge->setInformation(2 * Eigen::Matrix2d::Identity());
                edge->Xw_ = lamp_world_points[iter->second][j];
                edge->x_min_ = boxes[i][iter->first](0);
                edge->x_max_ = boxes[i][iter->first](2);
                edge->y_min_ = boxes[i][iter->first](1);
                edge->y_max_ = boxes[i][iter->first](3);

                optimizer.addEdge(edge);

                edges.push_back(edge);
            }
        }
        frame_edges.push_back(edges);
    }

    TicToc t_solve;
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);
    cout << "time cost for solution is: " << t_solve.toc() << endl;

    bool success = true;
    auto iter = matches[frames].begin();
    int point_num = lamp_world_points[iter->second].size();
    avg_error_ = 0.0;
    for(int j = 0; j < frame_edges[0].size(); j++){
        double box_length, box_width;
        box_length = boxes[frames][iter->first](2) - boxes[frames][iter->first](0);
        box_width = boxes[frames][iter->first](3) - boxes[frames][iter->first](1);

        EdgeSE3AllProjFixedHistPose* e = frame_edges[0][j];
        e->computeError();
        Eigen::Vector2d error = e->error();

        avg_error_ += error.squaredNorm();

        if(error.x() > 0.85 * box_length || error.y() > 0.85 * box_width){
            success = false;
        }
        if(j >= point_num - 1){
            ++iter;
            point_num = lamp_world_points[iter->second].size();
        }
    }

    avg_error_ /= frame_edges[0].size();

    g2o::SE3Quat update_pose = poseSE3->estimate();
    Eigen::Matrix3d delta_RR = update_pose.rotation().toRotationMatrix() * Rcw.transpose();
    Eigen::Vector3d delta_pp = update_pose.translation() - delta_RR * pcw;
    Eigen::AngleAxisd delta_rr(delta_RR);

    if(delta_rr.angle() > 0.1 || delta_pp.norm() > 0.16)
        success = false;

    if(success){
        Rcw = update_pose.rotation().toRotationMatrix();
        pcw = update_pose.translation();
        return true;
    }
    else{
        return false;
    }
}

bool Optimizer::PoseOptimizationEpline(Eigen::Matrix3d& Rcw, Eigen::Vector3d& pcw, const deq_mat3d& deq_Rcw, const deq_vec3d& deq_pcw, const deque<unordered_map<int, int>>& matches, const deque<vec_vec4d, Eigen::aligned_allocator<vec_vec4d>> boxes, const vec_vec3d& lamp_world_pos, const int& frames, int nIterations){
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
    BlockSolverType::LinearSolverType* linearSolver = new g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>();

    BlockSolverType* solver_ptr = new BlockSolverType(unique_ptr<BlockSolverType::LinearSolverType>(linearSolver));

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(unique_ptr<BlockSolverType>(solver_ptr));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    Eigen::Quaterniond qcw(Rcw);
    Eigen::Matrix<double, 7, 1> tmp;
    tmp << qcw.w(), qcw.x(), qcw.y(), qcw.z(), pcw.x(), pcw.y(), pcw.z();

    VertexSO3Expt* poseSE3 = new VertexSO3Expt();
    poseSE3->setEstimate(tmp);
    poseSE3->setId(0);
    poseSE3->setFixed(false);
    optimizer.addVertex(poseSE3);

    Eigen::Matrix3d delta_R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d delta_p = Eigen::Vector3d::Zero();

    vector<EdgeSE3CenterProjFixedHistPose*> edges_proj;
    for(auto iter = matches[frames].begin(); iter != matches[frames].end(); ++iter){
        EdgeSE3CenterProjFixedHistPose* edge_proj = new EdgeSE3CenterProjFixedHistPose(cam_fx, cam_fy, cam_cx, cam_cy, g2o::SE3Quat(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero()));
        edge_proj->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        
        Eigen::Vector2d box_center;
        box_center << 0.5 * (boxes[frames][iter->first](0) + boxes[frames][iter->first](2)), 0.5 * (boxes[frames][iter->first](1) + boxes[frames][iter->first](3));
        edge_proj->setMeasurement(box_center);
        edge_proj->setInformation(0.5 * Eigen::Matrix2d::Identity());
        edge_proj->Xw_ = lamp_world_pos[iter->second];
        
        optimizer.addEdge(edge_proj);
        
        edges_proj.push_back(edge_proj);
    }

    vector<EdgeSO3tEplineFixedHistPose*> edges_ep;
    for(int i = frames - 1; i > 0; i--){
        for(auto iter_cur = matches[frames].begin(); iter_cur != matches[frames].end(); ++iter_cur){
            Eigen::Vector3d box_center_cur;
            box_center_cur << 0.5 * (boxes[frames][iter_cur->first](0) + boxes[frames][iter_cur->first](2)), 0.5 * (boxes[frames][iter_cur->first](1) + boxes[frames][iter_cur->first](3)), 1;
            box_center_cur.x() = (box_center_cur.x() - cam_cx) / cam_fx;
            box_center_cur.y() = (box_center_cur.y() - cam_cy) / cam_cy;

            for(auto iter_hist = matches[i].begin(); iter_hist != matches[i].end(); ++iter_hist){
                if(iter_cur->second != iter_hist->second)
                    continue;

                EdgeSO3tEplineFixedHistPose* edge_ep = new EdgeSO3tEplineFixedHistPose(deq_Rcw[i], deq_pcw[i]);
                edge_ep->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));

                Eigen::Vector3d box_center_hist;
                box_center_hist << 0.5 * (boxes[i][iter_hist->first](0) + boxes[i][iter_hist->first](2)), 0.5 * (boxes[i][iter_hist->first](1) + boxes[i][iter_hist->first](3)), 1;
                box_center_hist.x() = (box_center_hist.x() - cam_cx) / cam_fx;
                box_center_hist.y() = (box_center_hist.y() - cam_cy) / cam_cy;

                edge_ep->setMeasurement(0);
                edge_ep->setInformation(0.5 * Eigen::Matrix<double, 1, 1>::Identity());
                edge_ep->liftup_cur_ = box_center_cur;
                edge_ep->liftup_hist_ = box_center_hist;

                optimizer.addEdge(edge_ep);

                edges_ep.push_back(edge_ep);
            }
        }
    }

    TicToc t_solve;
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);
    cout << "time cost for solution is: " << t_solve.toc() << endl;

    auto iter = matches[frames].begin();
    bool success = true;
    avg_error_ = 0.0;
    for(int i = 0; i < edges_proj.size(); i++){
        double box_length, box_width;
        box_length = boxes[frames][iter->first](2) - boxes[frames][iter->first](0);
        box_width = boxes[frames][iter->first](3) - boxes[frames][iter->first](1);

        EdgeSE3CenterProjFixedHistPose* e = edges_proj[i];
        e->computeError();
        Eigen::Vector2d error = e->error();

        avg_error_ += error.squaredNorm();

        if(error.x() > 0.75 * box_length || error.y() > 0.75 * box_width){
            success = false;
        }
        ++iter;
    }

    avg_error_ /= edges_proj.size();

    Eigen::Matrix<double, 7, 1> update_pose = poseSE3->estimate();
    Eigen::Quaterniond quat(update_pose(0), update_pose(1), update_pose(2), update_pose(3));
    quat.normalize();

    Eigen::Matrix3d delta_RR = quat.toRotationMatrix() * Rcw.transpose();
    Eigen::Vector3d delta_pp = update_pose.tail<3>() - delta_RR * pcw;
    Eigen::AngleAxisd delta_rr(delta_RR);

    if(delta_rr.angle() > 0.1 || delta_pp.norm() > 0.16)
        success = false;

    if(success){
        Rcw = quat.toRotationMatrix();
        pcw = update_pose.tail<3>();
        return true;
    }
    else{
        return false;
    }
}

bool Optimizer::PoseOptimizationAngle(Eigen::Matrix3d& Rcw, Eigen::Vector3d& pcw, const deq_mat3d& deq_Rcw, const deq_vec3d& deq_pcw, const deque<unordered_map<int, int>>& matches, const deque<vec_vec4d, Eigen::aligned_allocator<vec_vec4d>> boxes, const deque<map_fid_vec5d>& features, const vec_vec3d& lamp_world_pos, const int& frames, const Matrix6d& cov, int nIterations){
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
    BlockSolverType::LinearSolverType* linearSolver = new g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>();

    BlockSolverType* solver_ptr = new BlockSolverType(unique_ptr<BlockSolverType::LinearSolverType>(linearSolver));

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(unique_ptr<BlockSolverType>(solver_ptr));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    Eigen::Quaterniond qcw(Rcw);
    Eigen::Matrix<double, 7, 1> tmp;
    tmp << qcw.w(), qcw.x(), qcw.y(), qcw.z(), pcw.x(), pcw.y(), pcw.z();

    VertexSO3Expt* poseSE3 = new VertexSO3Expt();
    poseSE3->setEstimate(tmp);
    poseSE3->setId(0);
    poseSE3->setFixed(false);
    optimizer.addVertex(poseSE3);

    EdgeSO3tPriorPose* edge_pr = new EdgeSO3tPriorPose();
    edge_pr->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
    edge_pr->setMeasurement(g2o::SE3Quat(Rcw, pcw));
    edge_pr->setInformation(cov.inverse());

    optimizer.addEdge(edge_pr);

    vector<EdgeSO3tCenterProj*> edges_proj;
    for(auto iter = matches[frames].begin(); iter != matches[frames].end(); ++iter){
        EdgeSO3tCenterProj* edge_proj = new EdgeSO3tCenterProj(cam_fx, cam_fy, cam_cx, cam_cy);
        edge_proj->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        
        Eigen::Vector2d box_center;
        box_center << 0.5 * (boxes[frames][iter->first](0) + boxes[frames][iter->first](2)), 0.5 * (boxes[frames][iter->first](1) + boxes[frames][iter->first](3));
        edge_proj->setMeasurement(box_center);
        
        double dist = lamp_world_pos[iter->second].norm();
        dist = 1 - dist / 60;
        edge_proj->setInformation(40 * dist * Eigen::Matrix2d::Identity());
        
        edge_proj->Xw_ = lamp_world_pos[iter->second];
        
        optimizer.addEdge(edge_proj);
        
        edges_proj.push_back(edge_proj);
    }

    vector<EdgeSO3tForwardAngleFixedHistPose*> edges_fw;
    vector<EdgeSO3tBackwardAngleFixedHistPose*> edges_bw;
    for(int i = frames - 1; i > 0; i--){
        for(auto iter_cur = matches[frames].begin(); iter_cur != matches[frames].end(); ++iter_cur){
            
            Eigen::Vector3d box_center_cur;
            box_center_cur << 0.5 * (boxes[frames][iter_cur->first](0) + boxes[frames][iter_cur->first](2)), 0.5 * (boxes[frames][iter_cur->first](1) + boxes[frames][iter_cur->first](3)), 1;
            cout << "box_center_cur: " << endl << box_center_cur << endl;
            box_center_cur.x() = (box_center_cur.x() - cam_cx) / cam_fx;
            box_center_cur.y() = (box_center_cur.y() - cam_cy) / cam_fy;
            
            for(auto iter_hist = matches[i].begin(); iter_hist != matches[i].end(); ++iter_hist){
                if(iter_cur->second != iter_hist->second)
                    continue;

                // forward
                EdgeSO3tForwardAngleFixedHistPose* edge_fw = new EdgeSO3tForwardAngleFixedHistPose(deq_Rcw[i], deq_pcw[i]);
                edge_fw->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));

                Eigen::Vector3d box_center_hist;
                box_center_hist << 0.5 * (boxes[i][iter_hist->first](0) + boxes[i][iter_hist->first](2)), 0.5 * (boxes[i][iter_hist->first](1) + boxes[i][iter_hist->first](3)), 1;
                cout << "box_center_hist: " << endl << box_center_hist << endl;
                box_center_hist.x() = (box_center_hist.x() - cam_cx) / cam_fx;
                box_center_hist.y() = (box_center_hist.y() - cam_cy) / cam_fy;

                cout << "box_center_cur: " << endl << box_center_cur << endl << "box_center_hist: " << endl << box_center_hist << endl; 

                edge_fw->setMeasurement(box_center_cur.normalized());
                edge_fw->setInformation(0.1 * Eigen::Matrix3d::Identity());
                edge_fw->liftup_hist_ = box_center_hist;

                optimizer.addEdge(edge_fw);

                edges_fw.push_back(edge_fw);
                
                // backward
                // EdgeSO3tBackwardAngleFixedHistPose* edge_bw = new EdgeSO3tBackwardAngleFixedHistPose(deq_Rcw[i], deq_pcw[i]);
                // edge_bw->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));

                // edge_bw->setMeasurement(box_center_hist.normalized());
                // edge_bw->setInformation(0.25 * Eigen::Matrix3d::Identity());
                // edge_bw->liftup_cur_ = box_center_cur;

                // optimizer.addEdge(edge_bw);

                // edges_bw.push_back(edge_bw);
            }
        }
        // for(auto iter_cur = features[frames].begin(); iter_cur != features[frames].end(); ++iter_cur){

        //     int feature_id = iter_cur->first;
        //     Eigen::Vector3d feature_cur = iter_cur->second.head<3>();

        //     for(auto iter_hist = features[i].begin(); iter_hist != features[i].end(); ++iter_hist){
        //         if(iter_cur->first != iter_hist->first)
        //             continue;

        //         Eigen::Vector3d feature_hist = iter_hist->second.head<3>();

        //         // forward
        //         EdgeSO3tForwardAngleFixedHistPose* edge_fw = new EdgeSO3tForwardAngleFixedHistPose(deq_Rcw[i], deq_pcw[i]);
        //         edge_fw->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));

        //         edge_fw->setMeasurement(feature_cur.normalized());
        //         edge_fw->setInformation(0.15 * Eigen::Matrix3d::Identity());
        //         edge_fw->liftup_hist_ = feature_hist;

        //         g2o::RobustKernelCauchy* rk1 = new g2o::RobustKernelCauchy();
        //         edge_fw->setRobustKernel(rk1);
        //         rk1->setDelta(1.0);

        //         optimizer.addEdge(edge_fw);

        //         edges_fw.push_back(edge_fw);
                
        //         // // backward
        //         // EdgeSO3tBackwardAngleFixedHistPose* edge_bw = new EdgeSO3tBackwardAngleFixedHistPose(deq_Rcw[i], deq_pcw[i]);
        //         // edge_bw->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));

        //         // edge_bw->setMeasurement(feature_hist.normalized());
        //         // edge_bw->setInformation(0.25 * Eigen::Matrix3d::Identity());
        //         // edge_bw->liftup_cur_ = feature_cur;

        //         // g2o::RobustKernelCauchy* rk2 = new g2o::RobustKernelCauchy();
        //         // edge_bw->setRobustKernel(rk2);
        //         // rk2->setDelta(1.0);

        //         // optimizer.addEdge(edge_bw);

        //         // edges_bw.push_back(edge_bw);
        //     }
        // }
    }

    TicToc t_solve;
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);
    cout << "time cost for solution is: " << t_solve.toc() << endl;
    edges_fw.clear();
    // cout << "edges_fw size: " << edges_fw.size() << endl;

    // auto iter = matches[frames].begin();
    // bool success = true;
    // avg_error_ = 0.0;
    // for(int i = 0; i < edges_proj.size(); i++){
    //     double box_length, box_width;
    //     box_length = boxes[frames][iter->first](2) - boxes[frames][iter->first](0);
    //     box_width = boxes[frames][iter->first](3) - boxes[frames][iter->first](1);

    //     EdgeSO3tCenterProj* e = edges_proj[i];
    //     e->computeError();
    //     Eigen::Vector2d error = e->error();

    //     avg_error_ += error.squaredNorm();

    //     if(error.x() > 0.75 * box_length || error.y() > 0.75 * box_width){
    //         success = false;
    //     }
    //     ++iter;
    // }

    // avg_error_ /= edges_proj.size();

    Eigen::Matrix<double, 7, 1> update_pose = poseSE3->estimate();
    Eigen::Quaterniond quat(update_pose(0), update_pose(1), update_pose(2), update_pose(3));
    quat.normalize();

    Eigen::Matrix3d delta_RR = quat.toRotationMatrix() * Rcw.transpose();
    Eigen::Vector3d delta_pp = update_pose.tail<3>() - delta_RR * pcw;
    Eigen::AngleAxisd delta_rr(delta_RR);

    Rcw = quat.toRotationMatrix();
    pcw = update_pose.tail<3>();
    return true;

    // if(delta_rr.angle() > 0.1 || delta_pp.norm() > 0.16)
    //     success = false;

    // if(success){
    //     Rcw = quat.toRotationMatrix();
    //     pcw = update_pose.tail<3>();
    //     return true;
    // }
    // else{
    //     return false;
    // }
}

bool Optimizer::PoseOptimizationAngleBin(Eigen::Matrix3d& Rcw, Eigen::Vector3d& pcw, deq_mat3d& deq_Rcw, deq_vec3d& deq_pcw, const deque<unordered_map<int, int>>& matches, const deque<vec_vec4d, Eigen::aligned_allocator<vec_vec4d>> boxes, const deque<map_fid_vec5d>& features, const vec_vec3d& lamp_world_pos, const int& frames, const Matrix6d& cov, int nIterations){
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
    BlockSolverType::LinearSolverType* linearSolver = new g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>();

    BlockSolverType* solver_ptr = new BlockSolverType(unique_ptr<BlockSolverType::LinearSolverType>(linearSolver));

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(unique_ptr<BlockSolverType>(solver_ptr));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    Eigen::Quaterniond qcw(Rcw);
    Eigen::Matrix<double, 7, 1> tmp;
    tmp << qcw.w(), qcw.x(), qcw.y(), qcw.z(), pcw.x(), pcw.y(), pcw.z();

    vector<VertexSO3Expt*> poseSO3ts;

    VertexSO3Expt* poseSO3t = new VertexSO3Expt();
    poseSO3t->setEstimate(tmp);
    poseSO3t->setId(0);
    poseSO3t->setFixed(false);
    poseSO3ts.push_back(poseSO3t);
    optimizer.addVertex(poseSO3t);

    EdgeSO3tPriorPose* edge_pr = new EdgeSO3tPriorPose();
    edge_pr->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
    edge_pr->setMeasurement(g2o::SE3Quat(Rcw, pcw));
    edge_pr->setInformation(cov.inverse());

    optimizer.addEdge(edge_pr);

    vector<EdgeSO3tCenterProj*> edges_proj;
    for(auto iter = matches[frames].begin(); iter != matches[frames].end(); ++iter){
        EdgeSO3tCenterProj* edge_proj = new EdgeSO3tCenterProj(cam_fx, cam_fy, cam_cx, cam_cy);
        edge_proj->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        
        Eigen::Vector2d box_center;
        box_center << 0.5 * (boxes[frames][iter->first](0) + boxes[frames][iter->first](2)), 0.5 * (boxes[frames][iter->first](1) + boxes[frames][iter->first](3));
        edge_proj->setMeasurement(box_center);
        
        double dist = lamp_world_pos[iter->second].norm();
        dist = 1 - dist / 60;
        edge_proj->setInformation(40 * dist * Eigen::Matrix2d::Identity());
        
        edge_proj->Xw_ = lamp_world_pos[iter->second];
        
        optimizer.addEdge(edge_proj);
        
        edges_proj.push_back(edge_proj);
    }

    vector<EdgeSO3tForwardAngleNotFixedHistPose*> edges_fw;
    for(int i = frames - 1; i > 0; i--){
        Eigen::Quaterniond qcw_hist(deq_Rcw[i]);
        Eigen::Matrix<double, 7, 1> tmp_hist;
        tmp_hist << qcw_hist.w(), qcw_hist.x(), qcw_hist.y(), qcw_hist.z(), deq_pcw[i].x(), deq_pcw[i].y(), deq_pcw[i].z();

        VertexSO3Expt* poseSO3t_hist = new VertexSO3Expt();
        poseSO3t_hist->setEstimate(tmp_hist);
        poseSO3t_hist->setId(frames - i);
        poseSO3t_hist->setFixed(false);
        poseSO3ts.push_back(poseSO3t_hist);
        optimizer.addVertex(poseSO3t_hist);

        for(auto iter_cur = matches[frames].begin(); iter_cur != matches[frames].end(); ++iter_cur){
            
            Eigen::Vector3d box_center_cur;
            box_center_cur << 0.5 * (boxes[frames][iter_cur->first](0) + boxes[frames][iter_cur->first](2)), 0.5 * (boxes[frames][iter_cur->first](1) + boxes[frames][iter_cur->first](3)), 1;
            cout << "box_center_cur: " << endl << box_center_cur << endl;
            box_center_cur.x() = (box_center_cur.x() - cam_cx) / cam_fx;
            box_center_cur.y() = (box_center_cur.y() - cam_cy) / cam_fy;
            
            for(auto iter_hist = matches[i].begin(); iter_hist != matches[i].end(); ++iter_hist){
                if(iter_cur->second != iter_hist->second)
                    continue;

                // forward
                EdgeSO3tForwardAngleNotFixedHistPose* edge_fw = new EdgeSO3tForwardAngleNotFixedHistPose();
                edge_fw->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                edge_fw->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(frames - i)));

                Eigen::Vector3d box_center_hist;
                box_center_hist << 0.5 * (boxes[i][iter_hist->first](0) + boxes[i][iter_hist->first](2)), 0.5 * (boxes[i][iter_hist->first](1) + boxes[i][iter_hist->first](3)), 1;
                cout << "box_center_hist: " << endl << box_center_hist << endl;
                box_center_hist.x() = (box_center_hist.x() - cam_cx) / cam_fx;
                box_center_hist.y() = (box_center_hist.y() - cam_cy) / cam_fy;

                cout << "box_center_cur: " << endl << box_center_cur << endl << "box_center_hist: " << endl << box_center_hist << endl; 

                edge_fw->setMeasurement(box_center_cur.normalized());
                edge_fw->setInformation(0.1 * Eigen::Matrix3d::Identity());
                edge_fw->liftup_hist_ = box_center_hist;

                optimizer.addEdge(edge_fw);

                edges_fw.push_back(edge_fw);
                
                // backward
                // EdgeSO3tBackwardAngleFixedHistPose* edge_bw = new EdgeSO3tBackwardAngleFixedHistPose(deq_Rcw[i], deq_pcw[i]);
                // edge_bw->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));

                // edge_bw->setMeasurement(box_center_hist.normalized());
                // edge_bw->setInformation(0.25 * Eigen::Matrix3d::Identity());
                // edge_bw->liftup_cur_ = box_center_cur;

                // optimizer.addEdge(edge_bw);

                // edges_bw.push_back(edge_bw);
            }
        }
    }

    TicToc t_solve;
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);
    cout << "time cost for solution is: " << t_solve.toc() << endl;
    edges_fw.clear();
    // cout << "edges_fw size: " << edges_fw.size() << endl;

    // auto iter = matches[frames].begin();
    // bool success = true;
    // avg_error_ = 0.0;
    // for(int i = 0; i < edges_proj.size(); i++){
    //     double box_length, box_width;
    //     box_length = boxes[frames][iter->first](2) - boxes[frames][iter->first](0);
    //     box_width = boxes[frames][iter->first](3) - boxes[frames][iter->first](1);

    //     EdgeSO3tCenterProj* e = edges_proj[i];
    //     e->computeError();
    //     Eigen::Vector2d error = e->error();

    //     avg_error_ += error.squaredNorm();

    //     if(error.x() > 0.75 * box_length || error.y() > 0.75 * box_width){
    //         success = false;
    //     }
    //     ++iter;
    // }

    // avg_error_ /= edges_proj.size();

    Eigen::Matrix<double, 7, 1> update_pose = poseSO3t->estimate();
    Eigen::Quaterniond quat(update_pose(0), update_pose(1), update_pose(2), update_pose(3));
    quat.normalize();

    Eigen::Matrix3d delta_RR = quat.toRotationMatrix() * Rcw.transpose();
    Eigen::Vector3d delta_pp = update_pose.tail<3>() - delta_RR * pcw;
    Eigen::AngleAxisd delta_rr(delta_RR);

    Rcw = quat.toRotationMatrix();
    pcw = update_pose.tail<3>();

    for(int i = frames - 1; i > 0; i--){
        Eigen::Matrix<double, 7, 1> update_pose = poseSO3ts[frames - i]->estimate();
        Eigen::Quaterniond quat(update_pose(0), update_pose(1), update_pose(2), update_pose(3));
        quat.normalize();

        Eigen::Matrix3d delta_RR = quat.toRotationMatrix() * Rcw.transpose();
        Eigen::Vector3d delta_pp = update_pose.tail<3>() - delta_RR * pcw;
        Eigen::AngleAxisd delta_rr(delta_RR);

        deq_Rcw[i] = quat.toRotationMatrix();
        deq_pcw[i] = update_pose.tail<3>();
    }

    return true;

    // if(delta_rr.angle() > 0.1 || delta_pp.norm() > 0.16)
    //     success = false;

    // if(success){
    //     Rcw = quat.toRotationMatrix();
    //     pcw = update_pose.tail<3>();
    //     return true;
    // }
    // else{
    //     return false;
    // }
}