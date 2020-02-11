#include "optimizer/optimizer.h"

#include "util/converter.h"

#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/linear_solver_eigen.h"
#include "g2o/types/types_six_dof_expmap.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/solvers/linear_solver_dense.h"
#include "g2o/types/types_seven_dof_expmap.h"

#include <Eigen/StdVector>

#include <mutex>
#include <memory>

void Optimizer::GlobalBundleAdjustemnt(const std::shared_ptr<Map>& map, 
                                       const int num_iter, 
                                       bool* stop_flag, 
                                       const unsigned long loop_kf_index, 
                                       const bool is_robust) {
  const std::vector<KeyFrame*> keyframes = map->GetAllKeyFrames();
  const std::vector<MapPoint*> map_points = map->GetAllMapPoints();
  BundleAdjustment(keyframes, 
                   map_points, 
                   num_iter, 
                   stop_flag, 
                   loop_kf_index, 
                   is_robust);
}


void Optimizer::BundleAdjustment(const std::vector<KeyFrame*>& keyframes, 
                                 const std::vector<MapPoint*>& map_points,
                                 const int num_iter, 
                                 bool* stop_flag, 
                                 const unsigned long loop_kf_index, 
                                 const bool is_robust) {
  std::deque<bool> is_not_included_map_points(map_points.size(), false);

  // These pointers are deleted in the optimizer destructor
  g2o::BlockSolver_6_3::LinearSolverType* linearSolver 
      = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

  g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

  g2o::OptimizationAlgorithmLevenberg* algorithm 
      = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
  
  g2o::SparseOptimizer optimizer;
  optimizer.setAlgorithm(algorithm);

  if (stop_flag) {
    optimizer.setForceStopFlag(stop_flag);
  }

  long unsigned int max_keyframe_id = 0;

  // Set KeyFrame vertices
  for (KeyFrame* keyframe : keyframes) {
    if(keyframe->isBad()) {
      continue;
    }
    g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(keyframe->GetPose()));
    vSE3->setId(keyframe->Id());
    vSE3->setFixed(keyframe->Id()==0);
    optimizer.addVertex(vSE3);
    if (keyframe->Id() > max_keyframe_id) {
      max_keyframe_id = keyframe->Id();
    }
  }

  const float huber_thresh_2d = sqrt(5.99);
  const float huber_thresh_3d = sqrt(7.815);

  // Set MapPoint vertices
  for (size_t i = 0; i < map_points.size(); ++i) {
    MapPoint* map_point = map_points[i];
    if (map_point->isBad()) {
      continue;
    }
    g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
    vPoint->setEstimate(Converter::toVector3d(map_point->GetWorldPos()));
    const int id = map_point->GetId() + max_keyframe_id + 1;
    vPoint->setId(id);
    vPoint->setMarginalized(true);
    optimizer.addVertex(vPoint);

    const std::map<KeyFrame*,size_t> observations = map_point->GetObservations();

    int num_edges = 0;
    //SET EDGES
    for (auto mit = observations.begin();
              mit != observations.end(); 
              ++mit) {
      KeyFrame* keyframe = mit->first;
      if (keyframe->isBad() || keyframe->Id() > max_keyframe_id) {
        continue;
      }

      ++num_edges;

      const cv::KeyPoint& kpUn = keyframe->mvKeysUn[mit->second];

      if (keyframe->mvuRight[mit->second] < 0) {
        Eigen::Vector2d obs;
        obs << kpUn.pt.x, kpUn.pt.y;

        g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(keyframe->Id())));
        e->setMeasurement(obs);
        const float invSigma2 = keyframe->mvInvLevelSigma2[kpUn.octave];
        e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

        if (is_robust) {
          g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
          e->setRobustKernel(rk);
          rk->setDelta(huber_thresh_2d);
        }

        e->fx = keyframe->fx;
        e->fy = keyframe->fy;
        e->cx = keyframe->cx;
        e->cy = keyframe->cy;

        optimizer.addEdge(e);

      } else {
        Eigen::Vector3d obs;
        const float kp_ur = keyframe->mvuRight[mit->second];
        obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

        g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(keyframe->Id())));
        e->setMeasurement(obs);
        const float& invSigma2 = keyframe->mvInvLevelSigma2[kpUn.octave];
        Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
        e->setInformation(Info);

        if (is_robust) {
          g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
          e->setRobustKernel(rk);
          rk->setDelta(huber_thresh_3d);
        }

        e->fx = keyframe->fx;
        e->fy = keyframe->fy;
        e->cx = keyframe->cx;
        e->cy = keyframe->cy;
        e->bf = keyframe->mbf;

        optimizer.addEdge(e);
      }
    }

    if (num_edges == 0) {
      optimizer.removeVertex(vPoint);
      is_not_included_map_points[i] = true;
    }
  }

  // Optimize!
  optimizer.initializeOptimization();
  optimizer.optimize(num_iter);

  // Recover optimized data
  //Keyframes
  for (KeyFrame* keyframe : keyframes) {
    if (keyframe->isBad()) {
      continue;
    }
    g2o::VertexSE3Expmap* vSE3 
        = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(keyframe->Id()));
    g2o::SE3Quat SE3quat = vSE3->estimate();
    if (loop_kf_index == 0) {
      keyframe->SetPose(Converter::toCvMat(SE3quat));
    } else {
      keyframe->mTcwGBA = Converter::toCvMat(SE3quat);
      keyframe->bundle_adj_global_for_keyframe_id = loop_kf_index;
    }
  }

  //Points
  for (size_t i=0; i < map_points.size(); ++i) {
    if (map_points[i]->isBad() || is_not_included_map_points[i]) {
      continue;
    }
    MapPoint* map_point = map_points[i];

    g2o::VertexSBAPointXYZ* vPoint 
        = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(map_point->GetId() + max_keyframe_id + 1));

    if (loop_kf_index == 0) {
      map_point->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
      map_point->UpdateNormalAndDepth();
    } else {
      map_point->position_global_bundle_adj = Converter::toCvMat(vPoint->estimate());
      map_point->bundle_adj_global_for_keyframe_id = loop_kf_index;
    }
  }
}

int Optimizer::PoseOptimization(Frame& frame) {
  
  g2o::BlockSolver_6_3::LinearSolverType* linearSolver
      = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

  g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

  g2o::OptimizationAlgorithmLevenberg* algorithm 
      = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
  
  g2o::SparseOptimizer optimizer;
  optimizer.setAlgorithm(algorithm);

  int num_initial_correspondences = 0;

  // Set Frame vertex
  g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
  vSE3->setEstimate(Converter::toSE3Quat(frame.GetPose()));
  vSE3->setId(0);
  vSE3->setFixed(false);
  optimizer.addVertex(vSE3);

  // Set MapPoint vertices
  const int N = frame.NumKeypoints();

  std::vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
  std::vector<size_t> vnIndexEdgeMono;
  vpEdgesMono.reserve(N);
  vnIndexEdgeMono.reserve(N);

  std::vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;
  std::vector<size_t> vnIndexEdgeStereo;
  vpEdgesStereo.reserve(N);
  vnIndexEdgeStereo.reserve(N);

  const float delta_mono = std::sqrt(5.991);
  const float delta_stereo = std::sqrt(7.815);

  {
    std::unique_lock<std::mutex> lock(MapPoint::global_mutex);

    for (int i = 0; i < N; ++i) {
      MapPoint* pMP = frame.GetMapPoint(i);
      if (!pMP) {
        continue;
      }

      // Monocular observation
      if (frame.StereoCoordRight()[i] < 0) {
        ++num_initial_correspondences;
        frame.SetOutlier(i, false);

        Eigen::Vector2d obs;
        const cv::KeyPoint& kpUn = frame.GetUndistortedKeys()[i];
        obs << kpUn.pt.x, kpUn.pt.y;

        g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();

        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e->setMeasurement(obs);
        const float invSigma2 = frame.InvLevelSigma2()[kpUn.octave];
        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        rk->setDelta(delta_mono);
        e->setRobustKernel(rk);

        e->fx = frame.GetFx();
        e->fy = frame.GetFy();
        e->cx = frame.GetCx();
        e->cy = frame.GetCy();
        const cv::Mat Xw = pMP->GetWorldPos();
        e->Xw[0] = Xw.at<float>(0);
        e->Xw[1] = Xw.at<float>(1);
        e->Xw[2] = Xw.at<float>(2);

        optimizer.addEdge(e);

        vpEdgesMono.push_back(e);
        vnIndexEdgeMono.push_back(i);
      } else { // Stereo observation
        ++num_initial_correspondences;
        frame.SetOutlier(i, false);

        //SET EDGE
        Eigen::Vector3d obs;
        const cv::KeyPoint& kpUn = frame.GetUndistortedKeys()[i];
        const float& kp_ur = frame.StereoCoordRight()[i];
        obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

        g2o::EdgeStereoSE3ProjectXYZOnlyPose* e 
            = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e->setMeasurement(obs);
        const float invSigma2 = frame.InvLevelSigma2()[kpUn.octave];
        e->setInformation(Eigen::Matrix3d::Identity() * invSigma2);

        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        rk->setDelta(delta_stereo);
        e->setRobustKernel(rk);

        e->fx = frame.GetFx();
        e->fy = frame.GetFy();
        e->cx = frame.GetCx();
        e->cy = frame.GetCy();
        e->bf = frame.GetBaselineFx();
        const cv::Mat Xw = pMP->GetWorldPos();
        e->Xw[0] = Xw.at<float>(0);
        e->Xw[1] = Xw.at<float>(1);
        e->Xw[2] = Xw.at<float>(2);

        optimizer.addEdge(e);

        vpEdgesStereo.push_back(e);
        vnIndexEdgeStereo.push_back(i);
      }
    }
  }


  if (num_initial_correspondences < 3) {
    return 0;
  }

  // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
  // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
  constexpr int num_optim = 4;
  const std::array<float,num_optim> chi2Mono = {5.991, 5.991, 5.991, 5.991};
  const std::array<float,num_optim> chi2Stereo = {7.815, 7.815, 7.815, 7.815};
  const std::array<int,num_optim> num_iters = {10, 10, 10, 10};

  int is_bad = 0;
  for (size_t it = 0; it < num_optim; ++it) {

    vSE3->setEstimate(Converter::toSE3Quat(frame.GetPose()));
    optimizer.initializeOptimization(0);
    optimizer.optimize(num_iters[it]);

    is_bad=0;
    for (size_t i = 0; i < vpEdgesMono.size(); ++i) {
      g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

      const size_t idx = vnIndexEdgeMono[i];

      if (frame.IsOutlier(idx)) {
        e->computeError();
      }

      const float chi2 = e->chi2();

      if (chi2>chi2Mono[it]) {                
        frame.SetOutlier(idx, true);
        e->setLevel(1);
        ++is_bad;
      } else {
        frame.SetOutlier(idx, false);
        e->setLevel(0);
      }

      if (it == 2) {
        e->setRobustKernel(0);
      }
    }

    for (size_t i=0; i < vpEdgesStereo.size(); ++i) {
      g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = vpEdgesStereo[i];

      const size_t idx = vnIndexEdgeStereo[i];

      if (frame.IsOutlier(idx)) {
        e->computeError();
      }

      const float chi2 = e->chi2();

      if (chi2 > chi2Stereo[it]) {
        frame.SetOutlier(idx, true);
        e->setLevel(1);
        ++is_bad;
      } else {                
        e->setLevel(0);
        frame.SetOutlier(idx, false);
      }

      if (it==2) {
        e->setRobustKernel(0);
      }
    }

    if(optimizer.edges().size() < 10) {
      break;
    }
  }    

  // Recover optimized pose and return number of inliers
  g2o::VertexSE3Expmap* vSE3_recov 
      = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
  g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
  frame.SetPose(Converter::toCvMat(SE3quat_recov));

  return num_initial_correspondences - is_bad;
}

void Optimizer::LocalBundleAdjustment(KeyFrame* pKF, 
                                      bool* stop_flag, 
                                      const std::shared_ptr<Map>& map) {    
  // Local KeyFrames: First Breath Search from Current Keyframe
  std::list<KeyFrame*> local_keyframes;
  pKF->bundle_adj_local_id_for_keyframe = pKF->Id();
  local_keyframes.push_back(pKF);

  const std::vector<KeyFrame*> nbor_keyframes = pKF->GetVectorCovisibleKeyFrames();
  for (KeyFrame* nbor_keyframe : nbor_keyframes) {
    nbor_keyframe->bundle_adj_local_id_for_keyframe = pKF->Id();
    if (!nbor_keyframe->isBad()) {
      local_keyframes.push_back(nbor_keyframe);
    }
  }

  // Local MapPoints seen in Local KeyFrames
  std::list<MapPoint*> local_map_points;
  for (KeyFrame* local_keyframe : local_keyframes) {
    std::vector<MapPoint*> vpMPs = local_keyframe->GetMapPointMatches();
    for (MapPoint* pMP : vpMPs) {
      if(pMP 
         && !pMP->isBad()
         && pMP->bundle_adj_local_id_for_keyframe != pKF->Id()) {
        pMP->bundle_adj_local_id_for_keyframe = pKF->Id();
        local_map_points.push_back(pMP);
      }
    }
  }

  // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
  std::list<KeyFrame*> fixed_cameras;
  for(MapPoint* map_point : local_map_points) {
    std::map<KeyFrame*,size_t> observations = map_point->GetObservations();
    for (auto mit=observations.begin(); 
              mit != observations.end(); 
              ++mit) {
      KeyFrame* pKFi = mit->first;

      if (pKFi->bundle_adj_local_id_for_keyframe != pKF->Id() 
          && pKFi->mnBAFixedForKF != pKF->Id()) {                
        pKFi->mnBAFixedForKF=pKF->Id();
        if(!pKFi->isBad()) {
          fixed_cameras.push_back(pKFi);
        }
      }
    }
  }

  // Setup optimizer
  g2o::BlockSolver_6_3::LinearSolverType* linearSolver
      = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

  g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

  g2o::OptimizationAlgorithmLevenberg* algorithm
      = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
  
  g2o::SparseOptimizer optimizer;
  optimizer.setAlgorithm(algorithm);

  if (stop_flag) {
    optimizer.setForceStopFlag(stop_flag);
  }

  unsigned long max_keyframe_id = 0;

  // Set Local KeyFrame vertices
  // for(std::list<KeyFrame*>::iterator lit=local_keyframes.begin(), lend=local_keyframes.end(); lit!=lend; lit++)
  for (KeyFrame* keyframe : local_keyframes) {
    g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(keyframe->GetPose()));
    vSE3->setId(keyframe->Id());
    vSE3->setFixed(keyframe->Id() == 0);
    optimizer.addVertex(vSE3);
    if (keyframe->Id() > max_keyframe_id) {
      max_keyframe_id = keyframe->Id();
    }
  }

  // Set Fixed KeyFrame vertices
  for (KeyFrame* pKFi : fixed_cameras) {
    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
    vSE3->setId(pKFi->Id());
    vSE3->setFixed(true);
    optimizer.addVertex(vSE3);
    if(pKFi->Id() > max_keyframe_id) {
      max_keyframe_id = pKFi->Id();
    }
  }

  // Set MapPoint vertices
  const int expected_size = (local_keyframes.size() + fixed_cameras.size())* local_map_points.size();

  std::vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
  vpEdgesMono.reserve(expected_size);

  std::vector<KeyFrame*> vpEdgeKFMono;
  vpEdgeKFMono.reserve(expected_size);

  std::vector<MapPoint*> vpMapPointEdgeMono;
  vpMapPointEdgeMono.reserve(expected_size);

  std::vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
  vpEdgesStereo.reserve(expected_size);

  std::vector<KeyFrame*> vpEdgeKFStereo;
  vpEdgeKFStereo.reserve(expected_size);

  std::vector<MapPoint*> vpMapPointEdgeStereo;
  vpMapPointEdgeStereo.reserve(expected_size);

  const float huber_thresh_mono = std::sqrt(5.991);
  const float huber_thresh_stereo = std::sqrt(7.815);

  for (MapPoint* local_map_point : local_map_points) {
    g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
    vPoint->setEstimate(Converter::toVector3d(local_map_point->GetWorldPos()));
    const int id = local_map_point->GetId() + max_keyframe_id + 1;
    vPoint->setId(id);
    vPoint->setMarginalized(true);
    optimizer.addVertex(vPoint);

    const std::map<KeyFrame*,size_t> observations = local_map_point->GetObservations();

    //Set edges
    for (auto mit = observations.begin();
              mit != observations.end(); 
              ++mit) {
      KeyFrame* pKFi = mit->first;
      if (pKFi->isBad()) {
        continue;
      }
      const cv::KeyPoint& kpUn = pKFi->mvKeysUn[mit->second];

      // Monocular observation
      if(pKFi->mvuRight[mit->second] < 0) {
        Eigen::Vector2d obs;
        obs << kpUn.pt.x, kpUn.pt.y;

        g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->Id())));
        e->setMeasurement(obs);
        const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
        e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        rk->setDelta(huber_thresh_mono);
        e->setRobustKernel(rk);

        e->fx = pKFi->fx;
        e->fy = pKFi->fy;
        e->cx = pKFi->cx;
        e->cy = pKFi->cy;

        optimizer.addEdge(e);
        vpEdgesMono.push_back(e);
        vpEdgeKFMono.push_back(pKFi);
        vpMapPointEdgeMono.push_back(local_map_point);
      } else { // Stereo observation
        Eigen::Vector3d obs;
        const float kp_ur = pKFi->mvuRight[mit->second];
        obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

        g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->Id())));
        e->setMeasurement(obs);
        const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
        e->setInformation(Eigen::Matrix3d::Identity()*invSigma2);

        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        rk->setDelta(huber_thresh_stereo);
        e->setRobustKernel(rk);

        e->fx = pKFi->fx;
        e->fy = pKFi->fy;
        e->cx = pKFi->cx;
        e->cy = pKFi->cy;
        e->bf = pKFi->mbf;

        optimizer.addEdge(e);
        vpEdgesStereo.push_back(e);
        vpEdgeKFStereo.push_back(pKFi);
        vpMapPointEdgeStereo.push_back(local_map_point);
      }
      
    }
  }

  if (stop_flag && *stop_flag) {
    return;
  }

  optimizer.initializeOptimization();
  optimizer.optimize(5);

  bool do_more = true;

  if(stop_flag && *stop_flag) {
    do_more = false;
  }

  if(do_more) {

    // Check inlier observations
    for (size_t i = 0; i < vpEdgesMono.size(); ++i) {
      g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
      MapPoint* pMP = vpMapPointEdgeMono[i];

      if (pMP->isBad()) {
        continue;
      }

      if (e->chi2() > 5.991 || !e->isDepthPositive()) {
        e->setLevel(1);
      }

      e->setRobustKernel(0);
    }

    for(size_t i = 0; i < vpEdgesStereo.size(); ++i) {
      g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
      MapPoint* pMP = vpMapPointEdgeStereo[i];

      if(pMP->isBad()) {
        continue;
      }

      if (e->chi2() > 7.815 || !e->isDepthPositive()) {
        e->setLevel(1);
      }

      e->setRobustKernel(0);
    }

    // Optimize again without the outliers
    optimizer.initializeOptimization(0);
    optimizer.optimize(10);
  }

  std::vector<std::pair<KeyFrame*,MapPoint*>> vToErase;
  vToErase.reserve(vpEdgesMono.size()+vpEdgesStereo.size());

  // Check inlier observations       
  for (size_t i = 0; i < vpEdgesMono.size(); ++i) {
    g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
    MapPoint* pMP = vpMapPointEdgeMono[i];

    if(pMP->isBad()) {
      continue;
    }

    if(e->chi2()>5.991 || !e->isDepthPositive()) {
      KeyFrame* pKFi = vpEdgeKFMono[i];
      vToErase.push_back(std::make_pair(pKFi,pMP));
    }
  }

  for(size_t i = 0; i < vpEdgesStereo.size(); ++i) {
    g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
    MapPoint* pMP = vpMapPointEdgeStereo[i];

    if(pMP->isBad()) {
      continue;
    }

    if (e->chi2() > 7.815 || !e->isDepthPositive()) {
      KeyFrame* pKFi = vpEdgeKFStereo[i];
      vToErase.push_back(std::make_pair(pKFi,pMP));
    }
  }

  // Get Map Mutex
  std::unique_lock<std::mutex> lock(map->map_update_mutex);

  if (!vToErase.empty()) {
    for (size_t i = 0;i < vToErase.size(); ++i) {
      KeyFrame* pKFi = vToErase[i].first;
      MapPoint* pMPi = vToErase[i].second;
      pKFi->EraseMapPointMatch(pMPi);
      pMPi->EraseObservation(pKFi);
    }
  }

  // Recover optimized data
  //Keyframes
  for (KeyFrame* keyframe : local_keyframes) {
    g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(keyframe->Id()));
    g2o::SE3Quat SE3quat = vSE3->estimate();
    keyframe->SetPose(Converter::toCvMat(SE3quat));
  }

  //Points
  for (MapPoint* map_point : local_map_points) {
    g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(map_point->GetId() + max_keyframe_id + 1));
    map_point->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
    map_point->UpdateNormalAndDepth();
  }
}

void Optimizer::OptimizeEssentialGraph(const std::shared_ptr<Map>& pMap, 
                                       KeyFrame* pLoopKF, 
                                       KeyFrame* pCurKF,
                                       const LoopCloser::KeyFrameAndPose& non_corrected_sim3,
                                       const LoopCloser::KeyFrameAndPose& corrected_sim3,
                                       const std::map<KeyFrame*,std::set<KeyFrame*>>& LoopConnections, 
                                       const bool bFixScale) {
  // Setup optimizer
  g2o::BlockSolver_7_3::LinearSolverType * linearSolver =
          new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
  g2o::BlockSolver_7_3* solver_ptr = new g2o::BlockSolver_7_3(linearSolver);
  g2o::OptimizationAlgorithmLevenberg* algorithm 
      = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

  algorithm->setUserLambdaInit(1e-16);
  
  g2o::SparseOptimizer optimizer;
  optimizer.setVerbose(false);
  optimizer.setAlgorithm(algorithm);

  const std::vector<KeyFrame*> all_keyframes = pMap->GetAllKeyFrames();
  const std::vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

  const unsigned int nMaxKFid = pMap->GetMaxKeyframeId();

  std::vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid+1);
  std::vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid+1);
  std::vector<g2o::VertexSim3Expmap*> vpVertices(nMaxKFid+1);

  const int minFeat = 100;

  // Set KeyFrame vertices
  for (KeyFrame* pKF : all_keyframes) {
    if(pKF->isBad()) {
      continue;
    }
    g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

    const int nIDi = pKF->Id();
    const LoopCloser::KeyFrameAndPose::const_iterator it = corrected_sim3.find(pKF);
    if (it!= corrected_sim3.end()) {
      vScw[nIDi] = it->second;
      VSim3->setEstimate(it->second);
    } else {
      g2o::Sim3 Siw(Converter::toMatrix3d(pKF->GetRotation()), 
                    Converter::toVector3d(pKF->GetTranslation()),
                    1.0);
      vScw[nIDi] = Siw;
      VSim3->setEstimate(Siw);
    }

    if (pKF==pLoopKF) {
      VSim3->setFixed(true);
    }

    VSim3->setId(nIDi);
    VSim3->setMarginalized(false);
    VSim3->_fix_scale = bFixScale;

    optimizer.addVertex(VSim3);

    vpVertices[nIDi]=VSim3;
  }

  std::set<std::pair<long unsigned int,long unsigned int>> insterted_edges;

  // Set Loop edges
  for (auto mit = LoopConnections.begin(); mit != LoopConnections.end(); ++mit) {
    KeyFrame* pKF = mit->first;
    const long unsigned int nIDi = pKF->Id();
    const std::set<KeyFrame*> &spConnections = mit->second;
    const g2o::Sim3 Siw = vScw[nIDi];
    const g2o::Sim3 Swi = Siw.inverse();

    for(auto connection : spConnections) {
      const long unsigned int nIDj = connection->Id();
      if((nIDi!=pCurKF->Id() || nIDj!=pLoopKF->Id())
          && pKF->GetWeight(connection) < minFeat) {
        continue;
      }

      const g2o::Sim3 Sjw = vScw[nIDj];
      const g2o::Sim3 Sji = Sjw * Swi;

      g2o::EdgeSim3* e = new g2o::EdgeSim3();
      e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
      e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
      e->setMeasurement(Sji);
      e->information() =  Eigen::Matrix<double,7,7>::Identity();

      optimizer.addEdge(e);

      insterted_edges.insert(std::make_pair(std::min(nIDi,nIDj),std::max(nIDi,nIDj)));
    }
  }

  // Set normal edges
  for (KeyFrame* pKF : all_keyframes) {

    const int nIDi = pKF->Id();

    g2o::Sim3 Swi;

    const auto iti = non_corrected_sim3.find(pKF);
    if (iti != non_corrected_sim3.end()) {
      Swi = (iti->second).inverse();
    } else {
      Swi = vScw[nIDi].inverse();
    }

    KeyFrame* pParentKF = pKF->GetParent();

    // Spanning tree edge
    if (pParentKF) {
      int nIDj = pParentKF->Id();

      g2o::Sim3 Sjw;

      const auto itj = non_corrected_sim3.find(pParentKF);
      if (itj!=non_corrected_sim3.end()) {
        Sjw = itj->second;
      } else {
        Sjw = vScw[nIDj];
      }

      g2o::Sim3 Sji = Sjw * Swi;

      g2o::EdgeSim3* e = new g2o::EdgeSim3();
      e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
      e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
      e->setMeasurement(Sji);
      e->information() =  Eigen::Matrix<double,7,7>::Identity();
      optimizer.addEdge(e);
    }

    // Loop edges
    const std::set<KeyFrame*> loop_edges = pKF->GetLoopEdges();
    for (KeyFrame* pLKF : loop_edges) {
      if (pLKF->Id() < pKF->Id()) {
        g2o::Sim3 Slw;

        const auto itl = non_corrected_sim3.find(pLKF);
        if (itl != non_corrected_sim3.end()) {
          Slw = itl->second;
        } else {
          Slw = vScw[pLKF->Id()];
        }

        g2o::Sim3 Sli = Slw * Swi;
        g2o::EdgeSim3* el = new g2o::EdgeSim3();
        el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->Id())));
        el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
        el->setMeasurement(Sli);
        el->information() =  Eigen::Matrix<double,7,7>::Identity();
        optimizer.addEdge(el);
      }
    }

    // Covisibility graph edges
    const std::vector<KeyFrame*> connected_keyframes = pKF->GetCovisiblesByWeight(minFeat);
    for (KeyFrame* pKFn : connected_keyframes) {
      if(pKFn 
          && pKFn != pParentKF 
          && !pKF->hasChild(pKFn) 
          && !loop_edges.count(pKFn)) {
        if(!pKFn->isBad() && pKFn->Id()<pKF->Id()) {
          if (insterted_edges.count(std::make_pair(std::min(pKF->Id(),pKFn->Id()),
                                                    std::max(pKF->Id(),pKFn->Id())))) {
            continue;
          }

          g2o::Sim3 Snw;

          const auto itn = non_corrected_sim3.find(pKFn);
          if(itn!=non_corrected_sim3.end()) {
            Snw = itn->second;
          } else {
            Snw = vScw[pKFn->Id()];
          }

          g2o::Sim3 Sni = Snw * Swi;

          g2o::EdgeSim3* en = new g2o::EdgeSim3();
          en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->Id())));
          en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
          en->setMeasurement(Sni);
          en->information() =  Eigen::Matrix<double,7,7>::Identity();
          optimizer.addEdge(en);
        }
      }
    }
  }

  // Optimize!
  optimizer.initializeOptimization();
  optimizer.optimize(20);

  std::unique_lock<std::mutex> lock(pMap->map_update_mutex);

  // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
  for (KeyFrame* pKFi : all_keyframes) {
    const int nIDi = pKFi->Id();

    g2o::VertexSim3Expmap* VSim3 = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(nIDi));
    g2o::Sim3 CorrectedSiw = VSim3->estimate();
    vCorrectedSwc[nIDi] = CorrectedSiw.inverse();
    Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
    Eigen::Vector3d eigt = CorrectedSiw.translation();
    double s = CorrectedSiw.scale();

    eigt *=(1./s); //[R t/s;0 1]

    pKFi->SetPose(Converter::toCvSE3(eigR,eigt));
  }

  // Correct points. Transform to "non-optimized" reference keyframe pose 
  // and transform back with optimized pose
  for (MapPoint* pMP : vpMPs) {

    if(pMP->isBad()) {
      continue;
    }

    int nIDr;
    if (pMP->corrected_by_keyframe == pCurKF->Id()) {
      nIDr = pMP->corrected_reference;
    } else {
      KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
      nIDr = pRefKF->Id();
    }

    g2o::Sim3 Srw = vScw[nIDr];
    g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

    cv::Mat P3Dw = pMP->GetWorldPos();
    Eigen::Vector3d eigP3Dw = Converter::toVector3d(P3Dw);
    Eigen::Vector3d eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));

    pMP->SetWorldPos(Converter::toCvMat(eigCorrectedP3Dw));

    pMP->UpdateNormalAndDepth();
  }
}

int Optimizer::OptimizeSim3(KeyFrame* pKF1, 
                            KeyFrame* pKF2, 
                            std::vector<MapPoint*>& vpMatches1, 
                            g2o::Sim3& g2oS12, 
                            const float th2, 
                            const bool bFixScale) {
  
  g2o::BlockSolverX::LinearSolverType* linearSolver
      = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

  g2o::BlockSolverX* solver_ptr = new g2o::BlockSolverX(linearSolver);

  g2o::OptimizationAlgorithmLevenberg* algorithm 
      = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
  
  g2o::SparseOptimizer optimizer;
  optimizer.setAlgorithm(algorithm);

  // Calibration
  const cv::Mat& K1 = pKF1->mK;
  const cv::Mat& K2 = pKF2->mK;

  // Camera poses
  const cv::Mat R1w = pKF1->GetRotation();
  const cv::Mat t1w = pKF1->GetTranslation();
  const cv::Mat R2w = pKF2->GetRotation();
  const cv::Mat t2w = pKF2->GetTranslation();

  // Set Sim3 vertex
  g2o::VertexSim3Expmap* vSim3 = new g2o::VertexSim3Expmap();    
  vSim3->_fix_scale=bFixScale;
  vSim3->setEstimate(g2oS12);
  vSim3->setId(0);
  vSim3->setFixed(false);
  vSim3->_principle_point1[0] = K1.at<float>(0,2);
  vSim3->_principle_point1[1] = K1.at<float>(1,2);
  vSim3->_focal_length1[0] = K1.at<float>(0,0);
  vSim3->_focal_length1[1] = K1.at<float>(1,1);
  vSim3->_principle_point2[0] = K2.at<float>(0,2);
  vSim3->_principle_point2[1] = K2.at<float>(1,2);
  vSim3->_focal_length2[0] = K2.at<float>(0,0);
  vSim3->_focal_length2[1] = K2.at<float>(1,1);
  optimizer.addVertex(vSim3);

  // Set MapPoint vertices
  const int N = vpMatches1.size();
  const std::vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
  std::vector<g2o::EdgeSim3ProjectXYZ*> vpEdges12;
  std::vector<g2o::EdgeInverseSim3ProjectXYZ*> vpEdges21;
  std::vector<size_t> vnIndexEdge;

  vnIndexEdge.reserve(2*N);
  vpEdges12.reserve(2*N);
  vpEdges21.reserve(2*N);

  const float deltaHuber = sqrt(th2);

  int num_correspondences = 0;
  for (int i = 0; i < N; ++i) {
    if (!vpMatches1[i]) {
      continue;
    }

    MapPoint* pMP1 = vpMapPoints1[i];
    MapPoint* pMP2 = vpMatches1[i];

    const int id1 = 2*i+1;
    const int id2 = 2*(i+1);

    const int i2 = pMP2->GetIndexInKeyFrame(pKF2);

    if(pMP1 && pMP2) {
      if (!pMP1->isBad() && !pMP2->isBad() && i2>=0) {
        g2o::VertexSBAPointXYZ* vPoint1 = new g2o::VertexSBAPointXYZ();
        cv::Mat P3D1w = pMP1->GetWorldPos();
        cv::Mat P3D1c = R1w*P3D1w + t1w;
        vPoint1->setEstimate(Converter::toVector3d(P3D1c));
        vPoint1->setId(id1);
        vPoint1->setFixed(true);
        optimizer.addVertex(vPoint1);

        g2o::VertexSBAPointXYZ* vPoint2 = new g2o::VertexSBAPointXYZ();
        cv::Mat P3D2w = pMP2->GetWorldPos();
        cv::Mat P3D2c = R2w*P3D2w + t2w;
        vPoint2->setEstimate(Converter::toVector3d(P3D2c));
        vPoint2->setId(id2);
        vPoint2->setFixed(true);
        optimizer.addVertex(vPoint2);
      } else {
        continue;
      }
    } else {
      continue;
    }
    num_correspondences++;

    // Set edge x1 = S12*X2
    Eigen::Vector2d obs1;
    const cv::KeyPoint& kpUn1 = pKF1->mvKeysUn[i];
    obs1 << kpUn1.pt.x, kpUn1.pt.y;

    g2o::EdgeSim3ProjectXYZ* e12 = new g2o::EdgeSim3ProjectXYZ();
    e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id2)));
    e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
    e12->setMeasurement(obs1);
    const float invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];
    e12->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare1);

    g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
    rk1->setDelta(deltaHuber);
    e12->setRobustKernel(rk1);
    optimizer.addEdge(e12);

    // Set edge x2 = S21*X1
    Eigen::Vector2d obs2;
    const cv::KeyPoint& kpUn2 = pKF2->mvKeysUn[i2];
    obs2 << kpUn2.pt.x, kpUn2.pt.y;

    g2o::EdgeInverseSim3ProjectXYZ* e21 = new g2o::EdgeInverseSim3ProjectXYZ();

    e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id1)));
    e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
    e21->setMeasurement(obs2);
    float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];
    e21->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare2);

    g2o::RobustKernelHuber* rk2 = new g2o::RobustKernelHuber;
    rk2->setDelta(deltaHuber);
    e21->setRobustKernel(rk2);
    optimizer.addEdge(e21);

    vpEdges12.push_back(e12);
    vpEdges21.push_back(e21);
    vnIndexEdge.push_back(i);
  }

  // Optimize!
  optimizer.initializeOptimization();
  optimizer.optimize(5);

  // Check inliers
  int is_bad = 0;
  for(size_t i = 0; i < vpEdges12.size(); ++i) {
    g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
    g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
    if(!e12 || !e21) {
      continue;
    }

    if (e12->chi2() > th2 || e21->chi2() > th2) {
      size_t idx = vnIndexEdge[i];
      vpMatches1[idx] = static_cast<MapPoint*>(nullptr);
      optimizer.removeEdge(e12);
      optimizer.removeEdge(e21);
      vpEdges12[i] = static_cast<g2o::EdgeSim3ProjectXYZ*>(nullptr);
      vpEdges21[i] = static_cast<g2o::EdgeInverseSim3ProjectXYZ*>(nullptr);
      ++is_bad;
    }
  }

  const int more_iterations = (is_bad>0) ? 10 : 5;
  if ((num_correspondences - is_bad) < 10) {
    return 0;
  }

  // Optimize again only with inliers
  optimizer.initializeOptimization();
  optimizer.optimize(more_iterations);

  int nIn = 0;
  for (size_t i = 0; i < vpEdges12.size(); ++i) {
    g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
    g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
    if (!e12 || !e21) {
      continue;
    }

    if (e12->chi2() > th2 || e21->chi2() > th2) {
      size_t idx = vnIndexEdge[i];
      vpMatches1[idx] = static_cast<MapPoint*>(nullptr);
    } else {
      ++nIn;
    }
  }

  // Recover optimized Sim3
  g2o::VertexSim3Expmap* vSim3_recov = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(0));
  g2oS12 = vSim3_recov->estimate();

  return nIn;
}