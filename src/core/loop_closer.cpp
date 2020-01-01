#include "loop_closer.h"

#include "solvers/sim3solver.h"
#include "util/converter.h"
#include "optimizer/optimizer.h"
#include "orb_features/orb_matcher.h"

// #pragma GCC optimize ("O0")

LoopCloser::LoopCloser(const std::shared_ptr<Map>& map,
                       const std::shared_ptr<KeyframeDatabase>& keyframe_db,
                       const std::shared_ptr<OrbVocabulary>& orb_vocabulary,
                       const bool fix_scale)
      : map_(map)
      , keyframe_db_(keyframe_db)
      , orb_vocabulary_(orb_vocabulary)
      , fix_scale_(fix_scale) {

}

void LoopCloser::SetLocalMapper(const std::shared_ptr<LocalMapper>& local_mapper) { 
  local_mapper_ = local_mapper; 
}

void LoopCloser::Run() {
  is_finished_ = false;
  while (true) {
    // Check if there are keyframes in the queue
    if (CheckNewKeyFrames()) {
      // Detect loop candidates and check covisibility consistency
      if (DetectLoop()) {
        // Compute similarity transformation [sR|t]
        // In the stereo/RGBD case s=1
        if (ComputeSim3()) {
          // Perform loop fusion and pose graph optimization
          CorrectLoop();
        }
      }
    }       
    ResetIfRequested();
    if (CheckFinish()) {
      break;
    }
    usleep(5000);
  }
  SetFinish();
}

void LoopCloser::InsertKeyFrame(KeyFrame* keyframe) {
  std::unique_lock<std::mutex> lock(loop_queue_mutex_);
  if (keyframe->mnId != 0) {
    loop_keyframe_queue_.push_back(keyframe);
  }
}

void LoopCloser::RequestReset() {
  {
    std::unique_lock<std::mutex> lock(reset_mutex_);
    reset_requested_ = true;
  }

  while (true) {
    {
      std::unique_lock<std::mutex> lock(reset_mutex_);
      if (!reset_requested_) {
        break;
      }
    }
    usleep(5000);
  }
}

void LoopCloser::RunGlobalBundleAdjustment(unsigned long loop_kf_index) {
  std::cout << "Starting Global Bundle Adjustment\n";

  const int idx =  full_bundle_adj_idx_;
  Optimizer::GlobalBundleAdjustemnt(map_,
                                    10,
                                    &stop_global_bundle_adj_,
                                    loop_kf_index,
                                    false);

  // Update all MapPoints and KeyFrames
  // Local Mapping was active during BA, that means that there might be new keyframes
  // not included in the Global BA and they are not consistent with the updated map.
  // We need to propagate the correction through the spanning tree
  {
    std::unique_lock<std::mutex> lock(global_bundle_adj_mutex_);
    if (idx != full_bundle_adj_idx_) {
      return;
    }

    if (!stop_global_bundle_adj_) {
      std::cout << "Global Bundle Adjustment finished\n";
      std::cout << "Updating map ...\n";

      local_mapper_->RequestStop();
      // Wait until Local Mapping has effectively stopped

      while (!local_mapper_->IsStopped() && !local_mapper_->IsFinished()) {
        usleep(1000);
      }

      // Get Map Mutex
      std::unique_lock<std::mutex> lock(map_->map_update_mutex);

      // Correct keyframes starting at map first keyframe
      std::list<KeyFrame*> keyframes_to_check(map_->GetKeyframeOrigins().begin(),
                                              map_->GetKeyframeOrigins().end());

      while (!keyframes_to_check.empty()) {
        KeyFrame* keyframe = keyframes_to_check.front();
        cv::Mat Twc = keyframe->GetPoseInverse();
        const std::set<KeyFrame*> sChilds = keyframe->GetChilds();
        
        for (KeyFrame* pChild : sChilds) {
          if (pChild->bundle_adj_global_for_keyframe_id != loop_kf_index) {
            cv::Mat Tchildc = pChild->GetPose() * Twc;
            pChild->mTcwGBA = Tchildc * keyframe->mTcwGBA;
            pChild->bundle_adj_global_for_keyframe_id = loop_kf_index;
          }
          keyframes_to_check.push_back(pChild);
        }

        keyframe->mTcwBefGBA = keyframe->GetPose();
        keyframe->SetPose(keyframe->mTcwGBA);
        keyframes_to_check.pop_front();
      }

      // Correct MapPoints
      const std::vector<MapPoint*> vpMPs = map_->GetAllMapPoints();
      for (MapPoint* pMP : vpMPs) {
        if (pMP->isBad()) {
          continue;
        }

        if (pMP->bundle_adj_global_for_keyframe_id == loop_kf_index) {
          // If optimized by Global BA, just update
          pMP->SetWorldPos(pMP->position_global_bundle_adj);
        } else {
          // Update according to the correction of its reference keyframe
          KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();

          if(pRefKF->bundle_adj_global_for_keyframe_id != loop_kf_index) {
            continue;
          }

          // Map to non-corrected camera
          cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0,3).colRange(0,3);
          cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0,3).col(3);
          cv::Mat Xc = Rcw * pMP->GetWorldPos() + tcw;

          // Backproject using corrected camera
          cv::Mat Twc = pRefKF->GetPoseInverse();
          cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
          cv::Mat twc = Twc.rowRange(0,3).col(3);

          pMP->SetWorldPos(Rwc * Xc + twc);
        }
      }            

      map_->InformNewBigChange();

      local_mapper_->Release();

      std::cout << "Map updated!\n";
    }

    is_finished_global_budle_adj_ = true;
    is_running_global_budle_adj_ = false;
  }
}

bool LoopCloser::IsRunningGBA() const {
  std::unique_lock<std::mutex> lock(global_bundle_adj_mutex_);
  return is_running_global_budle_adj_;
}

void LoopCloser::RequestFinish() {
  std::unique_lock<std::mutex> lock(finish_mutex_);
  finish_requested_ = true;
}

bool LoopCloser::IsFinished() const {
  std::unique_lock<std::mutex> lock(finish_mutex_);
  return is_finished_;
}

bool LoopCloser::CheckNewKeyFrames() const {
  std::unique_lock<std::mutex> lock(loop_queue_mutex_);
  return !loop_keyframe_queue_.empty();
}

bool LoopCloser::DetectLoop() {
  {
    std::unique_lock<std::mutex> lock(loop_queue_mutex_);
    current_keyframe_ = loop_keyframe_queue_.front();
    loop_keyframe_queue_.pop_front();
    // Avoid that a keyframe can be erased while it is being process by this thread
    current_keyframe_->SetNotErase();
  }

  // If the map contains less than 10 KF or less than 10 KF have passed from last loop detection
  if (current_keyframe_->mnId < last_loop_kf_id_ + 10) {
    keyframe_db_->add(current_keyframe_);
    current_keyframe_->SetErase();
    return false;
  }

  // Compute reference BoW similarity score
  // This is the lowest score to a connected keyframe in the covisibility graph
  // We will impose loop candidates to have a higher similarity than this
  const DBoW2::BowVector& current_bow_vec = current_keyframe_->mBowVec;
  float min_score = 1;
  for (KeyFrame* keyframe : current_keyframe_->GetVectorCovisibleKeyFrames()) {
    if (keyframe->isBad()) {
      continue;
    }
    const float score = orb_vocabulary_->score(current_bow_vec, keyframe->mBowVec);
    min_score = std::min(min_score, score);
  }

  // Query the database imposing the minimum score
  const std::vector<KeyFrame*> candidate_keyframes = keyframe_db_->DetectLoopCandidates(current_keyframe_, min_score);

  // If there are no loop candidates, just add new keyframe and return false
  if (candidate_keyframes.empty()) {
    keyframe_db_->add(current_keyframe_);
    consistent_groups_.clear();
    current_keyframe_->SetErase();
    return false;
  }

  // For each loop candidate check consistency with previous loop candidates
  // Each candidate expands a covisibility group (keyframes connected 
  // to the loop candidate in the covisibility graph)
  // A group is consistent with a previous group if they share at least a keyframe
  // We must detect a consistent loop in several consecutive keyframes to accept it
  consistent_enough_candidates_.clear();

  std::vector<ConsistentGroup> current_consistent_groups;
  std::vector<bool> is_visited_consistent_group(consistent_groups_.size(),false); // TODO vector of bool :(
  
  for (KeyFrame* candidate_keyframe : candidate_keyframes) {

    std::set<KeyFrame*> candidate_group = candidate_keyframe->GetConnectedKeyFrames();
    candidate_group.insert(candidate_keyframe);

    bool has_enough_consistent = false;
    bool is_consistent_for_some_groups = false;
    for (size_t iG = 0; iG < consistent_groups_.size(); ++iG) {
      std::set<KeyFrame*> previous_group = consistent_groups_[iG].first;

      bool is_consistent = false;
      for (KeyFrame* kf : candidate_group) {
        if (previous_group.count(kf)) {
          is_consistent = true;
          is_consistent_for_some_groups = true;
          break;
        }
      }

      if (is_consistent) {
        const int n_previous_consistency = consistent_groups_[iG].second;
        const int n_current_consistency = n_previous_consistency + 1;
        
        if (!is_visited_consistent_group[iG]) {
          current_consistent_groups.emplace_back(candidate_group, n_current_consistency);
          is_visited_consistent_group[iG] = true; //this avoid to include the same group more than once
        }
        if (n_current_consistency >= covisibility_consistency_threshold_ 
            && !has_enough_consistent) {
          consistent_enough_candidates_.push_back(candidate_keyframe);
          has_enough_consistent = true; // This avoid to insert the same candidate more than once
        }
      }
    }

    // If the group is not consistent with any previous group insert with consistency counter set to zero
    if (!is_consistent_for_some_groups) {
      current_consistent_groups.emplace_back(candidate_group,0);
    }
  }

  // Update Covisibility Consistent Groups
  consistent_groups_ = current_consistent_groups;

  // Add Current Keyframe to database
  keyframe_db_->add(current_keyframe_);

  if (consistent_enough_candidates_.empty()) {
    current_keyframe_->SetErase();
    return false;
  } else {
    return true;
  } 
}

bool LoopCloser::ComputeSim3() {
  // For each consistent loop candidate we try to compute a Sim3
  const int num_initial_candidates = consistent_enough_candidates_.size();

  // We compute first ORB matches for each candidate
  // If enough matches are found, we setup a Sim3Solver
  OrbMatcher matcher(0.75, true);

  std::vector<std::unique_ptr<Sim3Solver>> sim3_solvers(num_initial_candidates);

  std::vector<std::vector<MapPoint*>> vvpMapPointMatches;
  vvpMapPointMatches.resize(num_initial_candidates);

  std::vector<bool> is_discarded(num_initial_candidates, false); //TODO vector of bool

  int num_candidates = 0; //candidates with enough matches

  for (int i = 0; i < num_initial_candidates; ++i) {
    KeyFrame* keyframe = consistent_enough_candidates_[i];

    // Avoid that local mapping erase it while it is being processed in this thread
    keyframe->SetNotErase();

    if (keyframe->isBad()) {
      is_discarded[i] = true;
      continue;
    }

    int num_matches = matcher.SearchByBoW(current_keyframe_,
                                       keyframe,
                                       vvpMapPointMatches[i]);

    if (num_matches < 20) {
      is_discarded[i] = true;
      continue;
    } else {
      sim3_solvers[i].reset(new Sim3Solver(current_keyframe_,
                                           keyframe,
                                           vvpMapPointMatches[i],
                                           fix_scale_));
      sim3_solvers[i]->SetRansacParameters(0.99, 20, 300);
    }
    ++num_candidates;
  }

  // Perform alternatively RANSAC iterations for each candidate
  // until one is succesful or all fail
  bool is_match = false;
  while (num_candidates > 0 && !is_match) {
    for (int i = 0; i < num_initial_candidates; ++i) {
      if (is_discarded[i]) {
        continue;
      }

      KeyFrame* keyframe = consistent_enough_candidates_[i];

      // Perform 5 Ransac Iterations
      std::vector<bool> is_inlier; //TODO vector of bool
      int num_inliers;
      bool no_more;
      cv::Mat Scm  = sim3_solvers[i]->iterate(5,
                                              no_more,
                                              is_inlier,
                                              num_inliers);

      // If Ransac reachs max. iterations discard keyframe
      if (no_more) {
        is_discarded[i] = true;
        --num_candidates;
      }

      // If RANSAC returns a Sim3, perform a guided matching and optimize with all correspondences
      if (!Scm.empty()) {
        std::vector<MapPoint*> map_point_matches(vvpMapPointMatches[i].size(), nullptr);
        for (size_t j = 0; j < is_inlier.size(); ++j) {
          if (is_inlier[j]) {
            map_point_matches[j] = vvpMapPointMatches[i][j];
          }
        }

        const float scale = sim3_solvers[i]->GetEstimatedScale();
        const cv::Mat rot = sim3_solvers[i]->GetEstimatedRotation();
        const cv::Mat trans = sim3_solvers[i]->GetEstimatedTranslation();
        matcher.SearchBySim3(current_keyframe_,
                             keyframe,
                             map_point_matches,
                             scale,
                             rot,
                             trans,
                             7.5f);

        g2o::Sim3 gScm(Converter::toMatrix3d(rot),
                       Converter::toVector3d(trans),
                       scale);
        const int num_inliers = Optimizer::OptimizeSim3(current_keyframe_, 
                                                        keyframe, 
                                                        map_point_matches, 
                                                        gScm, 
                                                        10, 
                                                        fix_scale_);

        // If optimization is succesful stop ransacs and continue
        if (num_inliers >= 20) {
          is_match = true;
          matched_keyframe_ = keyframe;
          const g2o::Sim3 gSmw(Converter::toMatrix3d(keyframe->GetRotation()),
                               Converter::toVector3d(keyframe->GetTranslation()),
                               1.0);
          g2o_Scw_ = gScm * gSmw;
          Scw_ = Converter::toCvMat(g2o_Scw_);
          cur_matched_points_ = map_point_matches;
          break;
        }
      }
    }
  }

  if (!is_match) {
    for (KeyFrame* keyframe : consistent_enough_candidates_) {  
      keyframe->SetErase();
    }
    current_keyframe_->SetErase();
    return false;
  }

  // Retrieve MapPoints seen in Loop Keyframe and neighbors
  std::vector<KeyFrame*> loop_connected_keyframes = matched_keyframe_->GetVectorCovisibleKeyFrames();
  loop_connected_keyframes.push_back(matched_keyframe_);
  loop_map_points_.clear();
  for (KeyFrame* keyframe : loop_connected_keyframes) {
    std::vector<MapPoint*> map_points = keyframe->GetMapPointMatches();
    for (MapPoint* map_point : map_points) {
      if (map_point
          && !map_point->isBad()
          && map_point->loop_point_for_keyframe_id != current_keyframe_->mnId) {
        loop_map_points_.push_back(map_point);
        map_point->loop_point_for_keyframe_id = current_keyframe_->mnId;
      }
    }
  }

  // Find more matches projecting with the computed Sim3
  matcher.SearchByProjection(current_keyframe_,
                             Scw_,
                             loop_map_points_,
                             cur_matched_points_,
                             10);

  // If enough matches accept Loop
  int num_total_matches = 0;
  for (size_t i = 0; i < cur_matched_points_.size(); ++i) {
    if (cur_matched_points_[i]) {
      ++num_total_matches;
    }
  }

  if (num_total_matches >= 40) {
    for (int i = 0; i < num_initial_candidates; ++i) {
      if (consistent_enough_candidates_[i] != matched_keyframe_) {
        consistent_enough_candidates_[i]->SetErase();
      }
    }
    return true;
  } else {
    for (int i = 0; i < num_initial_candidates; ++i) {
      consistent_enough_candidates_[i]->SetErase();
    }
    current_keyframe_->SetErase();
    return false;
  }
}

void LoopCloser::SearchAndFuse(const KeyFrameAndPose& corrected_poses_map) {
  
  OrbMatcher matcher(0.8);

  for (auto mit = corrected_poses_map.begin(); 
            mit != corrected_poses_map.end();
            ++mit) {
    KeyFrame* keyframe = mit->first;

    g2o::Sim3 g2oScw = mit->second;
    cv::Mat cvScw = Converter::toCvMat(g2oScw);

    std::vector<MapPoint*> vpReplacePoints(loop_map_points_.size(), nullptr);
    matcher.Fuse(keyframe,
                 cvScw,
                 loop_map_points_,
                 4,
                 vpReplacePoints);

    // Get Map Mutex
    std::unique_lock<std::mutex> lock(map_->map_update_mutex);
    for (size_t i = 0; i < loop_map_points_.size(); ++i) {
      MapPoint* replace_point = vpReplacePoints[i];
      if (replace_point) {
        replace_point->Replace(loop_map_points_[i]);
      }
    }
  }
}

void LoopCloser::CorrectLoop() {
  std::cout << "Loop detected!\n";

  // Send a stop signal to Local Mapping
  // Avoid new keyframes are inserted while correcting the loop
  local_mapper_->RequestStop();

  // If a Global Bundle Adjustment is running, abort it
  if (IsRunningGBA()) {
    std::unique_lock<std::mutex> lock(global_bundle_adj_mutex_);
    stop_global_bundle_adj_ = true;

    ++full_bundle_adj_idx_;

    if (global_bundle_adjustment_thread_) {
      global_bundle_adjustment_thread_->detach();
    }
  }

  // Wait until Local Mapping has effectively stopped
  while(!local_mapper_->IsStopped()) {
    usleep(1000);
  }

  // Ensure current keyframe is updated
  current_keyframe_->UpdateConnections();

  // Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
  std::vector<KeyFrame*> cur_connected_keyframes = current_keyframe_->GetVectorCovisibleKeyFrames();
  cur_connected_keyframes.push_back(current_keyframe_);

  KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
  CorrectedSim3[current_keyframe_] = g2o_Scw_;
  cv::Mat Twc = current_keyframe_->GetPoseInverse();

  {
    // Get Map Mutex
    std::unique_lock<std::mutex> lock(map_->map_update_mutex);

    for (std::vector<KeyFrame*>::iterator vit = cur_connected_keyframes.begin(); 
                                          vit != cur_connected_keyframes.end();
                                          ++vit) {
      KeyFrame* keyframe_i = *vit;

      cv::Mat Tiw = keyframe_i->GetPose();

      if (keyframe_i != current_keyframe_) {
        cv::Mat Tic = Tiw * Twc;
        cv::Mat Ric = Tic.rowRange(0,3).colRange(0,3);
        cv::Mat tic = Tic.rowRange(0,3).col(3);
        g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric),
                         Converter::toVector3d(tic),
                         1.0);
        g2o::Sim3 g2oCorrectedSiw = g2oSic * g2o_Scw_;
        //Pose corrected with the Sim3 of the loop closure
        CorrectedSim3[keyframe_i] = g2oCorrectedSiw;
      }

      cv::Mat Riw = Tiw.rowRange(0,3).colRange(0,3);
      cv::Mat tiw = Tiw.rowRange(0,3).col(3);
      g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw),
                       Converter::toVector3d(tiw),
                       1.0);
      //Pose without correction
      NonCorrectedSim3[keyframe_i] = g2oSiw;
    }

    // Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
    for (KeyFrameAndPose::iterator mit = CorrectedSim3.begin(); 
                                   mit != CorrectedSim3.end(); 
                                   ++mit) {
      KeyFrame* keyframe_i = mit->first;
      g2o::Sim3 g2oCorrectedSiw = mit->second;
      g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();

      g2o::Sim3 g2oSiw = NonCorrectedSim3[keyframe_i];

      std::vector<MapPoint*> vpMPsi = keyframe_i->GetMapPointMatches();
      for (size_t iMP = 0; iMP < vpMPsi.size(); ++iMP) {
        MapPoint* pMPi = vpMPsi[iMP];
        if(!pMPi || pMPi->isBad()) {
          continue;
        }
        if(pMPi->corrected_by_keyframe == current_keyframe_->mnId) {
          continue;
        }

        // Project with non-corrected pose and project back with corrected pose
        cv::Mat P3Dw = pMPi->GetWorldPos();
        Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
        Eigen::Matrix<double,3,1> eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw));

        cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
        pMPi->SetWorldPos(cvCorrectedP3Dw);
        pMPi->corrected_by_keyframe = current_keyframe_->mnId;
        pMPi->corrected_reference = keyframe_i->mnId;
        pMPi->UpdateNormalAndDepth();
      }

      // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
      Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
      Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
      double s = g2oCorrectedSiw.scale();

      eigt *=(1./s); //[R t/s;0 1]

      cv::Mat correctedTiw = Converter::toCvSE3(eigR,eigt);

      keyframe_i->SetPose(correctedTiw);

      // Make sure connections are updated
      keyframe_i->UpdateConnections();
    }

    // Start Loop Fusion
    // Update matched map points and replace if duplicated
    for (size_t i = 0; i < cur_matched_points_.size(); ++i) {
      if (cur_matched_points_[i]) {
          MapPoint* pLoopMP = cur_matched_points_[i];
          MapPoint* pCurMP = current_keyframe_->GetMapPoint(i);
          if (pCurMP) {
            pCurMP->Replace(pLoopMP);
          } else {
            current_keyframe_->AddMapPoint(pLoopMP, i);
            pLoopMP->AddObservation(current_keyframe_,i);
            pLoopMP->ComputeDistinctiveDescriptors();
          }
      }
    }
  } // mutex scope

  // Project MapPoints observed in the neighborhood of the loop keyframe
  // into the current keyframe and neighbors using corrected poses.
  // Fuse duplications.
  SearchAndFuse(CorrectedSim3);


  // After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
  std::map<KeyFrame*, std::set<KeyFrame*>> LoopConnections;
  for (std::vector<KeyFrame*>::iterator vit = cur_connected_keyframes.begin(); 
                                        vit != cur_connected_keyframes.end(); 
                                        ++vit) {
    KeyFrame* keyframe_i = *vit;
    std::vector<KeyFrame*> vpPreviousNeighbors = keyframe_i->GetVectorCovisibleKeyFrames();

    // Update connections. Detect new links.
    keyframe_i->UpdateConnections();
    LoopConnections[keyframe_i] = keyframe_i->GetConnectedKeyFrames();
    for (std::vector<KeyFrame*>::iterator vit_prev = vpPreviousNeighbors.begin(); 
                                          vit_prev != vpPreviousNeighbors.end(); 
                                          ++vit_prev) {
      LoopConnections[keyframe_i].erase(*vit_prev);
    }
    for (std::vector<KeyFrame*>::iterator vit = cur_connected_keyframes.begin(); 
                                          vit != cur_connected_keyframes.end(); 
                                          ++vit) {
        LoopConnections[keyframe_i].erase(*vit);
    }
  }

  // Optimize graph
  Optimizer::OptimizeEssentialGraph(map_, 
                                    matched_keyframe_, 
                                    current_keyframe_, 
                                    NonCorrectedSim3, 
                                    CorrectedSim3, 
                                    LoopConnections, 
                                    fix_scale_);

  map_->InformNewBigChange();

  // Add loop edge
  matched_keyframe_->AddLoopEdge(current_keyframe_);
  current_keyframe_->AddLoopEdge(matched_keyframe_);

  // Launch a new thread to perform Global Bundle Adjustment
  is_running_global_budle_adj_ = true;
  is_finished_global_budle_adj_ = false;
  stop_global_bundle_adj_ = false;
  global_bundle_adjustment_thread_.reset(new std::thread(&LoopCloser::RunGlobalBundleAdjustment, 
                                                         this, 
                                                         current_keyframe_->mnId));

  // Loop closed. Release Local Mapping.
  local_mapper_->Release();    
  last_loop_kf_id_ = current_keyframe_->mnId;   
}

void LoopCloser::ResetIfRequested() {
  std::unique_lock<std::mutex> lock(reset_mutex_);
  if (reset_requested_) {
    loop_keyframe_queue_.clear();
    last_loop_kf_id_ = 0;
    reset_requested_ = false;
  }
}

bool LoopCloser::CheckFinish() const {
  std::unique_lock<std::mutex> lock(finish_mutex_);
  return finish_requested_;
}

void LoopCloser::SetFinish() {
  std::unique_lock<std::mutex> lock(finish_mutex_);
  is_finished_ = true;
}
