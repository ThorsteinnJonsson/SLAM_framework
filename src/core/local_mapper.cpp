#include "local_mapper.h"

#include "orb_features/orb_matcher.h"
#include "optimizer/optimizer.h"

// #pragma GCC optimize ("O0")

LocalMapper::LocalMapper(const std::shared_ptr<Map>& map, 
                         const bool is_monocular)
    : is_monocular_(is_monocular)
    , reset_requested_(false)
    , finish_requested_(false)
    , is_finished_(true)
    , map_(map)
    , abort_bundle_adjustment_(false)
    , is_stopped_(false)
    , requested_stop_(false)
    , not_stop_(false)
    , is_accepting_keyframes_(true) {

}

void LocalMapper::SetLoopCloser(const std::shared_ptr<LoopCloser>& loop_closer) {
  loop_closer_ = loop_closer;
}

void LocalMapper::Run() {
  is_finished_ = false;
  while (true) {
    // Tracking will see that Local Mapping is busy
    SetAcceptKeyFrames(false);

    // Check if there are keyframes in the queue
    if (CheckNewKeyFrames()) {
      // BoW conversion and insertion in Map
      ProcessNewKeyFrame();

      // Check recent MapPoints
      MapPointCulling();

      // Triangulate new MapPoints
      CreateNewMapPoints();

      if (!CheckNewKeyFrames()) {
        // Find more matches in neighbor keyframes and fuse point duplications
        SearchInNeighbors();
      }

      abort_bundle_adjustment_ = false;
      if (!CheckNewKeyFrames() && !stopRequested()) {
        // Local BA
        if (map_->NumKeyFramesInMap() > 2) {
          Optimizer::LocalBundleAdjustment(current_keyframe_, 
                                           &abort_bundle_adjustment_, 
                                           map_);
        }

        // Check redundant local Keyframes
        KeyFrameCulling();
      }

      loop_closer_->InsertKeyFrame(current_keyframe_);

    } else if (Stop()){
      // Safe area to stop
      while (IsStopped() && !CheckFinish()) {
        usleep(3000);
      }
      if(CheckFinish()) {
        break;
      }
    }

    ResetIfRequested();

    // Tracking will see that Local Mapping is not busy
    SetAcceptKeyFrames(true);

    if(CheckFinish()) {
      break;
    }

    usleep(3000);
  }

  SetFinish();
}

void LocalMapper::InsertKeyFrame(KeyFrame* keyframe) {
  std::unique_lock<std::mutex> lock(new_keyframes_mutex_);
  new_keyframes_.push_back(keyframe);
  abort_bundle_adjustment_ = true;
}

void LocalMapper::RequestStop(){
  std::unique_lock<std::mutex> lock(stop_mutex_);
  requested_stop_ = true;
  std::unique_lock<std::mutex> lock2(new_keyframes_mutex_);
  abort_bundle_adjustment_ = true;
}

void LocalMapper::RequestReset(){
  {
    std::unique_lock<std::mutex> lock(reset_mutex_);
    reset_requested_ = true;
  }
  while (true) {
    {
      std::unique_lock<std::mutex> lock2(reset_mutex_);
      if (!reset_requested_) {
        break;
      }
    }
    usleep(3000);
  }
}

bool LocalMapper::Stop() {
  std::unique_lock<std::mutex> lock(stop_mutex_);
  if (requested_stop_ && !not_stop_) {
    is_stopped_ = true;
    std::cout << "Local Mapping STOP\n";
    return true;
  }
  return false;
}

void LocalMapper::Release() {
  std::unique_lock<std::mutex> lock(stop_mutex_);
  std::unique_lock<std::mutex> lock2(finished_mutex_);
  if (is_finished_) {
    return;
  }
  is_stopped_ = false;
  requested_stop_ = false;
  for (auto new_kf = new_keyframes_.begin(); 
            new_kf != new_keyframes_.end(); 
            ++new_kf) {
    delete *new_kf;
  }
  new_keyframes_.clear();
  std::cout << "Local Mapping RELEASE\n";
}

bool LocalMapper::IsStopped() const {
  std::unique_lock<std::mutex> lock(stop_mutex_);
  return is_stopped_;
}

bool LocalMapper::stopRequested() const {
  std::unique_lock<std::mutex> lock(stop_mutex_);
  return requested_stop_;
}

bool LocalMapper::IsAcceptingKeyFrames() const {
  std::unique_lock<std::mutex> lock(accept_keyframe_mutex_);
  return is_accepting_keyframes_;
}

void LocalMapper::SetAcceptKeyFrames(const bool flag) {
  std::unique_lock<std::mutex> lock(accept_keyframe_mutex_);
  is_accepting_keyframes_ = flag;
}

void LocalMapper::InterruptBA() {
  abort_bundle_adjustment_ = true;;
}

bool LocalMapper::SetNotStop(const bool flag) {
  std::unique_lock<std::mutex> lock(stop_mutex_);
  if (flag && is_stopped_) {
    return false;
  }
  not_stop_ = flag;
  return true;
}

void LocalMapper::RequestFinish() {
  std::unique_lock<std::mutex> lock(finished_mutex_);
  finish_requested_ = true;
}

bool LocalMapper::IsFinished() const {
  std::unique_lock<std::mutex> lock(finished_mutex_);
  return is_finished_;
}

int LocalMapper::NumKeyframesInQueue() const {
  std::unique_lock<std::mutex> lock(new_keyframes_mutex_);
  return new_keyframes_.size();  
}

bool LocalMapper::CheckNewKeyFrames() const {
  std::unique_lock<std::mutex> lock(new_keyframes_mutex_);
  return (!new_keyframes_.empty());
}

void LocalMapper::ProcessNewKeyFrame() {
  {
    std::unique_lock<std::mutex> lock(new_keyframes_mutex_);
    current_keyframe_ = new_keyframes_.front();
    new_keyframes_.pop_front();
  }

  // Compute Bags of Words structures
  current_keyframe_->ComputeBoW();

  // Associate MapPoints to the new keyframe and update normal and descriptor
  const std::vector<MapPoint*> map_point_matches = current_keyframe_->GetMapPointMatches();

  for (size_t i = 0; i < map_point_matches.size(); ++i) {
    MapPoint* map_point = map_point_matches[i];
    if(map_point && !map_point->isBad()) {
      if (!map_point->IsInKeyFrame(current_keyframe_)) {
        map_point->AddObservation(current_keyframe_, i);
        map_point->UpdateNormalAndDepth();
        map_point->ComputeDistinctiveDescriptors();
      } else {
        // this can only happen for new stereo points inserted by the Tracking
        recently_added_map_points_.push_back(map_point);
      }
    }
  }    

  // Update links in the Covisibility Graph
  current_keyframe_->UpdateConnections();

  // Insert Keyframe in Map
  map_->AddKeyFrame(current_keyframe_);
}

void LocalMapper::MapPointCulling() {
  // Check Recent Added MapPoints
  
  const long current_keyframe_id = static_cast<long>(current_keyframe_->Id());
  const int min_observations_thresh = is_monocular_? 2 : 3;

  std::list<MapPoint*>::iterator lit = recently_added_map_points_.begin();
  while (lit != recently_added_map_points_.end()){
    MapPoint* map_point = *lit;
    if (map_point->isBad()) {
      lit = recently_added_map_points_.erase(lit);
    } else if (map_point->GetFoundRatio() < 0.25f) {
      map_point->SetBadFlag();
      lit = recently_added_map_points_.erase(lit);
    } else if ((current_keyframe_id - map_point->GetFirstKeyframeID()) >= 2l &&
                map_point->NumObservations() <= min_observations_thresh) {
        map_point->SetBadFlag();
        lit = recently_added_map_points_.erase(lit);
    } else if ((current_keyframe_id - map_point->GetFirstKeyframeID()) >= 3l) {
        lit = recently_added_map_points_.erase(lit);
    } else {
     ++lit;
    }
  }
}

void LocalMapper::CreateNewMapPoints() {
  // Retrieve neighbor keyframes in covisibility graph
  const int num_kf = is_monocular_ ? 20 : 10;
  const std::vector<KeyFrame*> neighbor_keyframes = current_keyframe_->GetBestCovisibilityKeyFrames(num_kf);

  OrbMatcher matcher(0.6f, false);

  const cv::Mat rot_current_world = current_keyframe_->GetRotation();
  const cv::Mat rot_world_current = rot_current_world.t();
  const cv::Mat translation_current_world = current_keyframe_->GetTranslation();
  cv::Mat T_current_world(3,4,CV_32F);
  rot_current_world.copyTo(T_current_world.colRange(0,3));
  translation_current_world.copyTo(T_current_world.col(3));
  cv::Mat current_pos = current_keyframe_->GetCameraCenter();

  const float fx_cur = current_keyframe_->fx;
  const float fy_cur = current_keyframe_->fy;
  const float cx_cur = current_keyframe_->cx;
  const float cy_cur = current_keyframe_->cy;
  const float invfx_cur = current_keyframe_->invfx;
  const float invfy_cur = current_keyframe_->invfy;

  const float ratioFactor = 1.5f * current_keyframe_->mfScaleFactor;

  // Search matches with epipolar restriction and triangulate
  int num_added_points = 0;
  for (size_t i = 0; i < neighbor_keyframes.size(); ++i) {
    if (i > 0 && CheckNewKeyFrames()) {
      return;
    }
    KeyFrame* nbor_keyframe = neighbor_keyframes[i];

    // Check first that baseline is not too short
    const cv::Mat nbor_pos = nbor_keyframe->GetCameraCenter();
    const float baseline = cv::norm(nbor_pos - current_pos);

    if (is_monocular_) {
      const float nbor_median_depth = nbor_keyframe->ComputeSceneMedianDepth(2);
      const float baseline_depth_ratio = baseline / nbor_median_depth;
      if (baseline_depth_ratio < 0.01f) {
        continue;
      }
    } else {
      if (baseline < nbor_keyframe->mb) {
        continue;
      }
    }

    // Compute Fundamental Matrix
    const cv::Mat F12 = ComputeFundamentalMatrix(current_keyframe_, nbor_keyframe);

    // Search matches that fullfil epipolar constraint
    std::vector<std::pair<size_t,size_t>> vMatchedIndices;
    const bool only_stereo = false;
    matcher.SearchForTriangulation(current_keyframe_,
                                   nbor_keyframe,
                                   F12,
                                   vMatchedIndices,
                                   only_stereo);

    const cv::Mat rot_nbor_world = nbor_keyframe->GetRotation();
    const cv::Mat ros_world_nbor = rot_nbor_world.t();
    const cv::Mat translation_nbor_world = nbor_keyframe->GetTranslation();
    cv::Mat T_nbor_world(3,4,CV_32F);
    rot_nbor_world.copyTo(T_nbor_world.colRange(0,3));
    translation_nbor_world.copyTo(T_nbor_world.col(3));

    const float fx_nbor = nbor_keyframe->fx;
    const float fy_nbor = nbor_keyframe->fy;
    const float cx_nbor = nbor_keyframe->cx;
    const float cy_nbor = nbor_keyframe->cy;
    const float invfx_nbor = nbor_keyframe->invfx;
    const float invfy_nbor = nbor_keyframe->invfy;

    // Triangulate each match
    const int nmatches = vMatchedIndices.size();
    for(int ikp = 0; ikp < nmatches; ++ikp) {
      const int idx1 = vMatchedIndices[ikp].first;
      const int idx2 = vMatchedIndices[ikp].second;

      const cv::KeyPoint& kp1 = current_keyframe_->mvKeysUn[idx1];
      const float kp1_ur = current_keyframe_->mvuRight[idx1];
      bool bStereo1 = (kp1_ur >= 0);

      const cv::KeyPoint& kp2 = nbor_keyframe->mvKeysUn[idx2];
      const float kp2_ur = nbor_keyframe->mvuRight[idx2];
      bool bStereo2 = (kp2_ur >= 0);

      // Check parallax between rays
      cv::Mat xn1 = (cv::Mat_<float>(3,1) << (kp1.pt.x-cx_cur)*invfx_cur, 
                                             (kp1.pt.y-cy_cur)*invfy_cur, 
                                             1.0);
      cv::Mat xn2 = (cv::Mat_<float>(3,1) << (kp2.pt.x-cx_nbor)*invfx_nbor, 
                                             (kp2.pt.y-cy_nbor)*invfy_nbor, 
                                             1.0);
      cv::Mat ray1 = rot_world_current * xn1;
      cv::Mat ray2 = ros_world_nbor * xn2;
      const float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1)*cv::norm(ray2));

      float cosParallaxStereo = cosParallaxRays + 1;
      float cosParallaxStereo1 = cosParallaxStereo;
      float cosParallaxStereo2 = cosParallaxStereo;

      if (bStereo1) {
        cosParallaxStereo1 = std::cos(2*std::atan2(current_keyframe_->mb/2,
                                                   current_keyframe_->mvDepth[idx1]));
      } else if (bStereo2) {
        cosParallaxStereo2 = std::cos(2*std::atan2(nbor_keyframe->mb/2,
                                                   nbor_keyframe->mvDepth[idx2]));
      }

      cosParallaxStereo = std::min(cosParallaxStereo1,cosParallaxStereo2);

      cv::Mat x3D;
      if (cosParallaxRays < cosParallaxStereo && 
          cosParallaxRays > 0 && 
          (bStereo1 || bStereo2 || cosParallaxRays < 0.9998)) {
        // Linear Triangulation Method
        cv::Mat A(4,4,CV_32F);
        A.row(0) = xn1.at<float>(0)*T_current_world.row(2)-T_current_world.row(0);
        A.row(1) = xn1.at<float>(1)*T_current_world.row(2)-T_current_world.row(1);
        A.row(2) = xn2.at<float>(0)*T_nbor_world.row(2)-T_nbor_world.row(0);
        A.row(3) = xn2.at<float>(1)*T_nbor_world.row(2)-T_nbor_world.row(1);

        cv::Mat w,u,vt;
        cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

        x3D = vt.row(3).t();

        if (x3D.at<float>(3) == 0) {
          continue;
        }

        // Euclidean coordinates
        x3D = x3D.rowRange(0,3) / x3D.at<float>(3);
      }
      else if (bStereo1 && cosParallaxStereo1 < cosParallaxStereo2) {
        x3D = current_keyframe_->UnprojectStereo(idx1);
      }
      else if (bStereo2 && cosParallaxStereo2 < cosParallaxStereo1) {
        x3D = nbor_keyframe->UnprojectStereo(idx2);
      }
      else {
        continue; //No stereo and very low parallax
      }
      cv::Mat x3Dt = x3D.t();

      //Check triangulation in front of cameras
      const float z_cur = rot_current_world.row(2).dot(x3Dt) + T_current_world.at<float>(2,3);
      if (z_cur <= 0) {
        continue;
      }

      const float z_nbor = rot_nbor_world.row(2).dot(x3Dt) + T_nbor_world.at<float>(2,3);
      if (z_nbor <= 0) {
        continue;
      }

      //Check reprojection error in first keyframe
      const float sigmaSquare1 = current_keyframe_->mvLevelSigma2[kp1.octave];
      const float x1 = rot_current_world.row(0).dot(x3Dt) + T_current_world.at<float>(0,3);
      const float y1 = rot_current_world.row(1).dot(x3Dt) + T_current_world.at<float>(1,3);

      const float u1 = (fx_cur * x1) / z_cur + cx_cur;
      const float v1 = (fy_cur * y1) / z_cur + cy_cur;
      const float errX1 = u1 - kp1.pt.x;
      const float errY1 = v1 - kp1.pt.y;
      if (bStereo1) {
        const float u1_r = u1 - (current_keyframe_->mbf / z_cur);
        const float errX1_r = u1_r - kp1_ur;
        if ((errX1*errX1 + errY1*errY1 + errX1_r*errX1_r) > 7.8*sigmaSquare1) {
          continue;
        }  
      } else {
        if ((errX1*errX1 + errY1*errY1) > 5.991*sigmaSquare1) {
          continue;
        }
      }

      //Check reprojection error in second keyframe
      const float sigma_square_nbor = nbor_keyframe->mvLevelSigma2[kp2.octave];
      const float x_nbor = rot_nbor_world.row(0).dot(x3Dt) + T_nbor_world.at<float>(0,3);
      const float y_nbor = rot_nbor_world.row(1).dot(x3Dt) + T_nbor_world.at<float>(1,3);
      
      const float u2 = (fx_nbor * x_nbor) / z_nbor + cx_nbor;
      const float v2 = (fy_nbor * y_nbor) / z_nbor + cy_nbor;
      const float errX2 = u2 - kp2.pt.x;
      const float errY2 = v2 - kp2.pt.y;
      if (bStereo2) {
        const float u2_r = u2 - (current_keyframe_->mbf / z_nbor);
        const float errX2_r = u2_r - kp2_ur;
        if ((errX2*errX2 + errY2*errY2 + errX2_r*errX2_r) > 7.8f * sigma_square_nbor) {
          continue;
        } 
      } else {
        if ((errX2*errX2 + errY2*errY2) > 5.991*sigma_square_nbor) {
          continue;
        }
      }

      //Check scale consistency
      const float dist1 = cv::norm(x3D - current_pos);
      const float dist2 = cv::norm(x3D - nbor_pos);

      if (dist1==0 || dist2==0) {
        continue;
      }

      const float dist_ratio = dist2 / dist1;
      const float ratioOctave = current_keyframe_->mvScaleFactors[kp1.octave] / nbor_keyframe->mvScaleFactors[kp2.octave];

      if (dist_ratio * ratioFactor < ratioOctave || dist_ratio / ratioFactor > ratioOctave) {
        continue;
      }

      // Triangulation is succesfull
      MapPoint* pMP = new MapPoint(x3D,
                                   current_keyframe_,
                                   map_);

      pMP->AddObservation(current_keyframe_, idx1);            
      pMP->AddObservation(nbor_keyframe, idx2);

      current_keyframe_->AddMapPoint(pMP, idx1);
      nbor_keyframe->AddMapPoint(pMP, idx2);

      pMP->ComputeDistinctiveDescriptors();
      pMP->UpdateNormalAndDepth();

      map_->AddMapPoint(pMP);
      recently_added_map_points_.push_back(pMP);
      ++num_added_points;
    }
  }
}

void LocalMapper::SearchInNeighbors() {
  // Retrieve neighbor keyframes
  const int num_kf = is_monocular_? 20 : 10;
  std::vector<KeyFrame*> target_keyframes;
  for (KeyFrame* nbor_keyframe : current_keyframe_->GetBestCovisibilityKeyFrames(num_kf)) {
    if (nbor_keyframe->isBad() 
        || nbor_keyframe->mnFuseTargetForKF == current_keyframe_->Id()) {
      continue;
    }
    target_keyframes.push_back(nbor_keyframe);
    nbor_keyframe->mnFuseTargetForKF = current_keyframe_->Id();

    // Extend to some second neighbors
    for (KeyFrame* second_nbor : nbor_keyframe->GetBestCovisibilityKeyFrames(5)) {
      if (second_nbor->isBad() || 
          second_nbor->mnFuseTargetForKF == current_keyframe_->Id() || 
          second_nbor->Id() == current_keyframe_->Id()) {
        continue;
      }
      target_keyframes.push_back(second_nbor);
    }
  }

  // Search matches by projection from current KF in target KFs
  OrbMatcher matcher;
  std::vector<MapPoint*> map_point_matches = current_keyframe_->GetMapPointMatches();
  for (KeyFrame* target_keyframe : target_keyframes) {
    matcher.Fuse(target_keyframe, map_point_matches);
  }

  // Search matches by projection from target KFs in current KF
  std::vector<MapPoint*> fuse_candidates;
  fuse_candidates.reserve(target_keyframes.size() * map_point_matches.size());

  for (KeyFrame* target_keyframe : target_keyframes) {
    for (MapPoint* map_point : target_keyframe->GetMapPointMatches()) {
      if (!map_point
          || map_point->isBad()
          || map_point->fuse_candidate_id_for_keyframe == current_keyframe_->Id() ) {
        continue;
      }
      map_point->fuse_candidate_id_for_keyframe = current_keyframe_->Id();
      fuse_candidates.push_back(map_point);
    }
  }
  fuse_candidates.shrink_to_fit();
  matcher.Fuse(current_keyframe_, fuse_candidates);

  // Update points
  map_point_matches = current_keyframe_->GetMapPointMatches();
  for (size_t i = 0; i < map_point_matches.size(); ++i) {
    MapPoint* map_point = map_point_matches[i];
    if (map_point && !map_point->isBad()) {
      map_point->ComputeDistinctiveDescriptors();
      map_point->UpdateNormalAndDepth();
    }
  }

  // Update connections in covisibility graph
  current_keyframe_->UpdateConnections();
}

void LocalMapper::KeyFrameCulling() {
  // Check redundant keyframes (only local keyframes)
  // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
  // in at least other 3 keyframes (in the same or finer scale)
  // We only consider close stereo points
  for (KeyFrame* keyframe : current_keyframe_->GetVectorCovisibleKeyFrames()) {
    if (keyframe->Id() == 0) {
      continue;
    }
    const std::vector<MapPoint*> map_points = keyframe->GetMapPointMatches();

    const int obs_threshold = 3;
    int num_redundant_obs = 0;
    int num_points = 0;
    for (size_t i = 0; i < map_points.size(); ++i) {
      MapPoint* map_point = map_points[i];
      if(map_point && !map_point->isBad()) {
        if (!is_monocular_) {
          if (keyframe->mvDepth[i] > keyframe->mThDepth || 
              keyframe->mvDepth[i] < 0) {
            continue;
          }
        }

        ++num_points;
        if (map_point->NumObservations() > obs_threshold) {
          const int scale_level = keyframe->mvKeysUn[i].octave;
          const std::map<KeyFrame*,size_t> observations = map_point->GetObservations();
          int num_obs = 0;
          for (auto mit = observations.begin(); 
                    mit != observations.end(); 
                    ++mit) {
            const KeyFrame* obs_keyframe = mit->first;
            const size_t obs_idx = mit->second;
            if (obs_keyframe == keyframe) {
              continue;
            }
            const int obs_scale_level = obs_keyframe->mvKeysUn[obs_idx].octave;

            if (obs_scale_level <= scale_level + 1) {
              ++num_obs;
              if (num_obs >= obs_threshold) {
                break;
              }
            }
          }
          if (num_obs >= obs_threshold) {
            ++num_redundant_obs;
          }
        }
      }
    }  

    if (num_redundant_obs > 0.9f * num_points) {
      keyframe->SetBadFlag();
    }
  }
}

cv::Mat LocalMapper::ComputeFundamentalMatrix(KeyFrame* keyframe1, KeyFrame* keyframe2) const {
  const cv::Mat R1w = keyframe1->GetRotation();
  const cv::Mat t1w = keyframe1->GetTranslation();
  const cv::Mat R2w = keyframe2->GetRotation();
  const cv::Mat t2w = keyframe2->GetTranslation();

  const cv::Mat R12 = R1w * R2w.t();
  const cv::Mat t12 = -R1w * R2w.t() * t2w + t1w;

  const cv::Mat t12x = SkewSymmetricMatrix(t12);

  const cv::Mat& K1 = keyframe1->mK;
  const cv::Mat& K2 = keyframe2->mK;

  return K1.t().inv() * t12x * R12 * K2.inv();
}

cv::Mat LocalMapper::SkewSymmetricMatrix(const cv::Mat& v) const {
  return (cv::Mat_<float>(3,3) <<              0,  -v.at<float>(2),  v.at<float>(1),
                                   v.at<float>(2),              0,  -v.at<float>(0),
                                  -v.at<float>(1),  v.at<float>(0),              0);  
}

void LocalMapper::ResetIfRequested() {
  std::unique_lock<std::mutex> lock(reset_mutex_);
  if (reset_requested_) {
    new_keyframes_.clear();
    recently_added_map_points_.clear();
    reset_requested_ = false;
  }  
}

bool LocalMapper::CheckFinish() const {
  std::unique_lock<std::mutex> lock(finished_mutex_);
  return finish_requested_;
}

void LocalMapper::SetFinish() {
  std::unique_lock<std::mutex> lock(finished_mutex_);
  is_finished_ = true;    
  std::unique_lock<std::mutex> lock2(stop_mutex_);
  is_stopped_ = true;
}
