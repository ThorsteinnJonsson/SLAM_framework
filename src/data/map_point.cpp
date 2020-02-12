#include "map_point.h"

#include "orb_features/orb_matcher.h"

long unsigned int MapPoint::next_id = 0;
std::mutex MapPoint::global_mutex;

MapPoint::MapPoint(const cv::Mat& position, 
                   KeyFrame* reference_keyframe, 
                   const std::shared_ptr<Map>& map)
      : track_reference_id_for_frame(0)
      , last_frame_id_seen(0)
      , bundle_adj_local_id_for_keyframe(0)
      , fuse_candidate_id_for_keyframe(0)
      , loop_point_for_keyframe_id(0)
      , corrected_by_keyframe(0)
      , corrected_reference(0)
      , bundle_adj_global_for_keyframe_id(0)
      , first_keyframe_id_(reference_keyframe->Id())
      , first_frame_id_(reference_keyframe->FrameId())
      , num_observations_(0)
      , reference_keyframe_(reference_keyframe)
      , num_visible_(1)
      , num_found_(1)
      , is_bad_(false)
      , replaced_point_(nullptr)
      , min_dist_(0)
      , max_dist_(0)
      , map_(map) {

  position.copyTo(world_position_);
  normal_vector_ = cv::Mat::zeros(3,1,CV_32F);

  // MapPoints can be created from Tracking and Local Mapping. 
  // This mutex avoid conflicts with ID.
  std::unique_lock<std::mutex> lock(map_->point_creation_mutex);
  id_ = next_id++;
}

MapPoint::MapPoint(const cv::Mat& position, 
                   const std::shared_ptr<Map>& map, 
                   Frame* frame, 
                   const int &idxF)
      : track_reference_id_for_frame(0)
      , last_frame_id_seen(0)
      , bundle_adj_local_id_for_keyframe(0)
      , fuse_candidate_id_for_keyframe(0)
      , loop_point_for_keyframe_id(0)
      , corrected_by_keyframe(0)
      , corrected_reference(0)
      , bundle_adj_global_for_keyframe_id(0)
      , first_keyframe_id_(-1)
      , first_frame_id_(frame->Id())
      , num_observations_(0)
      , reference_keyframe_(nullptr)
      , num_visible_(1)
      , num_found_(1)
      , is_bad_(false)
      , replaced_point_(nullptr)
      , map_(map) {
  position.copyTo(world_position_);
  const cv::Mat Ow = frame->GetCameraCenter();
  normal_vector_ = world_position_ - Ow;
  normal_vector_ = normal_vector_ / cv::norm(normal_vector_);

  const cv::Mat PC = position - Ow;
  const float dist = cv::norm(PC);
  const int level = frame->GetUndistortedKeys()[idxF].octave;
  const float levelScaleFactor =  frame->ScaleFactors()[level];
  const int nLevels = frame->GetScaleLevel();

  max_dist_ = dist*levelScaleFactor;
  min_dist_ = max_dist_/frame->ScaleFactors()[nLevels-1];

  frame->GetDescriptors().row(idxF).copyTo(descriptor_);

  // MapPoints can be created from Tracking and Local Mapping. 
  //This mutex avoid conflicts with ID.
  std::unique_lock<std::mutex> lock(map_->point_creation_mutex);
  id_ = next_id++;
}

void MapPoint::SetWorldPos(const cv::Mat& position) {
  std::unique_lock<std::mutex> lock1(global_mutex);
  std::unique_lock<std::mutex> lock2(position_mutex_);
  position.copyTo(world_position_);
}

cv::Mat MapPoint::GetWorldPos() const {
  std::unique_lock<std::mutex> lock(position_mutex_);
  return world_position_.clone();
}

cv::Mat MapPoint::GetNormal() const {
  std::unique_lock<std::mutex> lock(position_mutex_);
  return normal_vector_.clone();
}

KeyFrame* MapPoint::GetReferenceKeyFrame() const {
  std::unique_lock<std::mutex> lock(feature_mutex_);
  return reference_keyframe_;
}

std::map<KeyFrame*,size_t> MapPoint::GetObservations() const {
  std::unique_lock<std::mutex> lock(feature_mutex_);
  return observations_;
}

int MapPoint::NumObservations() const {
  std::unique_lock<std::mutex> lock(feature_mutex_);
  return num_observations_;
}

void MapPoint::AddObservation(KeyFrame* keyframe, size_t idx) {
  std::unique_lock<std::mutex> lock(feature_mutex_);
  if (observations_.count(keyframe)) {
    return;
  }
  observations_[keyframe] = idx;
  if (keyframe->mvuRight[idx] >= 0) {
    num_observations_ += 2;
  } else {
    num_observations_ += 1;
  }
}

void MapPoint::EraseObservation(KeyFrame* keyframe) {
  bool bBad = false;
  { 
    std::unique_lock<std::mutex> lock(feature_mutex_);
    if (observations_.count(keyframe)) {
      const int idx = observations_[keyframe];
      if (keyframe->mvuRight[idx] >= 0) {
        num_observations_ -= 2;
      } else {
        num_observations_ -= 1;            
      }
      observations_.erase(keyframe);

      if (reference_keyframe_== keyframe) {
        reference_keyframe_ = observations_.begin()->first;
      }

      // If only 2 observations or less, discard point
      if (num_observations_ <= 2) {
        bBad = true;
      }
    }
  }
  if (bBad) {
    SetBadFlag();
  }
}

int MapPoint::GetIndexInKeyFrame(KeyFrame* keyframe) const {
  std::unique_lock<std::mutex> lock(feature_mutex_);
  return (observations_.count(keyframe)) ? observations_.at(keyframe) : -1;
}

bool MapPoint::IsInKeyFrame(KeyFrame* keyframe) const {
  std::unique_lock<std::mutex> lock(feature_mutex_);
  return observations_.count(keyframe);
}

void MapPoint::SetBadFlag() {
  std::map<KeyFrame*,size_t> obs;

  {
    std::unique_lock<std::mutex> lock1(feature_mutex_);
    std::unique_lock<std::mutex> lock2(position_mutex_);
    is_bad_ = true;
    obs = observations_;
    observations_.clear();
  }
  for (std::map<KeyFrame*,size_t>::iterator mit = obs.begin(); 
                                           mit != obs.end(); 
                                           ++mit) {
    KeyFrame* keyframe = mit->first;
    keyframe->EraseMapPointMatch(mit->second);
  }
  map_->EraseMapPoint(this);
}

bool MapPoint::isBad() const {
  std::unique_lock<std::mutex> lock1(feature_mutex_);
  std::unique_lock<std::mutex> lock2(position_mutex_);
  return is_bad_;
}

void MapPoint::Replace(MapPoint* point) {
  if (point->GetId() == this->id_) {
    return;
  }

  int num_visible, num_found;
  std::map<KeyFrame*,size_t> observations;
  
  {
    std::unique_lock<std::mutex> lock1(feature_mutex_);
    std::unique_lock<std::mutex> lock2(position_mutex_);
    is_bad_ = true;
    observations = observations_;
    observations_.clear();
    num_visible = num_visible_;
    num_found = num_found_;
    replaced_point_ = point;
  }

  for (const auto& observation : observations) {
    // Replace measurement in keyframe
    KeyFrame* keyframe = observation.first;
    const size_t index = observation.second;

    if (!point->IsInKeyFrame(keyframe)) {
      keyframe->ReplaceMapPointMatch(index, point);
      point->AddObservation(keyframe, index);
    } else {
      keyframe->EraseMapPointMatch(index);
    }
  } 
  point->IncreaseFound(num_found);
  point->IncreaseVisible(num_visible);
  point->ComputeDistinctiveDescriptors();

  map_->EraseMapPoint(this);
}

MapPoint* MapPoint::GetReplaced() const {
  std::unique_lock<std::mutex> lock1(feature_mutex_);
  std::unique_lock<std::mutex> lock2(position_mutex_);
  return replaced_point_;
}

void MapPoint::IncreaseVisible(int n) {
  std::unique_lock<std::mutex> lock(feature_mutex_);
  num_visible_ += n;
}

void MapPoint::IncreaseFound(int n) {
  std::unique_lock<std::mutex> lock(feature_mutex_);
  num_found_ += n;
}

float MapPoint::GetFoundRatio() const {
  std::unique_lock<std::mutex> lock(feature_mutex_);
  return static_cast<float>(num_found_) / num_visible_;
}

void MapPoint::ComputeDistinctiveDescriptors() {
  // Retrieve all observed descriptors
  std::map<KeyFrame*, size_t> observations;
  {
    std::unique_lock<std::mutex> lock1(feature_mutex_);
    if (is_bad_ || observations_.empty()) {
      return;
    }
    observations = observations_;
  }
  
  std::vector<cv::Mat> descriptors;
  descriptors.reserve(observations.size());
  for (std::map<KeyFrame*, size_t>::iterator mit = observations.begin(); 
                                             mit != observations.end(); 
                                             ++mit) {
    KeyFrame* keyframe = mit->first;
    if (!keyframe->isBad()) {
      descriptors.push_back(keyframe->mDescriptors.row(mit->second));
    }      
  }
  if (descriptors.empty()) {
    return;
  }

  // Compute distances between them
  const size_t N = descriptors.size();
  std::vector<std::vector<float>> distances(N, std::vector<float>(N));
  for (size_t i = 0; i < N; i++) {
    distances[i][i] = 0;
    for (size_t j = i + 1; j < N; j++) {
        const int dist_ij = OrbMatcher::DescriptorDistance(descriptors[i], descriptors[j]);
        distances[i][j] = dist_ij;
        distances[j][i] = dist_ij;
    }
  }

  // Take the descriptor with least median distance to the rest
  int best_median = std::numeric_limits<int>::max();
  int best_idx = 0;
  const int halfway_idx = 0.5*(N-1);
  for (size_t i = 0; i < N; ++i) {
    std::vector<int> dists(distances[i].begin(), distances[i].end());
    std::nth_element(dists.begin(), dists.begin()+halfway_idx, dists.end());
    const int median = dists[halfway_idx];
    if (median < best_median) {
      best_median = median;
      best_idx = i;
    }
  }

  {
    std::unique_lock<std::mutex> lock(feature_mutex_);
    descriptor_ = descriptors[best_idx].clone();
  }
}

cv::Mat MapPoint::GetDescriptor() const {
  std::unique_lock<std::mutex> lock(feature_mutex_);
  return descriptor_.clone();
}

void MapPoint::UpdateNormalAndDepth() {
  std::map<KeyFrame*, size_t> observations;
  KeyFrame* reference_keyframe;
  cv::Mat position;
  {
    std::unique_lock<std::mutex> lock1(feature_mutex_);
    std::unique_lock<std::mutex> lock2(position_mutex_);
    if (is_bad_) {
      return;
    }
    observations = observations_;
    reference_keyframe = reference_keyframe_;
    position = world_position_.clone();
  }

  if (observations.empty()) {
    return;
  }

  cv::Mat normal = cv::Mat::zeros(3, 1, CV_32F);
  int n = 0;
  for (std::map<KeyFrame*, size_t>::iterator mit = observations.begin(); 
                                             mit != observations.end(); 
                                             mit++) {
    KeyFrame* keyframe = mit->first;
    cv::Mat Owi = keyframe->GetCameraCenter();
    cv::Mat normali = world_position_ - Owi;
    normal = normal + normali / cv::norm(normali);
    ++n;
  }

  const cv::Mat PC = position - reference_keyframe->GetCameraCenter();
  const float dist = cv::norm(PC);
  const int level = reference_keyframe->undistorted_keypoints[observations[reference_keyframe]].octave;
  const float levelScaleFactor = reference_keyframe->scale_factors[level];
  const int nLevels = reference_keyframe->scale_levels;

  {
    std::unique_lock<std::mutex> lock3(position_mutex_);
    max_dist_ = dist * levelScaleFactor;
    min_dist_ = max_dist_ / reference_keyframe->scale_factors[nLevels - 1];
    normal_vector_ = normal / n;
  }
}

float MapPoint::GetMinDistanceInvariance() const {
  std::unique_lock<std::mutex> lock(position_mutex_);
  return 0.8f * min_dist_;
}

float MapPoint::GetMaxDistanceInvariance() const {
  std::unique_lock<std::mutex> lock(position_mutex_);
  return 1.2f * max_dist_;
}

int MapPoint::PredictScale(const float dist, KeyFrame* keyframe) const {
  float ratio;
  {
    std::unique_lock<std::mutex> lock(position_mutex_);
    ratio = max_dist_ / dist;
  }

  int nScale = std::ceil( std::log(ratio) / keyframe->log_scale_factor );
  if (nScale < 0) {
    nScale = 0;
  } else if (nScale >= keyframe->scale_levels) {
    nScale = keyframe->scale_levels - 1;
  }
  return nScale;
}

int MapPoint::PredictScale(const float dist, Frame* frame) const {
  float ratio;
  {
    std::unique_lock<std::mutex> lock(position_mutex_);
    ratio = max_dist_ / dist;
  }

  int nScale = std::ceil(std::log(ratio) / frame->GetLogScaleFactor());
  if (nScale < 0) {
    nScale = 0;
  } else if (nScale >= frame->GetScaleLevel()) {
    nScale = frame->GetScaleLevel() - 1;
  }
  return nScale;
}
