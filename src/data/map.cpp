#include "map.h"

Map::Map() : max_kf_id_(0)
           , big_change_idx_(0) {

}

void Map::AddKeyFrame(KeyFrame *pKF) {
  std::unique_lock<std::mutex> lock(map_mutex_);
  keyframes_.insert(pKF);
  if (pKF->mnId > max_kf_id_) {
    max_kf_id_ = pKF->mnId;
  } 
}

void Map::AddMapPoint(MapPoint* pMP) {
  std::unique_lock<std::mutex> lock(map_mutex_);
  map_points_.insert(pMP);
}

void Map::EraseMapPoint(MapPoint* pMP) {
  std::unique_lock<std::mutex> lock(map_mutex_);
  map_points_.erase(pMP);
  // TODO: This only erase the pointer.
  // Delete the MapPoint
}

void Map::EraseKeyFrame(KeyFrame* pKF) {
    std::unique_lock<std::mutex> lock(map_mutex_);
    keyframes_.erase(pKF);
    // TODO: This only erase the pointer.
    // Delete the KeyFrame
}

void Map::SetReferenceMapPoints(const std::vector<MapPoint*>& vpMPs) {
  std::unique_lock<std::mutex> lock(map_mutex_);
  reference_map_points_ = vpMPs;
}

void Map::InformNewBigChange() { 
  std::unique_lock<std::mutex> lock(map_mutex_);
  ++big_change_idx_;
}

int Map::GetLastBigChangeIdx() const {
  std::unique_lock<std::mutex> lock(map_mutex_);
  return big_change_idx_;
}

std::vector<KeyFrame*> Map::GetAllKeyFrames() const {
  std::unique_lock<std::mutex> lock(map_mutex_);
  return std::vector<KeyFrame*>(keyframes_.begin(),keyframes_.end());
}

std::vector<MapPoint*> Map::GetAllMapPoints() const {
  std::unique_lock<std::mutex> lock(map_mutex_);
  return std::vector<MapPoint*>(map_points_.begin(),map_points_.end());
}

std::vector<MapPoint*> Map::GetReferenceMapPoints() const {
  std::unique_lock<std::mutex> lock(map_mutex_);
  return reference_map_points_;
}

long unsigned  Map::NumKeyFramesInMap() const {
  std::unique_lock<std::mutex> lock(map_mutex_);
  return keyframes_.size();
}

long unsigned int Map::NumMapPointsInMap() const {
  std::unique_lock<std::mutex> lock(map_mutex_);
  return map_points_.size();
}

long unsigned int Map::GetMaxKeyframeId() const {
  std::unique_lock<std::mutex> lock(map_mutex_);
  return max_kf_id_;
}

void Map::Clear() {
  for (MapPoint* mp : map_points_) {
    delete mp;
  }
      
  for (KeyFrame* kf : keyframes_) {
    delete kf;
  }
      
  map_points_.clear();
  keyframes_.clear();
  max_kf_id_ = 0;
  reference_map_points_.clear();
  keyframe_origins_.clear();
}




