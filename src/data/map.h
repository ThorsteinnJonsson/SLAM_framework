#ifndef SRC_MAP_H_
#define SRC_MAP_H_

#include "data/map_point.h"
#include "data/keyframe.h"

#include <set>
#include <mutex>

class Map{
public:
  Map();
  ~Map() {}

  void AddKeyFrame(KeyFrame* pKF);
  void AddMapPoint(MapPoint* pMP);
  void EraseMapPoint(MapPoint* pMP);
  void EraseKeyFrame(KeyFrame* pKF);
  void SetReferenceMapPoints(const std::vector<MapPoint*>& vpMPs);
  void InformNewBigChange();
  int GetLastBigChangeIdx() const;

  std::vector<KeyFrame*> GetAllKeyFrames() const;
  std::vector<MapPoint*> GetAllMapPoints() const;
  std::vector<MapPoint*> GetReferenceMapPoints() const;

  long unsigned  NumKeyFramesInMap() const;
  long unsigned int NumMapPointsInMap() const;
  long unsigned int GetMaxKeyframeId() const;

  const std::vector<KeyFrame*>& GetKeyframeOrigins() { return keyframe_origins_; }
  void AddOrigin(KeyFrame* kf) {
    keyframe_origins_.push_back(kf);
  }

  void Clear();

  mutable std::mutex map_update_mutex;
  mutable std::mutex point_creation_mutex;

protected:
  std::set<KeyFrame*> keyframes_;
  std::set<MapPoint*> map_points_;

  std::vector<MapPoint*> reference_map_points_;

  std::vector<KeyFrame*> keyframe_origins_;
  
  long unsigned int max_kf_id_;
  
  // Index related to a big change in the map (loop closure, global BA)
  int big_change_idx_;

  mutable std::mutex map_mutex_;

};

#endif // SRC_MAP_H_