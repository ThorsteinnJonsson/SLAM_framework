#ifndef SRC_MAPPOINT_H_
#define SRC_MAPPOINT_H_

#include "data/keyframe.h"
#include "data/frame.h"
#include "data/map.h"

#include<opencv2/core/core.hpp>
#include<mutex>

class MapPoint{
public:
  MapPoint(const cv::Mat& position, 
           KeyFrame* reference_keyframe, 
           const std::shared_ptr<Map>& map);
  MapPoint(const cv::Mat& position, 
           const std::shared_ptr<Map>& map, 
           Frame* frame, 
           const int& idxF);
  ~MapPoint() {}

  void SetWorldPos(const cv::Mat& position);
  cv::Mat GetWorldPos() const;

  cv::Mat GetNormal() const;
  KeyFrame* GetReferenceKeyFrame() const;

  std::map<KeyFrame*,size_t> GetObservations() const;
  int NumObservations() const;

  void AddObservation(KeyFrame* keyframe,size_t idx);
  void EraseObservation(KeyFrame* keyframe);

  int GetIndexInKeyFrame(KeyFrame* keyframe) const;
  bool IsInKeyFrame(KeyFrame* keyframe) const;

  void SetBadFlag();
  bool isBad() const;

  void Replace(MapPoint* point);    
  MapPoint* GetReplaced() const;

  void IncreaseVisible(int n=1);
  void IncreaseFound(int n=1);
  float GetFoundRatio() const;

  void ComputeDistinctiveDescriptors();
  
  cv::Mat GetDescriptor() const;

  void UpdateNormalAndDepth();

  float GetMinDistanceInvariance() const;
  float GetMaxDistanceInvariance() const;

  int PredictScale(const float dist, KeyFrame* keyframe) const;
  int PredictScale(const float dist, Frame* frame) const;

  long unsigned int GetId() const { return id_; }
  long int GetFirstKeyframeID() const { return first_keyframe_id_; }

public:
  static std::mutex global_mutex;
  static long unsigned int next_id;

  // Variables used by the tracking
  float track_projected_x;
  float track_projected_y;
  float track_projected_x_right;
  bool track_is_in_view;
  int track_scale_level;
  float track_view_cos;
  long unsigned int track_reference_id_for_frame;
  long unsigned int last_frame_id_seen;

  // Variables used by local mapping
  long unsigned int bundle_adj_local_id_for_keyframe;
  long unsigned int fuse_candidate_id_for_keyframe;

  // Variables used by loop closing
  long unsigned int loop_point_for_keyframe_id;
  long unsigned int corrected_by_keyframe;
  long unsigned int corrected_reference;    
  cv::Mat position_global_bundle_adj;
  long unsigned int bundle_adj_global_for_keyframe_id;

protected:
  long unsigned int id_;
  long int first_keyframe_id_;
  long int first_frame_id_;
  int num_observations_;

  cv::Mat world_position_;

  // Keyframes observing the point and associated index in keyframe
  std::map<KeyFrame*,size_t> observations_;

  // Mean viewing direction
  cv::Mat normal_vector_;

  // Best descriptor to fast matching
  cv::Mat descriptor_;

  KeyFrame* reference_keyframe_;

  // Tracking counters
  int num_visible_;
  int num_found_;

  bool is_bad_;
  MapPoint* replaced_point_;

  // Scale invariance distances
  float min_dist_;
  float max_dist_;

  std::shared_ptr<Map> map_;

  mutable std::mutex position_mutex_;
  mutable std::mutex feature_mutex_;

};

#endif // SRC_MAPPOINT_H_