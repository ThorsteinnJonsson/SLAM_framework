#ifndef SRC_LOCAL_MAPPER_H_
#define SRC_LOCAL_MAPPER_H_

#include "core/loop_closer.h"
#include "data/map.h"
#include "data/keyframe.h"
#include "data/keyframe_database.h"

#include <mutex>

// Forward declarations
class LoopCloser;
class Map;

class LocalMapper {
public:
  explicit LocalMapper(const std::shared_ptr<Map>& map, 
                       const bool is_monocular);
  ~LocalMapper() {}

  void SetLoopCloser(const std::shared_ptr<LoopCloser>& loop_closer);

  void Run();

  void InsertKeyFrame(KeyFrame* keyframe);

  void RequestStop();
  void RequestReset();
  bool Stop();
  void Release();
  bool IsStopped() const;
  bool stopRequested() const;
  
  bool IsAcceptingKeyFrames() const;
  void SetAcceptKeyFrames(const bool flag);
  
  void InterruptBA();
  
  bool SetNotStop(const bool flag);
  
  void RequestFinish();
  bool IsFinished() const;

  int NumKeyframesInQueue() const;
  
protected:
  bool CheckNewKeyFrames() const;
  void ProcessNewKeyFrame();
  
  void MapPointCulling();
  void CreateNewMapPoints();
  void SearchInNeighbors();

  void KeyFrameCulling();

  cv::Mat ComputeFundamentalMatrix(KeyFrame* keyframe1, KeyFrame* keyframe2) const;

  cv::Mat SkewSymmetricMatrix(const cv::Mat &v) const;

  bool is_monocular_;

  void ResetIfRequested();
  bool reset_requested_;
  std::mutex reset_mutex_;

  bool CheckFinish() const;
  void SetFinish();
  bool finish_requested_;
  bool is_finished_;

  std::shared_ptr<Map> map_;
  std::shared_ptr<LoopCloser> loop_closer_;

  std::list<KeyFrame*> new_keyframes_;

  KeyFrame* current_keyframe_;

  std::list<MapPoint*> recently_added_map_points_;

  bool abort_bundle_adjustment_;

  bool is_stopped_;
  bool requested_stop_;
  bool not_stop_;
  
  bool is_accepting_keyframes_;
  
  mutable std::mutex finished_mutex_;
  mutable std::mutex new_keyframes_mutex_;
  mutable std::mutex stop_mutex_;
  mutable std::mutex accept_keyframe_mutex_;
  
};

#endif // SRC_LOCAL_MAPPER_H_