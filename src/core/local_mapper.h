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
  bool isStopped();
  bool stopRequested();
  
  bool IsAcceptingKeyFrames();
  void SetAcceptKeyFrames(const bool flag);
  
  void InterruptBA();
  
  bool SetNotStop(const bool flag);
  
  void RequestFinish();
  bool isFinished();

  int NumKeyframesInQueue();
  
protected:
  bool CheckNewKeyFrames();
  void ProcessNewKeyFrame();
  
  void MapPointCulling();
  void CreateNewMapPoints();
  void SearchInNeighbors();

  void KeyFrameCulling();

  cv::Mat ComputeF12(KeyFrame* pKF1, KeyFrame* pKF2) const;

  cv::Mat SkewSymmetricMatrix(const cv::Mat &v) const;

  bool is_monocular_;

  void ResetIfRequested();
  bool mbResetRequested;
  std::mutex mMutexReset;

  bool CheckFinish();
  void SetFinish();
  bool mbFinishRequested;
  bool mbFinished;
  std::mutex mMutexFinish;

  std::shared_ptr<Map> map_;

  std::shared_ptr<LoopCloser> loop_closer_;

  std::list<KeyFrame*> mlNewKeyFrames;

  KeyFrame* mpCurrentKeyFrame;

  std::list<MapPoint*> mlpRecentAddedMapPoints;

  std::mutex mMutexNewKFs;

  bool mbAbortBA;

  bool mbStopped;
  bool mbStopRequested;
  bool mbNotStop;
  std::mutex mMutexStop;

  bool is_accepting_keyframes_;
  std::mutex accept_keyframe_mutex_;
};

#endif // SRC_LOCAL_MAPPER_H_