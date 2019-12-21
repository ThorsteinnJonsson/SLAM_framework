#ifndef SRC_LOCAL_MAPPER_H_
#define SRC_LOCAL_MAPPER_H_

#include "keyframe.h"
#include "map.h"
#include "loop_closer.h"
#include "tracker.h"
#include "keyframe_database.h"

#include <mutex>

// Forward declarations
class LoopCloser;
class Tracker;
class Map;

class LocalMapper {
public:
  explicit LocalMapper(const std::shared_ptr<Map>& pMap, 
                       const bool bMonocular);
  ~LocalMapper() {}

  void SetLoopCloser(const std::shared_ptr<LoopCloser>& pLoopCloser);

  void Run();

  void InsertKeyFrame(KeyFrame* pKF);

  void RequestStop();
  void RequestReset();
  bool Stop();
  void Release();
  bool isStopped();
  bool stopRequested();
  
  bool AcceptKeyFrames();
  void SetAcceptKeyFrames(bool flag);
  
  void InterruptBA();
  
  bool SetNotStop(bool flag);
  
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

  bool mbMonocular;

  void ResetIfRequested();
  bool mbResetRequested;
  std::mutex mMutexReset;

  bool CheckFinish();
  void SetFinish();
  bool mbFinishRequested;
  bool mbFinished;
  std::mutex mMutexFinish;

  std::shared_ptr<Map> mpMap;

  std::shared_ptr<LoopCloser> mpLoopCloser;
  std::shared_ptr<Tracker> mpTracker;

  std::list<KeyFrame*> mlNewKeyFrames;

  KeyFrame* mpCurrentKeyFrame;

  std::list<MapPoint*> mlpRecentAddedMapPoints;

  std::mutex mMutexNewKFs;

  bool mbAbortBA;

  bool mbStopped;
  bool mbStopRequested;
  bool mbNotStop;
  std::mutex mMutexStop;

  bool mbAcceptKeyFrames;
  std::mutex mMutexAccept;
};

#endif // SRC_LOCAL_MAPPER_H_