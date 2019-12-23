#ifndef SRC_LOOP_CLOSER_H_
#define SRC_LOOP_CLOSER_H_

#include "map.h"
#include "data/keyframe.h"
#include "tracker.h"
#include "local_mapper.h"
#include "orb_vocabulary.h"
#include "keyframe_database.h"

#include <thread>
#include <mutex>

#include "g2o/types/types_seven_dof_expmap.h"

// Forward declarations
class Tracker;
class LocalMapper;
class KeyframeDatabase;

class LoopCloser {
public:
  typedef std::pair<std::set<KeyFrame*>,int> ConsistentGroup;    
  typedef std::map<KeyFrame*, 
                   g2o::Sim3,
                   std::less<KeyFrame*>,
                   Eigen::aligned_allocator<std::pair<const KeyFrame*, g2o::Sim3>>> KeyFrameAndPose;

public:
  explicit LoopCloser(const std::shared_ptr<Map>& map, 
                      const std::shared_ptr<KeyframeDatabase>& keyframe_db, 
                      const std::shared_ptr<OrbVocabulary>& orb_vocabulary, 
                      const bool fix_scale);
  ~LoopCloser() {}

  void SetLocalMapper(const std::shared_ptr<LocalMapper>& local_mapper);

  void Run();

  void InsertKeyFrame(KeyFrame *pKF);

  void RequestReset();

  // This function will run in a separate thread
  void RunGlobalBundleAdjustment(unsigned long nLoopKF);

  bool isRunningGBA();
  bool isFinishedGBA();

  void RequestFinish();
  bool isFinished();

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

protected:
  bool CheckNewKeyFrames();

  bool DetectLoop();

  bool ComputeSim3();

  void SearchAndFuse(const KeyFrameAndPose& corrected_poses_map);

  void CorrectLoop();

  void ResetIfRequested();
  bool mbResetRequested;
  std::mutex mMutexReset;

  bool CheckFinish();
  void SetFinish();
  bool mbFinishRequested;
  bool mbFinished;
  std::mutex mMutexFinish;

  std::shared_ptr<Map> map_;
  std::shared_ptr<LocalMapper> local_mapper_;

  const std::shared_ptr<KeyframeDatabase> keyframe_db_;
  const std::shared_ptr<OrbVocabulary> orb_vocabulary_;

  std::list<KeyFrame*> mlpLoopKeyFrameQueue;

  std::mutex mMutexLoopQueue;

  // Loop detector parameters
  const float covisibility_consistency_threshold;

  // Loop detector variables
  KeyFrame* mpCurrentKF;
  KeyFrame* mpMatchedKF;
  std::vector<ConsistentGroup> mvConsistentGroups;
  std::vector<KeyFrame*> mvpEnoughConsistentCandidates;
  std::vector<KeyFrame*> mvpCurrentConnectedKFs;
  std::vector<MapPoint*> mvpCurrentMatchedPoints;
  std::vector<MapPoint*> mvpLoopMapPoints;
  cv::Mat mScw;
  g2o::Sim3 mg2oScw;

  long unsigned int mLastLoopKFid;

  // Variables related to Global Bundle Adjustment
  bool mbRunningGBA;
  bool mbFinishedGBA;
  bool mbStopGBA;
  std::mutex mMutexGBA;
  std::unique_ptr<std::thread> global_bundle_adjustment_thread;

  // Fix scale in the stereo/RGB-D case
  bool mbFixScale;

  bool mnFullBAIdx;

};

#endif // SRC_LOOP_CLOSER_H_