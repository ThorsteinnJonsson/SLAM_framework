#ifndef SRC_LOOP_CLOSER_H_
#define SRC_LOOP_CLOSER_H_

#include "data/map.h"
#include "data/keyframe.h"
#include "data/keyframe_database.h"
#include "core/tracker.h"
#include "core/local_mapper.h"
#include "orb_features/orb_vocabulary.h"

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
  void RunGlobalBundleAdjustment(unsigned long loop_kf_index);

  bool IsRunningGBA() const;
  void RequestFinish();
  bool IsFinished()const ;

protected:
  bool CheckNewKeyFrames() const;

  bool DetectLoop();

  bool ComputeSim3();

  void SearchAndFuse(const KeyFrameAndPose& corrected_poses_map);

  void CorrectLoop();

  void ResetIfRequested();
  bool reset_requested_ = false;


  bool CheckFinish() const;
  void SetFinish();
  bool finish_requested_ = false;
  bool is_finished_ = false;


  std::shared_ptr<Map> map_;
  std::shared_ptr<LocalMapper> local_mapper_;

  const std::shared_ptr<KeyframeDatabase> keyframe_db_;
  const std::shared_ptr<OrbVocabulary> orb_vocabulary_;

  std::list<KeyFrame*> loop_keyframe_queue_;

  // Loop detector parameters
  const float covisibility_consistency_threshold_ = 3;

  // Loop detector variables
  KeyFrame* current_keyframe_ = nullptr;
  KeyFrame* matched_keyframe_ = nullptr;
  std::vector<ConsistentGroup> consistent_groups_;
  std::vector<KeyFrame*> consistent_enough_candidates_;
  // std::vector<KeyFrame*> cur_connected_keyframes_; doesn't need to be a member
  std::vector<MapPoint*> cur_matched_points_;
  std::vector<MapPoint*> loop_map_points_;
  cv::Mat Scw_;
  g2o::Sim3 g2o_Scw_;

  long unsigned int last_loop_kf_id_ = 0;

  // Variables related to Global Bundle Adjustment
  bool is_running_global_budle_adj_ = false;
  bool is_finished_global_budle_adj_ = true;
  bool stop_global_bundle_adj_ = false;
  std::unique_ptr<std::thread> global_bundle_adjustment_thread_ = nullptr;

  // Fix scale in the stereo/RGB-D case
  bool fix_scale_;

  bool full_bundle_adj_idx_ = 0;

  mutable std::mutex reset_mutex_;
  mutable std::mutex finish_mutex_;
  mutable std::mutex loop_queue_mutex_;
  mutable std::mutex global_bundle_adj_mutex_;

};

#endif // SRC_LOOP_CLOSER_H_