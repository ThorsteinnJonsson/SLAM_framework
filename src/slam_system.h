#ifndef SRC_STEREO_SLAM_H_
#define SRC_STEREO_SLAM_H_


#include <thread>
#include <mutex>
#include <memory>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include "util/sensor_type.h"
#include "util/tracking_state.h"
#include "data/map.h"
#include "tracker.h"
#include "local_mapper.h"
#include "loop_closer.h"


// Forward declarations
class Map;
class Tracker;
class LocalMapper;
class LoopCloser;

class SlamSystem {
public:
  explicit SlamSystem(const std::string& vocabulary_path, 
                      const std::string& settings_path, 
                      const SENSOR_TYPE sensor);
  ~SlamSystem();

  cv::Mat TrackStereo(const cv::Mat& imLeft, 
                      const cv::Mat& imRight,
                      const double timestamp);
  cv::Mat TrackRGBD(const cv::Mat& im, 
                    const cv::Mat& depthmap, 
                    const double timestamp);
  cv::Mat TrackMonocular(const cv::Mat& im, 
                         const double timestamp);

  // This stops local mapping thread (map building) and performs only camera tracking.
  void ActivateLocalizationMode();
  // This resumes local mapping thread and performs SLAM again.
  void DeactivateLocalizationMode();

  // All threads will be requested to finish.
  // It waits until all threads have finished.
  // This function must be called before saving the trajectory.
  void Shutdown();

  // Save camera trajectory in the KITTI dataset format.
  // Only for stereo and RGB-D. This method does not work for monocular.
  // Call Shutdown() before calling this function.
  // See format details at: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
  void SaveTrajectoryKITTI(const std::string& filename) const;

  // TODO: Save/Load functions (not implemented in the original ORB-SLAM)
  // SaveMap(const std::string &filename);
  // LoadMap(const std::string &filename);

  // Information from most recent processed frame
  // You can call this right after TrackMonocular (or stereo or RGBD)
  TrackingState GetTrackingState() const;
  std::vector<MapPoint*> GetTrackedMapPoints() const;
  std::vector<cv::KeyPoint> GetTrackedKeyPointsUn() const;

private:
  // Input sensor
  SENSOR_TYPE sensor_type_;

  // ORB vocabulary used for place recognition and feature matching.
  std::shared_ptr<OrbVocabulary> orb_vocabulary_;

  // KeyFrame database for place recognition (relocalization and loop detection).
  std::shared_ptr<KeyframeDatabase> keyframe_database_;

  // Map structure that stores the pointers to all KeyFrames and MapPoints.
  std::shared_ptr<Map> map_;

  // Tracker. It receives a frame and computes the associated camera pose.
  // It also decides when to insert a new keyframe, create some new MapPoints and
  // performs relocalization if tracking fails.
  std::shared_ptr<Tracker> tracker_;

  // Local Mapper. It manages the local map and performs local bundle adjustment.
  std::shared_ptr<LocalMapper> local_mapper_;

  // Loop Closer. It searches loops with every new keyframe. If there is a loop it performs
  // a pose graph optimization and full bundle adjustment (in a new thread) afterwards.
  std::shared_ptr<LoopCloser> loop_closer_;

  // System threads: Local Mapping, Loop Closing.
  // The Tracking thread "lives" in the main execution thread that creates the System object.
  std::unique_ptr<std::thread> local_mapping_thread_;
  std::unique_ptr<std::thread> loop_closing_thread_;

  // Change mode flags
  mutable std::mutex mode_mutex_;
  bool activate_localization_mode_;
  bool deactivate_localization_mode_;

  // Tracking state
  TrackingState tracking_state_;
  std::vector<MapPoint*> tracked_map_points_;
  std::vector<cv::KeyPoint> tracked_keypoints_un_;
  mutable std::mutex state_mutex_;  
 
};

# endif //SRC_STEREO_SLAM_H_