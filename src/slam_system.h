#ifndef SRC_STEREO_SLAM_H_
#define SRC_STEREO_SLAM_H_


#include <thread>
#include <mutex>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include "map.h"
#include "tracker.h"
#include "local_mapper.h"
#include "loop_closer.h"


// Forward declarations
class Map;
class Tracker;
class LocalMapper;
class LoopCloser;

enum SENSOR_TYPE {
  STEREO = 0,
  RGBD = 1,
  MONOCULAR = 2 // TODO not implemented
};

class SlamSystem {
public:
  SlamSystem(const std::string& strVocFile, 
             const std::string& strSettingsFile, 
             const SENSOR_TYPE sensor);
  ~SlamSystem();

  cv::Mat TrackStereo(const cv::Mat& imLeft, const cv::Mat& imRight, const double timestamp);
  // cv::Mat TrackRGBD(const cv::Mat& im, const cv::Mat& depthmap, const double timestamp); // TODO implement later
  // cv::Mat TrackMonocular(const cv::Mat &im, const double &timestamp); // TODO implement later

  // Reset the system (clear map)
  void Reset();

  // This stops local mapping thread (map building) and performs only camera tracking.
  void ActivateLocalizationMode();
  // This resumes local mapping thread and performs SLAM again.
  void DeactivateLocalizationMode();

  // Returns true if there have been a big map change (loop closure, global BA)
  // since last call to this function
  bool MapChanged();

  // All threads will be requested to finish.
  // It waits until all threads have finished.
  // This function must be called before saving the trajectory.
  void Shutdown();

  // Save camera trajectory in the KITTI dataset format.
  // Only for stereo and RGB-D. This method does not work for monocular.
  // Call first Shutdown()
  // See format details at: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
  void SaveTrajectoryKITTI(const string &filename) {} //TODO implement

  // TODO: Save/Load functions (not implemented in the original ORB-SLAM)
  // SaveMap(const string &filename);
  // LoadMap(const string &filename);

  // Information from most recent processed frame
  // You can call this right after TrackMonocular (or stereo or RGBD)
  int GetTrackingState();
  std::vector<MapPoint*> GetTrackedMapPoints();
  std::vector<cv::KeyPoint> GetTrackedKeyPointsUn();


private:
  // Input sensor
  SENSOR_TYPE mSensor;

  // ORB vocabulary used for place recognition and feature matching.
  OrbVocabulary* mpVocabulary;

  // KeyFrame database for place recognition (relocalization and loop detection).
  KeyframeDatabase* mpKeyFrameDatabase;

  // Map structure that stores the pointers to all KeyFrames and MapPoints.
  Map* mpMap;

  // Tracker. It receives a frame and computes the associated camera pose.
  // It also decides when to insert a new keyframe, create some new MapPoints and
  // performs relocalization if tracking fails.
  Tracker* mpTracker;

  // Local Mapper. It manages the local map and performs local bundle adjustment.
  LocalMapper* mpLocalMapper;

  // Loop Closer. It searches loops with every new keyframe. If there is a loop it performs
  // a pose graph optimization and full bundle adjustment (in a new thread) afterwards.
  LoopCloser* mpLoopCloser;

  // System threads: Local Mapping, Loop Closing.
  // The Tracking thread "lives" in the main execution thread that creates the System object.
  std::thread* mptLocalMapping;
  std::thread* mptLoopClosing;

  // Reset flag
  std::mutex mMutexReset;
  bool mbReset;

  // Change mode flags
  std::mutex mMutexMode;
  bool mbActivateLocalizationMode;
  bool mbDeactivateLocalizationMode;

  // Tracking state
  int mTrackingState;
  std::vector<MapPoint*> mTrackedMapPoints;
  std::vector<cv::KeyPoint> mTrackedKeyPointsUn;
  std::mutex mMutexState;  
 
};

# endif //SRC_STEREO_SLAM_H_