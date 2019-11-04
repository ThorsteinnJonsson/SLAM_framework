#ifndef SRC_STEREO_SLAM_H_
#define SRC_STEREO_SLAM_H_


#include <thread>
#include <mutex>

#include <opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>

#include "tracker.h"

class StereoSlam {
public:
  StereoSlam();
  ~StereoSlam();

  void TrackStereo(const cv::Mat& l_image, const cv::Mat& r_image, const double& timestamp);


private:
  // KeyFrame database for place recognition (relocalization and loop detection).
    // KeyFrameDatabase* mpKeyFrameDatabase;

    // Map structure that stores the pointers to all KeyFrames and MapPoints.
    // Map* mpMap;

    // Tracker. It receives a frame and computes the associated camera pose.
    // It also decides when to insert a new keyframe, create some new MapPoints and
    // performs relocalization if tracking fails.
    Tracker* tracker; //mpTracker

    // Local Mapper. It manages the local map and performs local bundle adjustment.
    // LocalMapping* mpLocalMapper;

    // Loop Closer. It searches loops with every new keyframe. If there is a loop it performs
    // a pose graph optimization and full bundle adjustment (in a new thread) afterwards.
    // LoopClosing* mpLoopCloser;

    std::thread* mptLocalMapping;
    std::thread* mptLoopClosing;

    // Reset flag
    // std::mutex mMutexReset;
    // bool mbReset;

    // Change mode flags
    // std::mutex mMutexMode;
    // bool mbActivateLocalizationMode;
    // bool mbDeactivateLocalizationMode;

    // Tracking state
    // int mTrackingState;
    // std::vector<MapPoint*> mTrackedMapPoints;
    // std::vector<cv::KeyPoint> mTrackedKeyPointsUn;
    // std::mutex mMutexState;

};

# endif //SRC_STEREO_SLAM_H_