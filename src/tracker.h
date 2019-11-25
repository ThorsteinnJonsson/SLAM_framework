#ifndef SRC_TRACKER_H_
#define SRC_TRACKER_H_

#include <opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

// #include"Viewer.h"
// #include"FrameDrawer.h"
#include"map.h"
#include"local_mapper.h"
#include"loop_closer.h"
#include"frame.h"
#include "orb_vocabulary.h"
#include "keyframe_database.h"
#include "orb_extractor.h"
// #include "Initializer.h"
// #include "MapDrawer.h"
#include "stereo_slam_system.h"

#include <mutex>

// Forward declerations
class Map;
class LocalMapper;
class LoopCloser;
class StereoSlamSystem;


class Tracker{
public:
  Tracker(StereoSlamSystem* pSys, 
          OrbVocabulary* pVoc, 
          Map* pMap,
          KeyframeDatabase* pKFDB, 
          const std::string& strSettingPath, 
          const int sensor);
  ~Tracker() {}

  // Preprocess the input and call Track(). Extract features and performs stereo matching.
  cv::Mat GrabImageStereo(const cv::Mat& imRectLeft,
                          const cv::Mat& imRectRight, 
                          const double timestamp);
  // cv::Mat GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp);
  

  void SetLocalMapper(LocalMapper* local_mapper) { mpLocalMapper = local_mapper; }
  void SetLoopCloser(LoopCloser* loop_closer) { mpLoopClosing = loop_closer; }

  // Load new settings
  // The focal lenght should be similar or scale prediction will fail when projecting points
  // TODO: Modify MapPoint::PredictScale to take into account focal lenght
  // void ChangeCalibration(const string &strSettingPath); // TODO doesn't seem to be used

  void Reset();

public:
  // Tracking states
  enum TrackingState {
    SYSTEM_NOT_READY=-1,
    NO_IMAGES_YET=0,
    NOT_INITIALIZED=1,
    OK=2,
    LOST=3
  };

  TrackingState mState;
  TrackingState mLastProcessedState;

  // Input sensor
  int mSensor;

  // Current Frame
  Frame mCurrentFrame;
  cv::Mat mImGray;

  // Initialization Variables (Monocular) # TODO maybe don't need since only monocular???
  // std::vector<int> mvIniLastMatches;
  // std::vector<int> mvIniMatches;
  // std::vector<cv::Point2f> mvbPrevMatched;
  // std::vector<cv::Point3f> mvIniP3D;
  // Frame mInitialFrame;

  // Lists used to recover the full camera trajectory at the end of the execution.
  // Basically we store the reference keyframe for each frame and its relative transformation
  std::list<cv::Mat> mlRelativeFramePoses;
  std::list<KeyFrame*> mlpReferences;
  std::list<double> mlFrameTimes;
  std::list<bool> mlbLost;

  // True if local mapping is deactivated and we are performing only localization
  bool mbOnlyTracking;

protected:
  // Map initialization for stereo and RGB-D
  void StereoInitialization();

  // Main tracking function. It is independent of the input sensor.
  void Track();

  void CheckReplacedInLastFrame();
  bool TrackReferenceKeyFrame();
  void UpdateLastFrame();
  bool TrackWithMotionModel();

  bool Relocalization();

  void UpdateLocalMap();
  void UpdateLocalPoints();
  void UpdateLocalKeyFrames();

  bool TrackLocalMap();
  void SearchLocalPoints();

  bool NeedNewKeyFrame();
  void CreateNewKeyFrame();

protected:

  // In case of performing only localization, this flag is true when there are no matches to
  // points in the map. Still tracking will continue if there are enough matches with 
  // temporal points. In that case we are doing visual odometry. 
  // The system will try to do relocalization to recover "zero-drift" localization to the map.
  bool mbVO;

  // Other Thread Pointers (TODO probably not the best way to do this)
  LocalMapper* mpLocalMapper;
  LoopCloser* mpLoopClosing;

  ORBextractor* mpORBextractorLeft;
  ORBextractor* mpORBextractorRight;
  ORBextractor* mpIniORBextractor;

  OrbVocabulary* mpORBVocabulary;
  KeyframeDatabase* mpKeyFrameDB;

  //Local Map
  KeyFrame* mpReferenceKF;
  std::vector<KeyFrame*> mvpLocalKeyFrames;
  std::vector<MapPoint*> mvpLocalMapPoints;

  StereoSlamSystem* mpSystem;

  Map* mpMap;

  // Calibration matrix
  cv::Mat mK;
  cv::Mat mDistCoef;
  float mbf;

  // New KeyFrame rules (according to fps)
  int mMinFrames;
  int mMaxFrames;

  // Threshold close/far points
  // Points seen as close by the stereo/RGBD sensor are considered reliable
  // and inserted from just one frame. Far points requiere a match in two keyframes.
  float mThDepth;

  // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
  float mDepthMapFactor;

  // Current matches in frame
  int mnMatchesInliers;

  // Last Frame, KeyFrame and Relocalisation Info
  KeyFrame* mpLastKeyFrame;
  Frame mLastFrame;
  unsigned int mnLastKeyFrameId;
  unsigned int mnLastRelocFrameId;

  //Motion Model
  cv::Mat mVelocity;

  //Color order (true RGB, false BGR, ignored if grayscale)
  bool mbRGB;

  std::list<MapPoint*> mlpTemporalPoints; // TODO do we ever use this
};


#endif //SRC_TRACKER_H_
