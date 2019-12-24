#ifndef SRC_TRACKER_H_
#define SRC_TRACKER_H_

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "util/sensor_type.h"
#include "util/tracking_state.h"
#include "map.h"
#include "local_mapper.h"
#include "loop_closer.h"
#include "data/frame.h"
#include "orb_vocabulary.h"
#include "keyframe_database.h"
#include "orb_extractor.h"
#include "initializer.h"

#include <mutex>
#include <memory>

// Forward declerations
class Map;
class LocalMapper;
class LoopCloser;
class SlamSystem;

class Tracker {
public:
  Tracker(const std::shared_ptr<OrbVocabulary>& orb_vocabulary, 
          const std::shared_ptr<Map>& map,
          const std::shared_ptr<KeyframeDatabase>& keyframe_db, 
          const std::string& settings_path, 
          const SENSOR_TYPE sensor);
  ~Tracker() {}

  // Preprocess the input and call Track(). Extract features and performs stereo matching.
  cv::Mat GrabImageStereo(const cv::Mat& left_image,
                          const cv::Mat& right_image, 
                          const double timestamp);
  cv::Mat GrabImageRGBD(const cv::Mat& rgbd_image,
                        const cv::Mat& depth_image, 
                        const double timestamp);
  cv::Mat GrabImageMonocular(const cv::Mat& image, 
                             const double timestamp);

  void SetLocalMapper(const std::shared_ptr<LocalMapper>& local_mapper) { 
    local_mapper_ = local_mapper; 
  }

  void SetLoopCloser(const std::shared_ptr<LoopCloser>& loop_closer) { 
    loop_closer_ = loop_closer;
  }

  // Use this function if you have deactivated local mapping and you only want to localize the camera.
  void InformOnlyTracking(const bool flag) { mbOnlyTracking = flag; }

  bool NeedSystemReset() const;

  void Reset();

  TrackingState GetState() const { return state_; }

  const Frame& GetCurrentFrame() const { return current_frame_; }
  
  const std::list<cv::Mat>& GetRelativeFramePoses() { return mlRelativeFramePoses; }
  const std::list<KeyFrame*>& GetReferenceKeyframes() { return mlpReferences; }
  const std::list<double>& GetFrameTimes() { return mlFrameTimes; }

protected:
  // Map initialization for stereo and RGB-D
  void StereoInitialization();

  // Map initialization for monocular
  void MonocularInitialization();
  void CreateInitialMapMonocular();

  // Main tracking function. It is independent of the input sensor.
  void Track();

  void ReplaceInLastFrame();
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
  bool use_visual_odometry_;

  // Other Thread Pointers
  std::shared_ptr<LocalMapper> local_mapper_;
  std::shared_ptr<LoopCloser> loop_closer_;

  std::shared_ptr<ORBextractor> orb_extractor_left_;
  std::shared_ptr<ORBextractor> orb_extractor_right_;
  std::shared_ptr<ORBextractor> mpIniORBextractor;

  const std::shared_ptr<OrbVocabulary> orb_vocabulary_;
  const std::shared_ptr<KeyframeDatabase> keyframe_db_;

  // Initalization (only for monocular)
  std::unique_ptr<Initializer> mpInitializer = nullptr; // TODO unique_ptr?

  //Local Map
  KeyFrame* mpReferenceKF;
  std::vector<KeyFrame*> mvpLocalKeyFrames;
  std::vector<MapPoint*> mvpLocalMapPoints;

  std::shared_ptr<Map> mpMap;

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

private:
  TrackingState state_;
  TrackingState last_processed_state_;

  SENSOR_TYPE sensor_type_;

  Frame current_frame_;

  // Lists used to recover the full camera trajectory at the end of the execution.
  // Basically we store the reference keyframe for each frame and its relative transformation
  std::list<cv::Mat> mlRelativeFramePoses;
  std::list<KeyFrame*> mlpReferences;
  std::list<double> mlFrameTimes;
  std::list<bool> mlbLost;

  // True if local mapping is deactivated and we are performing only localization
  bool mbOnlyTracking;

  // Initialization Variables (Monocular)
  std::vector<int> mvIniMatches; // TODO Should probably be private
  std::vector<cv::Point2f> mvbPrevMatched;
  std::vector<cv::Point3f> mvIniP3D;
  Frame mInitialFrame;

  const int num_required_matches_ = 15;
  bool system_reset_needed_ = false;
};


#endif //SRC_TRACKER_H_
