#ifndef SRC_TRACKER_H_
#define SRC_TRACKER_H_

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "core/local_mapper.h"
#include "core/loop_closer.h"
#include "data/map.h"
#include "data/frame.h"
#include "data/keyframe_database.h"
#include "orb_features/orb_vocabulary.h"
#include "orb_features/orb_extractor.h"
#include "util/initializer.h"
#include "util/sensor_type.h"
#include "util/tracking_state.h"
#include "json.hpp"

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
          const nlohmann::json& config, 
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
  void InformOnlyTracking(const bool flag) { is_only_tracking_ = flag; }

  bool NeedSystemReset() const;

  void Reset();

  TrackingState GetState() const { return state_; }

  const Frame& GetCurrentFrame() const { return current_frame_; }
  
  const std::list<cv::Mat>& GetRelativeFramePoses() { return relative_frame_poses_; }
  const std::list<KeyFrame*>& GetReferenceKeyframes() { return reference_keyframes_; }
  const std::list<double>& GetFrameTimes() { return frame_times_; }
  const std::list<bool>& GetLost() {return is_lost_; }

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
  std::shared_ptr<ORBextractor> orb_extractor_ini_;

  const std::shared_ptr<OrbVocabulary> orb_vocabulary_;
  const std::shared_ptr<KeyframeDatabase> keyframe_db_;

  // Initalization (only for monocular)
  std::unique_ptr<Initializer> mpInitializer = nullptr;

  //Local Map
  KeyFrame* reference_keyframe_;
  std::vector<KeyFrame*> local_keyframes_;
  std::vector<MapPoint*> local_map_points_;

  std::shared_ptr<Map> map_;

  // Calibration matrix
  cv::Mat calibration_mat_;
  cv::Mat dist_coeff_;
  float scaled_baseline_; // Stereo baseline times fx

  // New KeyFrame rules (according to fps)
  int min_frames_;
  int max_frames_;

  // Threshold close/far points
  // Points seen as close by the stereo/RGBD sensor are considered reliable
  // and inserted from just one frame. Far points requiere a match in two keyframes.
  float depth_threshold_;

  // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
  float depth_map_factor_;

  // Current matches in frame
  int num_inlier_matches_;

  // Last Frame, KeyFrame and Relocalisation Info
  KeyFrame* last_keyframe_;
  Frame last_frame_;
  unsigned int last_keyframe_id_;
  unsigned int last_relocation_frame_id_;

  //Motion Model
  cv::Mat velocity_;

  //Color order (true RGB, false BGR, ignored if grayscale)
  bool is_rgb_;

private:
  TrackingState state_;
  TrackingState last_processed_state_;

  SENSOR_TYPE sensor_type_;

  Frame current_frame_;

  // Lists used to recover the full camera trajectory at the end of the execution.
  // Basically we store the reference keyframe for each frame 
  // and its relative transformation //TODO put this into a struct
  std::list<cv::Mat> relative_frame_poses_;
  std::list<KeyFrame*> reference_keyframes_;
  std::list<double> frame_times_;
  std::list<bool> is_lost_;

  // True if local mapping is deactivated and we are performing only localization
  bool is_only_tracking_;

  // Initialization Variables (Monocular)
  std::vector<int> init_matches_;
  std::vector<cv::Point2f> prev_matched_;
  std::vector<cv::Point3f> init_points_;
  Frame initial_frame_;

  const int num_required_matches_ = 15;
  bool system_reset_needed_ = false;
};


#endif //SRC_TRACKER_H_
