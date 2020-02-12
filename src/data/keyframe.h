#ifndef SRC_KEYFRAME_H_
#define SRC_KEYFRAME_H_


#include "data/map_point.h"
#include "data/frame.h"
#include "DBoW2/BowVector.h"
#include "DBoW2/FeatureVector.h"
#include "orb_features/orb_vocabulary.h"
#include "orb_features/orb_extractor.h"
#include "data/keyframe_database.h"

#include <mutex>
#include <set>

// Forward declarations
class Map;
class MapPoint;
class Frame;
class KeyframeDatabase;


class KeyFrame{
public:
  KeyFrame(const Frame& frame, 
           const std::shared_ptr<Map>& pMap, 
           const std::shared_ptr<KeyframeDatabase>& pKFDB);
  ~KeyFrame() {}

  void SetPose(const cv::Mat& Tcw_);
  cv::Mat GetPose();
  cv::Mat GetPoseInverse();
  cv::Mat GetCameraCenter();
  cv::Mat GetStereoCenter();
  cv::Mat GetRotation();
  cv::Mat GetTranslation();

  // Bag of Words Representation
  void ComputeBoW();

  // Covisibility graph functions
  void AddConnection(KeyFrame* pKF, const int weight);
  void EraseConnection(KeyFrame* pKF);
  void UpdateConnections();
  std::set<KeyFrame*> GetConnectedKeyFrames();
  std::vector<KeyFrame*> GetVectorCovisibleKeyFrames();
  std::vector<KeyFrame*> GetBestCovisibilityKeyFrames(const int N);
  std::vector<KeyFrame*> GetCovisiblesByWeight(const int w);
  int GetWeight(KeyFrame* pKF);

  // Spanning tree functions
  void AddChild(KeyFrame* pKF);
  void EraseChild(KeyFrame* pKF);
  void ChangeParent(KeyFrame* pKF);
  std::set<KeyFrame*> GetChilds();
  KeyFrame* GetParent();
  bool hasChild(KeyFrame* pKF);

  // Loop Edges
  void AddLoopEdge(KeyFrame* pKF);
  std::set<KeyFrame*> GetLoopEdges();

  // MapPoint observation functions
  void AddMapPoint(MapPoint* pMP, const size_t idx);
  void EraseMapPointMatch(const size_t idx);
  void EraseMapPointMatch(MapPoint* pMP);
  void ReplaceMapPointMatch(const size_t idx, MapPoint* pMP);
  std::set<MapPoint*> GetMapPoints();
  std::vector<MapPoint*> GetMapPointMatches();
  int TrackedMapPoints(const int minObs);
  MapPoint* GetMapPoint(const size_t idx);

  // KeyPoint functions
  std::vector<size_t> GetFeaturesInArea(const float x, const float y, const float r) const;
  cv::Mat UnprojectStereo(int i);

  bool IsInImage(const float x, const float y) const;

  void SetNotErase();
  void SetErase();
  
  void SetBadFlag();
  bool isBad() const;

  // Compute Scene Depth (q=2 median). Used in monocular.
  float ComputeSceneMedianDepth(const int q);

  static bool LesserId(KeyFrame* pKF1, KeyFrame* pKF2){ return pKF1->Id() < pKF2->Id(); }

private:
  void UpdateBestCovisibles();

// The following variables are accesed from only one thread or never change (no mutex needed)
public:
  long unsigned int Id() const { return id_; }
  void SetId(long unsigned int id) { id_ = id; }
  long unsigned int FrameId() const { return frame_id_; }

  static void ResetId() { next_id_ = 0;}

  // Grid (to speed up feature matching)
  static constexpr int grid_rows = 48;
  static constexpr int grid_cols = 64;

  // Variables used by the tracking
  long unsigned int track_reference_for_frame = 0;
  long unsigned int fuse_target_for_kf = 0;

  // Variables used by the local mapping
  long unsigned int bundle_adj_local_id_for_keyframe = 0;
  long unsigned int bundle_adj_fixed_for_keyframe = 0;

  // Variables used by the keyframe database
  long unsigned int mnLoopQuery = 0;
  int mnLoopWords = 0;
  float mLoopScore;
  long unsigned int mnRelocQuery = 0;
  int mnRelocWords = 0;
  float mRelocScore;

  // Variables used by loop closing
  cv::Mat Tcw_global_bundle_adj;
  cv::Mat Tcw_before_global_bundle_adj;
  long unsigned int bundle_adj_global_for_keyframe_id;

  // Calibration parameters
  const cv::Mat calib_mat;
  static float fx;
  static float fy;
  static float cx;
  static float cy;
  static float invfx;
  static float invfy;
  static bool initial_computations;

  // Stereo baseline multiplied by fx.
  const float mbf;

  // Stereo baseline in meters.
  const float mb;

  // Threshold close/far points. Close points are inserted from 1 view.
  // Far points are inserted as in the monocular case from 2 views.
  const float depth_threshold;

  // Number of KeyPoints
  const int num_keyframes;

  // KeyPoints, stereo coordinate and descriptors (all associated by an index)
  const std::vector<cv::KeyPoint> keypoints;
  const std::vector<cv::KeyPoint> undistorted_keypoints;
  
  const std::vector<float> right_coords; // negative value for monocular points
  const std::vector<float> depths; // negative value for monocular points
  const cv::Mat descriptors;

  //BoW
  DBoW2::BowVector bow_vec;
  DBoW2::FeatureVector feature_vec;

  // Pose relative to parent (this is computed when bad flag is activated)
  cv::Mat Tcp;

  // Scale
  const int scale_levels;
  const float scale_factor;
  const float log_scale_factor;
  const std::vector<float> scale_factors;
  const std::vector<float> level_sigma_sq;
  const std::vector<float> inv_level_sigma_sq;

private:
  static long unsigned int next_id_;
  long unsigned int id_;
  const long unsigned int frame_id_;
  const double timestamp_;

// The following variables need to be accessed trough a mutex to be thread safe.
protected:
  // Image bounds
  static int min_x_;
  static int min_y_;
  static int max_x_;
  static int max_y_;
  // SE3 Pose and camera center
  cv::Mat Tcw_;
  cv::Mat Twc_;
  cv::Mat Ow_;

  cv::Mat Cw_; // Stereo middle point. Only for visualization

  // MapPoints associated to keypoints
  std::vector<MapPoint*> map_points_;

  // BoW
  std::shared_ptr<KeyframeDatabase> keyframe_db_;
  std::shared_ptr<OrbVocabulary> orb_vocabulary_;

  // Grid over the image to speed up feature matching
  static float grid_element_width_;
  static float grid_element_height_;

  using FrameGrid = std::array<std::array<std::vector<std::size_t>, grid_rows>,grid_cols>; 
  FrameGrid grid_;

  std::map<KeyFrame*,int> connected_keyframe_weights_;
  std::vector<KeyFrame*> ordered_connected_keyframes_;
  std::vector<int> ordered_weights_;

  // Spanning Tree and Loop Edges
  bool first_connection_ = true;
  KeyFrame* parent_ = nullptr;
  std::set<KeyFrame*> children_;
  std::set<KeyFrame*> loop_edges_;

  // Bad flags
  bool not_erase_ = false;
  bool to_be_erased_ = false;
  bool bad_ = false;    

  std::shared_ptr<Map> map_;

  mutable std::mutex pose_mutex_;
  mutable std::mutex connection_mutex_;
  mutable std::mutex feature_mutex_;
};

#endif // SRC_KEYFRAME_H_