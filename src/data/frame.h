#ifndef SRC_FRAME_H
#define SRC_FRAME_H

#include "data/map_point.h"
#include "DBoW2/BowVector.h"
#include "DBoW2/FeatureVector.h"
#include "orb_features/orb_vocabulary.h"
#include "data/keyframe.h"
#include "orb_features/orb_extractor.h"

#include <opencv2/opencv.hpp>

#include <memory>
#include <deque>

// Forward declarations
class MapPoint;
class KeyFrame;

class Frame {
public:
  Frame() {}
  ~Frame() {}

  // Copy constructor.
  Frame(const Frame& frame);

  // Constructor for stereo cameras.
  Frame(const cv::Mat& left_image, 
        const cv::Mat& right_image, 
        const double timestamp, 
        const std::shared_ptr<ORBextractor>& left_extractor, 
        const std::shared_ptr<ORBextractor>& right_extractor, 
        const std::shared_ptr<OrbVocabulary>& vocabulary, 
        cv::Mat& K, 
        cv::Mat& dist_coeff, 
        const float bf, 
        const float depth_threshold);

  // Constructor for RGB-D cameras.
  Frame(const cv::Mat& gray_image, 
        const cv::Mat& depth_image, 
        const double timestamp, 
        const std::shared_ptr<ORBextractor>& extractor,
        const std::shared_ptr<OrbVocabulary>& vocabulary, 
        cv::Mat& K, 
        cv::Mat& dist_coeff, 
        const float bf, 
        const float depth_threshold);

  // Constructor for Monocular cameras.
  Frame(const cv::Mat& gray_image, 
        const double timestamp, 
        const std::shared_ptr<ORBextractor>& extractor,
        const std::shared_ptr<OrbVocabulary>& vocabulary,
        cv::Mat& K, 
        cv::Mat& dist_coeff, 
        const float bf, 
        const float depth_threshold);

  // Extract ORB on the image. 0 for left image and 1 for right image.
  void ExtractORB(int flag, const cv::Mat& im);

  // Compute Bag of Words representation.
  void ComputeBoW();

  // Set the camera pose.
  void SetPose(const cv::Mat& Tcw);

  // Computes rotation, translation and camera center matrices from the camera pose.
  void UpdatePoseMatrices();

  // Returns the camera center.
  inline cv::Mat GetCameraCenter() { return Ow_.clone(); }

  // Returns inverse of rotation
  inline cv::Mat GetRotationInverse() { return Rwc_.clone(); }

  // Check if a MapPoint is in the frustum of the camera
  // and fill variables of the MapPoint to be used by the tracking
  bool isInFrustum(MapPoint* pMP, float viewingCosLimit);

  // Compute the cell of a keypoint (return false if outside the grid)
  bool PosInGrid(const cv::KeyPoint& kp, 
                 int& posX, 
                 int& posY);

  std::vector<size_t> GetFeaturesInArea(const float x, 
                                        const float y, 
                                        const float r, 
                                        const int minLevel=-1, 
                                        const int maxLevel=-1) const;

  // Search a match for each keypoint in the left image to a keypoint in the right image.
  // If there is a match, depth is computed and the right coordinate associated 
  // to the left keypoint is stored.
  void ComputeStereoMatches();

  // Associate a "right" coordinate to a keypoint if there is valid depth in the depthmap.
  void ComputeStereoFromRGBD(const cv::Mat& imDepth);

  // Backprojects a keypoint (if stereo/depth info available) into 3D world coordinates.
  cv::Mat UnprojectStereo(const int i);

public:
  static constexpr int grid_rows = 48;
  static constexpr int grid_cols = 64;

  // Calibration matrix and OpenCV distortion parameters.
  const cv::Mat& GetCalibMat() const { return calib_mat_; }
  const float GetFx() const { return fx_; }
  const float GetFy() const { return fy_; }
  const float GetCx() const { return cx_; }
  const float GetCy() const { return cy_; }
  const float GetInvFx() const { return invfx_; } 
  const float GetInvFy() const { return invfy_; } 

  // // Stereo baseline multiplied by fx.
  const float GetBaselineFx() const { return baseline_fx_; }

  // Stereo baseline in meters.
  const float GetBaseline() const { return baseline_; }

  // Threshold close/far points. Close points are inserted from 1 view.
  // Far points are inserted as in the monocular case from 2 views.
  const float GetDepthThrehold() const { return thresh_depth_; }

  // Number of KeyPoints.
  const int NumKeypoints() const { return num_keypoints_; }

  const std::shared_ptr<OrbVocabulary>& GetVocabulary() const { return orb_vocabulary_; } 
  const std::shared_ptr<ORBextractor>& GetLeftOrbExtractor() const {return left_orb_extractor_; } 
  const std::shared_ptr<ORBextractor>& GetRightOrbExtractor() const {return right_orb_extractor_; } 
  const double GetTimestamp() const { return timestamp_; } 

  const std::vector<cv::KeyPoint>& GetKeys() const { return keypoints_; }
  const std::vector<cv::KeyPoint>& GetRightKeys() const { return right_keypoints_; }
  const std::vector<cv::KeyPoint>& GetUndistortedKeys() const { return undistorted_keypoints_; }
  const std::vector<float>& StereoCoordRight() const {return stereo_coords_; }
  const std::vector<float>& StereoDepth() const {return depths_; }

  const DBoW2::BowVector& GetBowVector() const { return bow_vec_; }
  const DBoW2::FeatureVector& GetFeatureVector() const { return feature_vec_; }

  // // ORB descriptor, each row associated to a keypoint.
  const cv::Mat& GetDescriptors() const { return descriptors_; }
  const cv::Mat& GetRightDescriptors() const { return right_descriptors_; }

  // // MapPoints associated to keypoints, NULL pointer if no association.
  const std::vector<MapPoint*>& GetMapPoints() const {return map_points_; }
  std::vector<MapPoint*>& GetMapPoints() {return map_points_; }
  void SetMapPoints(const std::vector<MapPoint*>& pts) { map_points_ = pts; }
  MapPoint* GetMapPoint(int idx) const {return map_points_[idx]; };
  void SetMapPoint(int idx, MapPoint* pt) { map_points_[idx] = pt; }

  // // Flag to identify outlier associations.
  const std::deque<bool>& GetOutliers() const {return is_outlier_; }
  void SetOutliers(const std::deque<bool>& is_outlier) { is_outlier_ = is_outlier; }
  void SetOutlier(int idx, bool val) { is_outlier_[idx] = val; }
  bool IsOutlier(int idx) const { return is_outlier_[idx]; }

  // Keypoints are assigned to cells in a grid to reduce matching complexity 
  // when projecting MapPoints.
  static float mfGridElementWidthInv;
  static float mfGridElementHeightInv;

  const std::array<std::array<std::vector<std::size_t>,grid_rows>,grid_cols>& GetGrid() const { return grid_; }

  // Camera pose.
  cv::Mat mTcw;

  // Current and Next Frame id.
  static long unsigned int nNextId;
  long unsigned int mnId;

  // Reference Keyframe.
  KeyFrame* mpReferenceKF;

  // Scale pyramid info.
  int mnScaleLevels;
  float mfScaleFactor;
  float mfLogScaleFactor;
  std::vector<float> mvScaleFactors;
  std::vector<float> mvInvScaleFactors;
  std::vector<float> mvLevelSigma2;
  std::vector<float> mvInvLevelSigma2;

  const float GetMinX() const { return min_x_; }
  const float GetMaxX() const { return max_x_; }
  const float GetMinY() const { return min_y_; }
  const float GetMaxY() const { return max_y_; }

private:
  static bool mbInitialComputations;

  // Undistorted Image Bounds (computed once).
  static float min_x_;
  static float max_x_;
  static float min_y_;
  static float max_y_;

  // Calibration matrix and OpenCV distortion parameters.
  cv::Mat calib_mat_;
  static float fx_;
  static float fy_;
  static float cx_;
  static float cy_;
  static float invfx_;
  static float invfy_;
  cv::Mat dist_coeff_;

    // Stereo baseline multiplied by fx.
  float baseline_fx_;

  // Stereo baseline in meters.
  float baseline_;

  // Threshold close/far points. Close points are inserted from 1 view.
  // Far points are inserted as in the monocular case from 2 views.
  float thresh_depth_;

private:
  void MakeInitialComputations(const cv::Mat& image, 
                               cv::Mat& calibration_mat);

  // Undistort keypoints given OpenCV distortion parameters.
  // Only for the RGB-D case. Stereo must be already rectified!
  // (called in the constructor).
  void UndistortKeyPoints();

  // Computes image bounds for the undistorted image (called in the constructor).
  void ComputeImageBounds(const cv::Mat& imLeft);

  // Assign keypoints to the grid for speed up feature matching (called in the constructor).
  std::array<std::array<std::vector<std::size_t>,grid_rows>,grid_cols> grid_;
  void AssignFeaturesToGrid();

  // Rotation, translation and camera center
  cv::Mat Rcw_;
  cv::Mat tcw_;
  cv::Mat Rwc_;
  cv::Mat Ow_; // same as twc_

  std::shared_ptr<OrbVocabulary> orb_vocabulary_ = nullptr;
  std::shared_ptr<ORBextractor> left_orb_extractor_;
  std::shared_ptr<ORBextractor> right_orb_extractor_;
  
  double timestamp_;

  int num_keypoints_;

  // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
  // In the stereo case, undistorted_keypoints_ is redundant as images must be rectified.
  // In the RGB-D case, RGB images can be distorted.
  std::vector<cv::KeyPoint> keypoints_;
  std::vector<cv::KeyPoint> right_keypoints_;
  std::vector<cv::KeyPoint> undistorted_keypoints_;

  // Corresponding stereo coordinate and depth for each keypoint.
  // "Monocular" keypoints have a negative value.
  std::vector<float> stereo_coords_;
  std::vector<float> depths_;

  // Bag of Words Vector structures.
  DBoW2::BowVector bow_vec_;
  DBoW2::FeatureVector feature_vec_;

  // ORB descriptor, each row associated to a keypoint.
  cv::Mat descriptors_;
  cv::Mat right_descriptors_;

  // MapPoints associated to keypoints, NULL pointer if no association.
  std::vector<MapPoint*> map_points_;

  // Flag to identify outlier associations.
  std::deque<bool> is_outlier_;

};

#endif // SRC_FRAME_H