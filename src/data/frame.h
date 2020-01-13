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
  Frame(const cv::Mat& imLeft, 
        const cv::Mat& imRight, 
        const double timeStamp, 
        const std::shared_ptr<ORBextractor>& extractorLeft, 
        const std::shared_ptr<ORBextractor>& extractorRight, 
        const std::shared_ptr<OrbVocabulary>& voc, 
        cv::Mat& K, 
        cv::Mat& distCoef, 
        const float bf, 
        const float thDepth);

  // Constructor for RGB-D cameras.
  Frame(const cv::Mat& imGray, 
        const cv::Mat& imDepth, 
        const double timeStamp, 
        const std::shared_ptr<ORBextractor>& extractor,
        const std::shared_ptr<OrbVocabulary>& voc, 
        cv::Mat& K, 
        cv::Mat& distCoef, 
        const float bf, 
        const float thDepth);

  // Constructor for Monocular cameras.
  Frame(const cv::Mat& imGray, 
        const double timeStamp, 
        const std::shared_ptr<ORBextractor>& extractor,
        const std::shared_ptr<OrbVocabulary>& voc, 
        cv::Mat& K, 
        cv::Mat& distCoef, 
        const float bf, 
        const float thDepth);

  // Extract ORB on the image. 0 for left image and 1 for right image.
  void ExtractORB(int flag, const cv::Mat& im);

  // Compute Bag of Words representation.
  void ComputeBoW();

  // Set the camera pose.
  void SetPose(const cv::Mat& Tcw);

  // Computes rotation, translation and camera center matrices from the camera pose.
  void UpdatePoseMatrices();

  // Returns the camera center.
  inline cv::Mat GetCameraCenter() { return mOw.clone(); }

  // Returns inverse of rotation
  inline cv::Mat GetRotationInverse() { return mRwc.clone(); }

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

  // Vocabulary used for relocalization.
  std::shared_ptr<OrbVocabulary> mpORBvocabulary = nullptr;

  // Feature extractor. The right is used only in the stereo case.
  std::shared_ptr<ORBextractor> mpORBextractorLeft;
  std::shared_ptr<ORBextractor> mpORBextractorRight;

  // Frame timestamp.
  double mTimeStamp;

  // Calibration matrix and OpenCV distortion parameters.
  cv::Mat mK;
  static float fx;
  static float fy;
  static float cx;
  static float cy;
  static float invfx;
  static float invfy;
  cv::Mat mDistCoef;

  // Stereo baseline multiplied by fx.
  float mbf;

  // Stereo baseline in meters.
  float mb;

  // Threshold close/far points. Close points are inserted from 1 view.
  // Far points are inserted as in the monocular case from 2 views.
  float mThDepth;

  // Number of KeyPoints.
  int mN;

  // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
  // In the stereo case, mvKeysUn is redundant as images must be rectified.
  // In the RGB-D case, RGB images can be distorted.
  std::vector<cv::KeyPoint> mvKeys;
  std::vector<cv::KeyPoint> mvKeysRight;
  std::vector<cv::KeyPoint> mvKeysUn;

  // Corresponding stereo coordinate and depth for each keypoint.
  // "Monocular" keypoints have a negative value.
  std::vector<float> mvuRight;
  std::vector<float> mvDepth;

  // Bag of Words Vector structures.
  DBoW2::BowVector mBowVec;
  DBoW2::FeatureVector mFeatVec;

  // ORB descriptor, each row associated to a keypoint.
  cv::Mat mDescriptors;
  cv::Mat mDescriptorsRight;

  // MapPoints associated to keypoints, NULL pointer if no association.
  std::vector<MapPoint*> mvpMapPoints;

  // Flag to identify outlier associations.
  std::deque<bool> mvbOutlier; // TODO vector of bools is not good, replace

  // Keypoints are assigned to cells in a grid to reduce matching complexity 
  // when projecting MapPoints.
  static float mfGridElementWidthInv;
  static float mfGridElementHeightInv;

  const std::array<std::array<std::vector<std::size_t>, grid_rows>, grid_cols>& GetGrid() const { return grid_; }

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

  // Undistorted Image Bounds (computed once).
  static float mnMinX;
  static float mnMaxX;
  static float mnMinY;
  static float mnMaxY;

  static bool mbInitialComputations;

private:
  // Undistort keypoints given OpenCV distortion parameters.
  // Only for the RGB-D case. Stereo must be already rectified!
  // (called in the constructor).
  void UndistortKeyPoints();

  // Computes image bounds for the undistorted image (called in the constructor).
  void ComputeImageBounds(const cv::Mat& imLeft);

  // Assign keypoints to the grid for speed up feature matching (called in the constructor).
  std::array<std::array<std::vector<std::size_t>, grid_rows>, grid_cols> grid_;
  void AssignFeaturesToGrid();

  // Rotation, translation and camera center
  cv::Mat mRcw;
  cv::Mat mtcw;
  cv::Mat mRwc;
  cv::Mat mOw; //==mtwc

};

#endif // SRC_FRAME_H