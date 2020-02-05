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

  void SetPose(const cv::Mat& Tcw);
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

  // Image
  bool IsInImage(const float x, const float y) const;

  // Enable/Disable bad flag changes
  void SetNotErase();
  void SetErase();
  
  // Set/check bad flag
  void SetBadFlag();
  bool isBad() const;

  // Compute Scene Depth (q=2 median). Used in monocular.
  float ComputeSceneMedianDepth(const int q);

  static bool lId(KeyFrame* pKF1, KeyFrame* pKF2){ return pKF1->mnId < pKF2->mnId; }

private:
  void UpdateBestCovisibles();

// The following variables are accesed from only 1 thread or never change (no mutex needed).
public:
  static long unsigned int nNextId;
  long unsigned int mnId;
  const long unsigned int mnFrameId;

  const double timestamp_;

  // Grid (to speed up feature matching)
  static constexpr int grid_rows = 48;
  static constexpr int grid_cols = 64;

  // Variables used by the tracking
  long unsigned int mnTrackReferenceForFrame;
  long unsigned int mnFuseTargetForKF;

  // Variables used by the local mapping
  long unsigned int bundle_adj_local_id_for_keyframe;
  long unsigned int mnBAFixedForKF;

  // Variables used by the keyframe database
  long unsigned int mnLoopQuery;
  int mnLoopWords;
  float mLoopScore;
  long unsigned int mnRelocQuery;
  int mnRelocWords;
  float mRelocScore;

  // Variables used by loop closing
  cv::Mat mTcwGBA;
  cv::Mat mTcwBefGBA;
  long unsigned int bundle_adj_global_for_keyframe_id;

  // Calibration parameters
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
  const float mThDepth;

  // Number of KeyPoints
  const int N;

  // KeyPoints, stereo coordinate and descriptors (all associated by an index)
  const std::vector<cv::KeyPoint> mvKeys;
  const std::vector<cv::KeyPoint> mvKeysUn;
  
  const std::vector<float> mvuRight; // negative value for monocular points
  const std::vector<float> mvDepth; // negative value for monocular points
  const cv::Mat mDescriptors;

  //BoW
  DBoW2::BowVector mBowVec;
  DBoW2::FeatureVector mFeatVec;

  // Pose relative to parent (this is computed when bad flag is activated)
  cv::Mat mTcp;

  // Scale
  const int mnScaleLevels;
  const float mfScaleFactor;
  const float mfLogScaleFactor;
  const std::vector<float> mvScaleFactors;
  const std::vector<float> mvLevelSigma2;
  const std::vector<float> mvInvLevelSigma2;

  // Calibration matrix
  const cv::Mat mK;  

// The following variables need to be accessed trough a mutex to be thread safe.
protected:

  // Image bounds
  static int mnMinX;
  static int mnMinY;
  static int mnMaxX;
  static int mnMaxY;
  // SE3 Pose and camera center
  cv::Mat Tcw;
  cv::Mat Twc;
  cv::Mat Ow;

  cv::Mat Cw; // Stereo middle point. Only for visualization

  // MapPoints associated to keypoints
  std::vector<MapPoint*> mvpMapPoints;

  // BoW
  std::shared_ptr<KeyframeDatabase> mpKeyFrameDB;
  std::shared_ptr<OrbVocabulary> mpORBvocabulary;

  // Grid over the image to speed up feature matching
  static float grid_element_width_;
  static float grid_element_height_;

  std::array<std::array<std::vector<std::size_t>, grid_rows>, grid_cols> grid_;

  std::map<KeyFrame*,int> mConnectedKeyFrameWeights;
  std::vector<KeyFrame*> mvpOrderedConnectedKeyFrames;
  std::vector<int> mvOrderedWeights;

  // Spanning Tree and Loop Edges
  bool mbFirstConnection;
  KeyFrame* mpParent;
  std::set<KeyFrame*> mspChildrens;
  std::set<KeyFrame*> mspLoopEdges;

  // Bad flags
  bool mbNotErase;
  bool mbToBeErased;
  bool mbBad;    

  float mHalfBaseline; // Only for visualization

  std::shared_ptr<Map> mpMap;

  mutable std::mutex pose_mutex_;
  mutable std::mutex connection_mutex_;
  mutable std::mutex feature_mutex_;


};

#endif // SRC_KEYFRAME_H_