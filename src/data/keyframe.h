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
  KeyFrame(const Frame& F, 
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
  void UpdateBestCovisibles(); // TODO can't this be private??
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

  static bool weightComp(int a, int b) { return a > b; } // TODO just use lambda

  static bool lId(KeyFrame* pKF1, KeyFrame* pKF2){ return pKF1->mnId < pKF2->mnId; }


// The following variables are accesed from only 1 thread or never change (no mutex needed).
public:
  static long unsigned int nNextId;
  long unsigned int mnId;
  const long unsigned int mnFrameId;

  const double mTimeStamp;

  // Grid (to speed up feature matching)
  const int mnGridCols;
  const int mnGridRows;
  const float mfGridElementWidthInv;
  const float mfGridElementHeightInv;

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
  const float fx, fy, cx, cy, invfx, invfy, mbf, mb, mThDepth; // TODO seems exessive to copy for every keyframe

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

  // Image bounds and calibration
  const int mnMinX;
  const int mnMinY;
  const int mnMaxX;
  const int mnMaxY;
  const cv::Mat mK;  

// The following variables need to be accessed trough a mutex to be thread safe.
protected:
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
  std::vector<std::vector<std::vector<size_t>>> mGrid;

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

  mutable std::mutex mMutexPose;
  mutable std::mutex mMutexConnections;
  mutable std::mutex mMutexFeatures;


};

#endif // SRC_KEYFRAME_H_