#ifndef SRC_OPTIMIZER_H_
#define SRC_OPTIMIZER_H_

#include "core/loop_closer.h"
#include "data/frame.h"
#include "data/keyframe.h"
#include "data/map_point.h"
#include "data/map.h"


#include "g2o/types/types_seven_dof_expmap.h"

class Optimizer {
public:
  // TODO this function can probably be private
  static void BundleAdjustment(const std::vector<KeyFrame*>& vpKF, 
                               const std::vector<MapPoint*>& vpMP,
                               int nIterations=5, 
                               bool* pbStopFlag=nullptr, 
                               const unsigned long nLoopKF=0,
                               const bool bRobust=true);

  static void GlobalBundleAdjustemnt(const std::shared_ptr<Map>& pMap, 
                                     int nIterations=5, 
                                     bool *pbStopFlag=nullptr,
                                     const unsigned long nLoopKF=0, 
                                     const bool bRobust=true);

  static void LocalBundleAdjustment(KeyFrame* pKF, 
                                    bool* pbStopFlag, 
                                    const std::shared_ptr<Map>& pMap);

  static int PoseOptimization(Frame* pFrame);

  
  // if bFixScale is true, 6DoF optimization (stereo,rgbd), 7DoF otherwise (mono)
  static void OptimizeEssentialGraph(const std::shared_ptr<Map>& pMap, 
                                     KeyFrame* pLoopKF, 
                                     KeyFrame* pCurKF,
                                     const LoopCloser::KeyFrameAndPose& NonCorrectedSim3,
                                     const LoopCloser::KeyFrameAndPose& CorrectedSim3,
                                     const std::map<KeyFrame*,std::set<KeyFrame*>>& LoopConnections,
                                     const bool bFixScale);

  // if bFixScale is true, optimize SE3 (stereo,rgbd), Sim3 otherwise (mono)
  static int OptimizeSim3(KeyFrame* pKF1, 
                          KeyFrame* pKF2, 
                          std::vector<MapPoint*>& vpMatches1,
                          g2o::Sim3& g2oS12, 
                          const float th2, 
                          const bool bFixScale);
};

#endif // SRC_OPTIMIZER_H_