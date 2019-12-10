#ifndef SRC_OPTIMIZER_H_
#define SRC_OPTIMIZER_H_

#include "map.h"
#include "map_point.h"
#include "keyframe.h"
#include "loop_closer.h"
#include "frame.h"

#include "g2o/types/types_seven_dof_expmap.h"

class Optimizer {
public:
  Optimizer();
  ~Optimizer();

  int static PoseOptimization(Frame* pFrame) { return -99999;} // TODO
  void static GlobalBundleAdjustemnt(Map* pMap, 
                                     int nIterations=5, 
                                     bool *pbStopFlag=NULL,
                                     const unsigned long nLoopKF=0, 
                                     const bool bRobust = true) {} // TODO

  void static LocalBundleAdjustment(KeyFrame* pKF, bool* pbStopFlag, Map* pMap);

  // if bFixScale is true, optimize SE3 (stereo,rgbd), Sim3 otherwise (mono)
  static int OptimizeSim3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<MapPoint*>& vpMatches1,
                          g2o::Sim3& g2oS12, const float th2, const bool bFixScale) {
                            return -1;
                          } //TODO

  // if bFixScale is true, 6DoF optimization (stereo,rgbd), 7DoF otherwise (mono)
  void static OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                     const LoopCloser::KeyFrameAndPose& NonCorrectedSim3,
                                     const LoopCloser::KeyFrameAndPose& CorrectedSim3,
                                     const map<KeyFrame *, set<KeyFrame*>>& LoopConnections,
                                     const bool bFixScale) {

                                     } //TODO

};

#endif // SRC_OPTIMIZER_H_