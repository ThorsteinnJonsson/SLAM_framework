#ifndef SRC_OPTIMIZER_H_
#define SRC_OPTIMIZER_H_

#include "frame.h"

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

  // if bFixScale is true, optimize SE3 (stereo,rgbd), Sim3 otherwise (mono)
  static int OptimizeSim3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<MapPoint*>& vpMatches1,
                          g2o::Sim3& g2oS12, const float th2, const bool bFixScale); //TODO
private:

};

#endif // SRC_OPTIMIZER_H_