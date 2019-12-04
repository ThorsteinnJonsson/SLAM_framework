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

private:

};

#endif // SRC_OPTIMIZER_H_