#ifndef SRC_OPTIMIZER_H_
#define SRC_OPTIMIZER_H_

#include "frame.h"

class Optimizer {
public:
  Optimizer();
  ~Optimizer();

  int static PoseOptimization(Frame* pFrame) { return -99999;} // TODO

private:

};

#endif // SRC_OPTIMIZER_H_