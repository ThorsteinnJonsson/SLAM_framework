#ifndef SRC_LOCAL_MAPPER_H_
#define SRC_LOCAL_MAPPER_H_

#include "keyframe.h"
#include "map.h"
#include "loop_closer.h"
#include "tracker.h"
#include "keyframe_database.h"

class LocalMapper {
public:
  LocalMapper(Map* pMap, const float bMonocular); // TODO why is this float???
  ~LocalMapper();

  void Run() { while(true){} } // TODO

  void RequestStop() {} //TODO implement
  void RequestReset() {} //TODO implement
  bool Stop() { return false; } //TODO implement
  void Release() {} //TODO implement
  void InsertKeyFrame(KeyFrame* pKF) {} //TODO implement
  bool isStopped() { return false; } //TODO implement
  bool stopRequested() { return false; } //TODO implement
  bool AcceptKeyFrames() { return false; } //TODO implement
  void InterruptBA(); //TODO implement
  int KeyframesInQueue() {return -1;} //TODO implement
  bool SetNotStop(bool flag) {return false;} //TODO implement

  void SetLoopCloser(LoopCloser* pLoopCloser) {} // TODO
  void SetTracker(Tracker* pTracker) {} // TOOD
  
  void RequestFinish() {} //TODO
  bool isFinished() { return false; } //TODO
  
protected:

};

#endif // SRC_LOCAL_MAPPER_H_