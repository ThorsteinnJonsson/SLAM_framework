#ifndef SRC_LOOP_CLOSER_H_
#define SRC_LOOP_CLOSER_H_

#include "keyframe.h"
#include "local_mapper.h"
#include "map.h"
#include "orb_vocabulary.h"
#include "tracker.h"
#include "keyframe_database.h"

#include <thread>
#include <mutex>
#include "third_party/g2o/g2o/types/types_seven_dof_expmap.h"

// Forward declarations
class Tracker;
class LocalMapper;
class KeyframeDatabase;

class LoopCloser {
public:
  LoopCloser(Map* pMap, KeyframeDatabase* pDB, OrbVocabulary* pVoc, const bool bFixScale) {} // TODO
  ~LoopCloser();

  void Run() { while(true){} } // TODO

  void RequestReset() {} // TODO

  void SetTracker(Tracker* pTracker) {}  // TODO

  void SetLocalMapper(LocalMapper* pLocalMapper) {} // TODO

  bool isRunningGBA() { return false; } //TODO
  bool isFinishedGBA() { return false; } //TODO

  void RequestFinish() {} //TODO
  bool isFinished() { return false; } //TODO
  
private:

};

#endif // SRC_LOOP_CLOSER_H_