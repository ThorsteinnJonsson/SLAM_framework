#ifndef SRC_KEYFRAME_DATABASE_H_
#define SRC_KEYFRAME_DATABASE_H_


#include "keyframe.h"
#include "frame.h"
#include "orb_vocabulary.h"

#include<mutex>

class KeyFrame;
class Frame;


class KeyframeDatabase {
public:
  KeyframeDatabase(const OrbVocabulary& voc) {} // TODO
  ~KeyframeDatabase() {}

  void add(KeyFrame* pKF) {} //TODO
  void erase(KeyFrame* pKF) {} //TODO
  void clear() {}
  std::vector<KeyFrame*> DetectRelocalizationCandidates(Frame* F) { return std::vector<KeyFrame*>(); } // TODO this can probably be passed as a reference


private:

};

#endif // SRC_KEYFRAME_DATABASE_H_