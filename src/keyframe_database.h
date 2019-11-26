#ifndef SRC_KEYFRAME_DATABASE_H_
#define SRC_KEYFRAME_DATABASE_H_

#include "orb_vocabulary.h"

class KeyframeDatabase {
public:
  KeyframeDatabase(const OrbVocabulary& voc) {} // TODO
  ~KeyframeDatabase() {}

  void clear() {}
  std::vector<KeyFrame*> DetectRelocalizationCandidates(Frame* F) { return std::vector<KeyFrame*>(); } // TODO this can probably be passed as a reference

private:

};

#endif // SRC_KEYFRAME_DATABASE_H_