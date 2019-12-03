#ifndef SRC_KEYFRAME_DATABASE_H_
#define SRC_KEYFRAME_DATABASE_H_

#include <vector>
#include <list>
#include <set>
#include <mutex>

#include "keyframe.h"
#include "frame.h"
#include "orb_vocabulary.h"



class KeyFrame;
class Frame;


class KeyframeDatabase {
public:
  KeyframeDatabase(const OrbVocabulary& voc);
  ~KeyframeDatabase() {}

  void add(KeyFrame* pKF);
  void erase(KeyFrame* pKF);
  void clear();

  std::vector<KeyFrame *> DetectLoopCandidates(KeyFrame* pKF, float minScore);
  std::vector<KeyFrame*> DetectRelocalizationCandidates(Frame* F);// TODO this can probably be passed as a reference

protected:
  const OrbVocabulary* mpVoc;
  std::vector<std::list<KeyFrame*>> mvInvertedFile;
  std::mutex mMutex;
};

#endif // SRC_KEYFRAME_DATABASE_H_