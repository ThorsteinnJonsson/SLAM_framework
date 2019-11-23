#ifndef SRC_KEYFRAME_DATABASE_H_
#define SRC_KEYFRAME_DATABASE_H_

class KeyframeDatabase {
public:
  KeyframeDatabase() {}
  ~KeyframeDatabase() {}

  void clear() {}
  std::vector<KeyFrame*> DetectRelocalizationCandidates(Frame* F) { return std::vector<KeyFrame*>(); } // TODO this can probably be passed as a reference

private:

};

#endif // SRC_KEYFRAME_DATABASE_H_