#ifndef SRC_LOCAL_MAPPER_H_
#define SRC_LOCAL_MAPPER_H_

class LocalMapper {
public:
  LocalMapper();
  ~LocalMapper();

  void RequestReset() {} //TODO implement
  void InsertKeyFrame(KeyFrame* pKF) {} //TODO implement
  bool isStopped() { return false; } //TODO implement
  bool stopRequested() { return false; } //TODO implement
  bool AcceptKeyFrames() { return false; } //TODO implement
  void InterruptBA(); //TODO implement
  int KeyframesInQueue() {return -1;} //TODO implement
protected:

};

#endif // SRC_LOCAL_MAPPER_H_