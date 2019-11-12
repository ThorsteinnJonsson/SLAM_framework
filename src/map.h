#ifndef SRC_MAP_H_
#define SRC_MAP_H_

#include "map_point.h"
#include "keyframe.h"

#include <set>
#include <mutex>

class Map{
public:
  Map();
  ~Map() {}

  void AddKeyFrame(KeyFrame* pKF);
  void AddMapPoint(MapPoint* pMP);
  void EraseMapPoint(MapPoint* pMP);
  void EraseKeyFrame(KeyFrame* pKF);
  void SetReferenceMapPoints(const std::vector<MapPoint*>& vpMPs);
  void InformNewBigChange();
  int GetLastBigChangeIdx();

  std::vector<KeyFrame*> GetAllKeyFrames();
  std::vector<MapPoint*> GetAllMapPoints();
  std::vector<MapPoint*> GetReferenceMapPoints();

  long unsigned  KeyFramesInMap();
  long unsigned int MapPointsInMap();

  long unsigned int GetMaxKFid();

  void clear();


  std::vector<KeyFrame*> mvpKeyFrameOrigins;

  std::mutex mMutexMapUpdate;
  std::mutex mMutexPointCreation;


protected:
  std::set<KeyFrame*> mspKeyFrames;
  std::set<MapPoint*> mspMapPoints;

  std::vector<MapPoint*> mvpReferenceMapPoints;
  
  long unsigned int mnMaxKFid;
  
  // Index related to a big change in the map (loop closure, global BA)
  int mnBigChangeIdx;

  std::mutex mMutexMap;

};

#endif // SRC_MAP_H_