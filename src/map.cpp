#include "map.h"

Map::Map() : mnMaxKFid(0)
           , mnBigChangeIdx(0) 
{

}

void Map::AddKeyFrame(KeyFrame *pKF) {
  std::unique_lock<std::mutex> lock(mMutexMap);
  mspKeyFrames.insert(pKF);
  if (pKF->mnId > mnMaxKFid) {
    mnMaxKFid = pKF->mnId;
  } 
}

void Map::AddMapPoint(MapPoint* pMP) {
  std::unique_lock<std::mutex> lock(mMutexMap);
  mspMapPoints.insert(pMP);
}

void Map::EraseMapPoint(MapPoint* pMP) {
  std::unique_lock<std::mutex> lock(mMutexMap);
  mspMapPoints.erase(pMP);
  // TODO: This only erase the pointer.
  // Delete the MapPoint
}

void Map::EraseKeyFrame(KeyFrame* pKF) {
    std::unique_lock<std::mutex> lock(mMutexMap);
    mspKeyFrames.erase(pKF);
    // TODO: This only erase the pointer.
    // Delete the KeyFrame
}

void Map::SetReferenceMapPoints(const std::vector<MapPoint*>& vpMPs) {
  // TODO only for visualization??
  std::unique_lock<std::mutex> lock(mMutexMap);
  mvpReferenceMapPoints = vpMPs;
}

void Map::InformNewBigChange() {
  std::unique_lock<std::mutex> lock(mMutexMap);
  ++mnBigChangeIdx;
}

int Map::GetLastBigChangeIdx() {
  std::unique_lock<std::mutex> lock(mMutexMap);
  return mnBigChangeIdx;
}

std::vector<KeyFrame*> Map::GetAllKeyFrames() {
  std::unique_lock<std::mutex> lock(mMutexMap);
  return std::vector<KeyFrame*>(mspKeyFrames.begin(),mspKeyFrames.end());
}

std::vector<MapPoint*> Map::GetAllMapPoints() {
  std::unique_lock<std::mutex> lock(mMutexMap);
  return std::vector<MapPoint*>(mspMapPoints.begin(),mspMapPoints.end());
}

std::vector<MapPoint*> Map::GetReferenceMapPoints() {
  std::unique_lock<std::mutex> lock(mMutexMap);
  return mvpReferenceMapPoints;
}

long unsigned  Map::KeyFramesInMap() {
  std::unique_lock<std::mutex> lock(mMutexMap);
  return mspKeyFrames.size();
}

long unsigned int Map::MapPointsInMap() {
  std::unique_lock<std::mutex> lock(mMutexMap);
  return mspMapPoints.size();
}

long unsigned int Map::GetMaxKFid() {
  std::unique_lock<std::mutex> lock(mMutexMap);
  return mnMaxKFid;
}

void Map::clear() {
  for(std::set<MapPoint*>::iterator sit = mspMapPoints.begin(); 
                                    sit != mspMapPoints.end(); 
                                    sit++) {
    delete *sit;
    // TODO probably not the best way to do this
  }
      

  for(std::set<KeyFrame*>::iterator sit = mspKeyFrames.begin(); 
                                    sit != mspKeyFrames.end(); 
                                    sit++) {
    delete *sit;
  }
      
  mspMapPoints.clear();
  mspKeyFrames.clear();
  mnMaxKFid = 0;
  mvpReferenceMapPoints.clear();
  mvpKeyFrameOrigins.clear();
}




