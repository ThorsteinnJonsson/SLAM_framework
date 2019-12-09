#include "keyframe.h"
#include "converter.h"

// Define static variables with initial values
long unsigned int KeyFrame::nNextId = 0;

KeyFrame::KeyFrame(Frame& F, 
                   Map* pMap, 
                   KeyframeDatabase* pKFDB) 
      : mnFrameId(F.mnId)
      , mTimeStamp(F.mTimeStamp)
      , mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS)
      , mfGridElementWidthInv(F.mfGridElementWidthInv)
      , mfGridElementHeightInv(F.mfGridElementHeightInv)
      , mnTrackReferenceForFrame(0)
      , mnFuseTargetForKF(0)
      , mnBALocalForKF(0)
      , mnBAFixedForKF(0)
      , mnLoopQuery(0)
      , mnLoopWords(0)
      , mnRelocQuery(0)
      , mnRelocWords(0)
      , mnBAGlobalForKF(0)
      , fx(F.fx)
      , fy(F.fy)
      , cx(F.cx)
      , cy(F.cy)
      , invfx(F.invfx)
      , invfy(F.invfy)
      , mbf(F.mbf)
      , mb(F.mb)
      , mThDepth(F.mThDepth)
      , N(F.mN)
      , mvKeys(F.mvKeys)
      , mvKeysUn(F.mvKeysUn)
      , mvuRight(F.mvuRight)
      , mvDepth(F.mvDepth)
      , mDescriptors(F.mDescriptors.clone())
      , mBowVec(F.mBowVec)
      , mFeatVec(F.mFeatVec)
      , mnScaleLevels(F.mnScaleLevels)
      , mfScaleFactor(F.mfScaleFactor)
      , mfLogScaleFactor(F.mfLogScaleFactor)
      , mvScaleFactors(F.mvScaleFactors)
      , mvLevelSigma2(F.mvLevelSigma2)
      , mvInvLevelSigma2(F.mvInvLevelSigma2)
      , mnMinX(F.mnMinX)
      , mnMinY(F.mnMinY)
      , mnMaxX(F.mnMaxX)
      , mnMaxY(F.mnMaxY)
      , mK(F.mK)
      , mvpMapPoints(F.mvpMapPoints)
      , mpKeyFrameDB(pKFDB)
      , mpORBvocabulary(F.mpORBvocabulary)
      , mbFirstConnection(true)
      , mpParent(NULL)
      , mbNotErase(false)
      , mbToBeErased(false)
      , mbBad(false)
      , mHalfBaseline(F.mb/2)
      , mpMap(pMap) 
{
  mnId = ++nNextId;

  mGrid.resize(mnGridCols);
  for(int i=0; i < mnGridCols; ++i) {
    mGrid[i].resize(mnGridRows);
    for(int j=0; j < mnGridRows; ++j) {
      mGrid[i][j] = F.mGrid[i][j];
    }
  }

  SetPose(F.mTcw);  
}

void KeyFrame::SetPose(const cv::Mat& input_Tcw) {
  std::unique_lock<std::mutex> lock(mMutexPose);
  input_Tcw.copyTo(Tcw);
  cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
  cv::Mat tcw = Tcw.rowRange(0,3).col(3);
  cv::Mat Rwc = Rcw.t();
  Ow = -Rwc*tcw;

  Twc = cv::Mat::eye(4,4,Tcw.type());
  Rwc.copyTo(Twc.rowRange(0,3).colRange(0,3));
  Ow.copyTo(Twc.rowRange(0,3).col(3));
  cv::Mat center = (cv::Mat_<float>(4,1) << mHalfBaseline, 0 , 0, 1);
  Cw = Twc*center;
}

cv::Mat KeyFrame::GetPose() {
  std::unique_lock<std::mutex> lock(mMutexPose);
  return Tcw.clone();
}

cv::Mat KeyFrame::GetPoseInverse() {
  std::unique_lock<std::mutex> lock(mMutexPose);
  return Twc.clone();
}

cv::Mat KeyFrame::GetCameraCenter() {
  std::unique_lock<std::mutex> lock(mMutexPose);
  return Ow.clone();
}

cv::Mat KeyFrame::GetStereoCenter() {
  std::unique_lock<std::mutex> lock(mMutexPose);
  return Cw.clone();
}

cv::Mat KeyFrame::GetRotation() {
  std::unique_lock<std::mutex> lock(mMutexPose);
  return Tcw.rowRange(0,3).colRange(0,3).clone();
}

cv::Mat KeyFrame::GetTranslation() {
  std::unique_lock<std::mutex> lock(mMutexPose);
  return Tcw.rowRange(0,3).col(3).clone();
}

void KeyFrame::ComputeBoW() {
    if(mBowVec.empty() || mFeatVec.empty()) {
        std::vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        // Feature vector associates features with nodes in the 4th level (from leaves up)
        // We assume the vocabulary tree has 6 levels, change the 4 otherwise
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

void KeyFrame::AddConnection(KeyFrame* pKF, const int weight) {
  {
    std::unique_lock<std::mutex> lock(mMutexConnections);
    if( !mConnectedKeyFrameWeights.count(pKF) ) {
      mConnectedKeyFrameWeights[pKF] = weight;
    } else if( mConnectedKeyFrameWeights[pKF] != weight ) {
      mConnectedKeyFrameWeights[pKF] = weight;
    } else {
      return;
    }
  }
  UpdateBestCovisibles();
}

void KeyFrame::EraseConnection(KeyFrame* pKF) {
  bool bUpdate = false;
  {
    std::unique_lock<std::mutex> lock(mMutexConnections);
    if(mConnectedKeyFrameWeights.count(pKF)) {
      mConnectedKeyFrameWeights.erase(pKF);
      bUpdate=true;
    }
  }

  if(bUpdate) {
    UpdateBestCovisibles();
  }
}

void KeyFrame::UpdateBestCovisibles() {
  std::unique_lock<std::mutex> lock(mMutexConnections);
  std::vector<std::pair<int,KeyFrame*>> vPairs;
  vPairs.reserve(mConnectedKeyFrameWeights.size());
  for(std::map<KeyFrame*,int>::iterator mit = mConnectedKeyFrameWeights.begin(); 
      mit != mConnectedKeyFrameWeights.end(); 
      mit++) {
    vPairs.push_back(std::make_pair(mit->second,mit->first));
  }

  std::sort(vPairs.begin(),vPairs.end());
  std::list<KeyFrame*> lKFs;
  std::list<int> lWs;
  for(size_t i=0; i < vPairs.size(); ++i) {
    lKFs.push_front(vPairs[i].second);
    lWs.push_front(vPairs[i].first);
  }

  mvpOrderedConnectedKeyFrames = std::vector<KeyFrame*>(lKFs.begin(),lKFs.end());
  mvOrderedWeights = std::vector<int>(lWs.begin(), lWs.end());    
}

void KeyFrame::UpdateConnections() {
  std::map<KeyFrame*,int> KFcounter;

  std::vector<MapPoint*> vpMP;
  {
    std::unique_lock<std::mutex> lockMPs(mMutexFeatures);
    vpMP = mvpMapPoints;
  }

  //For all map points in keyframe check in which other keyframes are they seen
  //Increase counter for those keyframes
  for (std::vector<MapPoint*>::iterator vit = vpMP.begin(); 
                                        vit != vpMP.end(); 
                                        ++vit) {
    MapPoint* pMP = *vit;

    if(!pMP || pMP->isBad()) {
      continue;     
    }

    std::map<KeyFrame*,size_t> observations = pMP->GetObservations();

    for (std::map<KeyFrame*,size_t>::iterator mit = observations.begin();
                                              mit != observations.end(); 
                                              mit++) {
      if ( mit->first->mnId == mnId ) {
        continue;
      }
      KFcounter[mit->first]++;
    }
  }

  // This should not happen
  assert(KFcounter.empty() == false);
  if (KFcounter.empty()) {
    return;
  }

  //If the counter is greater than threshold add connection
  //In case no keyframe counter is over threshold add the one with maximum counter
  int nmax = 0;
  KeyFrame* pKFmax=NULL;
  int th = 15;

  std::vector<std::pair<int,KeyFrame*>> vPairs;
  vPairs.reserve(KFcounter.size());
  for (std::map<KeyFrame*,int>::iterator mit = KFcounter.begin(); 
                                         mit != KFcounter.end(); 
                                         ++mit)
  {
    KeyFrame* other_KF = mit->first;
    int other_KF_count = mit->second;
    if (other_KF_count > nmax) {
      nmax = other_KF_count;
      pKFmax = other_KF;
    }
    if (other_KF_count >= th) {
      vPairs.push_back(std::make_pair(other_KF_count, other_KF));
      other_KF->AddConnection(this, other_KF_count);
    }
  }

  if(vPairs.empty()) {
    vPairs.push_back(std::make_pair(nmax, pKFmax));
    pKFmax->AddConnection(this, nmax);
  }

  std::sort(vPairs.begin(), vPairs.end());
  std::list<KeyFrame*> lKFs;
  std::list<int> lWs;
  for (size_t i=0; i < vPairs.size(); ++i) {
    lKFs.push_front(vPairs[i].second);
    lWs.push_front(vPairs[i].first);
  }

  {
    std::unique_lock<std::mutex> lockCon(mMutexConnections);
    mConnectedKeyFrameWeights = KFcounter;
    mvpOrderedConnectedKeyFrames = std::vector<KeyFrame*>(lKFs.begin(), lKFs.end());
    mvOrderedWeights = std::vector<int>(lWs.begin(), lWs.end());

    if (mbFirstConnection && mnId != 0) {
      mpParent = mvpOrderedConnectedKeyFrames.front();
      mpParent->AddChild(this);
      mbFirstConnection = false;
    }
  }
}

std::set<KeyFrame*> KeyFrame::GetConnectedKeyFrames() {
  std::unique_lock<std::mutex> lock(mMutexConnections);
  std::set<KeyFrame*> s;
  for(std::map<KeyFrame*,int>::iterator mit = mConnectedKeyFrameWeights.begin();
                                        mit != mConnectedKeyFrameWeights.end();
                                        mit++) {
    s.insert(mit->first);
  }
  return s;
}

std::vector<KeyFrame*> KeyFrame::GetVectorCovisibleKeyFrames() {
  std::unique_lock<std::mutex> lock(mMutexConnections);
  return mvpOrderedConnectedKeyFrames;
}

std::vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(const int N) {
  std::unique_lock<std::mutex> lock(mMutexConnections);
  if ( static_cast<int>(mvpOrderedConnectedKeyFrames.size()) < N) {
    return mvpOrderedConnectedKeyFrames;
  } else {
    return std::vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(),
                                  mvpOrderedConnectedKeyFrames.begin() + N);
  }
}

std::vector<KeyFrame*> KeyFrame::GetCovisiblesByWeight(const int w) {
  std::unique_lock<std::mutex> lock(mMutexConnections);

  if(mvpOrderedConnectedKeyFrames.empty()) {
    return std::vector<KeyFrame*>();
  }

  std::vector<int>::iterator it = std::upper_bound(mvOrderedWeights.begin(),
                                                   mvOrderedWeights.end(),
                                                   w,
                                                   KeyFrame::weightComp);
  if (it == mvOrderedWeights.end()) {
    return std::vector<KeyFrame*>();
  } else {
    int n = it - mvOrderedWeights.begin();
    return std::vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), 
                                  mvpOrderedConnectedKeyFrames.begin() + n);
  }
}

int KeyFrame::GetWeight(KeyFrame* pKF) {
  std::unique_lock<std::mutex> lock(mMutexConnections);
  if (mConnectedKeyFrameWeights.count(pKF)) {
    return mConnectedKeyFrameWeights[pKF];
  } else {
    return 0;
  }
}

void KeyFrame::AddChild(KeyFrame* pKF) {
  std::unique_lock<std::mutex> lock(mMutexConnections);
  mspChildrens.insert(pKF);
}

void KeyFrame::EraseChild(KeyFrame* pKF) {
  std::unique_lock<std::mutex> lock(mMutexConnections);
  mspChildrens.erase(pKF);
}

void KeyFrame::ChangeParent(KeyFrame* pKF) {
  std::unique_lock<std::mutex> lock(mMutexConnections);
  mpParent = pKF;
  pKF->AddChild(this);
}

std::set<KeyFrame*> KeyFrame::GetChilds() {
  std::unique_lock<std::mutex> lock(mMutexConnections);
  return mspChildrens;
}

KeyFrame* KeyFrame::GetParent() {
  std::unique_lock<std::mutex> lock(mMutexConnections);
  return mpParent;
}

bool KeyFrame::hasChild(KeyFrame* pKF) {
  std::unique_lock<std::mutex> lock(mMutexConnections);
  return mspChildrens.count(pKF);
}

void KeyFrame::AddLoopEdge(KeyFrame* pKF) {
  std::unique_lock<std::mutex> lock(mMutexConnections);
  mbNotErase = true;
  mspLoopEdges.insert(pKF);
}

std::set<KeyFrame*> KeyFrame::GetLoopEdges() {
  std::unique_lock<std::mutex> lock(mMutexConnections);
  return mspLoopEdges;
}

void KeyFrame::AddMapPoint(MapPoint* pMP, const size_t idx) {
  std::unique_lock<std::mutex> lock(mMutexFeatures);
  mvpMapPoints[idx] = pMP;
}

void KeyFrame::EraseMapPointMatch(const size_t idx) {
  std::unique_lock<std::mutex> lock(mMutexFeatures);
  mvpMapPoints[idx] = nullptr;
}

void KeyFrame::EraseMapPointMatch(MapPoint* pMP) {
  int idx = pMP->GetIndexInKeyFrame(this);
  if(idx >= 0) {
    // TODO why is there no mutex here??
    mvpMapPoints[idx] = nullptr;
  }
}

void KeyFrame::ReplaceMapPointMatch(const size_t idx, MapPoint* pMP) {
  // TODO why is there no mutex here??
  mvpMapPoints[idx] = pMP;
}

std::set<MapPoint*> KeyFrame::GetMapPoints() {
  std::unique_lock<std::mutex> lock(mMutexFeatures);
  std::set<MapPoint*> s;
  for (size_t i=0; i < mvpMapPoints.size(); ++i) {
    if (!mvpMapPoints[i]) {
      continue;
    }
    MapPoint* pMP = mvpMapPoints[i];
    if (!pMP->isBad()) {
      s.insert(pMP);
    }
  }
  return s;
}

std::vector<MapPoint*> KeyFrame::GetMapPointMatches() {
  std::unique_lock<std::mutex> lock(mMutexFeatures);
  return mvpMapPoints;
}

int KeyFrame::TrackedMapPoints(const int minObs) {
  std::unique_lock<std::mutex> lock(mMutexFeatures);

  int nPoints = 0;
  const bool bCheckObs = minObs > 0;
  for (int i = 0; i < N; ++i) {
    MapPoint* pMP = mvpMapPoints[i];
    if (pMP && !pMP->isBad()) {
      if (bCheckObs) {
        if (mvpMapPoints[i]->Observations() >= minObs) {
          nPoints++;
        }
      }
      else {
        nPoints++;
      }
    }
  }
  return nPoints;
}

MapPoint* KeyFrame::GetMapPoint(const size_t idx) {
  std::unique_lock<std::mutex> lock(mMutexFeatures);
  return mvpMapPoints[idx];
}

std::vector<size_t> KeyFrame::GetFeaturesInArea(const float x, 
                                                const float y, 
                                                const float r) const 
{
  std::vector<size_t> vIndices;
  vIndices.reserve(N);

  const int nMinCellX = std::max(0,
                                 static_cast<int>(std::floor((x-mnMinX-r) * mfGridElementWidthInv)));
  const int nMaxCellX = std::min(static_cast<int>(mnGridCols-1), // TODO is this needed? 
                                 static_cast<int>(std::ceil((x-mnMinX+r)*mfGridElementWidthInv)));
  const int nMinCellY = std::max(0,
                                 static_cast<int>(std::floor((y-mnMinY-r) * mfGridElementHeightInv)));
  const int nMaxCellY = std::min(static_cast<int>(mnGridRows-1), // TODO is this needed? 
                                 static_cast<int>(std::ceil((y-mnMinY+r)*mfGridElementHeightInv)));
  if(nMaxCellX < 0 || nMinCellX >= mnGridCols) {
    return vIndices;
  }    
  if(nMaxCellY < 0 || nMinCellY >= mnGridRows) {
    return vIndices;
  }
      
  for (int ix = nMinCellX; ix <= nMaxCellX; ++ix) {
    for (int iy = nMinCellY; iy <= nMaxCellY; ++iy) {
      const std::vector<size_t> vCell = mGrid[ix][iy];
      for (size_t j = 0; j < vCell.size(); ++j) {
        const cv::KeyPoint& kpUn = mvKeysUn[vCell[j]];
        const float distx = kpUn.pt.x - x;
        const float disty = kpUn.pt.y - y;
        if(std::fabs(distx) < r && std::fabs(disty) < r) {
          vIndices.push_back(vCell[j]);
        }
      }
    }
  }
  return vIndices;
}

cv::Mat KeyFrame::UnprojectStereo(int i) {
  const float z = mvDepth[i];
  if (z > 0) {
    const float u = mvKeys[i].pt.x;
    const float v = mvKeys[i].pt.y;
    const float x = (u - cx) * z * invfx;
    const float y = (v - cy) * z * invfy;
    cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);

    std::unique_lock<std::mutex> lock(mMutexPose);
    return Twc.rowRange(0,3).colRange(0,3) * x3Dc + Twc.rowRange(0,3).col(3);
  } else {
    return cv::Mat();
  }      
}

bool KeyFrame::IsInImage (const float x, const float y) const {
  return ( x >= mnMinX && x < mnMaxX && y >= mnMinY && y < mnMaxY);
}

void KeyFrame::SetNotErase() {
  std::unique_lock<std::mutex> lock(mMutexConnections);
  mbNotErase = true;
}

void KeyFrame::SetErase() {
  {
    std::unique_lock<std::mutex> lock(mMutexConnections);
    if (mspLoopEdges.empty()) {
      mbNotErase = false;
    }
  }
  if (mbToBeErased) {
    SetBadFlag();
  }
}

void KeyFrame::SetBadFlag() {
  {
    std::unique_lock<std::mutex> lock(mMutexConnections);
    if (mnId == 0) {
      return;
    } else if (mbNotErase) {
      mbToBeErased = true;
      return;
    }
  }

  for (std::map<KeyFrame*,int>::iterator mit = mConnectedKeyFrameWeights.begin(); 
                                        mit != mConnectedKeyFrameWeights.end(); 
                                        mit++) 
  {
    mit->first->EraseConnection(this);
  }

  for (size_t i = 0; i < mvpMapPoints.size(); ++i) {
    if (mvpMapPoints[i]) {
      mvpMapPoints[i]->EraseObservation(this);
    }
  }
  
  {
    std::unique_lock<std::mutex> lockCon(mMutexConnections);
    std::unique_lock<std::mutex> lockFeat(mMutexFeatures);

    mConnectedKeyFrameWeights.clear();
    mvpOrderedConnectedKeyFrames.clear();

    // Update Spanning Tree
    std::set<KeyFrame*> sParentCandidates;
    sParentCandidates.insert(mpParent);

    // Assign at each iteration one children with a parent (the pair with highest covisibility weight)
    // Include that children as new parent candidate for the rest
    while (!mspChildrens.empty()) {
      bool bContinue = false;

      int max = -1;
      KeyFrame* pC;
      KeyFrame* pP;

      for (std::set<KeyFrame*>::iterator sit = mspChildrens.begin(); 
                                        sit != mspChildrens.end(); 
                                        sit++)
      {
        KeyFrame* pKF = *sit;
        if(pKF->isBad()) {
          continue;
        }

        // Check if a parent candidate is connected to the keyframe
        std::vector<KeyFrame*> vpConnected = pKF->GetVectorCovisibleKeyFrames();
        for (size_t i = 0; i < vpConnected.size(); i++) {
          for (std::set<KeyFrame*>::iterator spcit = sParentCandidates.begin(); 
                                             spcit != sParentCandidates.end(); 
                                             spcit++) 
          {
            if (vpConnected[i]->mnId == (*spcit)->mnId) {
              int w = pKF->GetWeight(vpConnected[i]);
              if(w > max) {
                pC = pKF;
                pP = vpConnected[i];
                max = w;
                bContinue = true;
              }
            }
          }
        }
      }

      if (bContinue) {
          pC->ChangeParent(pP);
          sParentCandidates.insert(pC);
          mspChildrens.erase(pC);
      } else {
        break;
      }
    }

    // If a children has no covisibility links with any parent candidate, assign to the original parent of this KF
    if(!mspChildrens.empty()) {
      for(std::set<KeyFrame*>::iterator sit = mspChildrens.begin(); 
                                        sit != mspChildrens.end(); 
                                        sit++)
      {
        (*sit)->ChangeParent(mpParent);
      }
    }

    mpParent->EraseChild(this);
    mTcp = Tcw * mpParent->GetPoseInverse();
    mbBad = true;
  }

  mpMap->EraseKeyFrame(this);
  mpKeyFrameDB->erase(this);
}

bool KeyFrame::isBad() {
  std::unique_lock<std::mutex> lock(mMutexConnections);
  return mbBad;
}


