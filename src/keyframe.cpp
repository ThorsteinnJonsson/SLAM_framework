#include "keyframe.h"
#include "converter.h"

KeyFrame::KeyFrame(Frame& F, Map* pMap, KeyFrameDatabase* pKFDB) 
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
  for(std::vector<MapPoint*>::iterator vit = vpMP.begin(); 
      vit != vpMP.end(); 
      vit++) 
  {
    MapPoint* pMP = *vit;

    if(!pMP || pMP->isBad()) {
      continue;     
    }

    std::map<KeyFrame*,size_t> observations = pMP->GetObservations();

    for(std::map<KeyFrame*,size_t>::iterator mit = observations.begin();
        mit != observations.end(); 
        mit++)
    {
      if ( mit->first->mnId == mnId ) {
        continue;
      }
      KFcounter[mit->first]++;
    }
  }

  // This should not happen
  assert(KFcounter.empty() == false);
  if(KFcounter.empty()) {
    return;
  }

  //If the counter is greater than threshold add connection
  //In case no keyframe counter is over threshold add the one with maximum counter
  int nmax = 0;
  KeyFrame* pKFmax=NULL;
  int th = 15;

  std::vector<std::pair<int,KeyFrame*>> vPairs;
  vPairs.reserve(KFcounter.size());
  for (std::map<KeyFrame*,int>::iterator mit=KFcounter.begin(); 
      mit != KFcounter.end(); 
      mit++)
  {
    KeyFrame* other_KF = mit->first;
    int other_KF_count = mit->second;
    if(other_KF_count > nmax) {
      nmax = other_KF_count;
      pKFmax =other_KF
    }
    if(other_KF_count >= th) {
      vPairs.push_back(std::make_pair(other_KF_count, other_KF);
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
  // TODO
}

