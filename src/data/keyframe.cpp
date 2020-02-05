#include "keyframe.h"

#include "util/converter.h"

// Define static variables with initial values
long unsigned int KeyFrame::nNextId = 0;

bool KeyFrame::initial_computations = true;

float KeyFrame::cx, KeyFrame::cy; 
float KeyFrame::fx, KeyFrame::fy;
float KeyFrame::invfx, KeyFrame::invfy;

int KeyFrame::mnMinX, KeyFrame::mnMinY;
int KeyFrame::mnMaxX, KeyFrame::mnMaxY;

float KeyFrame::grid_element_width_;
float KeyFrame::grid_element_height_;


KeyFrame::KeyFrame(const Frame& frame, 
                   const std::shared_ptr<Map>& pMap, 
                   const std::shared_ptr<KeyframeDatabase>& pKFDB) 
      : mnFrameId(frame.mnId)
      , timestamp_(frame.GetTimestamp())
      , mnTrackReferenceForFrame(0)
      , mnFuseTargetForKF(0)
      , bundle_adj_local_id_for_keyframe(0)
      , mnBAFixedForKF(0)
      , mnLoopQuery(0)
      , mnLoopWords(0)
      , mnRelocQuery(0)
      , mnRelocWords(0)
      , bundle_adj_global_for_keyframe_id(0)
      , mbf(frame.GetBaselineFx())
      , mb(frame.GetBaseline())
      , mThDepth(frame.GetDepthThrehold())
      , N(frame.NumKeypoints())
      , mvKeys(frame.GetKeys())
      , mvKeysUn(frame.GetUndistortedKeys())
      , mvuRight(frame.StereoCoordRight())
      , mvDepth(frame.StereoDepth())
      , mDescriptors(frame.GetDescriptors().clone())
      , mBowVec(frame.GetBowVector())
      , mFeatVec(frame.GetFeatureVector())
      , mnScaleLevels(frame.mnScaleLevels)
      , mfScaleFactor(frame.mfScaleFactor)
      , mfLogScaleFactor(frame.mfLogScaleFactor)
      , mvScaleFactors(frame.mvScaleFactors)
      , mvLevelSigma2(frame.mvLevelSigma2)
      , mvInvLevelSigma2(frame.mvInvLevelSigma2)
      , mK(frame.GetCalibMat())
      , mvpMapPoints(frame.GetMapPoints())
      , mpKeyFrameDB(pKFDB)
      , mpORBvocabulary(frame.GetVocabulary())
      , mbFirstConnection(true)
      , mpParent(nullptr)
      , mbNotErase(false)
      , mbToBeErased(false)
      , mbBad(false)
      , mHalfBaseline(frame.GetBaseline() / 2)
      , mpMap(pMap) {

  if (initial_computations) {
    fx = frame.GetFx();
    fy = frame.GetFy();
    cx = frame.GetCx();
    cy = frame.GetCy();
    invfx = frame.GetInvFx();
    invfy = frame.GetInvFy();

    mnMinX = frame.GetMinX();
    mnMinY = frame.GetMinY();
    mnMaxX = frame.GetMaxX();
    mnMaxY = frame.GetMaxY();

    grid_element_width_ = frame.GridElementWidth();
    grid_element_height_ = frame.GridElementHeight();

    initial_computations = false;
  }

  mnId = nNextId++;

  grid_ = frame.GetGrid();

  SetPose(frame.GetPose());  
}

void KeyFrame::SetPose(const cv::Mat& input_Tcw) {
  std::unique_lock<std::mutex> lock(pose_mutex_);
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
  std::unique_lock<std::mutex> lock(pose_mutex_);
  return Tcw.clone();
}

cv::Mat KeyFrame::GetPoseInverse() {
  std::unique_lock<std::mutex> lock(pose_mutex_);
  return Twc.clone();
}

cv::Mat KeyFrame::GetCameraCenter() {
  std::unique_lock<std::mutex> lock(pose_mutex_);
  return Ow.clone();
}

cv::Mat KeyFrame::GetStereoCenter() {
  std::unique_lock<std::mutex> lock(pose_mutex_);
  return Cw.clone();
}

cv::Mat KeyFrame::GetRotation() {
  std::unique_lock<std::mutex> lock(pose_mutex_);
  return Tcw.rowRange(0,3).colRange(0,3).clone();
}

cv::Mat KeyFrame::GetTranslation() {
  std::unique_lock<std::mutex> lock(pose_mutex_);
  return Tcw.rowRange(0,3).col(3).clone();
}

void KeyFrame::ComputeBoW() {
  if (mBowVec.empty() || mFeatVec.empty()) {
    std::vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
    // Feature vector associates features with nodes in the 4th level (from leaves up)
    // We assume the vocabulary tree has 6 levels, change the 4 otherwise
    mpORBvocabulary->transform(vCurrentDesc,
                               mBowVec,
                               mFeatVec,
                               4);
  }
}

void KeyFrame::AddConnection(KeyFrame* pKF, const int weight) {
  {
    std::unique_lock<std::mutex> lock(connection_mutex_);
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
    std::unique_lock<std::mutex> lock(connection_mutex_);
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
  std::unique_lock<std::mutex> lock(connection_mutex_);
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
    std::unique_lock<std::mutex> lockMPs(feature_mutex_);
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
    std::unique_lock<std::mutex> lockCon(connection_mutex_);
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
  std::unique_lock<std::mutex> lock(connection_mutex_);
  std::set<KeyFrame*> s;
  for(std::map<KeyFrame*,int>::iterator mit = mConnectedKeyFrameWeights.begin();
                                        mit != mConnectedKeyFrameWeights.end();
                                        mit++) {
    s.insert(mit->first);
  }
  return s;
}

std::vector<KeyFrame*> KeyFrame::GetVectorCovisibleKeyFrames() {
  std::unique_lock<std::mutex> lock(connection_mutex_);
  return mvpOrderedConnectedKeyFrames;
}

std::vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(const int N) {
  std::unique_lock<std::mutex> lock(connection_mutex_);
  if ( static_cast<int>(mvpOrderedConnectedKeyFrames.size()) < N) {
    return mvpOrderedConnectedKeyFrames;
  } else {
    return std::vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(),
                                  mvpOrderedConnectedKeyFrames.begin() + N);
  }
}

std::vector<KeyFrame*> KeyFrame::GetCovisiblesByWeight(const int w) {
  std::unique_lock<std::mutex> lock(connection_mutex_);

  if(mvpOrderedConnectedKeyFrames.empty()) {
    return std::vector<KeyFrame*>();
  }

  std::vector<int>::iterator it = std::upper_bound(mvOrderedWeights.begin(),
                                                   mvOrderedWeights.end(),
                                                   w,
                                                   [](int a, int b){ return a > b; });
  if (it == mvOrderedWeights.end()) {
    return std::vector<KeyFrame*>();
  } else {
    int n = it - mvOrderedWeights.begin();
    return std::vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), 
                                  mvpOrderedConnectedKeyFrames.begin() + n);
  }
}

int KeyFrame::GetWeight(KeyFrame* pKF) {
  std::unique_lock<std::mutex> lock(connection_mutex_);
  if (mConnectedKeyFrameWeights.count(pKF)) {
    return mConnectedKeyFrameWeights[pKF];
  } else {
    return 0;
  }
}

void KeyFrame::AddChild(KeyFrame* pKF) {
  std::unique_lock<std::mutex> lock(connection_mutex_);
  mspChildrens.insert(pKF);
}

void KeyFrame::EraseChild(KeyFrame* pKF) {
  std::unique_lock<std::mutex> lock(connection_mutex_);
  mspChildrens.erase(pKF);
}

void KeyFrame::ChangeParent(KeyFrame* pKF) {
  std::unique_lock<std::mutex> lock(connection_mutex_);
  mpParent = pKF;
  pKF->AddChild(this);
}

std::set<KeyFrame*> KeyFrame::GetChilds() {
  std::unique_lock<std::mutex> lock(connection_mutex_);
  return mspChildrens;
}

KeyFrame* KeyFrame::GetParent() {
  std::unique_lock<std::mutex> lock(connection_mutex_);
  return mpParent;
}

bool KeyFrame::hasChild(KeyFrame* pKF) {
  std::unique_lock<std::mutex> lock(connection_mutex_);
  return mspChildrens.count(pKF);
}

void KeyFrame::AddLoopEdge(KeyFrame* pKF) {
  std::unique_lock<std::mutex> lock(connection_mutex_);
  mbNotErase = true;
  mspLoopEdges.insert(pKF);
}

std::set<KeyFrame*> KeyFrame::GetLoopEdges() {
  std::unique_lock<std::mutex> lock(connection_mutex_);
  return mspLoopEdges;
}

void KeyFrame::AddMapPoint(MapPoint* pMP, const size_t idx) {
  std::unique_lock<std::mutex> lock(feature_mutex_);
  mvpMapPoints[idx] = pMP;
}

void KeyFrame::EraseMapPointMatch(const size_t idx) {
  std::unique_lock<std::mutex> lock(feature_mutex_);
  mvpMapPoints[idx] = nullptr;
}

void KeyFrame::EraseMapPointMatch(MapPoint* pMP) {
  int idx = pMP->GetIndexInKeyFrame(this);
  if(idx >= 0) {
    EraseMapPointMatch(idx);
  }
}

void KeyFrame::ReplaceMapPointMatch(const size_t idx, MapPoint* pMP) {
  std::unique_lock<std::mutex> lock(feature_mutex_);
  mvpMapPoints[idx] = pMP;
}

std::set<MapPoint*> KeyFrame::GetMapPoints() {
  std::unique_lock<std::mutex> lock(feature_mutex_);
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
  std::unique_lock<std::mutex> lock(feature_mutex_);
  return mvpMapPoints;
}

int KeyFrame::TrackedMapPoints(const int minObs) {
  std::unique_lock<std::mutex> lock(feature_mutex_);

  int nPoints = 0;
  const bool bCheckObs = minObs > 0;
  for (int i = 0; i < N; ++i) {
    MapPoint* pMP = mvpMapPoints[i];
    if (pMP && !pMP->isBad()) {
      if (bCheckObs) {
        if (mvpMapPoints[i]->NumObservations() >= minObs) {
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
  std::unique_lock<std::mutex> lock(feature_mutex_);
  return mvpMapPoints[idx];
}

std::vector<size_t> KeyFrame::GetFeaturesInArea(const float x, 
                                                const float y, 
                                                const float r) const 
{
  std::vector<size_t> vIndices;
  vIndices.reserve(N);

  const int nMinCellX = std::max(0,
                                 static_cast<int>(std::floor((x-mnMinX-r) / grid_element_width_)));
  const int nMaxCellX = std::min(grid_cols-1, 
                                 static_cast<int>(std::ceil((x-mnMinX+r) / grid_element_width_)));
  const int nMinCellY = std::max(0,
                                 static_cast<int>(std::floor((y-mnMinY-r) / grid_element_height_)));
  const int nMaxCellY = std::min(grid_rows-1, 
                                 static_cast<int>(std::ceil((y-mnMinY+r) / grid_element_height_)));
  if (   nMaxCellX < 0 || nMinCellX >= grid_cols
      || nMaxCellY < 0 || nMinCellY >= grid_rows) {
    return vIndices;
  }    
      
  for (int ix = nMinCellX; ix <= nMaxCellX; ++ix) {
    for (int iy = nMinCellY; iy <= nMaxCellY; ++iy) {
      const std::vector<size_t>& vCell = grid_[ix][iy];
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

    std::unique_lock<std::mutex> lock(pose_mutex_);
    return Twc.rowRange(0,3).colRange(0,3) * x3Dc + Twc.rowRange(0,3).col(3);
  } else {
    return cv::Mat();
  }      
}

bool KeyFrame::IsInImage (const float x, const float y) const {
  return ( x >= mnMinX && x < mnMaxX && y >= mnMinY && y < mnMaxY);
}

void KeyFrame::SetNotErase() {
  std::unique_lock<std::mutex> lock(connection_mutex_);
  mbNotErase = true;
}

void KeyFrame::SetErase() {
  {
    std::unique_lock<std::mutex> lock(connection_mutex_);
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
    std::unique_lock<std::mutex> lock(connection_mutex_);
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
    std::unique_lock<std::mutex> lockCon(connection_mutex_);
    std::unique_lock<std::mutex> lockFeat(feature_mutex_);

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

bool KeyFrame::isBad() const {
  std::unique_lock<std::mutex> lock(connection_mutex_);
  return mbBad;
}

float KeyFrame::ComputeSceneMedianDepth(const int q) {
  std::vector<MapPoint*> vpMapPoints;
  cv::Mat Tcw_;
  {
    std::unique_lock<std::mutex> lock(feature_mutex_);
    std::unique_lock<std::mutex> lock2(pose_mutex_);
    vpMapPoints = mvpMapPoints;
    Tcw_ = Tcw.clone();
  }

  std::vector<float> vDepths;
  vDepths.reserve(N);
  cv::Mat Rcw2 = Tcw_.row(2).colRange(0,3);
  Rcw2 = Rcw2.t();
  float zcw = Tcw_.at<float>(2,3);
  for (int i = 0; i < N; ++i) {
    if (mvpMapPoints[i]) {
      MapPoint* pMP = mvpMapPoints[i];
      cv::Mat x3Dw = pMP->GetWorldPos();
      float z = Rcw2.dot(x3Dw)+zcw;
      vDepths.push_back(z);
    }
  }
  std::sort(vDepths.begin(),vDepths.end());
  return vDepths[(vDepths.size()-1)/q];
}

