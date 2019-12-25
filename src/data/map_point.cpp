#include "map_point.h"

#include "orb_features/orb_matcher.h"

long unsigned int MapPoint::nNextId = 0;
std::mutex MapPoint::mGlobalMutex;

MapPoint::MapPoint(const cv::Mat& position, 
                   KeyFrame* pRefKF, 
                   const std::shared_ptr<Map>& pMap)
      : mnFirstKFid(pRefKF->mnId)
      , mnFirstFrame(pRefKF->mnFrameId)
      , nObs(0)
      , mnTrackReferenceForFrame(0)
      , mnLastFrameSeen(0)
      , mnBALocalForKF(0)
      , mnFuseCandidateForKF(0)
      , mnLoopPointForKF(0)
      , mnCorrectedByKF(0)
      , mnCorrectedReference(0)
      , mnBAGlobalForKF(0)
      , mpRefKF(pRefKF)
      , mnVisible(1)
      , mnFound(1)
      , mbBad(false)
      , mpReplaced(nullptr)
      , mfMinDistance(0)
      , mfMaxDistance(0)
      , mpMap(pMap) {
  position.copyTo(world_position_);
  mNormalVector = cv::Mat::zeros(3,1,CV_32F);

  // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
  std::unique_lock<std::mutex> lock(mpMap->mMutexPointCreation);
  mnId = nNextId++;
}

MapPoint::MapPoint(const cv::Mat& position, 
                   const std::shared_ptr<Map>& pMap, 
                   Frame* pFrame, 
                   const int &idxF)
      : mnFirstKFid(-1)
      , mnFirstFrame(pFrame->mnId)
      , nObs(0)
      , mnTrackReferenceForFrame(0)
      , mnLastFrameSeen(0)
      , mnBALocalForKF(0)
      , mnFuseCandidateForKF(0)
      , mnLoopPointForKF(0)
      , mnCorrectedByKF(0)
      , mnCorrectedReference(0)
      , mnBAGlobalForKF(0)
      , mpRefKF(nullptr)
      , mnVisible(1)
      , mnFound(1)
      , mbBad(false)
      , mpReplaced(nullptr)
      , mpMap(pMap) {
  position.copyTo(world_position_);
  cv::Mat Ow = pFrame->GetCameraCenter();
  mNormalVector = world_position_ - Ow;
  mNormalVector = mNormalVector/cv::norm(mNormalVector);

  cv::Mat PC = position - Ow;
  const float dist = cv::norm(PC);
  const int level = pFrame->mvKeysUn[idxF].octave;
  const float levelScaleFactor =  pFrame->mvScaleFactors[level];
  const int nLevels = pFrame->mnScaleLevels;

  mfMaxDistance = dist*levelScaleFactor;
  mfMinDistance = mfMaxDistance/pFrame->mvScaleFactors[nLevels-1];

  pFrame->mDescriptors.row(idxF).copyTo(mDescriptor);

  // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
  std::unique_lock<std::mutex> lock(mpMap->mMutexPointCreation);
  mnId = nNextId++;
}

void MapPoint::SetWorldPos(const cv::Mat& position) {
  std::unique_lock<std::mutex> lock1(mGlobalMutex);
  std::unique_lock<std::mutex> lock2(mMutexPos);
  position.copyTo(world_position_);
}

cv::Mat MapPoint::GetWorldPos() {
  std::unique_lock<std::mutex> lock(mMutexPos);
  return world_position_.clone();
}

cv::Mat MapPoint::GetNormal() {
  std::unique_lock<std::mutex> lock(mMutexPos);
  return mNormalVector.clone();
}

KeyFrame* MapPoint::GetReferenceKeyFrame() {
  std::unique_lock<std::mutex> lock(mMutexFeatures);
  return mpRefKF;
}

std::map<KeyFrame*,size_t> MapPoint::GetObservations() {
  std::unique_lock<std::mutex> lock(mMutexFeatures);
  return mObservations;
}

// TODO chance to NumObservations
int MapPoint::Observations() {
  std::unique_lock<std::mutex> lock(mMutexFeatures);
  return nObs;
}

void MapPoint::AddObservation(KeyFrame* pKF, size_t idx) {
  std::unique_lock<std::mutex> lock(mMutexFeatures);
  if (mObservations.count(pKF)) {
    return;
  }
  mObservations[pKF] = idx;
  if (pKF->mvuRight[idx] >= 0) {
    nObs += 2;
  } else {
    nObs += 1;
  }
}

void MapPoint::EraseObservation(KeyFrame* pKF) {
  bool bBad = false;
  { 
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    if (mObservations.count(pKF)) {
      int idx = mObservations[pKF];
      if (pKF->mvuRight[idx] >= 0) {
        nObs -= 2;
      } else {
        nObs -= 1;            
      }
      mObservations.erase(pKF);

      if (mpRefKF==pKF) {
        mpRefKF = mObservations.begin()->first;
      }

      // If only 2 observations or less, discard point
      if (nObs <= 2) {
        bBad = true;
      }
    }
  }

  if(bBad) {
    SetBadFlag();
  }
}

int MapPoint::GetIndexInKeyFrame(KeyFrame* pKF) {
  std::unique_lock<std::mutex> lock(mMutexFeatures);
  if(mObservations.count(pKF)) {
    return mObservations[pKF];
  } else {
    return -1;
  }
}

bool MapPoint::IsInKeyFrame(KeyFrame* pKF) {
  std::unique_lock<std::mutex> lock(mMutexFeatures);
  return mObservations.count(pKF);
}

void MapPoint::SetBadFlag() {
  std::map<KeyFrame*,size_t> obs;

  {
    std::unique_lock<std::mutex> lock1(mMutexFeatures);
    std::unique_lock<std::mutex> lock2(mMutexPos);
    mbBad = true;
    obs = mObservations;
    mObservations.clear();
  }
  for(std::map<KeyFrame*,size_t>::iterator mit = obs.begin(); 
                                           mit != obs.end(); 
                                           ++mit) {
    KeyFrame* pKF = mit->first;
    pKF->EraseMapPointMatch(mit->second);
  }
  mpMap->EraseMapPoint(this);
}

bool MapPoint::isBad() {
  std::unique_lock<std::mutex> lock1(mMutexFeatures);
  std::unique_lock<std::mutex> lock2(mMutexPos);
  return mbBad;
}

void MapPoint::Replace(MapPoint* pMP) {
  if (pMP->mnId == this->mnId) {
    return;
  }

  int nvisible, nfound;
  std::map<KeyFrame*,size_t> observations;
  
  {
    std::unique_lock<std::mutex> lock1(mMutexFeatures);
    std::unique_lock<std::mutex> lock2(mMutexPos);
    mbBad = true;
    observations = mObservations;
    mObservations.clear();
    nvisible = mnVisible;
    nfound = mnFound;
    mpReplaced = pMP;
  }

  for (const auto& observation : observations) {
    // Replace measurement in keyframe
    KeyFrame* keyframe = observation.first;
    const size_t index = observation.second;

    if (!pMP->IsInKeyFrame(keyframe)) {
      keyframe->ReplaceMapPointMatch(index, pMP);
      pMP->AddObservation(keyframe, index);
    } else {
      keyframe->EraseMapPointMatch(index);
    }
  } 
  pMP->IncreaseFound(nfound);
  pMP->IncreaseVisible(nvisible);
  pMP->ComputeDistinctiveDescriptors();

  mpMap->EraseMapPoint(this);
}

MapPoint* MapPoint::GetReplaced() {
  std::unique_lock<std::mutex> lock1(mMutexFeatures);
  std::unique_lock<std::mutex> lock2(mMutexPos);
  return mpReplaced;
}

void MapPoint::IncreaseVisible(int n) {
  std::unique_lock<std::mutex> lock(mMutexFeatures);
  mnVisible += n;
}

void MapPoint::IncreaseFound(int n) {
  std::unique_lock<std::mutex> lock(mMutexFeatures);
  mnFound += n;
}

float MapPoint::GetFoundRatio() {
  std::unique_lock<std::mutex> lock(mMutexFeatures);
  return static_cast<float>(mnFound) / mnVisible;
}

void MapPoint::ComputeDistinctiveDescriptors() {
  // Retrieve all observed descriptors
  std::vector<cv::Mat> vDescriptors;
  
  std::map<KeyFrame*, size_t> observations;

  {
    std::unique_lock<std::mutex> lock1(mMutexFeatures);
    if (mbBad) {
      return;
    }
    observations = mObservations;
  }
  
  if (observations.empty()) {
    return;
  }

  vDescriptors.reserve(observations.size());
  for (std::map<KeyFrame*, size_t>::iterator mit = observations.begin(); 
                                             mit != observations.end(); 
                                             ++mit) {
    KeyFrame* pKF = mit->first;
    if (!pKF->isBad()) {
      vDescriptors.push_back(pKF->mDescriptors.row(mit->second));
    }      
  }
  if (vDescriptors.empty()) {
    return;
  }

  // Compute distances between them
  const size_t N = vDescriptors.size();
  float Distances[N][N]; // TODO some better way than this using vectors or something
  for (size_t i = 0; i < N; i++) {
    Distances[i][i] = 0;
    for (size_t j = i + 1; j < N; j++) {
        int distij = OrbMatcher::DescriptorDistance(vDescriptors[i], vDescriptors[j]);
        Distances[i][j] = distij;
        Distances[j][i] = distij;
    }
  }

  // Take the descriptor with least median distance to the rest
  int BestMedian = INT_MAX;
  int BestIdx = 0;
  for (size_t i = 0; i < N; i++) {
    std::vector<int> vDists(Distances[i], Distances[i] + N);
    std::sort(vDists.begin(), vDists.end());
    int median = vDists[0.5 * (N - 1)];

    if (median < BestMedian) {
      BestMedian = median;
      BestIdx = i;
    }
  }

  {
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mDescriptor = vDescriptors[BestIdx].clone();
  }
}

cv::Mat MapPoint::GetDescriptor() {
  std::unique_lock<std::mutex> lock(mMutexFeatures);
  return mDescriptor.clone();
}

void MapPoint::UpdateNormalAndDepth() {
  std::map<KeyFrame*, size_t> observations;
  KeyFrame* pRefKF;
  cv::Mat Pos;
  {
    std::unique_lock<std::mutex> lock1(mMutexFeatures);
    std::unique_lock<std::mutex> lock2(mMutexPos);
    if (mbBad) {
      return;
    }
    observations = mObservations;
    pRefKF = mpRefKF;
    Pos = world_position_.clone();
  }

  if (observations.empty()) {
    return;
  }

  cv::Mat normal = cv::Mat::zeros(3, 1, CV_32F);
  int n = 0;
  for (std::map<KeyFrame*, size_t>::iterator mit = observations.begin(); 
       mit != observations.end(); 
       mit++) 
  {
    KeyFrame* pKF = mit->first;
    cv::Mat Owi = pKF->GetCameraCenter();
    cv::Mat normali = world_position_ - Owi;
    normal = normal + normali / cv::norm(normali);
    ++n;
  }

  cv::Mat PC = Pos - pRefKF->GetCameraCenter();
  const float dist = cv::norm(PC);
  const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;
  const float levelScaleFactor = pRefKF->mvScaleFactors[level];
  const int nLevels = pRefKF->mnScaleLevels;

  {
    std::unique_lock<std::mutex> lock3(mMutexPos);
    mfMaxDistance = dist * levelScaleFactor;
    mfMinDistance = mfMaxDistance / pRefKF->mvScaleFactors[nLevels - 1];
    mNormalVector = normal / n;
  }
}

float MapPoint::GetMinDistanceInvariance() {
  std::unique_lock<std::mutex> lock(mMutexPos);
  return 0.8f * mfMinDistance;
  // TODO where does this number come from??
}

float MapPoint::GetMaxDistanceInvariance() {
  std::unique_lock<std::mutex> lock(mMutexPos);
  return 1.2f * mfMaxDistance;
  // TODO where does this number come from??
}

int MapPoint::PredictScale(const float currentDist, KeyFrame* pKF) {
  float ratio;
  {
    std::unique_lock<std::mutex> lock(mMutexPos);
    ratio = mfMaxDistance / currentDist;
  }

  int nScale = ceil( log(ratio) / pKF->mfLogScaleFactor );
  if (nScale < 0) {
    nScale = 0;
  } else if (nScale >= pKF->mnScaleLevels) {
    nScale = pKF->mnScaleLevels - 1;
  }
  return nScale;
}

int MapPoint::PredictScale(const float currentDist, Frame* pF) {
  float ratio;
  {
    std::unique_lock<std::mutex> lock(mMutexPos);
    ratio = mfMaxDistance / currentDist;
  }

  int nScale = ceil(log(ratio) / pF->mfLogScaleFactor);
  if (nScale < 0) {
    nScale = 0;
  } else if (nScale >= pF->mnScaleLevels) {
    nScale = pF->mnScaleLevels - 1;
  }
  return nScale;
}
