#include "loop_closer.h"

#include "sim3solver.h"
#include "converter.h"
#include "optimizer.h"
#include "orb_matcher.h"

LoopCloser::LoopCloser(Map* pMap,
                       KeyframeDatabase* pDB,
                       OrbVocabulary* pVoc,
                       const bool bFixScale) 
      : mbResetRequested(false)
      , mbFinishRequested(false)
      , mbFinished(true)
      , mpMap(pMap)
      , mpKeyFrameDB(pDB)
      , mpORBVocabulary(pVoc)
      , mnCovisibilityConsistencyTh(3)
      , mpMatchedKF(NULL)
      , mLastLoopKFid(0)
      , mbRunningGBA(false)
      , mbFinishedGBA(true)
      , mbStopGBA(false)
      , mpThreadGBA(NULL)
      , mbFixScale(bFixScale)
      , mnFullBAIdx(0) {

}

void LoopCloser::SetTracker(Tracker* pTracker) {
  mpTracker = pTracker;
}

void LoopCloser::SetLocalMapper(LocalMapper* pLocalMapper) {
  mpLocalMapper = pLocalMapper;
}

void LoopCloser::Run() {
  mbFinished = false;
  while (true) {
    // Check if there are keyframes in the queue
    if (CheckNewKeyFrames()) {
      // Detect loop candidates and check covisibility consistency
      if (DetectLoop()) {
        // Compute similarity transformation [sR|t]
        // In the stereo/RGBD case s=1
        if (ComputeSim3()) {
          // Perform loop fusion and pose graph optimization
          CorrectLoop();
        }
      }
    }       
    ResetIfRequested();
    if (CheckFinish()) {
      break;
    }
    usleep(5000);
  }
  SetFinish();
}

void LoopCloser::InsertKeyFrame(KeyFrame *pKF) {
  std::unique_lock<std::mutex> lock(mMutexLoopQueue);
  if (pKF->mnId != 0) {
    mlpLoopKeyFrameQueue.push_back(pKF);
  }
}

void LoopCloser::RequestReset() {
  {
    std::unique_lock<std::mutex> lock(mMutexReset);
    mbResetRequested = true;
  }

  while (true) {
    {
      std::unique_lock<std::mutex> lock(mMutexReset);
      if (!mbResetRequested) {
        break;
      }
    }
    usleep(5000);
  }
}

void LoopCloser::RunGlobalBundleAdjustment(unsigned long nLoopKF) {
  std::cout << "Starting Global Bundle Adjustment\n";

  const int idx =  mnFullBAIdx;
  Optimizer::GlobalBundleAdjustemnt(mpMap,
                                    10,
                                    &mbStopGBA,
                                    nLoopKF,
                                    false);

  // Update all MapPoints and KeyFrames
  // Local Mapping was active during BA, that means that there might be new keyframes
  // not included in the Global BA and they are not consistent with the updated map.
  // We need to propagate the correction through the spanning tree
  {
    std::unique_lock<std::mutex> lock(mMutexGBA);
    if (idx != mnFullBAIdx) {
      return;
    }

    if (!mbStopGBA) {
      std::cout << "Global Bundle Adjustment finished\n";
      std::cout << "Updating map ...\n";

      mpLocalMapper->RequestStop();
      // Wait until Local Mapping has effectively stopped

      while (!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished()) {
        usleep(1000);
      }

      // Get Map Mutex
      std::unique_lock<std::mutex> lock(mpMap->mMutexMapUpdate);

      // Correct keyframes starting at map first keyframe
      std::list<KeyFrame*> lpKFtoCheck(mpMap->mvpKeyFrameOrigins.begin(),
                                       mpMap->mvpKeyFrameOrigins.end());

      while (!lpKFtoCheck.empty()) {
        KeyFrame* pKF = lpKFtoCheck.front();
        cv::Mat Twc = pKF->GetPoseInverse();
        const std::set<KeyFrame*> sChilds = pKF->GetChilds();
        
        for (std::set<KeyFrame*>::const_iterator sit = sChilds.begin();
                                                 sit != sChilds.end();
                                                 ++sit)
        {
          KeyFrame* pChild = *sit;
          if (pChild->mnBAGlobalForKF != nLoopKF) {
            cv::Mat Tchildc = pChild->GetPose() * Twc;
            pChild->mTcwGBA = Tchildc * pKF->mTcwGBA;
            pChild->mnBAGlobalForKF = nLoopKF;
          }
          lpKFtoCheck.push_back(pChild);
        }

        pKF->mTcwBefGBA = pKF->GetPose();
        pKF->SetPose(pKF->mTcwGBA);
        lpKFtoCheck.pop_front();
      }

      // Correct MapPoints
      const std::vector<MapPoint*> vpMPs = mpMap->GetAllMapPoints();

      for (size_t i = 0; i < vpMPs.size(); ++i) {
        MapPoint* pMP = vpMPs[i];

        if (pMP->isBad()) {
          continue;
        }

        if (pMP->mnBAGlobalForKF == nLoopKF) {
          // If optimized by Global BA, just update
          pMP->SetWorldPos(pMP->mPosGBA);
        } else {
          // Update according to the correction of its reference keyframe
          KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();

          if(pRefKF->mnBAGlobalForKF != nLoopKF) {
            continue;
          }

          // Map to non-corrected camera
          cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0,3).colRange(0,3);
          cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0,3).col(3);
          cv::Mat Xc = Rcw * pMP->GetWorldPos() + tcw;

          // Backproject using corrected camera
          cv::Mat Twc = pRefKF->GetPoseInverse();
          cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
          cv::Mat twc = Twc.rowRange(0,3).col(3);

          pMP->SetWorldPos(Rwc * Xc + twc);
        }
      }            

      mpMap->InformNewBigChange();

      mpLocalMapper->Release();

      std::cout << "Map updated!\n";
    }

    mbFinishedGBA = true;
    mbRunningGBA = false;
  }
}

bool LoopCloser::isRunningGBA() {
  std::unique_lock<std::mutex> lock(mMutexGBA);
  return mbRunningGBA
}

bool LoopCloser::isFinishedGBA() {
  std::unique_lock<std::mutex> lock(mMutexGBA);
  return mbFinishedGBA;
}

void LoopCloser::RequestFinish() {
  std::unique_lock<std::mutex> lock(mMutexFinish);
  mbFinishRequested = true;
}

bool LoopCloser::isFinished() {
  std::unique_lock<std::mutex> lock(mMutexFinish);
  return mbFinished;
}

bool LoopCloser::CheckNewKeyFrames() {
  std::unique_lock<std::mutex> lock(mMutexLoopQueue);
  return !mlpLoopKeyFrameQueue.empty();
}

bool LoopCloser::DetectLoop() {
  {
    std::unique_lock<std::mutex> lock(mMutexLoopQueue);
    mpCurrentKF = mlpLoopKeyFrameQueue.front();
    mlpLoopKeyFrameQueue.pop_front();
    // Avoid that a keyframe can be erased while it is being process by this thread
    mpCurrentKF->SetNotErase();
  }

  //If the map contains less than 10 KF or less than 10 KF have passed from last loop detection
  if (mpCurrentKF->mnId < mLastLoopKFid + 10) {
    mpKeyFrameDB->add(mpCurrentKF);
    mpCurrentKF->SetErase();
    return false;
  }

  // Compute reference BoW similarity score
  // This is the lowest score to a connected keyframe in the covisibility graph
  // We will impose loop candidates to have a higher similarity than this
  const vector<KeyFrame*> vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames();
  const DBoW2::BowVector& CurrentBowVec = mpCurrentKF->mBowVec;
  float minScore = 1;
  for (size_t i = 0; i < vpConnectedKeyFrames.size(); ++i) {
    KeyFrame* pKF = vpConnectedKeyFrames[i];
    if (pKF->isBad()) {
      continue;
    }
    const DBoW2::BowVector& BowVec = pKF->mBowVec;
    const float score = mpORBVocabulary->score(CurrentBowVec, BowVec);
    minScore = std::min(minScore, score);
  }

  // Query the database imposing the minimum score
  std::vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectLoopCandidates(mpCurrentKF, minScore);

  // If there are no loop candidates, just add new keyframe and return false
  if (vpCandidateKFs.empty()) {
    mpKeyFrameDB->add(mpCurrentKF);
    mvConsistentGroups.clear();
    mpCurrentKF->SetErase();
    return false;
  }

  // For each loop candidate check consistency with previous loop candidates
  // Each candidate expands a covisibility group (keyframes connected 
  // to the loop candidate in the covisibility graph)
  // A group is consistent with a previous group if they share at least a keyframe
  // We must detect a consistent loop in several consecutive keyframes to accept it
  mvpEnoughConsistentCandidates.clear();

  std::vector<ConsistentGroup> vCurrentConsistentGroups;
  std::vector<bool> vbConsistentGroup(mvConsistentGroups.size(),false); // TODO vector of bool :(
  
  for (size_t i = 0; i < vpCandidateKFs.size(); ++i) {
    KeyFrame* pCandidateKF = vpCandidateKFs[i];

    std::set<KeyFrame*> spCandidateGroup = pCandidateKF->GetConnectedKeyFrames();
    spCandidateGroup.insert(pCandidateKF);

    bool bEnoughConsistent = false;
    bool bConsistentForSomeGroup = false;
    for (size_t iG = 0; iG < mvConsistentGroups.size(); ++iG) {
      std::set<KeyFrame*> sPreviousGroup = mvConsistentGroups[iG].first;

      bool bConsistent = false;
      for (std::set<KeyFrame*>::iterator sit = spCandidateGroup.begin(); 
                                         sit != spCandidateGroup.end();
                                         ++sit)
      {
        if (sPreviousGroup.count(*sit)) {
          bConsistent = true;
          bConsistentForSomeGroup = true;
          break;
        }
      }

      if (bConsistent) {
        const int nPreviousConsistency = mvConsistentGroups[iG].second;
        const int nCurrentConsistency = nPreviousConsistency + 1;
        
        if (!vbConsistentGroup[iG]) {
          ConsistentGroup cg = std::make_pair(spCandidateGroup, nCurrentConsistency);
          vCurrentConsistentGroups.push_back(cg);
          vbConsistentGroup[iG] = true; //this avoid to include the same group more than once
        }
        if (nCurrentConsistency >= mnCovisibilityConsistencyTh && !bEnoughConsistent) {
          mvpEnoughConsistentCandidates.push_back(pCandidateKF);
          bEnoughConsistent = true; //this avoid to insert the same candidate more than once
        }
      }
    }

    // If the group is not consistent with any previous group insert with consistency counter set to zero
    if (!bConsistentForSomeGroup) {
      ConsistentGroup cg = std::make_pair(spCandidateGroup,0);
      vCurrentConsistentGroups.push_back(cg);
    }
  }

  // Update Covisibility Consistent Groups
  mvConsistentGroups = vCurrentConsistentGroups;

  // Add Current Keyframe to database
  mpKeyFrameDB->add(mpCurrentKF);

  if (mvpEnoughConsistentCandidates.empty()) {
    mpCurrentKF->SetErase();
    return false;
  } else {
    return true;
  }
  
  // TODO does this ever get reached??
  mpCurrentKF->SetErase();
  return false;
}

bool LoopCloser::ComputeSim3() {
  //TODO
}




