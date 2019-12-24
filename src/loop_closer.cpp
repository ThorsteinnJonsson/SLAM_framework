#include "loop_closer.h"

#include "solvers/sim3solver.h"
#include "util/converter.h"
#include "optimizer.h"
#include "orb_matcher.h"

LoopCloser::LoopCloser(const std::shared_ptr<Map>& map,
                       const std::shared_ptr<KeyframeDatabase>& keyframe_db,
                       const std::shared_ptr<OrbVocabulary>& orb_vocabulary,
                       const bool fix_scale) 
      : mbResetRequested(false)
      , mbFinishRequested(false)
      , mbFinished(true)
      , map_(map)
      , keyframe_db_(keyframe_db)
      , orb_vocabulary_(orb_vocabulary)
      , covisibility_consistency_threshold(3)
      , mpMatchedKF(nullptr)
      , mLastLoopKFid(0)
      , mbRunningGBA(false)
      , mbFinishedGBA(true)
      , mbStopGBA(false)
      , global_bundle_adjustment_thread(nullptr)
      , mbFixScale(fix_scale)
      , mnFullBAIdx(0) {

}

void LoopCloser::SetLocalMapper(const std::shared_ptr<LocalMapper>& local_mapper) { 
  local_mapper_ = local_mapper; 
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

void LoopCloser::InsertKeyFrame(KeyFrame* keyframe) {
  std::unique_lock<std::mutex> lock(mMutexLoopQueue);
  if (keyframe->mnId != 0) {
    mlpLoopKeyFrameQueue.push_back(keyframe);
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
  Optimizer::GlobalBundleAdjustemnt(map_,
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

      local_mapper_->RequestStop();
      // Wait until Local Mapping has effectively stopped

      while (!local_mapper_->isStopped() && !local_mapper_->isFinished()) {
        usleep(1000);
      }

      // Get Map Mutex
      std::unique_lock<std::mutex> lock(map_->mMutexMapUpdate);

      // Correct keyframes starting at map first keyframe
      std::list<KeyFrame*> lpKFtoCheck(map_->mvpKeyFrameOrigins.begin(),
                                       map_->mvpKeyFrameOrigins.end());

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
      const std::vector<MapPoint*> vpMPs = map_->GetAllMapPoints();

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

      map_->InformNewBigChange();

      local_mapper_->Release();

      std::cout << "Map updated!\n";
    }

    mbFinishedGBA = true;
    mbRunningGBA = false;
  }
}

bool LoopCloser::isRunningGBA() {
  std::unique_lock<std::mutex> lock(mMutexGBA);
  return mbRunningGBA;
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
    keyframe_db_->add(mpCurrentKF);
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
    const float score = orb_vocabulary_->score(CurrentBowVec, BowVec);
    minScore = std::min(minScore, score);
  }

  // Query the database imposing the minimum score
  std::vector<KeyFrame*> vpCandidateKFs = keyframe_db_->DetectLoopCandidates(mpCurrentKF, minScore);

  // If there are no loop candidates, just add new keyframe and return false
  if (vpCandidateKFs.empty()) {
    keyframe_db_->add(mpCurrentKF);
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
        if (nCurrentConsistency >= covisibility_consistency_threshold && !bEnoughConsistent) {
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
  keyframe_db_->add(mpCurrentKF);

  if (mvpEnoughConsistentCandidates.empty()) {
    mpCurrentKF->SetErase();
    return false;
  } else {
    return true;
  } 
}

bool LoopCloser::ComputeSim3() {
  // For each consistent loop candidate we try to compute a Sim3
  const int nInitialCandidates = mvpEnoughConsistentCandidates.size();

  // We compute first ORB matches for each candidate
  // If enough matches are found, we setup a Sim3Solver
  OrbMatcher matcher(0.75, true);

  std::vector<Sim3Solver*> vpSim3Solvers;
  vpSim3Solvers.resize(nInitialCandidates);

  std::vector<std::vector<MapPoint*>> vvpMapPointMatches;
  vvpMapPointMatches.resize(nInitialCandidates);

  std::vector<bool> vbDiscarded; //TODO vector of bool
  vbDiscarded.resize(nInitialCandidates);

  int nCandidates = 0; //candidates with enough matches

  for (int i = 0; i < nInitialCandidates; ++i) {
    KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

    // Avoid that local mapping erase it while it is being processed in this thread
    pKF->SetNotErase();

    if (pKF->isBad()) {
      vbDiscarded[i] = true;
      continue;
    }

    int nmatches = matcher.SearchByBoW(mpCurrentKF,
                                       pKF,
                                       vvpMapPointMatches[i]);

    if (nmatches < 20) {
      vbDiscarded[i] = true;
      continue;
    } else {
      Sim3Solver* pSolver = new Sim3Solver(mpCurrentKF,
                                           pKF,
                                           vvpMapPointMatches[i],
                                           mbFixScale);
      pSolver->SetRansacParameters(0.99, 20, 300);
      vpSim3Solvers[i] = pSolver;
    }
    ++nCandidates;
  }

  // Perform alternatively RANSAC iterations for each candidate
  // until one is succesful or all fail
  bool bMatch = false;
  while (nCandidates > 0 && !bMatch) {
    for (int i = 0; i < nInitialCandidates; ++i) {
      if (vbDiscarded[i]) {
        continue;
      }

      KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

      // Perform 5 Ransac Iterations
      std::vector<bool> vbInliers; //TODO vector of bool
      int nInliers;
      bool bNoMore;
      Sim3Solver* pSolver = vpSim3Solvers[i];
      cv::Mat Scm  = pSolver->iterate(5,
                                      bNoMore,
                                      vbInliers,
                                      nInliers);

      // If Ransac reachs max. iterations discard keyframe
      if (bNoMore) {
        vbDiscarded[i] = true;
        --nCandidates;
      }

      // If RANSAC returns a Sim3, perform a guided matching and optimize with all correspondences
      if(!Scm.empty()) {
        std::vector<MapPoint*> vpMapPointMatches(vvpMapPointMatches[i].size(), nullptr);
        for (size_t j = 0; j < vbInliers.size(); ++j) {
          if (vbInliers[j]) {
            vpMapPointMatches[j] = vvpMapPointMatches[i][j];
          }
        }

        cv::Mat R = pSolver->GetEstimatedRotation();
        cv::Mat t = pSolver->GetEstimatedTranslation();
        const float s = pSolver->GetEstimatedScale();
        matcher.SearchBySim3(mpCurrentKF,
                             pKF,
                             vpMapPointMatches,
                             s,
                             R,
                             t,
                             7.5);

        g2o::Sim3 gScm(Converter::toMatrix3d(R),
                       Converter::toVector3d(t),
                       s);
        const int nInliers = Optimizer::OptimizeSim3(mpCurrentKF, 
                                                     pKF, 
                                                     vpMapPointMatches, 
                                                     gScm, 
                                                     10, 
                                                     mbFixScale);

        // If optimization is succesful stop ransacs and continue
        if (nInliers >= 20) {
          bMatch = true;
          mpMatchedKF = pKF;
          g2o::Sim3 gSmw(Converter::toMatrix3d(pKF->GetRotation()),
                         Converter::toVector3d(pKF->GetTranslation()),
                         1.0);
          mg2oScw = gScm * gSmw;
          mScw = Converter::toCvMat(mg2oScw);

          mvpCurrentMatchedPoints = vpMapPointMatches;
          break;
        }
      }
    }
  }

  if (!bMatch) {
    for (int i = 0; i < nInitialCandidates; ++i){
      mvpEnoughConsistentCandidates[i]->SetErase();
    }
    mpCurrentKF->SetErase();
    return false;
  }

  // Retrieve MapPoints seen in Loop Keyframe and neighbors
  std::vector<KeyFrame*> vpLoopConnectedKFs = mpMatchedKF->GetVectorCovisibleKeyFrames();
  vpLoopConnectedKFs.push_back(mpMatchedKF);
  mvpLoopMapPoints.clear();
  for (std::vector<KeyFrame*>::iterator vit = vpLoopConnectedKFs.begin(); 
                                        vit != vpLoopConnectedKFs.end(); 
                                        ++vit)
  {
    KeyFrame* pKF = *vit;
    std::vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();
    for (size_t i = 0; i < vpMapPoints.size(); ++i) {
      MapPoint* pMP = vpMapPoints[i];
      if (pMP) {
        if (!pMP->isBad() && pMP->mnLoopPointForKF != mpCurrentKF->mnId) {
          mvpLoopMapPoints.push_back(pMP);
          pMP->mnLoopPointForKF = mpCurrentKF->mnId;
        }
      }
    }
  }

  // Find more matches projecting with the computed Sim3
  matcher.SearchByProjection(mpCurrentKF, 
                             mScw, 
                             mvpLoopMapPoints, 
                             mvpCurrentMatchedPoints,
                             10);

  // If enough matches accept Loop
  int nTotalMatches = 0;
  for (size_t i = 0; i < mvpCurrentMatchedPoints.size(); ++i) {
    if (mvpCurrentMatchedPoints[i]) {
      ++nTotalMatches;
    }
  }

  if (nTotalMatches >= 40) {
    for (int i = 0; i < nInitialCandidates; ++i) {
      if (mvpEnoughConsistentCandidates[i] != mpMatchedKF) {
        mvpEnoughConsistentCandidates[i]->SetErase();
      }
    }
    return true;
  } else {
    for (int i = 0; i < nInitialCandidates; ++i) {
      mvpEnoughConsistentCandidates[i]->SetErase();
    }
    mpCurrentKF->SetErase();
    return false;
  }
}

void LoopCloser::SearchAndFuse(const KeyFrameAndPose& corrected_poses_map) {
  
  OrbMatcher matcher(0.8);

  for (auto mit = corrected_poses_map.begin(); 
            mit != corrected_poses_map.end();
            ++mit) {
    KeyFrame* pKF = mit->first;

    g2o::Sim3 g2oScw = mit->second;
    cv::Mat cvScw = Converter::toCvMat(g2oScw);

    std::vector<MapPoint*> vpReplacePoints(mvpLoopMapPoints.size(), nullptr);
    matcher.Fuse(pKF,
                 cvScw,
                 mvpLoopMapPoints,
                 4,
                 vpReplacePoints);

    // Get Map Mutex
    std::unique_lock<std::mutex> lock(map_->mMutexMapUpdate);
    const int nLP = mvpLoopMapPoints.size();
    for (int i = 0; i < nLP; ++i) {
      MapPoint* pRep = vpReplacePoints[i];
      if (pRep) {
        pRep->Replace(mvpLoopMapPoints[i]);
      }
    }
  }
}

void LoopCloser::CorrectLoop() {
  std::cout << "Loop detected!\n";

  // Send a stop signal to Local Mapping
  // Avoid new keyframes are inserted while correcting the loop
  local_mapper_->RequestStop();

  // If a Global Bundle Adjustment is running, abort it
  if (isRunningGBA()) {
    std::unique_lock<std::mutex> lock(mMutexGBA);
    mbStopGBA = true;

    ++mnFullBAIdx;

    if (global_bundle_adjustment_thread) {
      global_bundle_adjustment_thread->detach();
      delete global_bundle_adjustment_thread;
    }
  }

  // Wait until Local Mapping has effectively stopped
  while(!local_mapper_->isStopped()) {
    usleep(1000);
  }

  // Ensure current keyframe is updated
  mpCurrentKF->UpdateConnections();

  // Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
  mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
  mvpCurrentConnectedKFs.push_back(mpCurrentKF);

  KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
  CorrectedSim3[mpCurrentKF] = mg2oScw;
  cv::Mat Twc = mpCurrentKF->GetPoseInverse();

  {
    // Get Map Mutex
    std::unique_lock<std::mutex> lock(map_->mMutexMapUpdate);

    for (std::vector<KeyFrame*>::iterator vit = mvpCurrentConnectedKFs.begin(); 
                                          vit != mvpCurrentConnectedKFs.end();
                                          ++vit) {
      KeyFrame* pKFi = *vit;

      cv::Mat Tiw = pKFi->GetPose();

      if (pKFi != mpCurrentKF) {
        cv::Mat Tic = Tiw * Twc;
        cv::Mat Ric = Tic.rowRange(0,3).colRange(0,3);
        cv::Mat tic = Tic.rowRange(0,3).col(3);
        g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric),
                         Converter::toVector3d(tic),
                         1.0);
        g2o::Sim3 g2oCorrectedSiw = g2oSic * mg2oScw;
        //Pose corrected with the Sim3 of the loop closure
        CorrectedSim3[pKFi] = g2oCorrectedSiw;
      }

      cv::Mat Riw = Tiw.rowRange(0,3).colRange(0,3);
      cv::Mat tiw = Tiw.rowRange(0,3).col(3);
      g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw),
                       Converter::toVector3d(tiw),
                       1.0);
      //Pose without correction
      NonCorrectedSim3[pKFi] = g2oSiw;
    }

    // Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
    for (KeyFrameAndPose::iterator mit = CorrectedSim3.begin(); 
                                   mit != CorrectedSim3.end(); 
                                   ++mit) {
      KeyFrame* pKFi = mit->first;
      g2o::Sim3 g2oCorrectedSiw = mit->second;
      g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();

      g2o::Sim3 g2oSiw = NonCorrectedSim3[pKFi];

      std::vector<MapPoint*> vpMPsi = pKFi->GetMapPointMatches();
      for (size_t iMP = 0; iMP < vpMPsi.size(); ++iMP) {
        MapPoint* pMPi = vpMPsi[iMP];
        if(!pMPi || pMPi->isBad()) {
          continue;
        }
        if(pMPi->mnCorrectedByKF == mpCurrentKF->mnId) {
          continue;
        }

        // Project with non-corrected pose and project back with corrected pose
        cv::Mat P3Dw = pMPi->GetWorldPos();
        Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
        Eigen::Matrix<double,3,1> eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw));

        cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
        pMPi->SetWorldPos(cvCorrectedP3Dw);
        pMPi->mnCorrectedByKF = mpCurrentKF->mnId;
        pMPi->mnCorrectedReference = pKFi->mnId;
        pMPi->UpdateNormalAndDepth();
      }

      // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
      Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
      Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
      double s = g2oCorrectedSiw.scale();

      eigt *=(1./s); //[R t/s;0 1]

      cv::Mat correctedTiw = Converter::toCvSE3(eigR,eigt);

      pKFi->SetPose(correctedTiw);

      // Make sure connections are updated
      pKFi->UpdateConnections();
    }

    // Start Loop Fusion
    // Update matched map points and replace if duplicated
    for (size_t i = 0; i < mvpCurrentMatchedPoints.size(); ++i) {
      if (mvpCurrentMatchedPoints[i]) {
          MapPoint* pLoopMP = mvpCurrentMatchedPoints[i];
          MapPoint* pCurMP = mpCurrentKF->GetMapPoint(i);
          if (pCurMP) {
            pCurMP->Replace(pLoopMP);
          } else {
            mpCurrentKF->AddMapPoint(pLoopMP, i);
            pLoopMP->AddObservation(mpCurrentKF,i);
            pLoopMP->ComputeDistinctiveDescriptors();
          }
      }
    }
  } // mutex scope

  // Project MapPoints observed in the neighborhood of the loop keyframe
  // into the current keyframe and neighbors using corrected poses.
  // Fuse duplications.
  SearchAndFuse(CorrectedSim3);


  // After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
  std::map<KeyFrame*, std::set<KeyFrame*>> LoopConnections;
  for (std::vector<KeyFrame*>::iterator vit = mvpCurrentConnectedKFs.begin(); 
                                        vit != mvpCurrentConnectedKFs.end(); 
                                        ++vit) {
    KeyFrame* pKFi = *vit;
    std::vector<KeyFrame*> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();

    // Update connections. Detect new links.
    pKFi->UpdateConnections();
    LoopConnections[pKFi] = pKFi->GetConnectedKeyFrames();
    for (std::vector<KeyFrame*>::iterator vit_prev = vpPreviousNeighbors.begin(); 
                                          vit_prev != vpPreviousNeighbors.end(); 
                                          ++vit_prev) {
      LoopConnections[pKFi].erase(*vit_prev);
    }
    for (std::vector<KeyFrame*>::iterator vit = mvpCurrentConnectedKFs.begin(); 
                                          vit != mvpCurrentConnectedKFs.end(); 
                                          ++vit) {
        LoopConnections[pKFi].erase(*vit);
    }
  }

  // Optimize graph
  Optimizer::OptimizeEssentialGraph(map_, 
                                    mpMatchedKF, 
                                    mpCurrentKF, 
                                    NonCorrectedSim3, 
                                    CorrectedSim3, 
                                    LoopConnections, 
                                    mbFixScale);

  map_->InformNewBigChange();

  // Add loop edge
  mpMatchedKF->AddLoopEdge(mpCurrentKF);
  mpCurrentKF->AddLoopEdge(mpMatchedKF);

  // Launch a new thread to perform Global Bundle Adjustment
  mbRunningGBA = true;
  mbFinishedGBA = false;
  mbStopGBA = false;
  global_bundle_adjustment_thread  = new std::thread(&LoopCloser::RunGlobalBundleAdjustment, 
                                                     this, 
                                                     mpCurrentKF->mnId);

  // Loop closed. Release Local Mapping.
  local_mapper_->Release();    
  mLastLoopKFid = mpCurrentKF->mnId;   
}


void LoopCloser::ResetIfRequested() {
  std::unique_lock<std::mutex> lock(mMutexReset);
  if (mbResetRequested) {
    mlpLoopKeyFrameQueue.clear();
    mLastLoopKFid = 0;
    mbResetRequested = false;
  }
}

bool LoopCloser::CheckFinish() {
  std::unique_lock<std::mutex> lock(mMutexFinish);
  return mbFinishRequested;
}

void LoopCloser::SetFinish() {
  std::unique_lock<std::mutex> lock(mMutexFinish);
  mbFinished = true;
}
