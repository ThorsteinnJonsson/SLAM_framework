#include "local_mapper.h"

#include "orb_features/orb_matcher.h"
#include "optimizer/optimizer.h"

LocalMapper::LocalMapper(const std::shared_ptr<Map>& map, 
                         const bool is_monocular)
    : is_monocular_(is_monocular)
    , mbResetRequested(false)
    , mbFinishRequested(false)
    , mbFinished(true)
    , map_(map)
    , mbAbortBA(false)
    , mbStopped(false)
    , mbStopRequested(false)
    , mbNotStop(false)
    , mbAcceptKeyFrames(true) {

}

void LocalMapper::SetLoopCloser(const std::shared_ptr<LoopCloser>& loop_closer) {
  loop_closer_ = loop_closer;
}

void LocalMapper::Run() {
  mbFinished = false;
  while (true) {
    // Tracking will see that Local Mapping is busy
    SetAcceptKeyFrames(false);

    // Check if there are keyframes in the queue
    const bool has_new_kfs = CheckNewKeyFrames();
    if (has_new_kfs) {
      // BoW conversion and insertion in Map
      ProcessNewKeyFrame();

      // Check recent MapPoints
      MapPointCulling();

      // Triangulate new MapPoints
      CreateNewMapPoints();

      if (!CheckNewKeyFrames()) {
        // Find more matches in neighbor keyframes and fuse point duplications
        SearchInNeighbors();
      }

      mbAbortBA = false;
      if (!CheckNewKeyFrames() && !stopRequested()) {
        // Local BA
        if (map_->KeyFramesInMap() > 2) {
          Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame, &mbAbortBA, map_);
        }

        // Check redundant local Keyframes
        KeyFrameCulling();
      }

      loop_closer_->InsertKeyFrame(mpCurrentKeyFrame);

    } else if (Stop()){
      // Safe area to stop
      while (isStopped() && !CheckFinish()) {
        usleep(3000);
      }
      if(CheckFinish()) {
        break;
      }
    }

    ResetIfRequested();

    // Tracking will see that Local Mapping is not busy
    SetAcceptKeyFrames(true);

    if(CheckFinish()) {
      break;
    }

    usleep(3000);
  }

  SetFinish();
}

void LocalMapper::InsertKeyFrame(KeyFrame* keyframe) {
  std::unique_lock<std::mutex> lock(mMutexNewKFs);
  mlNewKeyFrames.push_back(keyframe);
  mbAbortBA = true;
}

void LocalMapper::RequestStop(){
  std::unique_lock<std::mutex> lock(mMutexStop);
  mbStopRequested = true;
  std::unique_lock<std::mutex> lock2(mMutexNewKFs);
  mbAbortBA = true;
}

void LocalMapper::RequestReset(){
  {
    std::unique_lock<std::mutex> lock(mMutexReset);
    mbResetRequested = true;
  }
  while (true) {
    {
      std::unique_lock<std::mutex> lock2(mMutexReset);
      if (!mbResetRequested) {
        break;
      }
    }
    usleep(3000);
  }
}

bool LocalMapper::Stop(){
  std::unique_lock<std::mutex> lock(mMutexStop);
  if (mbStopRequested && !mbNotStop) {
    mbStopped = true;
    std::cout << "Local Mapping STOP\n";
    return true;
  }
  return false;
}

void LocalMapper::Release(){
  std::unique_lock<std::mutex> lock(mMutexStop);
  std::unique_lock<std::mutex> lock2(mMutexFinish);
  if (mbFinished) {
    return;
  }
  mbStopped = false;
  mbStopRequested = false;
  for (std::list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(); 
                                      lit != mlNewKeyFrames.end(); 
                                      ++lit) {
    delete *lit;
  }
  mlNewKeyFrames.clear();
  std::cout << "Local Mapping RELEASE\n";
}

bool LocalMapper::isStopped(){
  std::unique_lock<std::mutex> lock(mMutexStop);
  return mbStopped;
}

bool LocalMapper::stopRequested(){
  std::unique_lock<std::mutex> lock(mMutexStop);
  return mbStopRequested;
}

bool LocalMapper::AcceptKeyFrames() {
  std::unique_lock<std::mutex> lock(mMutexAccept);
  return mbAcceptKeyFrames;
}

void LocalMapper::SetAcceptKeyFrames(const bool flag) {
  std::unique_lock<std::mutex> lock(mMutexAccept);
  mbAcceptKeyFrames = flag;
}

void LocalMapper::InterruptBA() {
  // TODO why no mutex??
  mbAbortBA = true;;
}

bool LocalMapper::SetNotStop(const bool flag) {
  std::unique_lock<std::mutex> lock(mMutexStop);
  if (flag && mbStopped) {
    return false;
  }
  mbNotStop = flag;
  return true;
}

void LocalMapper::RequestFinish() {
  std::unique_lock<std::mutex> lock(mMutexFinish);
  mbFinishRequested = true;
}

bool LocalMapper::isFinished() {
  std::unique_lock<std::mutex> lock(mMutexFinish);
  return mbFinished;
}

int LocalMapper::NumKeyframesInQueue() {
  std::unique_lock<std::mutex> lock(mMutexNewKFs);
  return mlNewKeyFrames.size();  
}

bool LocalMapper::CheckNewKeyFrames() {
  std::unique_lock<std::mutex> lock(mMutexNewKFs);
  return (!mlNewKeyFrames.empty());
}

void LocalMapper::ProcessNewKeyFrame() {
  {
    std::unique_lock<std::mutex> lock(mMutexNewKFs);
    mpCurrentKeyFrame = mlNewKeyFrames.front();
    mlNewKeyFrames.pop_front();
  }

  // Compute Bags of Words structures
  mpCurrentKeyFrame->ComputeBoW();

  // Associate MapPoints to the new keyframe and update normal and descriptor
  const std::vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

  for (size_t i = 0; i < vpMapPointMatches.size(); ++i) {
    MapPoint* pMP = vpMapPointMatches[i];
    if(pMP && !pMP->isBad()) {
      if (!pMP->IsInKeyFrame(mpCurrentKeyFrame)) {
        pMP->AddObservation(mpCurrentKeyFrame, i);
        pMP->UpdateNormalAndDepth();
        pMP->ComputeDistinctiveDescriptors();
      } else {
        // this can only happen for new stereo points inserted by the Tracking
        mlpRecentAddedMapPoints.push_back(pMP);
      }
    }
  }    

  // Update links in the Covisibility Graph
  mpCurrentKeyFrame->UpdateConnections();

  // Insert Keyframe in Map
  map_->AddKeyFrame(mpCurrentKeyFrame);
}

void LocalMapper::MapPointCulling() {
  // Check Recent Added MapPoints
  std::list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();
  // const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;
  const int nCurrentKFid = static_cast<int>(mpCurrentKeyFrame->mnId);

  const int cnThObs = is_monocular_? 2 : 3;

  while (lit != mlpRecentAddedMapPoints.end()){
    MapPoint* pMP = *lit;
    if (pMP->isBad()) {
      lit = mlpRecentAddedMapPoints.erase(lit);
    } else if (pMP->GetFoundRatio() < 0.25f) {
      pMP->SetBadFlag();
      lit = mlpRecentAddedMapPoints.erase(lit);
    } else if ((nCurrentKFid - static_cast<int>(pMP->GetFirstKeyframeID())) >= 2 &&
                pMP->NumObservations() <= cnThObs) {
        pMP->SetBadFlag();
        lit = mlpRecentAddedMapPoints.erase(lit);
    } else if((nCurrentKFid-static_cast<int>(pMP->GetFirstKeyframeID())) >= 3) {
        lit = mlpRecentAddedMapPoints.erase(lit);
    } else {
     ++lit;
    }
  }
}

void LocalMapper::CreateNewMapPoints() {
  // Retrieve neighbor keyframes in covisibility graph
  const int nn = is_monocular_ ? 20 : 10;
  const std::vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

  OrbMatcher matcher(0.6f, false);

  cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
  cv::Mat Rwc1 = Rcw1.t();
  cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
  cv::Mat Tcw1(3,4,CV_32F);
  Rcw1.copyTo(Tcw1.colRange(0,3));
  tcw1.copyTo(Tcw1.col(3));
  cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

  const float fx1 = mpCurrentKeyFrame->fx;
  const float fy1 = mpCurrentKeyFrame->fy;
  const float cx1 = mpCurrentKeyFrame->cx;
  const float cy1 = mpCurrentKeyFrame->cy;
  const float invfx1 = mpCurrentKeyFrame->invfx;
  const float invfy1 = mpCurrentKeyFrame->invfy;

  const float ratioFactor = 1.5f * mpCurrentKeyFrame->mfScaleFactor;

  // Search matches with epipolar restriction and triangulate
  int nnew = 0;
  for (size_t i = 0; i < vpNeighKFs.size(); ++i) {
    if (i > 0 && CheckNewKeyFrames()) {
      return;
    }
    KeyFrame* pKF2 = vpNeighKFs[i];

    // Check first that baseline is not too short
    cv::Mat Ow2 = pKF2->GetCameraCenter();
    cv::Mat vBaseline = Ow2 - Ow1;
    const float baseline = cv::norm(vBaseline);

    if (!is_monocular_) {
      if (baseline < pKF2->mb) {
        continue;
      }
    } else {
      const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
      const float ratioBaselineDepth = baseline / medianDepthKF2;
      if (ratioBaselineDepth < 0.01f) {
        continue;
      }
    }

    // Compute Fundamental Matrix
    cv::Mat F12 = ComputeF12(mpCurrentKeyFrame, pKF2);

    // Search matches that fullfil epipolar constraint
    std::vector<std::pair<size_t,size_t>> vMatchedIndices;
    matcher.SearchForTriangulation(mpCurrentKeyFrame,
                                   pKF2,
                                   F12,
                                   vMatchedIndices,
                                   false);

    cv::Mat Rcw2 = pKF2->GetRotation();
    cv::Mat Rwc2 = Rcw2.t();
    cv::Mat tcw2 = pKF2->GetTranslation();
    cv::Mat Tcw2(3,4,CV_32F);
    Rcw2.copyTo(Tcw2.colRange(0,3));
    tcw2.copyTo(Tcw2.col(3));

    const float fx2 = pKF2->fx;
    const float fy2 = pKF2->fy;
    const float cx2 = pKF2->cx;
    const float cy2 = pKF2->cy;
    const float invfx2 = pKF2->invfx;
    const float invfy2 = pKF2->invfy;

    // Triangulate each match
    const int nmatches = vMatchedIndices.size();
    for(int ikp = 0; ikp < nmatches; ++ikp) {
      const int idx1 = vMatchedIndices[ikp].first;
      const int idx2 = vMatchedIndices[ikp].second;

      const cv::KeyPoint& kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
      const float kp1_ur = mpCurrentKeyFrame->mvuRight[idx1];
      bool bStereo1 = kp1_ur >= 0;

      const cv::KeyPoint& kp2 = pKF2->mvKeysUn[idx2];
      const float kp2_ur = pKF2->mvuRight[idx2];
      bool bStereo2 = (kp2_ur >= 0);

      // Check parallax between rays
      cv::Mat xn1 = cv::Mat(3,1, CV_32F, {(kp1.pt.x-cx1)*invfx1, 
                                          (kp1.pt.y-cy1)*invfy1, 
                                           1.0});
      cv::Mat xn2 = cv::Mat(3,1, CV_32F, {(kp2.pt.x-cx2)*invfx2, 
                                          (kp2.pt.y-cy2)*invfy2, 
                                           1.0});
      cv::Mat ray1 = Rwc1 * xn1;
      cv::Mat ray2 = Rwc2 * xn2;
      const float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1)*cv::norm(ray2));

      float cosParallaxStereo = cosParallaxRays + 1;
      float cosParallaxStereo1 = cosParallaxStereo;
      float cosParallaxStereo2 = cosParallaxStereo;

      if(bStereo1) {
        cosParallaxStereo1 = std::cos(2*std::atan2(mpCurrentKeyFrame->mb/2,
                                                   mpCurrentKeyFrame->mvDepth[idx1]));
      } else if(bStereo2) {
        cosParallaxStereo2 = std::cos(2*std::atan2(pKF2->mb/2,
                                                   pKF2->mvDepth[idx2]));
      }

      cosParallaxStereo = std::min(cosParallaxStereo1,cosParallaxStereo2);

      cv::Mat x3D;
      if (cosParallaxRays < cosParallaxStereo && 
          cosParallaxRays > 0 && 
          (bStereo1 || bStereo2 || cosParallaxRays < 0.9998)) {
        // Linear Triangulation Method
        cv::Mat A(4,4,CV_32F);
        A.row(0) = xn1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
        A.row(1) = xn1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
        A.row(2) = xn2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
        A.row(3) = xn2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

        cv::Mat w,u,vt;
        cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

        x3D = vt.row(3).t();

        if (x3D.at<float>(3) == 0) {
          continue;
        }

        // Euclidean coordinates
        x3D = x3D.rowRange(0,3) / x3D.at<float>(3);
      }
      else if (bStereo1 && cosParallaxStereo1 < cosParallaxStereo2) {
        x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);
      }
      else if (bStereo2 && cosParallaxStereo2 < cosParallaxStereo1) {
        x3D = pKF2->UnprojectStereo(idx2);
      }
      else {
        continue; //No stereo and very low parallax
      }
      cv::Mat x3Dt = x3D.t();

      //Check triangulation in front of cameras
      const float z1 = Rcw1.row(2).dot(x3Dt) + tcw1.at<float>(2);
      if (z1 <= 0) {
        continue;
      }

      const float z2 = Rcw2.row(2).dot(x3Dt) + tcw2.at<float>(2);
      if(z2 <= 0) {
        continue;
      }

      //Check reprojection error in first keyframe
      const float sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
      const float x1 = Rcw1.row(0).dot(x3Dt)+tcw1.at<float>(0);
      const float y1 = Rcw1.row(1).dot(x3Dt)+tcw1.at<float>(1);
      const float invz1 = 1.0 / z1;

      if (!bStereo1) {
        const float u1 = fx1*x1*invz1+cx1;
        const float v1 = fy1*y1*invz1+cy1;
        const float errX1 = u1 - kp1.pt.x;
        const float errY1 = v1 - kp1.pt.y;
        if ((errX1*errX1 + errY1*errY1) > 5.991*sigmaSquare1) {
          continue;
        }
      } else {
          const float u1 = fx1*x1*invz1+cx1;
          const float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1;
          const float v1 = fy1*y1*invz1+cy1;
          const float errX1 = u1 - kp1.pt.x;
          const float errY1 = v1 - kp1.pt.y;
          const float errX1_r = u1_r - kp1_ur;
          if ((errX1*errX1 + errY1*errY1 + errX1_r*errX1_r) > 7.8*sigmaSquare1) {
            continue;
          }
      }

      //Check reprojection error in second keyframe
      const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
      const float x2 = Rcw2.row(0).dot(x3Dt)+tcw2.at<float>(0);
      const float y2 = Rcw2.row(1).dot(x3Dt)+tcw2.at<float>(1);
      const float invz2 = 1.0/z2;
      if (!bStereo2) {
        const float u2 = fx2*x2*invz2+cx2;
        const float v2 = fy2*y2*invz2+cy2;
        const float errX2 = u2 - kp2.pt.x;
        const float errY2 = v2 - kp2.pt.y;
        if((errX2*errX2 + errY2*errY2) > 5.991*sigmaSquare2) {
          continue;
        }
      } else {
        const float u2 = fx2*x2*invz2+cx2;
        const float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
        const float v2 = fy2*y2*invz2+cy2;
        const float errX2 = u2 - kp2.pt.x;
        const float errY2 = v2 - kp2.pt.y;
        const float errX2_r = u2_r - kp2_ur;
        if((errX2*errX2 + errY2*errY2 + errX2_r*errX2_r) > 7.8*sigmaSquare2) {
          continue;
        }
      }

      //Check scale consistency
      cv::Mat normal1 = x3D - Ow1;
      const float dist1 = cv::norm(normal1);

      cv::Mat normal2 = x3D - Ow2;
      const float dist2 = cv::norm(normal2);

      if (dist1==0 || dist2==0) {
        continue;
      }

      const float ratioDist = dist2 / dist1;
      const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave] / pKF2->mvScaleFactors[kp2.octave];

      if (ratioDist*ratioFactor < ratioOctave || ratioDist > ratioOctave*ratioFactor) {
        continue;
      }

      // Triangulation is succesfull
      MapPoint* pMP = new MapPoint(x3D,
                                   mpCurrentKeyFrame,
                                   map_);

      pMP->AddObservation(mpCurrentKeyFrame, idx1);            
      pMP->AddObservation(pKF2, idx2);

      mpCurrentKeyFrame->AddMapPoint(pMP, idx1);
      pKF2->AddMapPoint(pMP, idx2);

      pMP->ComputeDistinctiveDescriptors();
      pMP->UpdateNormalAndDepth();

      map_->AddMapPoint(pMP);
      mlpRecentAddedMapPoints.push_back(pMP);
      ++nnew;
    }
  }
}

void LocalMapper::SearchInNeighbors() {
  // Retrieve neighbor keyframes
  const int nn = is_monocular_? 20 : 10;
  const std::vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
  std::vector<KeyFrame*> vpTargetKFs;
  for (std::vector<KeyFrame*>::const_iterator vit = vpNeighKFs.begin(); 
                                              vit != vpNeighKFs.end(); 
                                              ++vit) {
    KeyFrame* pKFi = *vit;
    if (pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId) {
      continue;
    }
    vpTargetKFs.push_back(pKFi);
    pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;

    // Extend to some second neighbors
    const std::vector<KeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
    for(std::vector<KeyFrame*>::const_iterator vit2 = vpSecondNeighKFs.begin();
                                               vit2 != vpSecondNeighKFs.end(); 
                                               ++vit2) {
      KeyFrame* pKFi2 = *vit2;
      if (pKFi2->isBad() || 
          pKFi2->mnFuseTargetForKF == mpCurrentKeyFrame->mnId || 
          pKFi2->mnId == mpCurrentKeyFrame->mnId) {
        continue;
      }
      vpTargetKFs.push_back(pKFi2);
    }
  }

  // Search matches by projection from current KF in target KFs
  OrbMatcher matcher;
  std::vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
  for (std::vector<KeyFrame*>::iterator vit = vpTargetKFs.begin(); 
                                        vit != vpTargetKFs.end(); 
                                        ++vit) {
    KeyFrame* pKFi = *vit;
    matcher.Fuse(pKFi, vpMapPointMatches);
  }

  // Search matches by projection from target KFs in current KF
  std::vector<MapPoint*> vpFuseCandidates;
  vpFuseCandidates.reserve(vpTargetKFs.size() * vpMapPointMatches.size());

  for (std::vector<KeyFrame*>::iterator vitKF = vpTargetKFs.begin(); 
                                        vitKF != vpTargetKFs.end(); 
                                        ++vitKF) {
    KeyFrame* pKFi = *vitKF;

    std::vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();

    for (std::vector<MapPoint*>::iterator vitMP = vpMapPointsKFi.begin();
                                          vitMP != vpMapPointsKFi.end(); 
                                          ++vitMP) {
      MapPoint* pMP = *vitMP;
      if(!pMP ||
          pMP->isBad() ||
          pMP->fuse_candidate_id_for_keyframe == mpCurrentKeyFrame->mnId ) {
          continue;
      }
      pMP->fuse_candidate_id_for_keyframe = mpCurrentKeyFrame->mnId;
      vpFuseCandidates.push_back(pMP);
    }
  }
  matcher.Fuse(mpCurrentKeyFrame, vpFuseCandidates);

  // Update points
  vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
  for (size_t i = 0; i < vpMapPointMatches.size(); ++i) {
    MapPoint* pMP = vpMapPointMatches[i];
    if (pMP && !pMP->isBad()) {
      pMP->ComputeDistinctiveDescriptors();
      pMP->UpdateNormalAndDepth();
    }
  }

  // Update connections in covisibility graph
  mpCurrentKeyFrame->UpdateConnections();
}

void LocalMapper::KeyFrameCulling() {
  // Check redundant keyframes (only local keyframes)
  // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
  // in at least other 3 keyframes (in the same or finer scale)
  // We only consider close stereo points
  std::vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

  for (std::vector<KeyFrame*>::iterator vit = vpLocalKeyFrames.begin(); 
                                        vit != vpLocalKeyFrames.end(); 
                                        ++vit) {
    KeyFrame* pKF = *vit;
    if (pKF->mnId == 0) {
      continue;
    }
    const std::vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

    const int thObs = 3;
    int nRedundantObservations = 0;
    int nMPs = 0;
    for (size_t i = 0; i < vpMapPoints.size(); ++i) {
      MapPoint* pMP = vpMapPoints[i];
      if(pMP && !pMP->isBad()) {
        if (!is_monocular_) {
          if (pKF->mvDepth[i] > pKF->mThDepth || 
              pKF->mvDepth[i] < 0) {
            continue;
          }
        }

        ++nMPs;
        if (pMP->NumObservations() > thObs) {
          const int scaleLevel = pKF->mvKeysUn[i].octave;
          const std::map<KeyFrame*,size_t> observations = pMP->GetObservations();
          int nObs = 0;
          for (std::map<KeyFrame*,size_t>::const_iterator mit = observations.begin(); 
                                                          mit != observations.end(); 
                                                          ++mit) {
            KeyFrame* pKFi = mit->first;
            if (pKFi==pKF) {
              continue;
            }
            const int scaleLeveli = pKFi->mvKeysUn[mit->second].octave;

            if(scaleLeveli <= scaleLevel + 1) {
              nObs++;
              if (nObs >= thObs) {
                break;
              }
            }
          }
          if (nObs >= thObs) {
            ++nRedundantObservations;
          }
        }
      }
    }  

    if (nRedundantObservations > 0.9f * nMPs) {
      pKF->SetBadFlag();
    }
  }
}

cv::Mat LocalMapper::ComputeF12(KeyFrame* pKF1, KeyFrame* pKF2) const {
  const cv::Mat R1w = pKF1->GetRotation();
  const cv::Mat t1w = pKF1->GetTranslation();
  const cv::Mat R2w = pKF2->GetRotation();
  const cv::Mat t2w = pKF2->GetTranslation();

  const cv::Mat R12 = R1w*R2w.t();
  const cv::Mat t12 = -R1w*R2w.t()*t2w + t1w;

  const cv::Mat t12x = SkewSymmetricMatrix(t12);

  const cv::Mat& K1 = pKF1->mK;
  const cv::Mat& K2 = pKF2->mK;

  return K1.t().inv() * t12x * R12 * K2.inv();
}

cv::Mat LocalMapper::SkewSymmetricMatrix(const cv::Mat& v) const {
  return (cv::Mat_<float>(3,3) <<              0,  -v.at<float>(2),  v.at<float>(1),
                                   v.at<float>(2),              0,  -v.at<float>(0),
                                  -v.at<float>(1),  v.at<float>(0),              0);  
}

void LocalMapper::ResetIfRequested() {
  std::unique_lock<std::mutex> lock(mMutexReset);
  if (mbResetRequested) {
    mlNewKeyFrames.clear();
    mlpRecentAddedMapPoints.clear();
    mbResetRequested = false;
  }  
}

bool LocalMapper::CheckFinish() {
  std::unique_lock<std::mutex> lock(mMutexFinish);
  return mbFinishRequested;
}

void LocalMapper::SetFinish() {
  std::unique_lock<std::mutex> lock(mMutexFinish);
  mbFinished = true;    
  std::unique_lock<std::mutex> lock2(mMutexStop);
  mbStopped = true;
}
