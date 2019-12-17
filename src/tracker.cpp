#include "tracker.h"

#include <unistd.h> // usleep

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "orb_matcher.h"
#include "converter.h"
#include "map.h"

#include "optimizer.h"
#include "pnp_solver.h"

#include <iostream>

#include <mutex>

Tracker::Tracker(SlamSystem* pSys, 
                 OrbVocabulary* pVoc, 
                 Map* pMap,
                 KeyframeDatabase* pKFDB, 
                 const std::string& strSettingPath, // TODO
                 const int sensor)
      : mState(TrackingState::NO_IMAGES_YET)
      , mSensor(sensor)
      , mbOnlyTracking(false)
      , mbVO(false)
      , mpORBVocabulary(pVoc)
      , mpKeyFrameDB(pKFDB)
      , mpSystem(pSys)
      , mpMap(pMap)
      , mnLastRelocFrameId(0)
{
      // Load camera parameters from settings file
  // cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
  // float fx = fSettings["Camera.fx"];
  // float fy = fSettings["Camera.fy"];
  // float cx = fSettings["Camera.cx"];
  // float cy = fSettings["Camera.cy"];
  float fx = 718.856f; // TODO, just use from kitti00 for now
  float fy = 718.856f;
  float cx = 607.1928f;
  float cy = 185.2157f;

  cv::Mat K = cv::Mat::eye(3,3,CV_32F);
  K.at<float>(0,0) = fx;
  K.at<float>(1,1) = fy;
  K.at<float>(0,2) = cx;
  K.at<float>(1,2) = cy;
  K.copyTo(mK);

  cv::Mat DistCoef(4,1,CV_32F);
  // DistCoef.at<float>(0) = fSettings["Camera.k1"];
  // DistCoef.at<float>(1) = fSettings["Camera.k2"];
  // DistCoef.at<float>(2) = fSettings["Camera.p1"];
  // DistCoef.at<float>(3) = fSettings["Camera.p2"];
  // const float k3 = fSettings["Camera.k3"];
  DistCoef.at<float>(0) = 0.0f; // TODO just use kitti00 params for now
  DistCoef.at<float>(1) = 0.0f;
  DistCoef.at<float>(2) = 0.0f;
  DistCoef.at<float>(3) = 0.0f;
  const float k3 = 0.0f;
  if (k3 != 0) {
    DistCoef.resize(5);
    DistCoef.at<float>(4) = k3;
  }
  DistCoef.copyTo(mDistCoef);

  // mbf = fSettings["Camera.bf"];
  mbf = 386.1448f; // TODO just used from kitti00

  // float fps = fSettings["Camera.fps"];
  float fps = 10.0f; // TODO from kitti00
  if (fps == 0) {
    fps = 30.0f;
  }

  // Max/Min Frames to insert keyframes and to check relocalisation
  mMinFrames = 0;
  mMaxFrames = fps;

  // mbRGB = fSettings["Camera.RGB"];
  mbRGB = 1;

  // Load ORB params
  // int nFeatures = fSettings["ORBextractor.nFeatures"];
  // float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
  // int nLevels = fSettings["ORBextractor.nLevels"];
  // int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
  // int fMinThFAST = fSettings["ORBextractor.minThFAST"];
  int nFeatures = 2000;
  float fScaleFactor = 1.2f;
  int nLevels = 8;
  int fIniThFAST = 20;
  int fMinThFAST = 7;

  mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
  if (sensor == SENSOR_TYPE::STEREO) {
    mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);  
  }

  // mThDepth = mbf * static_cast<float>(fSettings["ThDepth"]) / fx;
  mThDepth = mbf * static_cast<float>(35) / fx;
  
  if(sensor == SENSOR_TYPE::RGBD) {
    // mDepthMapFactor = fSettings["DepthMapFactor"];
    mDepthMapFactor = 0;
    if (std::fabs(mDepthMapFactor) < 1e-5) {
      mDepthMapFactor = 1;
    } else {
      mDepthMapFactor = 1.0f / mDepthMapFactor;
    }
  }
}


cv::Mat Tracker::GrabImageStereo(const cv::Mat& imRectLeft,
                                 const cv::Mat& imRectRight, 
                                 const double timestamp) {
  mImGray = imRectLeft;
  cv::Mat imGrayRight = imRectRight;

  // Convert to grayscale if image is RGB
  if (mImGray.channels() == 3) {
    if (mbRGB) {
      cv::cvtColor(mImGray,mImGray, CV_RGB2GRAY);
      cv::cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
    } else {
      cvtColor(mImGray,mImGray,CV_BGR2GRAY);
      cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
    }
  } else if (mImGray.channels() == 4) {
    if (mbRGB) {
      cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
      cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
    } else {
      cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
      cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
    }
  } 

  mCurrentFrame = Frame(mImGray,
                        imGrayRight,
                        timestamp,
                        mpORBextractorLeft,
                        mpORBextractorRight,
                        mpORBVocabulary,
                        mK,
                        mDistCoef,
                        mbf,
                        mThDepth);
  Track();
  return mCurrentFrame.mTcw.clone();;
}

void Tracker::Reset() {
  std::cout << "System Reseting" << std::endl;
  // if(mpViewer) {
  //   mpViewer->RequestStop();
  //   while (!mpViewer->isStopped()) {
  //     usleep(3000);
  //   }
  // }
  
  // Reset local mapper
  mpLocalMapper->RequestReset();

  // Reset loop closing
  mpLoopClosing->RequestReset();

  // Clear BoW database
  mpKeyFrameDB->clear();

  // Clear map
  mpMap->clear();

  KeyFrame::nNextId = 0;
  Frame::nNextId = 0;
  mState = TrackingState::NO_IMAGES_YET;

  // if(mpInitializer) { // only for mono
  //   delete mpInitializer;
  //   mpInitializer = static_cast<Initializer*>(NULL);
  // }

  mlRelativeFramePoses.clear();
  mlpReferences.clear();
  mlFrameTimes.clear();
  mlbLost.clear();
}


void Tracker::StereoInitialization() {
  if (mCurrentFrame.mN > 500) {
    // Set Frame pose to the origin
    mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));

    // Create KeyFrame
    KeyFrame* pKFini = new KeyFrame(mCurrentFrame,
                                    mpMap,
                                    mpKeyFrameDB);

    // Insert KeyFrame in the map
    mpMap->AddKeyFrame(pKFini);

    // Create MapPoints and asscoiate to KeyFrame
    for (int i=0; i < mCurrentFrame.mN; ++i) {
      float z = mCurrentFrame.mvDepth[i];
      if (z > 0) {
        cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
        MapPoint* pNewMP = new MapPoint(x3D, pKFini, mpMap);
        pNewMP->AddObservation(pKFini,i);
        pKFini->AddMapPoint(pNewMP,i);
        pNewMP->ComputeDistinctiveDescriptors();
        pNewMP->UpdateNormalAndDepth();
        mpMap->AddMapPoint(pNewMP);

        mCurrentFrame.mvpMapPoints[i] = pNewMP;
      }
    }

    std::cout << "New map created with " << mpMap->MapPointsInMap() << " points" << std::endl;

    mpLocalMapper->InsertKeyFrame(pKFini);

    mLastFrame = Frame(mCurrentFrame);
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKFini;

    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints = mpMap->GetAllMapPoints();

    mpReferenceKF = pKFini;
    mCurrentFrame.mpReferenceKF = pKFini;

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    mState = TrackingState::OK;
  }
}

void Tracker::Track() {

  if (mState == TrackingState::NO_IMAGES_YET) {
    mState = NOT_INITIALIZED;
  }

  mLastProcessedState = mState;

  std::unique_lock<std::mutex> lock(mpMap->mMutexMapUpdate);

  if (mState == TrackingState::NOT_INITIALIZED) {
    StereoInitialization();
    if (mState != OK) {
      return;
    }
  } else {
    // System is initialized. Track Frame.
    bool bOK = false;

    // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
    if (!mbOnlyTracking) {
      // Local Mapping is activated. This is the normal behaviour, unless
      // you explicitly activate the "only tracking" mode.
      if (mState == TrackingState::OK) {
        // Local Mapping might have changed some MapPoints tracked in last frame
        CheckReplacedInLastFrame();

        if (mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId + 2) {
          bOK = TrackReferenceKeyFrame();
        } else {
          bOK = TrackWithMotionModel();
          if (!bOK) {
            bOK = TrackReferenceKeyFrame();
          }
        }
      } else {
        bOK = Relocalization();
      }
    }  else { // TODO maybe don't need to do this. Comment out for now 
      // std::cout << "Localization-only Mode disabled for now\n";  
      //   // Localization Mode: Local Mapping is deactivated

      //   if(mState==LOST)
      //   {
      //       bOK = Relocalization();
      //   }
      //   else
      //   {
      //       if(!mbVO)
      //       {
      //           // In last frame we tracked enough MapPoints in the map

      //           if(!mVelocity.empty())
      //           {
      //               bOK = TrackWithMotionModel();
      //           }
      //           else
      //           {
      //               bOK = TrackReferenceKeyFrame();
      //           }
      //       }
      //       else
      //       {
      //           // In last frame we tracked mainly "visual odometry" points.

      //           // We compute two camera poses, one from motion model and one doing relocalization.
      //           // If relocalization is sucessfull we choose that solution, otherwise we retain
      //           // the "visual odometry" solution.

      //           bool bOKMM = false;
      //           bool bOKReloc = false;
      //           vector<MapPoint*> vpMPsMM;
      //           vector<bool> vbOutMM;
      //           cv::Mat TcwMM;
      //           if(!mVelocity.empty())
      //           {
      //               bOKMM = TrackWithMotionModel();
      //               vpMPsMM = mCurrentFrame.mvpMapPoints;
      //               vbOutMM = mCurrentFrame.mvbOutlier;
      //               TcwMM = mCurrentFrame.mTcw.clone();
      //           }
      //           bOKReloc = Relocalization();

      //           if(bOKMM && !bOKReloc)
      //           {
      //               mCurrentFrame.SetPose(TcwMM);
      //               mCurrentFrame.mvpMapPoints = vpMPsMM;
      //               mCurrentFrame.mvbOutlier = vbOutMM;

      //               if(mbVO)
      //               {
      //                   for(int i =0; i<mCurrentFrame.N; i++)
      //                   {
      //                       if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
      //                       {
      //                           mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
      //                       }
      //                   }
      //               }
      //           }
      //           else if(bOKReloc)
      //           {
      //               mbVO = false;
      //           }

      //           bOK = bOKReloc || bOKMM;
      //       }
      //   }
    }

    mCurrentFrame.mpReferenceKF = mpReferenceKF;

    // If we have an initial estimation of the camera pose and matching. Track the local map.
    if (!mbOnlyTracking) {
      if (bOK) {
        bOK = TrackLocalMap();
      }
    } else {
      // TODO Disabled for now since we don't wanna use only tracking
        // // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
        // // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
        // // the camera we will use the local map again.
        // if(bOK && !mbVO)
        //     bOK = TrackLocalMap();
    }

    if (bOK) {
      mState = TrackingState::OK;
    } else {
      mState = TrackingState::LOST;
    }
        
    // If tracking were good, check if we insert a keyframe
    if (bOK) {
      // Update motion model
      if (!mLastFrame.mTcw.empty()) {
        cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
        mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
        mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
        mVelocity = mCurrentFrame.mTcw*LastTwc;
      } else {
        mVelocity = cv::Mat();
      }

      // Clean VO matches
      for (int i=0; i < mCurrentFrame.mN; ++i) {
        MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
        if (pMP && pMP->Observations() < 1) {
          mCurrentFrame.mvbOutlier[i] = false; 
          mCurrentFrame.mvpMapPoints[i] = nullptr;
        }
      }

      // Check if we need to insert a new keyframe
      if (NeedNewKeyFrame()) {
        CreateNewKeyFrame();
      }

      // We allow points with high innovation (considererd outliers by the Huber Function)
      // to pass to the new keyframe, so that bundle adjustment will finally decide
      // if they are outliers or not. We don't want next frame to estimate its position
      // with those points so we discard them in the frame.
      for (int i=0; i < mCurrentFrame.mN; ++i) {
        if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i]) {
          mCurrentFrame.mvpMapPoints[i] = nullptr;
        }
      }
    }

    // Reset if the camera gets lost soon after initialization
    if (mState == TrackingState::LOST) {
      if (mpMap->KeyFramesInMap() <= 5) {
        std::cout << "Track lost soon after initialisation, resetting..." << std::endl;
        mpSystem->FlagReset();
        return;
      }
    }

    if (!mCurrentFrame.mpReferenceKF) {
      mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }

    mLastFrame = Frame(mCurrentFrame);
  }

  // Store frame pose information to retrieve the complete camera trajectory afterwards.
  if (!mCurrentFrame.mTcw.empty()) {
    cv::Mat Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
    mlRelativeFramePoses.push_back(Tcr);
    mlpReferences.push_back(mpReferenceKF);
    mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
    mlbLost.push_back( mState == TrackingState::LOST );
  } else {
    // This can happen if tracking is lost
    mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
    mlpReferences.push_back(mlpReferences.back());
    mlFrameTimes.push_back(mlFrameTimes.back());
    mlbLost.push_back( mState == TrackingState::LOST );
  }
}

void Tracker::CheckReplacedInLastFrame() {
  for (int i=0; i < mLastFrame.mN; ++i) {
    MapPoint* pMP = mLastFrame.mvpMapPoints[i];
    if (pMP) {
      MapPoint* pRep = pMP->GetReplaced();
      if (pRep) {
        mLastFrame.mvpMapPoints[i] = pRep;
      }
    }
  }
}

bool Tracker::TrackReferenceKeyFrame() {
  // Compute Bag of Words vector
  mCurrentFrame.ComputeBoW();

  // We perform first an ORB matching with the reference keyframe
  // If enough matches are found we setup a PnP solver
  OrbMatcher matcher(0.7, true);
  std::vector<MapPoint*> vpMapPointMatches;

  int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);
  if (nmatches < num_required_matches_) {
    return false;
  }

  mCurrentFrame.mvpMapPoints = vpMapPointMatches;
  mCurrentFrame.SetPose(mLastFrame.mTcw);

  Optimizer::PoseOptimization(&mCurrentFrame);

  // Discard outliers
  int nmatchesMap = 0;
  for (int i = 0; i < mCurrentFrame.mN; ++i) {
    if (mCurrentFrame.mvpMapPoints[i]) {
      if (mCurrentFrame.mvbOutlier[i]) {
        MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

        // mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
        mCurrentFrame.mvpMapPoints[i] = nullptr;
        mCurrentFrame.mvbOutlier[i] = false;
        pMP->mbTrackInView = false;
        pMP->mnLastFrameSeen = mCurrentFrame.mnId;
        --nmatches; // TODO never seem to use  this again
      } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0) {
        nmatchesMap++;
      }
    }
  }

  return (nmatchesMap >= 10);
}

void Tracker::UpdateLastFrame() {
  // Update pose according to reference keyframe
  KeyFrame* pRef = mLastFrame.mpReferenceKF;
  cv::Mat Tlr = mlRelativeFramePoses.back();

  mLastFrame.SetPose(Tlr * pRef->GetPose());

  if (mnLastKeyFrameId == mLastFrame.mnId 
      || mSensor==SENSOR_TYPE::MONOCULAR 
      || !mbOnlyTracking) {
    return;
  }

  // Create "visual odometry" MapPoints
  // We sort points according to their measured depth by the stereo/RGB-D sensor
  std::vector<std::pair<float,int>> vDepthIdx;
  vDepthIdx.reserve(mLastFrame.mN);
  for (int i = 0; i < mLastFrame.mN; ++i) {
    float z = mLastFrame.mvDepth[i];
    if(z > 0) {
      vDepthIdx.push_back(std::make_pair(z,i));
    }
  }

  if(vDepthIdx.empty()) {
    return;
  }

  std::sort(vDepthIdx.begin(), vDepthIdx.end());

  // We insert all close points (depth < mThDepth)
  // If less than 100 close points, we insert the 100 closest ones.
  int nPoints = 0;
  for (size_t j = 0; j < vDepthIdx.size(); ++j) {
    int i = vDepthIdx[j].second;
    bool bCreateNew = false;
    MapPoint* pMP = mLastFrame.mvpMapPoints[i];
    if (!pMP) {
      bCreateNew = true;
    } else if (pMP->Observations() < 1) {
      bCreateNew = true;
    }

    if (bCreateNew) {
      cv::Mat x3D = mLastFrame.UnprojectStereo(i);
      MapPoint* pNewMP = new MapPoint(x3D,
                                      mpMap,
                                      &mLastFrame,
                                      i);

      mLastFrame.mvpMapPoints[i] = pNewMP;
    } 
    ++nPoints;

    if (vDepthIdx[j].first > mThDepth && nPoints > 100) {
      break;
    }
  }
}

bool Tracker::TrackWithMotionModel() {
  OrbMatcher matcher(0.9, true);

  // Update last frame pose according to its reference keyframe
  // Create "visual odometry" points if in Localization Mode
  UpdateLastFrame();

  mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);

  std::fill(mCurrentFrame.mvpMapPoints.begin(),
            mCurrentFrame.mvpMapPoints.end(),
            nullptr);

  // Project points seen in previous frame
  int th;
  if (mSensor != SENSOR_TYPE::STEREO) {
    th=15;
  } else {
    th=7;
  }
  int nmatches = matcher.SearchByProjection(mCurrentFrame,
                                            mLastFrame,
                                            th,
                                            mSensor==SENSOR_TYPE::MONOCULAR);

  // If few matches, uses a wider window search
  if(nmatches < 20) {
    std::fill(mCurrentFrame.mvpMapPoints.begin(),
              mCurrentFrame.mvpMapPoints.end(),
              nullptr);
    nmatches = matcher.SearchByProjection(mCurrentFrame,
                                          mLastFrame,
                                          2*th,
                                          mSensor==SENSOR_TYPE::MONOCULAR);
  }

  if (nmatches < 20) {
    return false;
  }
      
  // Optimize frame pose with all matches
  Optimizer::PoseOptimization(&mCurrentFrame);

  // Discard outliers
  int nmatchesMap = 0;
  for (int i = 0; i < mCurrentFrame.mN; ++i) {
    if (mCurrentFrame.mvpMapPoints[i]) {
      if (mCurrentFrame.mvbOutlier[i]) {
        MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

        mCurrentFrame.mvpMapPoints[i] = nullptr;
        mCurrentFrame.mvbOutlier[i] = false;
        pMP->mbTrackInView = false;
        pMP->mnLastFrameSeen = mCurrentFrame.mnId;
        --nmatches;
      } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0) {
        ++nmatchesMap;
      }
    }
  }    

  if (mbOnlyTracking) {
    mbVO = (nmatchesMap < 10);
    return (nmatches > 20);
  } else {
    return (nmatchesMap >= 10);
  }  
}

bool Tracker::Relocalization() {
  // Compute Bag of Words Vector
  mCurrentFrame.ComputeBoW();

  // Relocalization is performed when tracking is lost
  // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
  std::vector<KeyFrame*> vpCandidateKFs = 
                mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

  if (vpCandidateKFs.empty()) {
    return false;
  }

  const int nKFs = vpCandidateKFs.size();

  // We perform first an ORB matching with each candidate
  // If enough matches are found we setup a PnP solver
  OrbMatcher matcher(0.75, true);

  std::vector<std::unique_ptr<PnPsolver>> vpPnPsolvers;
  vpPnPsolvers.resize(nKFs);

  std::vector<std::vector<MapPoint*>> vvpMapPointMatches;
  vvpMapPointMatches.resize(nKFs);

  std::vector<bool> vbDiscarded; // TODO bool of vectors
  vbDiscarded.resize(nKFs);

  int nCandidates = 0;

  for (int i = 0; i < nKFs; ++i) {
    KeyFrame* pKF = vpCandidateKFs[i];
    if (pKF->isBad()) {
      vbDiscarded[i] = true;
    } else {
      int nmatches = matcher.SearchByBoW(pKF, 
                                         mCurrentFrame, 
                                         vvpMapPointMatches[i]);
      if(nmatches < 15) {
        vbDiscarded[i] = true;
        continue;
      } else {
        std::unique_ptr<PnPsolver> pSolver;
        pSolver.reset(new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]));
        pSolver->SetRansacParameters(0.99,
                                     10,
                                     300,
                                     4,
                                     0.5,
                                     5.991);
        vpPnPsolvers[i] = std::move(pSolver);
        ++nCandidates;
      }
    }
  }

  // Alternatively perform some iterations of P4P RANSAC
  // Until we found a camera pose supported by enough inliers
  bool bMatch = false;
  OrbMatcher matcher2(0.9, true);

  while (nCandidates > 0 && !bMatch) {
    for (int i = 0; i < nKFs; ++i) {
      if(vbDiscarded[i]) {
        continue;
      }

      // Perform 5 Ransac Iterations
      std::vector<bool> vbInliers; // TODO vector of bools
      int nInliers;
      bool bNoMore;

      constexpr int num_iter = 5;
      cv::Mat Tcw = vpPnPsolvers[i]->iterate(num_iter,
                                             bNoMore,
                                             vbInliers,
                                             nInliers);

      // If Ransac reachs max iterations discard keyframe
      if (bNoMore) {
        vbDiscarded[i] = true;
        --nCandidates;
      }

      // If a Camera Pose is computed, optimize
      if(!Tcw.empty()) {
        Tcw.copyTo(mCurrentFrame.mTcw);

        std::set<MapPoint*> sFound;

        for (size_t j = 0; j < vbInliers.size(); ++j) {
          if (vbInliers[j]) {
            mCurrentFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j];
            sFound.insert(vvpMapPointMatches[i][j]);
          } else {
            mCurrentFrame.mvpMapPoints[j] = nullptr;
          }
        }

        int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

        if(nGood < 10) {
          continue;
        }

        for (int io = 0; io < mCurrentFrame.mN; ++io) {
          if (mCurrentFrame.mvbOutlier[io]) {
            mCurrentFrame.mvpMapPoints[io] = nullptr;
          }
        }

        // If few inliers, search by projection in a coarse window and optimize again
        if (nGood < 50) {
          int nadditional = matcher2.SearchByProjection(mCurrentFrame,
                                                        vpCandidateKFs[i],
                                                        sFound,
                                                        10,
                                                        100);

          if (nadditional + nGood >= 50) {
            nGood = Optimizer::PoseOptimization(&mCurrentFrame);

            // If many inliers but still not enough, search by projection again in a narrower window
            // the camera has been already optimized with many points
            if (nGood > 30 && nGood < 50) {
              sFound.clear();
              for (int ip = 0; ip < mCurrentFrame.mN; ++ip) {
                if (mCurrentFrame.mvpMapPoints[ip]) {
                  sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                }
              }
              nadditional = matcher2.SearchByProjection(mCurrentFrame,
                                                        vpCandidateKFs[i],
                                                        sFound,
                                                        3,
                                                        64);

              // Final optimization
              if (nGood + nadditional >= 50) {
                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                for (int io = 0; io < mCurrentFrame.mN; ++io) {
                  if (mCurrentFrame.mvbOutlier[io]) {
                    mCurrentFrame.mvpMapPoints[io] = nullptr;
                  }
                }
              }
            }
          }
        }

        // If the pose is supported by enough inliers stop ransacs and continue
        if (nGood >= 50) {
          bMatch = true;
          break;
        }
      }
    }
  }

  if (!bMatch) {
    return false;
  } else {
    mnLastRelocFrameId = mCurrentFrame.mnId;
    return true;
  }
}

void Tracker::UpdateLocalMap() {
  // This is for visualization
  mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

  // Update
  UpdateLocalKeyFrames();
  UpdateLocalPoints();
}

void Tracker::UpdateLocalPoints() {
  mvpLocalMapPoints.clear();

  for (std::vector<KeyFrame*>::const_iterator itKF = mvpLocalKeyFrames.begin(); 
                                              itKF != mvpLocalKeyFrames.end(); 
                                              ++itKF)
  {
    KeyFrame* pKF = *itKF;
    const std::vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

    for(std::vector<MapPoint*>::const_iterator itMP = vpMPs.begin();
                                               itMP != vpMPs.end();
                                               ++itMP)
    {
      MapPoint* pMP = *itMP;
      if (!pMP || (pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId)) {
        continue;
      }

      if (!pMP->isBad()) {
        mvpLocalMapPoints.push_back(pMP);
        pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
      }
    }
  }
}

void Tracker::UpdateLocalKeyFrames() {
  // Each map point vote for the keyframes in which it has been observed
  std::map<KeyFrame*,int> keyframeCounter;
  for (int i = 0; i < mCurrentFrame.mN; ++i) {
    if (mCurrentFrame.mvpMapPoints[i]) {
      MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
      if (!pMP->isBad()) {
        const std::map<KeyFrame*,size_t> observations = pMP->GetObservations();
        for (std::map<KeyFrame*,size_t>::const_iterator it = observations.begin(); 
                                                        it != observations.end(); 
                                                        ++it) 
        {
          keyframeCounter[it->first]++;
        }
      } else {
        mCurrentFrame.mvpMapPoints[i] = nullptr;
      }
    }
  }

  if (keyframeCounter.empty()) {
    return;
  }

  int max = 0;
  KeyFrame* pKFmax= nullptr;

  mvpLocalKeyFrames.clear();
  mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

  // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
  for (std::map<KeyFrame*,int>::const_iterator it = keyframeCounter.begin();
                                               it != keyframeCounter.end(); 
                                               ++it)
  {
      KeyFrame* pKF = it->first;

      if (pKF->isBad()) {
        continue;
      }

      if (it->second > max) {
        max = it->second;
        pKFmax = pKF;
      }

      mvpLocalKeyFrames.push_back(it->first);
      pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
  }


  // Include also some not-already-included keyframes that are neighbors to already-included keyframes
  for (std::vector<KeyFrame*>::const_iterator itKF = mvpLocalKeyFrames.begin();
                                              itKF != mvpLocalKeyFrames.end(); 
                                              ++itKF)
  {
    // Limit the number of keyframes
    if(mvpLocalKeyFrames.size() > 80) {
      break;
    }

    KeyFrame* pKF = *itKF;

    const std::vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

    for (std::vector<KeyFrame*>::const_iterator itNeighKF = vNeighs.begin(); 
                                                itNeighKF != vNeighs.end(); 
                                                ++itNeighKF)
    {
      KeyFrame* pNeighKF = *itNeighKF;
      if (!pNeighKF->isBad() 
          && pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
        mvpLocalKeyFrames.push_back(pNeighKF);
        pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
        break;
      }
    }

    const std::set<KeyFrame*> spChilds = pKF->GetChilds();
    for (std::set<KeyFrame*>::const_iterator sit = spChilds.begin(); 
                                             sit != spChilds.end(); 
                                             ++sit)
    {
      KeyFrame* pChildKF = *sit;
      if (!pChildKF->isBad() 
          && pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
        mvpLocalKeyFrames.push_back(pChildKF);
        pChildKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
        break;
      }
    }

    KeyFrame* pParent = pKF->GetParent();
    if (pParent
        && pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
      mvpLocalKeyFrames.push_back(pParent);
      pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;
      break;
    }
  }

  if (pKFmax) {
    mpReferenceKF = pKFmax;
    mCurrentFrame.mpReferenceKF = mpReferenceKF;
  }
}

bool Tracker::TrackLocalMap() {
  // We have an estimation of the camera pose and some map points tracked in the frame.
  // We retrieve the local map and try to find matches to points in the local map.

  UpdateLocalMap();
  SearchLocalPoints();

  // Optimize Pose
  Optimizer::PoseOptimization(&mCurrentFrame);
  mnMatchesInliers = 0;

  // Update MapPoints Statistics
  for (int i = 0; i < mCurrentFrame.mN; ++i) {
    if (mCurrentFrame.mvpMapPoints[i]) {
      if (!mCurrentFrame.mvbOutlier[i]) {
        mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
        if (!mbOnlyTracking) {
          if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0) {
            ++mnMatchesInliers;
          }
        } else {
          ++mnMatchesInliers;
        }
      } else if (mSensor==SENSOR_TYPE::STEREO) {
        mCurrentFrame.mvpMapPoints[i] = nullptr;
      }
    }
  }

  // Decide if the tracking was succesful
  // More restrictive if there was a relocalization recently
  if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames 
      && mnMatchesInliers < 50) {
    return false;
  }
  if (mnMatchesInliers < 30) {
    return false;
  }
  return true;
}

void Tracker::SearchLocalPoints() {
  // Do not search map points already matched
  for (std::vector<MapPoint*>::iterator vit = mCurrentFrame.mvpMapPoints.begin(); 
                                        vit != mCurrentFrame.mvpMapPoints.end(); 
                                        ++vit)
  {
    MapPoint* pMP = *vit;
    if (pMP) {
      if (pMP->isBad()) {
        // *vit = nullptr;
        pMP = nullptr;
      } else {
        pMP->IncreaseVisible();
        pMP->mnLastFrameSeen = mCurrentFrame.mnId;
        pMP->mbTrackInView = false;
      }
    }
  }

  // Project points in frame and check its visibility
  int nToMatch = 0;
  for (std::vector<MapPoint*>::iterator vit = mvpLocalMapPoints.begin(); 
                                        vit != mvpLocalMapPoints.end(); 
                                        ++vit)
  {
    MapPoint* pMP = *vit;
    if (pMP->mnLastFrameSeen == mCurrentFrame.mnId
        || pMP->isBad()) {
      continue;
    }
    // Project (this fills MapPoint variables for matching)
    if (mCurrentFrame.isInFrustum(pMP, 0.5)) {
      pMP->IncreaseVisible();
      ++nToMatch;
    }
  }

  if (nToMatch > 0) {
    OrbMatcher matcher(0.8);
    int th = 1;
    if (mSensor == SENSOR_TYPE::RGBD) {
      th = 3;
    }
    // If the camera has been relocalised recently, perform a coarser search
    if (mCurrentFrame.mnId < mnLastRelocFrameId+2) {
      th = 5;
    }
    matcher.SearchByProjection(mCurrentFrame,
                               mvpLocalMapPoints,
                               th);
  }
}

bool Tracker::NeedNewKeyFrame() {
  if (mbOnlyTracking) {
    return false;
  }

  // If Local Mapping is frozen by a loop closure do not insert keyframes
  if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested()) {
    return false;
  }

  const int nKFs = mpMap->KeyFramesInMap();

  // Do not insert keyframes if not enough frames have passed from last relocalisation
  if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && nKFs > mMaxFrames) {
    return false;
  }

  // Tracked MapPoints in the reference keyframe
  int nMinObs = 3;
  if (nKFs <= 2) {
    nMinObs = 2;
  }
  int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

  // Local Mapping accept keyframes?
  bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

  // Check how many "close" points are being tracked and how many could be potentially created.
  int nNonTrackedClose = 0;
  int nTrackedClose = 0;
  if (mSensor != SENSOR_TYPE::MONOCULAR) {
    for (int i = 0; i < mCurrentFrame.mN; ++i) {
      if (mCurrentFrame.mvDepth[i] > 0 && mCurrentFrame.mvDepth[i] < mThDepth) {
        if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i]) {
          ++nTrackedClose;
        } else {
          ++nNonTrackedClose;
        }
      }
    }
  }

  bool bNeedToInsertClose = ((nTrackedClose < 100) && (nNonTrackedClose > 70));

  // Thresholds
  float thRefRatio = 0.75f;
  if (nKFs < 2) {
    thRefRatio = 0.4f;
  }

  if (mSensor==SENSOR_TYPE::MONOCULAR) {
    thRefRatio = 0.9f;
  }

  // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
  const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames;
  // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
  const bool c1b = (mCurrentFrame.mnId >= mnLastKeyFrameId + mMinFrames && bLocalMappingIdle);
  //Condition 1c: tracking is weak
  const bool c1c =  mSensor != SENSOR_TYPE::MONOCULAR && (mnMatchesInliers < 0.25*nRefMatches || bNeedToInsertClose);
  // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
  const bool c2 = ((mnMatchesInliers < nRefMatches * thRefRatio || bNeedToInsertClose) && mnMatchesInliers > 15);

  if ((c1a||c1b||c1c) && c2) {
    // If the mapping accepts keyframes, insert keyframe.
    // Otherwise send a signal to interrupt BA
    if (bLocalMappingIdle) {
      return true;
    } else {
      mpLocalMapper->InterruptBA();
      if (mSensor != SENSOR_TYPE::MONOCULAR) {
        if (mpLocalMapper->NumKeyframesInQueue() < 3) {
          return true;
        } else {
          return false;
        }
      } else {
        return false;
      }
    }
  } else {
    return false;
  }
}

void Tracker::CreateNewKeyFrame() {
  
  if (!mpLocalMapper->SetNotStop(true)) {
    return;
  }

  KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

  mpReferenceKF = pKF;
  mCurrentFrame.mpReferenceKF = pKF;

  if (mSensor != SENSOR_TYPE::MONOCULAR) {
    mCurrentFrame.UpdatePoseMatrices();

    // We sort points by the measured depth by the stereo/RGBD sensor.
    // We create all those MapPoints whose depth < mThDepth.
    // If there are less than 100 close points we create the 100 closest.
    std::vector<std::pair<float,int>> vDepthIdx;
    vDepthIdx.reserve(mCurrentFrame.mN);
    for (int i = 0; i < mCurrentFrame.mN; ++i) {
      float z = mCurrentFrame.mvDepth[i];
      if (z > 0) {
        vDepthIdx.push_back(std::make_pair(z,i));
      }
    }

    if (!vDepthIdx.empty()) {
      std::sort(vDepthIdx.begin(), vDepthIdx.end());

      int nPoints = 0;
      for (size_t j = 0; j < vDepthIdx.size(); ++j) {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
        if (!pMP) {
          bCreateNew = true;
        } else if (pMP->Observations() < 1) {
          bCreateNew = true;
          mCurrentFrame.mvpMapPoints[i] = nullptr;
        }

        if (bCreateNew) {
          cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
          MapPoint* pNewMP = new MapPoint(x3D, pKF, mpMap);
          pNewMP->AddObservation(pKF, i);
          pKF->AddMapPoint(pNewMP, i);
          pNewMP->ComputeDistinctiveDescriptors();
          pNewMP->UpdateNormalAndDepth();
          mpMap->AddMapPoint(pNewMP);

          mCurrentFrame.mvpMapPoints[i] = pNewMP;
        }
        ++nPoints;

        if (vDepthIdx[j].first > mThDepth && nPoints > 100) {
          break;
        }
      }
    }
  }

  mpLocalMapper->InsertKeyFrame(pKF);
  mpLocalMapper->SetNotStop(false);

  mnLastKeyFrameId = mCurrentFrame.mnId;
  mpLastKeyFrame = pKF;
}
