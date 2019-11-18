#include "tracker.h"

#include <unistd.h> // usleep

Tracker::Tracker(StereoSlamSystem* pSys, 
                 OrbVocabulary* pVoc, 
                 Map* pMap,
                 KeyframeDatabase* pKFDB, 
                 const std::string& strSettingPath, 
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
    bool bOK;

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
    } else { // TODO maybe don't need to do this. Comment out for now
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
      if(bOK) {
        bOK = TrackLocalMap();
      }
    } else {
      // TODO Disabled for noow since we don't wanna use only tracking
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

      // Delete temporal MapPoints
      for (std::list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(); 
                                          lit != mlpTemporalPoints.end(); 
                                          lit++) 
      {
        MapPoint* pMP = *lit;
        delete pMP;
      }
      mlpTemporalPoints.clear();

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
        mpSystem->Reset(); // TODO maybe rename to FlagReset or something?
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

}

bool Tracker::TrackReferenceKeyFrame() {

}

void Tracker::UpdateLastFrame() {

}

bool Tracker::TrackWithMotionModel() {

}
