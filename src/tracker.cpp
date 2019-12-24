#include "tracker.h"

#include <unistd.h> // usleep
#include <iostream>
#include <mutex>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "orb_matcher.h"
#include "util/converter.h"
#include "map.h"
#include "optimizer.h"
#include "solvers/pnp_solver.h"


Tracker::Tracker(const std::shared_ptr<OrbVocabulary>& orb_vocabulary, 
                 const std::shared_ptr<Map>& map,
                 const std::shared_ptr<KeyframeDatabase>& keyframe_db, 
                 const std::string& settings_path, // TODO
                 const SENSOR_TYPE sensor)
      : use_visual_odometry_(false)
      , orb_vocabulary_(orb_vocabulary)
      , keyframe_db_(keyframe_db)
      , mpMap(map)
      , mnLastRelocFrameId(0)
      , state_(TrackingState::NO_IMAGES_YET)
      , sensor_type_(sensor)
      , mbOnlyTracking(false) {
  // Load camera parameters from settings file
  // cv::FileStorage fSettings(settings_path, cv::FileStorage::READ);
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

  orb_extractor_left_ = std::make_shared<ORBextractor>(nFeatures,
                                                       fScaleFactor,
                                                       nLevels,
                                                       fIniThFAST,
                                                       fMinThFAST);
  if (sensor == SENSOR_TYPE::STEREO) {
    orb_extractor_right_ = std::make_shared<ORBextractor>(nFeatures,
                                                          fScaleFactor,
                                                          nLevels,
                                                          fIniThFAST,
                                                          fMinThFAST);
  }
  if (sensor == SENSOR_TYPE::MONOCULAR) {
    mpIniORBextractor = std::make_shared<ORBextractor>(2*nFeatures,
                                                       fScaleFactor,
                                                       nLevels,
                                                       fIniThFAST,
                                                       fMinThFAST);
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


cv::Mat Tracker::GrabImageStereo(const cv::Mat& left_image,
                                 const cv::Mat& right_image, 
                                 const double timestamp) {
  cv::Mat gray_image = left_image;
  cv::Mat imGrayRight = right_image;

  // Convert to grayscale if image is RGB
  if (gray_image.channels() == 3) {
    if (mbRGB) {
      cv::cvtColor(gray_image,gray_image, CV_RGB2GRAY);
      cv::cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
    } else {
      cvtColor(gray_image,gray_image,CV_BGR2GRAY);
      cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
    }
  } else if (gray_image.channels() == 4) {
    if (mbRGB) {
      cvtColor(gray_image,gray_image,CV_RGBA2GRAY);
      cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
    } else {
      cvtColor(gray_image,gray_image,CV_BGRA2GRAY);
      cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
    }
  } 

  current_frame_ = Frame(gray_image,
                         imGrayRight,
                         timestamp,
                         orb_extractor_left_,
                         orb_extractor_right_,
                         orb_vocabulary_,
                         mK,
                         mDistCoef,
                         mbf,
                         mThDepth);
  Track();
  return current_frame_.mTcw.clone();
}

cv::Mat Tracker::GrabImageRGBD(const cv::Mat& rgbd_image,
                               const cv::Mat& depth_image, 
                               const double timestamp) {
  cv::Mat gray_image = rgbd_image;
  cv::Mat imDepth = depth_image;

  if (gray_image.channels() == 3) {
    if(mbRGB) {
      cvtColor(gray_image,gray_image,CV_RGB2GRAY);
    } else {
      cvtColor(gray_image,gray_image,CV_BGR2GRAY);
    }
  } else if (gray_image.channels() == 4) {
    if(mbRGB) {
      cvtColor(gray_image,gray_image,CV_RGBA2GRAY);
    } else {
      cvtColor(gray_image,gray_image,CV_BGRA2GRAY);
    }
  }

  if ((std::fabs(mDepthMapFactor-1.0f) > 1e-5) || imDepth.type()!=CV_32F) {
    imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);
  }

  current_frame_ = Frame(gray_image,
                         imDepth,
                         timestamp,
                         orb_extractor_left_,
                          orb_vocabulary_,
                          mK,
                         mDistCoef,
                          mbf,
                         mThDepth);
  Track();
  return current_frame_.mTcw.clone();                      
}

cv::Mat Tracker::GrabImageMonocular(const cv::Mat& image, 
                                    const double timestamp) {
  cv::Mat gray_image = image;

  if (gray_image.channels() == 3) {
    if(mbRGB) {
      cvtColor(gray_image,gray_image,CV_RGB2GRAY);
    } else {
      cvtColor(gray_image,gray_image,CV_BGR2GRAY);
    }
  } else if(gray_image.channels() == 4) {
    if(mbRGB) {
      cvtColor(gray_image,gray_image,CV_RGBA2GRAY);
    } else {
      cvtColor(gray_image,gray_image,CV_BGRA2GRAY);
    }
  }

  if (state_ == NOT_INITIALIZED || state_ == NO_IMAGES_YET) {
    current_frame_ = Frame(gray_image,
                           timestamp,
                           mpIniORBextractor,
                           orb_vocabulary_,
                           mK,
                           mDistCoef,
                           mbf,
                           mThDepth);
  } else {
    current_frame_ = Frame(gray_image,
                           timestamp,
                           orb_extractor_left_,
                            orb_vocabulary_,
                            mK,
                           mDistCoef,
                            mbf,
                           mThDepth);
  }
  Track();
  return current_frame_.mTcw.clone();
}

bool Tracker::NeedSystemReset() const {
  return system_reset_needed_;
}

void Tracker::Reset() {
  std::cout << "System resetting...\n";
  
  local_mapper_->RequestReset();
  loop_closer_->RequestReset();

  keyframe_db_->clear();

  mpMap->clear();

  KeyFrame::nNextId = 0;
  Frame::nNextId = 0;
  state_ = TrackingState::NO_IMAGES_YET;

  mpInitializer.reset(nullptr);

  mlRelativeFramePoses.clear();
  mlpReferences.clear();
  mlFrameTimes.clear();
  mlbLost.clear();
  system_reset_needed_ = false;
}


void Tracker::StereoInitialization() {
  if (current_frame_.mN > 500) {
    // Set Frame pose to the origin
    current_frame_.SetPose(cv::Mat::eye(4,4,CV_32F));

    // Create KeyFrame
    KeyFrame* pKFini = new KeyFrame(current_frame_, mpMap, keyframe_db_);

    // Insert KeyFrame in the map
    mpMap->AddKeyFrame(pKFini);

    // Create MapPoints and asscoiate to KeyFrame
    for (int i=0; i < current_frame_.mN; ++i) {
      float z = current_frame_.mvDepth[i];
      if (z > 0) {
        cv::Mat x3D = current_frame_.UnprojectStereo(i);
        MapPoint* pNewMP = new MapPoint(x3D, pKFini, mpMap);
        pNewMP->AddObservation(pKFini,i);
        pKFini->AddMapPoint(pNewMP,i);
        pNewMP->ComputeDistinctiveDescriptors();
        pNewMP->UpdateNormalAndDepth();
        mpMap->AddMapPoint(pNewMP);

        current_frame_.mvpMapPoints[i] = pNewMP;
      }
    }

    std::cout << "New map created with " << mpMap->MapPointsInMap() << " points" << std::endl;

    local_mapper_->InsertKeyFrame(pKFini);

    mLastFrame = Frame(current_frame_);
    mnLastKeyFrameId = current_frame_.mnId;
    mpLastKeyFrame = pKFini;

    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints = mpMap->GetAllMapPoints();

    mpReferenceKF = pKFini;
    current_frame_.mpReferenceKF = pKFini;

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    state_ = TrackingState::OK;
  }
}

void Tracker::MonocularInitialization() {

  if (!mpInitializer) {
    // Set Reference Frame
    if (current_frame_.mvKeys.size() > 100) {
      mInitialFrame = Frame(current_frame_);
      mLastFrame = Frame(current_frame_);

      mvbPrevMatched.resize(current_frame_.mvKeysUn.size());
      for (size_t i = 0; i < current_frame_.mvKeysUn.size(); ++i) {
        mvbPrevMatched[i]=current_frame_.mvKeysUn[i].pt;
      }

      mpInitializer.reset(new Initializer(current_frame_, 1.0, 200));

      std::fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
      return;
    }
  } else {
    // Try to initialize
    if (current_frame_.mvKeys.size() <= 100u) {
      mpInitializer.reset(nullptr);
      std::fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
      return;
    }

    // Find correspondences
    OrbMatcher matcher(0.9, true);
    int nmatches = matcher.SearchForInitialization(mInitialFrame,
                                                   current_frame_,
                                                   mvbPrevMatched,
                                                   mvIniMatches,
                                                   100);

    // Check if there are enough correspondences
    if (nmatches < 100) {
      mpInitializer.reset(nullptr);
      return;
    }

    cv::Mat Rcw; // Current Camera Rotation
    cv::Mat tcw; // Current Camera Translation
    std::vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches) // TODO boolvector

    if (mpInitializer->Initialize(current_frame_, 
                                  mvIniMatches,
                                  Rcw,
                                  tcw,
                                  mvIniP3D,
                                  vbTriangulated)) {
      for (size_t i = 0; i < mvIniMatches.size(); ++i) {
        if (mvIniMatches[i] >= 0 && !vbTriangulated[i]) {
          mvIniMatches[i] = -1;
          --nmatches;
        }
      }

      // Set Frame Poses
      mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
      cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
      Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
      tcw.copyTo(Tcw.rowRange(0,3).col(3));
      current_frame_.SetPose(Tcw);

      CreateInitialMapMonocular();
    }
  }
}

void Tracker::CreateInitialMapMonocular() {
  // Create KeyFrames
  KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpMap,keyframe_db_);
  KeyFrame* pKFcur = new KeyFrame(current_frame_,mpMap,keyframe_db_);

  pKFini->ComputeBoW();
  pKFcur->ComputeBoW();

  // Insert KFs in the map
  mpMap->AddKeyFrame(pKFini);
  mpMap->AddKeyFrame(pKFcur);

  // Create MapPoints and asscoiate to keyframes
  for (size_t i=0; i<mvIniMatches.size();++i) {
    if(mvIniMatches[i] < 0) {
      continue;
    }

    //Create MapPoint.
    cv::Mat worldPos(mvIniP3D[i]);

    MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpMap);

    pKFini->AddMapPoint(pMP,i);
    pKFcur->AddMapPoint(pMP,mvIniMatches[i]);

    pMP->AddObservation(pKFini,i);
    pMP->AddObservation(pKFcur,mvIniMatches[i]);

    pMP->ComputeDistinctiveDescriptors();
    pMP->UpdateNormalAndDepth();

    //Fill Current Frame structure
    current_frame_.mvpMapPoints[mvIniMatches[i]] = pMP;
    current_frame_.mvbOutlier[mvIniMatches[i]] = false;

    //Add to Map
    mpMap->AddMapPoint(pMP);
  }

  // Update Connections
  pKFini->UpdateConnections();
  pKFcur->UpdateConnections();

  // Bundle Adjustment
  std::cout << "New Map created with " << mpMap->MapPointsInMap() << " points\n";

  Optimizer::GlobalBundleAdjustemnt(mpMap,20);

  // Set median depth to 1
  float medianDepth = pKFini->ComputeSceneMedianDepth(2);
  float invMedianDepth = 1.0f / medianDepth;

  if(medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 100) {
    std::cout << "Wrong initialization, reseting...\n";
    Reset();
    return;
  }

  // Scale initial baseline
  cv::Mat Tc2w = pKFcur->GetPose();
  Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3) * invMedianDepth;
  pKFcur->SetPose(Tc2w);

  // Scale points
  std::vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
  for (size_t iMP = 0; iMP < vpAllMapPoints.size(); ++iMP) {
    if (vpAllMapPoints[iMP]) {
      MapPoint* pMP = vpAllMapPoints[iMP];
      pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
    }
  }

  local_mapper_->InsertKeyFrame(pKFini);
  local_mapper_->InsertKeyFrame(pKFcur);

  current_frame_.SetPose(pKFcur->GetPose());
  mnLastKeyFrameId = current_frame_.mnId;
  mpLastKeyFrame = pKFcur;

  mvpLocalKeyFrames.push_back(pKFcur);
  mvpLocalKeyFrames.push_back(pKFini);
  mvpLocalMapPoints = mpMap->GetAllMapPoints();
  mpReferenceKF = pKFcur;
  current_frame_.mpReferenceKF = pKFcur;

  mLastFrame = Frame(current_frame_);

  mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

  mpMap->mvpKeyFrameOrigins.push_back(pKFini);

  state_ = TrackingState::OK;
}

void Tracker::Track() {

  if (state_ == TrackingState::NO_IMAGES_YET) {
    state_ = NOT_INITIALIZED;
  }

  last_processed_state_ = state_;

  std::unique_lock<std::mutex> lock(mpMap->mMutexMapUpdate);

  if (state_ == TrackingState::NOT_INITIALIZED) {
    if(sensor_type_ == SENSOR_TYPE::STEREO || sensor_type_ == SENSOR_TYPE::RGBD) {
      StereoInitialization();
    } else {
      MonocularInitialization();
    }
    if (state_ != OK) {
      return;
    }
  } else {
    // System is initialized. Track Frame.
    bool bOK = false;

    // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
    if (!mbOnlyTracking) {
      // Local Mapping is activated. This is the normal behaviour, unless
      // you explicitly activate the "only tracking" mode.
      if (state_ == TrackingState::OK) {
        // Local Mapping might have changed some MapPoints tracked in last frame
        ReplaceInLastFrame();

        if (mVelocity.empty() || current_frame_.mnId < mnLastRelocFrameId + 2) {
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
    }  else {
      std::cout << "Localization-only Mode\n";  
      // Localization Mode: Local Mapping is deactivated
      if (state_ == LOST) {
        bOK = Relocalization();
      } else {
        if (use_visual_odometry_) {
          // In last frame we tracked mainly "visual odometry" points.
          // We compute two camera poses, one from motion model and one doing relocalization.
          // If relocalization is sucessfull we choose that solution, otherwise we retain
          // the "visual odometry" solution.
          bool bOKMM = false;
          bool bOKReloc = false;
          std::vector<MapPoint*> vpMPsMM;
          std::vector<bool> vbOutMM;
          cv::Mat TcwMM;
          if (!mVelocity.empty()) {
            bOKMM = TrackWithMotionModel();
            vpMPsMM = current_frame_.mvpMapPoints;
            vbOutMM = current_frame_.mvbOutlier;
            TcwMM = current_frame_.mTcw.clone();
          }
          bOKReloc = Relocalization();

          if (bOKMM && !bOKReloc) {
            current_frame_.SetPose(TcwMM);
            current_frame_.mvpMapPoints = vpMPsMM;
            current_frame_.mvbOutlier = vbOutMM;

            if (use_visual_odometry_) {
              for (int i = 0; i < current_frame_.mN; ++i) {
                if (current_frame_.mvpMapPoints[i] && !current_frame_.mvbOutlier[i]) {
                  current_frame_.mvpMapPoints[i]->IncreaseFound();
                }
              }
            }
          } else if(bOKReloc) {
            use_visual_odometry_ = false;
          }
          bOK = (bOKReloc || bOKMM);
        } else {
          // In last frame we tracked enough MapPoints in the map
          if (!mVelocity.empty()) {
            bOK = TrackWithMotionModel();
          } else {
            bOK = TrackReferenceKeyFrame();
          }
        }
      }
    }
    current_frame_.mpReferenceKF = mpReferenceKF;

    // If we have an initial estimation of the camera pose and matching. Track the local map.
    if (!mbOnlyTracking) {
      if (bOK) {
        bOK = TrackLocalMap();
      }
    } else {
      // use_visual_odometry_ true means that there are few matches to MapPoints in the map. We cannot retrieve
      // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
      // the camera we will use the local map again.
      if (bOK && !use_visual_odometry_) {
        bOK = TrackLocalMap();
      }
    }

    if (bOK) {
      state_ = TrackingState::OK;
    } else {
      state_ = TrackingState::LOST;
    }
        
    // If tracking were good, check if we insert a keyframe
    if (bOK) {
      // Update motion model
      if (!mLastFrame.mTcw.empty()) {
        cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
        mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
        mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
        mVelocity = current_frame_.mTcw*LastTwc;
      } else {
        mVelocity = cv::Mat();
      }

      // Clean VO matches
      for (int i=0; i < current_frame_.mN; ++i) {
        MapPoint* pMP = current_frame_.mvpMapPoints[i];
        if (pMP && pMP->Observations() < 1) {
          current_frame_.mvbOutlier[i] = false; 
          current_frame_.mvpMapPoints[i] = nullptr;
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
      for (int i=0; i < current_frame_.mN; ++i) {
        if (current_frame_.mvpMapPoints[i] && current_frame_.mvbOutlier[i]) {
          current_frame_.mvpMapPoints[i] = nullptr;
        }
      }
    }

    // Reset if the camera gets lost soon after initialization
    if (state_ == TrackingState::LOST) {
      if (mpMap->KeyFramesInMap() <= 5) {
        std::cout << "Track lost soon after initialisation, resetting...\n";
        system_reset_needed_ = true;
        return;
      }
    }

    if (!current_frame_.mpReferenceKF) {
      current_frame_.mpReferenceKF = mpReferenceKF;
    }

    mLastFrame = Frame(current_frame_);
  }

  // Store frame pose information to retrieve the complete camera trajectory afterwards.
  if (!current_frame_.mTcw.empty()) {
    cv::Mat Tcr = current_frame_.mTcw * current_frame_.mpReferenceKF->GetPoseInverse();
    mlRelativeFramePoses.push_back(Tcr);
    mlpReferences.push_back(mpReferenceKF);
    mlFrameTimes.push_back(current_frame_.mTimeStamp);
    mlbLost.push_back( state_ == TrackingState::LOST );
  } else {
    // This can happen if tracking is lost
    mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
    mlpReferences.push_back(mlpReferences.back());
    mlFrameTimes.push_back(mlFrameTimes.back());
    mlbLost.push_back( state_ == TrackingState::LOST );
  }
}

void Tracker::ReplaceInLastFrame() {
  for (int i = 0; i < mLastFrame.mN; ++i) {
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
  current_frame_.ComputeBoW();

  // We perform first an ORB matching with the reference keyframe
  // If enough matches are found we setup a PnP solver
  OrbMatcher matcher(0.7, true);
  std::vector<MapPoint*> vpMapPointMatches;

  int nmatches = matcher.SearchByBoW(mpReferenceKF,current_frame_,vpMapPointMatches);
  if (nmatches < num_required_matches_) {
    return false;
  }

  current_frame_.mvpMapPoints = vpMapPointMatches;
  current_frame_.SetPose(mLastFrame.mTcw);

  Optimizer::PoseOptimization(&current_frame_);

  // Discard outliers
  int nmatchesMap = 0;
  for (int i = 0; i < current_frame_.mN; ++i) {
    if (current_frame_.mvpMapPoints[i]) {
      if (current_frame_.mvbOutlier[i]) {
        MapPoint* pMP = current_frame_.mvpMapPoints[i];

        current_frame_.mvpMapPoints[i] = nullptr;
        current_frame_.mvbOutlier[i] = false;
        pMP->mbTrackInView = false;
        pMP->mnLastFrameSeen = current_frame_.mnId;
        --nmatches; // TODO never seem to use  this again
      } else if (current_frame_.mvpMapPoints[i]->Observations() > 0) {
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
      || sensor_type_==SENSOR_TYPE::MONOCULAR 
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

  current_frame_.SetPose(mVelocity * mLastFrame.mTcw);

  std::fill(current_frame_.mvpMapPoints.begin(),
            current_frame_.mvpMapPoints.end(),
            nullptr);

  // Project points seen in previous frame
  int th;
  if (sensor_type_ != SENSOR_TYPE::STEREO) {
    th=15;
  } else {
    th=7;
  }
  int nmatches = matcher.SearchByProjection(current_frame_,
                                            mLastFrame,
                                            th,
                                            sensor_type_==SENSOR_TYPE::MONOCULAR);

  // If few matches, uses a wider window search
  if(nmatches < 20) {
    std::fill(current_frame_.mvpMapPoints.begin(),
              current_frame_.mvpMapPoints.end(),
              nullptr);
    nmatches = matcher.SearchByProjection(current_frame_,
                                          mLastFrame,
                                          2*th,
                                          sensor_type_==SENSOR_TYPE::MONOCULAR);
  }

  if (nmatches < 20) {
    return false;
  }
      
  // Optimize frame pose with all matches
  Optimizer::PoseOptimization(&current_frame_);

  // Discard outliers
  int nmatchesMap = 0;
  for (int i = 0; i < current_frame_.mN; ++i) {
    if (current_frame_.mvpMapPoints[i]) {
      if (current_frame_.mvbOutlier[i]) {
        MapPoint* pMP = current_frame_.mvpMapPoints[i];

        current_frame_.mvpMapPoints[i] = nullptr;
        current_frame_.mvbOutlier[i] = false;
        pMP->mbTrackInView = false;
        pMP->mnLastFrameSeen = current_frame_.mnId;
        --nmatches;
      } else if (current_frame_.mvpMapPoints[i]->Observations() > 0) {
        ++nmatchesMap;
      }
    }
  }    

  if (mbOnlyTracking) {
    use_visual_odometry_ = (nmatchesMap < 10);
    return (nmatches > 20);
  } else {
    return (nmatchesMap >= 10);
  }  
}

bool Tracker::Relocalization() {
  // Compute Bag of Words Vector
  current_frame_.ComputeBoW();

  // Relocalization is performed when tracking is lost
  // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
  std::vector<KeyFrame*> vpCandidateKFs = keyframe_db_->DetectRelocalizationCandidates(&current_frame_);

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
                                         current_frame_, 
                                         vvpMapPointMatches[i]);
      if(nmatches < 15) {
        vbDiscarded[i] = true;
        continue;
      } else {
        std::unique_ptr<PnPsolver> pSolver;
        pSolver.reset(new PnPsolver(current_frame_,vvpMapPointMatches[i]));
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
        Tcw.copyTo(current_frame_.mTcw);

        std::set<MapPoint*> sFound;

        for (size_t j = 0; j < vbInliers.size(); ++j) {
          if (vbInliers[j]) {
            current_frame_.mvpMapPoints[j] = vvpMapPointMatches[i][j];
            sFound.insert(vvpMapPointMatches[i][j]);
          } else {
            current_frame_.mvpMapPoints[j] = nullptr;
          }
        }

        int nGood = Optimizer::PoseOptimization(&current_frame_);

        if(nGood < 10) {
          continue;
        }

        for (int io = 0; io < current_frame_.mN; ++io) {
          if (current_frame_.mvbOutlier[io]) {
            current_frame_.mvpMapPoints[io] = nullptr;
          }
        }

        // If few inliers, search by projection in a coarse window and optimize again
        if (nGood < 50) {
          int nadditional = matcher2.SearchByProjection(current_frame_,
                                                        vpCandidateKFs[i],
                                                        sFound,
                                                        10,
                                                        100);

          if (nadditional + nGood >= 50) {
            nGood = Optimizer::PoseOptimization(&current_frame_);

            // If many inliers but still not enough, search by projection again in a narrower window
            // the camera has been already optimized with many points
            if (nGood > 30 && nGood < 50) {
              sFound.clear();
              for (int ip = 0; ip < current_frame_.mN; ++ip) {
                if (current_frame_.mvpMapPoints[ip]) {
                  sFound.insert(current_frame_.mvpMapPoints[ip]);
                }
              }
              nadditional = matcher2.SearchByProjection(current_frame_,
                                                        vpCandidateKFs[i],
                                                        sFound,
                                                        3,
                                                        64);

              // Final optimization
              if (nGood + nadditional >= 50) {
                nGood = Optimizer::PoseOptimization(&current_frame_);

                for (int io = 0; io < current_frame_.mN; ++io) {
                  if (current_frame_.mvbOutlier[io]) {
                    current_frame_.mvpMapPoints[io] = nullptr;
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
    mnLastRelocFrameId = current_frame_.mnId;
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
      if (!pMP || (pMP->mnTrackReferenceForFrame == current_frame_.mnId)) {
        continue;
      }

      if (!pMP->isBad()) {
        mvpLocalMapPoints.push_back(pMP);
        pMP->mnTrackReferenceForFrame = current_frame_.mnId;
      }
    }
  }
}

void Tracker::UpdateLocalKeyFrames() {
  // Each map point vote for the keyframes in which it has been observed
  std::map<KeyFrame*,int> keyframeCounter;
  for (int i = 0; i < current_frame_.mN; ++i) {
    if (current_frame_.mvpMapPoints[i]) {
      MapPoint* pMP = current_frame_.mvpMapPoints[i];
      if (!pMP->isBad()) {
        const std::map<KeyFrame*,size_t> observations = pMP->GetObservations();
        for (std::map<KeyFrame*,size_t>::const_iterator it = observations.begin(); 
                                                        it != observations.end(); 
                                                        ++it) 
        {
          keyframeCounter[it->first]++;
        }
      } else {
        current_frame_.mvpMapPoints[i] = nullptr;
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
      pKF->mnTrackReferenceForFrame = current_frame_.mnId;
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
          && pNeighKF->mnTrackReferenceForFrame != current_frame_.mnId) {
        mvpLocalKeyFrames.push_back(pNeighKF);
        pNeighKF->mnTrackReferenceForFrame = current_frame_.mnId;
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
          && pChildKF->mnTrackReferenceForFrame != current_frame_.mnId) {
        mvpLocalKeyFrames.push_back(pChildKF);
        pChildKF->mnTrackReferenceForFrame = current_frame_.mnId;
        break;
      }
    }

    KeyFrame* pParent = pKF->GetParent();
    if (pParent
        && pParent->mnTrackReferenceForFrame != current_frame_.mnId) {
      mvpLocalKeyFrames.push_back(pParent);
      pParent->mnTrackReferenceForFrame = current_frame_.mnId;
      break;
    }
  }

  if (pKFmax) {
    mpReferenceKF = pKFmax;
    current_frame_.mpReferenceKF = mpReferenceKF;
  }
}

bool Tracker::TrackLocalMap() {
  // We have an estimation of the camera pose and some map points tracked in the frame.
  // We retrieve the local map and try to find matches to points in the local map.

  UpdateLocalMap();
  SearchLocalPoints();

  // Optimize Pose
  Optimizer::PoseOptimization(&current_frame_);
  mnMatchesInliers = 0;

  // Update MapPoints Statistics
  for (int i = 0; i < current_frame_.mN; ++i) {
    if (current_frame_.mvpMapPoints[i]) {
      if (!current_frame_.mvbOutlier[i]) {
        current_frame_.mvpMapPoints[i]->IncreaseFound();
        if (!mbOnlyTracking) {
          if (current_frame_.mvpMapPoints[i]->Observations() > 0) {
            ++mnMatchesInliers;
          }
        } else {
          ++mnMatchesInliers;
        }
      } else if (sensor_type_==SENSOR_TYPE::STEREO) {
        current_frame_.mvpMapPoints[i] = nullptr;
      }
    }
  }

  // Decide if the tracking was succesful
  // More restrictive if there was a relocalization recently
  if (current_frame_.mnId < mnLastRelocFrameId + mMaxFrames 
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
  for (std::vector<MapPoint*>::iterator vit = current_frame_.mvpMapPoints.begin(); 
                                        vit != current_frame_.mvpMapPoints.end(); 
                                        ++vit)
  {
    MapPoint* pMP = *vit;
    if (pMP) {
      if (pMP->isBad()) {
        // *vit = nullptr;
        pMP = nullptr;
      } else {
        pMP->IncreaseVisible();
        pMP->mnLastFrameSeen = current_frame_.mnId;
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
    if (pMP->mnLastFrameSeen == current_frame_.mnId
        || pMP->isBad()) {
      continue;
    }
    // Project (this fills MapPoint variables for matching)
    if (current_frame_.isInFrustum(pMP, 0.5)) {
      pMP->IncreaseVisible();
      ++nToMatch;
    }
  }

  if (nToMatch > 0) {
    OrbMatcher matcher(0.8);
    int th = 1;
    if (sensor_type_ == SENSOR_TYPE::RGBD) {
      th = 3;
    }
    // If the camera has been relocalised recently, perform a coarser search
    if (current_frame_.mnId < mnLastRelocFrameId+2) {
      th = 5;
    }
    matcher.SearchByProjection(current_frame_,
                               mvpLocalMapPoints,
                               th);
  }
}

bool Tracker::NeedNewKeyFrame() {
  if (mbOnlyTracking) {
    return false;
  }

  // If Local Mapping is frozen by a loop closure do not insert keyframes
  if (local_mapper_->isStopped() || local_mapper_->stopRequested()) {
    return false;
  }

  const int nKFs = mpMap->KeyFramesInMap();

  // Do not insert keyframes if not enough frames have passed from last relocalisation
  if (current_frame_.mnId < mnLastRelocFrameId + mMaxFrames && nKFs > mMaxFrames) {
    return false;
  }

  // Tracked MapPoints in the reference keyframe
  const int nMinObs = (nKFs <= 2) ? 2 : 3;
  int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

  // Local Mapping accept keyframes?
  bool bLocalMappingIdle = local_mapper_->AcceptKeyFrames();

  // Check how many "close" points are being tracked and how many could be potentially created.
  int nNonTrackedClose = 0;
  int nTrackedClose = 0;
  if (sensor_type_ != SENSOR_TYPE::MONOCULAR) {
    for (int i = 0; i < current_frame_.mN; ++i) {
      if (current_frame_.mvDepth[i] > 0 && current_frame_.mvDepth[i] < mThDepth) {
        if (current_frame_.mvpMapPoints[i] && !current_frame_.mvbOutlier[i]) {
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

  if (sensor_type_==SENSOR_TYPE::MONOCULAR) {
    thRefRatio = 0.9f;
  }

  // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
  const bool c1a = current_frame_.mnId >= mnLastKeyFrameId + mMaxFrames;
  // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
  const bool c1b = (current_frame_.mnId >= mnLastKeyFrameId + mMinFrames && bLocalMappingIdle);
  //Condition 1c: tracking is weak
  const bool c1c =  sensor_type_ != SENSOR_TYPE::MONOCULAR && (mnMatchesInliers < 0.25*nRefMatches || bNeedToInsertClose);
  // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
  const bool c2 = ((mnMatchesInliers < nRefMatches * thRefRatio || bNeedToInsertClose) && mnMatchesInliers > 15);

  if ((c1a||c1b||c1c) && c2) {
    // If the mapping accepts keyframes, insert keyframe.
    // Otherwise send a signal to interrupt BA
    if (bLocalMappingIdle) {
      return true;
    } else {
      local_mapper_->InterruptBA();
      if (sensor_type_ != SENSOR_TYPE::MONOCULAR) {
        if (local_mapper_->NumKeyframesInQueue() < 3) {
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
  
  if (!local_mapper_->SetNotStop(true)) {
    return;
  }

  KeyFrame* pKF = new KeyFrame(current_frame_, mpMap, keyframe_db_);

  mpReferenceKF = pKF;
  current_frame_.mpReferenceKF = pKF;

  if (sensor_type_ != SENSOR_TYPE::MONOCULAR) {
    current_frame_.UpdatePoseMatrices();

    // We sort points by the measured depth by the stereo/RGBD sensor.
    // We create all those MapPoints whose depth < mThDepth.
    // If there are less than 100 close points we create the 100 closest.
    std::vector<std::pair<float,int>> vDepthIdx;
    vDepthIdx.reserve(current_frame_.mN);
    for (int i = 0; i < current_frame_.mN; ++i) {
      float z = current_frame_.mvDepth[i];
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

        MapPoint* pMP = current_frame_.mvpMapPoints[i];
        if (!pMP) {
          bCreateNew = true;
        } else if (pMP->Observations() < 1) {
          bCreateNew = true;
          current_frame_.mvpMapPoints[i] = nullptr;
        }

        if (bCreateNew) {
          cv::Mat x3D = current_frame_.UnprojectStereo(i);
          MapPoint* pNewMP = new MapPoint(x3D, pKF, mpMap);
          pNewMP->AddObservation(pKF, i);
          pKF->AddMapPoint(pNewMP, i);
          pNewMP->ComputeDistinctiveDescriptors();
          pNewMP->UpdateNormalAndDepth();
          mpMap->AddMapPoint(pNewMP);

          current_frame_.mvpMapPoints[i] = pNewMP;
        }
        ++nPoints;

        if (vDepthIdx[j].first > mThDepth && nPoints > 100) {
          break;
        }
      }
    }
  }

  local_mapper_->InsertKeyFrame(pKF);
  local_mapper_->SetNotStop(false);

  mnLastKeyFrameId = current_frame_.mnId;
  mpLastKeyFrame = pKF;
}

