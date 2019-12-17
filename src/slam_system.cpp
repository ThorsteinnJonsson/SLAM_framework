#include "slam_system.h"
#include <unistd.h> // usleep


SlamSystem::SlamSystem(const std::string& strVocFile, 
                       const std::string& strSettingsFile, 
                       const SENSOR_TYPE sensor) 
        : mSensor(sensor) 
        , reset_flag_(false)
        , mbActivateLocalizationMode(false)
        , mbDeactivateLocalizationMode(false) {
    
  // Output welcome message
  std::cout << "\n" << "Starting SLAM system." << "\n";

  std::cout << "Input sensor was set to: ";
  if (mSensor == SENSOR_TYPE::MONOCULAR) {
    std::cout << "Monocular" << "\n";
  } else if (mSensor == SENSOR_TYPE::STEREO) {
    std::cout << "Stereo" << "\n";
  } else if (mSensor == SENSOR_TYPE::RGBD) {
    std::cout << "RGB-D" << "\n";
  }

  //Load ORB Vocabulary
  std::cout << "\n" << "Loading ORB Vocabulary. This could take a while..." << "\n";

  mpVocabulary = new OrbVocabulary();
  bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
  if (!bVocLoad) {
      std::cerr << "Wrong path to vocabulary. " << "\n";
      std::cerr << "Failed to open: " << strVocFile << "\n";
      exit(-1);
  }
  std::cout << "Vocabulary loaded!" << "\n" << "\n";

  //Create KeyFrame Database
  mpKeyFrameDatabase = new KeyframeDatabase(*mpVocabulary);

  //Create the Map
  mpMap = new Map();

  //Initialize the Tracking thread
  //(it will live in the main thread of execution, the one that called this constructor)
  mpTracker = std::make_shared<Tracker>(this, 
                                        mpVocabulary, 
                                        mpMap, 
                                        mpKeyFrameDatabase, 
                                        strSettingsFile, 
                                        mSensor);

  //Initialize the Local Mapping thread and launch
  mpLocalMapper = std::make_shared<LocalMapper>(mpMap, 
                                                mSensor==SENSOR_TYPE::MONOCULAR);
  mptLocalMapping.reset(new thread(&LocalMapper::Run, mpLocalMapper));

  //Initialize the Loop Closing thread and launch
  mpLoopCloser = std::make_shared<LoopCloser>(mpMap, 
                                              mpKeyFrameDatabase, 
                                              mpVocabulary, 
                                              mSensor != SENSOR_TYPE::MONOCULAR);
  mptLoopClosing.reset(new thread(&LoopCloser::Run, mpLoopCloser));

  //Set pointers between threads
  mpTracker->SetLocalMapper(mpLocalMapper);
  mpTracker->SetLoopCloser(mpLoopCloser);

  mpLocalMapper->SetLoopCloser(mpLoopCloser);

  mpLoopCloser->SetLocalMapper(mpLocalMapper);

}

SlamSystem::~SlamSystem() {

}

cv::Mat SlamSystem::TrackStereo(const cv::Mat& imLeft, 
                                const cv::Mat& imRight, 
                                const double timestamp) {
  if (mSensor != SENSOR_TYPE::STEREO) {
    cerr << "ERROR: you called TrackStereo but input sensor was not set to STEREO." << endl;
    exit(-1);
  }   

  // Check mode change
  {
    std::unique_lock<std::mutex> lock(mMutexMode);
    if (mbActivateLocalizationMode) {
      mpLocalMapper->RequestStop();

      // Wait until Local Mapping has effectively stopped
      while (!mpLocalMapper->isStopped()) {
        usleep(1000);
      }

      mpTracker->InformOnlyTracking(true);
      mbActivateLocalizationMode = false;
    }
    if (mbDeactivateLocalizationMode) {
      mpTracker->InformOnlyTracking(false);
      mpLocalMapper->Release();
      mbDeactivateLocalizationMode = false;
    }
  }

  // Check reset
  {
    std::unique_lock<std::mutex> lock(mMutexReset);
    if (reset_flag_) {
      mpTracker->Reset();
      reset_flag_ = false;
    }
  }

  cv::Mat Tcw = mpTracker->GrabImageStereo(imLeft, imRight, timestamp);

  std::unique_lock<std::mutex> lock2(mMutexState);
  mTrackingState = mpTracker->mState;
  mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
  mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
  return Tcw;
}


void SlamSystem::FlagReset() {
  std::unique_lock<std::mutex> lock(mMutexReset);
  reset_flag_ = true;  
}

void SlamSystem::ActivateLocalizationMode() {
  std::unique_lock<std::mutex> lock(mMutexMode);
  mbActivateLocalizationMode = true;
}

void SlamSystem::DeactivateLocalizationMode() {
  std::unique_lock<std::mutex> lock(mMutexMode);
  mbDeactivateLocalizationMode = true;
}

bool SlamSystem::MapChanged() {
  static int n = 0;
  int curn = mpMap->GetLastBigChangeIdx();
  if (n < curn) {
    n = curn;
    return true;
  } else {
    return false;
  }
}

void SlamSystem::Shutdown() {
  mpLocalMapper->RequestFinish();
  mpLoopCloser->RequestFinish();

  // Wait until all thread have effectively stopped
  while (!mpLocalMapper->isFinished() || 
         !mpLoopCloser->isFinished()  || 
          mpLoopCloser->isRunningGBA()) {
    usleep(5000);
  }
  mptLocalMapping->join();
  mptLoopClosing->join();
}

int SlamSystem::GetTrackingState() {
  std::unique_lock<std::mutex> lock(mMutexState);
  return mTrackingState;
}

std::vector<MapPoint*> SlamSystem::GetTrackedMapPoints() {
  std::unique_lock<std::mutex> lock(mMutexState);
  return mTrackedMapPoints;
}

std::vector<cv::KeyPoint> SlamSystem::GetTrackedKeyPointsUn() {
  std::unique_lock<std::mutex> lock(mMutexState);
  return mTrackedKeyPointsUn;
}

void SlamSystem::SaveTrajectoryKITTI(const string& filename) const {
  std::cout << std::endl << "Saving camera trajectory to " << filename << " ..." << std::endl;
  if (mSensor==MONOCULAR) {
    std::cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular.\n";
    return;
  }

  std::vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
  std::sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

  // Transform all keyframes so that the first keyframe is at the origin.
  // After a loop closure the first keyframe might not be at the origin.
  cv::Mat T_world_origin = vpKFs[0]->GetPoseInverse();

  std::ofstream f;
  f.open(filename.c_str());
  f << std::fixed;

  // Frame pose is stored relative to its reference keyframe 
  // (which is optimized by BA and pose graph).
  // We need to get first the keyframe pose and then concatenate the relative transformation.
  // Frames not localized (tracking failure) are not saved.

  // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
  // which is true when tracking failed (lbL).
  std::list<KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
  std::list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
  for (std::list<cv::Mat>::iterator lit = mpTracker->mlRelativeFramePoses.begin();
                                    lit != mpTracker->mlRelativeFramePoses.end();
                                    ++lit, ++lRit, ++lT) {
    KeyFrame* pKF = *lRit;

    cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);
    while(pKF->isBad()) {
      Trw = Trw * pKF->mTcp;
      pKF = pKF->GetParent();
    }

    Trw = Trw * pKF->GetPose() * T_world_origin;

    cv::Mat Tcw = (*lit) * Trw;
    cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
    cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

    f << std::setprecision(9) 
      <<  Rwc.at<float>(0,0) << " " << Rwc.at<float>(0,1)  << " " << Rwc.at<float>(0,2) << " "  << twc.at<float>(0) << " " <<
          Rwc.at<float>(1,0) << " " << Rwc.at<float>(1,1)  << " " << Rwc.at<float>(1,2) << " "  << twc.at<float>(1) << " " <<
          Rwc.at<float>(2,0) << " " << Rwc.at<float>(2,1)  << " " << Rwc.at<float>(2,2) << " "  << twc.at<float>(2) << std::endl;
  }
  f.close();
  std::cout << std::endl << "Trajectory saved!" << endl;
}
