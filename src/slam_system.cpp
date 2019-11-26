#include "slam_system.h"
#include <unistd.h> // usleep

SlamSystem::SlamSystem(const std::string& strVocFile, 
                       const std::string& strSettingsFile, 
                       const SENSOR_TYPE sensor) 
        : mSensor(sensor) 
        , mbReset(false)
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
  mpTracker = new Tracker(this, 
                          mpVocabulary, 
                          mpMap, 
                          mpKeyFrameDatabase, 
                          strSettingsFile, 
                          mSensor);

  //Initialize the Local Mapping thread and launch
  mpLocalMapper = new LocalMapper(mpMap, 
                                  mSensor==SENSOR_TYPE::MONOCULAR);
  mptLocalMapping = new thread(&LocalMapper::Run, mpLocalMapper);

  //Initialize the Loop Closing thread and launch
  mpLoopCloser = new LoopCloser(mpMap, 
                                mpKeyFrameDatabase, 
                                mpVocabulary, 
                                mSensor != SENSOR_TYPE::MONOCULAR);
  mptLoopClosing = new thread(&LoopCloser::Run, mpLoopCloser);

  //Set pointers between threads
  mpTracker->SetLocalMapper(mpLocalMapper);
  mpTracker->SetLoopCloser(mpLoopCloser);

  mpLocalMapper->SetTracker(mpTracker);
  mpLocalMapper->SetLoopCloser(mpLoopCloser);

  mpLoopCloser->SetTracker(mpTracker);
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
    if (mbReset) {
      mpTracker->Reset();
      mbReset = false;
    }
  }

  cv::Mat Tcw = mpTracker->GrabImageStereo(imLeft, imRight, timestamp);

  std::unique_lock<std::mutex> lock2(mMutexState);
  mTrackingState = mpTracker->mState;
  mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
  mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
  return Tcw;
}


void SlamSystem::Reset() {
  std::unique_lock<std::mutex> lock(mMutexReset);
  mbReset = true;  
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
return false;//TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

}