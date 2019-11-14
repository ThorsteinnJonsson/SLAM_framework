#include "stereo_slam_system.h"

StereoSlamSystem::StereoSlamSystem() {

  //Create KeyFrame Database
  // mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

  //Create the Map
  // mpMap = new Map();


  // //(it will live in the main thread of execution, the one that called this constructor)
  // mpTracker = new Tracking(this, mpVocabulary, mpFrameDrawer, mpMapDrawer,
  //                           mpMap, mpKeyFrameDatabase, strSettingsFile, mSensor);
  tracker = new Tracker(this, mpVocabulary, mpMap, mpKeyFrameDatabase, strSettingsFile, SENSOR_TYPE::STEREO); //TODO fix

  // //Initialize the Local Mapping thread and launch
  // mpLocalMapper = new LocalMapping(mpMap, mSensor==MONOCULAR);
  // mptLocalMapping = new thread(&ORB_SLAM2::LocalMapping::Run,mpLocalMapper);

  // //Initialize the Loop Closing thread and launch
  // mpLoopCloser = new LoopClosing(mpMap, mpKeyFrameDatabase, mpVocabulary, mSensor!=MONOCULAR);
  // mptLoopClosing = new thread(&ORB_SLAM2::LoopClosing::Run, mpLoopCloser);


}

StereoSlamSystem::~StereoSlamSystem() {

}

void StereoSlamSystem::TrackStereo(const cv::Mat& l_image, const cv::Mat& r_image, const double& timestamp) {

  // TODO lots of stuff

  cv::Mat tcw = tracker->GrabImageStereo(l_image, r_image, timestamp);
}