#include "tracker.h"

Tracker::Tracker() {

}

Tracker::~Tracker() {
  
}


cv::Mat Tracker::GrabImageStereo(const cv::Mat& imRectLeft,
                                 const cv::Mat& imRectRight, 
                                 const double timestamp) {
  mImGray = imRectLeft;
  cv::Mat imGrayRight = imRectRight;

  // Convert to grayscale if image is RGB
  if(mImGray.channels()==3) {
    cv::cvtColor(mImGray,mImGray, CV_RGB2GRAY);
    cv::cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
  }

  // mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
  
  Track();

  // return return mCurrentFrame.mTcw.clone(); // TODO
  cv::Mat rip;
  return rip;

}

void Tracker::Track() {
  // TODO
  // Get Map Mutex -> Map cannot be changed
    // unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

  if (this->mState == TrackingState::NOT_INITIALIZED) {
    StereoInitialization();
    if (this->mState != TrackingState::OK) {
      return;
    }
  } else {

  }
}

void Tracker::StereoInitialization() {

}