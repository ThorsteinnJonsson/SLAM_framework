#ifndef SRC_STEREO_TRACKER_H_
#define SRC_STEREO_TRACKER_H_

#include <opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>


class Tracker{
public:
  Tracker();
  ~Tracker();

  // Tracking states
  enum TrackingState {
    SYSTEM_NOT_READY=-1,
    NO_IMAGES_YET=0,
    NOT_INITIALIZED=1,
    OK=2,
    LOST=3
  };

  // Preprocess the input and call Track(). Extract features and performs stereo matching.
  cv::Mat GrabImageStereo(const cv::Mat& imRectLeft,
                          const cv::Mat& imRectRight, 
                          const double timestamp);
  // Main tracking function
  void Track();
private:
  void StereoInitialization();

private:

  TrackingState mState = TrackingState::NOT_INITIALIZED;

  // Current Frame
  // Frame mCurrentFrame;
  cv::Mat mImGray;

  
};


#endif //SRC_STEREO_TRACKER_H_
