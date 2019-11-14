#include "tracker.h"

Tracker::Tracker(StereoSlamSystem* pSys, 
                 OrbVocabulary* pVoc, 
                 Map* pMap,
                 KeyFrameDatabase* pKFDB, 
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
  // TODO DO THIS NEXT
}

void Tracker::Track() {
  // TODO

}

void Tracker::StereoInitialization() {

}