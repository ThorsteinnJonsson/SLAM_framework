#include "frame.h"

#include <thread>

#include "converter.h"
#include "orb_matcher.h"


Frame::Frame(const Frame& frame)
  : mpORBvocabulary(frame.mpORBvocabulary)
  , mpORBextractorLeft(frame.mpORBextractorLeft)
  , mpORBextractorRight(frame.mpORBextractorRight)
  , mTimeStamp(frame.mTimeStamp)
  , mK(frame.mK.clone())
  , mDistCoef(frame.mDistCoef.clone())
  , mbf(frame.mbf)
  , mb(frame.mb)
  , mThDepth(frame.mThDepth)
  , N(frame.N)
  , mvKeys(frame.mvKeys)
  , mvKeysRight(frame.mvKeysRight)
  , mvKeysUn(frame.mvKeysUn)
  , mvuRight(frame.mvuRight)
  , mvDepth(frame.mvDepth)
  , mBowVec(frame.mBowVec)
  , mFeatVec(frame.mFeatVec)
  , mDescriptors(frame.mDescriptors.clone())
  , mDescriptorsRight(frame.mDescriptorsRight.clone())
  , mvpMapPoints(frame.mvpMapPoints)
  , mvbOutlier(frame.mvbOutlier)
  , mnId(frame.mnId)
  , mpReferenceKF(frame.mpReferenceKF)
  , mnScaleLevels(frame.mnScaleLevels)
  , mfScaleFactor(frame.mfScaleFactor)
  , mfLogScaleFactor(frame.mfLogScaleFactor)
  , mvScaleFactors(frame.mvScaleFactors)
  , mvInvScaleFactors(frame.mvInvScaleFactors)
  , mvLevelSigma2(frame.mvLevelSigma2)
  , mvInvLevelSigma2(frame.mvInvLevelSigma2) 
{
  for(int i=0; i<FRAME_GRID_COLS; ++i) {
    for(int j=0; j<FRAME_GRID_ROWS; ++j) {
      mGrid[i][j] = frame.mGrid[i][j];
    }
  }
  if(!frame.mTcw.empty()) {
    SetPose(frame.mTcw);
  }
}

Frame::Frame(const cv::Mat& imLeft, 
             const cv::Mat& imRight, 
             const double& timeStamp, 
             OrbExtractor* extractorLeft, 
             OrbExtractor* extractorRight, 
             OrbVocabulary* voc, 
             cv::Mat& K, 
             cv::Mat& distCoef, 
             const float bf, 
             const float thDepth)
    : mpORBvocabulary(voc)
    , mpORBextractorLeft(extractorLeft)
    , mpORBextractorRight(extractorRight)
    , mTimeStamp(timeStamp)
    , mK(K.clone())
    , mDistCoef(distCoef.clone())
    , mbf(bf)
    , mThDepth(thDepth)
    , mpReferenceKF(nullptr) 
{
  // Frame ID
  mnId=nNextId++;

  // Scale Level Info
  mnScaleLevels = mpORBextractorLeft->GetLevels();
  mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
  mfLogScaleFactor = log(mfScaleFactor);
  mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
  mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
  mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
  mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

  // ORB extraction
  std::thread thread_left(&Frame::ExtractORB, this, 0, imLeft);
  std::thread thread_right(&Frame::ExtractORB, this, 1, imRight);
  thread_left.join();
  thread_right.join();

  N = mvKeys.size();
  if(mvKeys.empty()) {
    return;
  }

  UndistortKeyPoints();
  ComputeStereoMatches();

  mvpMapPoints = std::vector<MapPoint*>(N,nullptr);    
  mvbOutlier = std::vector<bool>(N,false); // vector of bools is bad

  // This is done only for the first Frame (or after a change in the calibration)
  if(mbInitialComputations) 
  {
      ComputeImageBounds(imLeft);

      mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
      mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

      fx = K.at<float>(0,0);
      fy = K.at<float>(1,1);
      cx = K.at<float>(0,2);
      cy = K.at<float>(1,2);
      invfx = 1.0f/fx;
      invfy = 1.0f/fy;

      mbInitialComputations=false;
  }

  mb = mbf/fx;
  
  AssignFeaturesToGrid();
}

void Frame::AssignFeaturesToGrid() {
  int nReserve = 0.5f * N / (FRAME_GRID_COLS * FRAME_GRID_ROWS);
  for(unsigned int i=0; i < FRAME_GRID_COLS; ++i) {
    for (unsigned int j=0; j < FRAME_GRID_ROWS; ++j) {
      mGrid[i][j].reserve(nReserve);
    }
  }
  for(int i=0; i < N; ++i) {
    const cv::KeyPoint& kp = mvKeysUn[i];
    int nGridPosX, nGridPosY;
    if(PosInGrid(kp,nGridPosX,nGridPosY)) {
      mGrid[nGridPosX][nGridPosY].push_back(i);
    }
  }
}

void Frame::ExtractORB(int flag, const cv::Mat& im) {
  if(flag==0) {
    (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors); // TODO unclear syntax, why do we need to overload the () operator??
  } else {
    (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight);
  }    
}

void Frame::ComputeBoW() {
  //TODO
}






















