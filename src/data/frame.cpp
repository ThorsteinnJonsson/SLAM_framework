#include "data/frame.h"

#include <thread>

#include "util/converter.h"
#include "orb_features/orb_matcher.h"

// Define static variables with initial values
long unsigned int Frame::nNextId = 0;
bool Frame::mbInitialComputations = true;
float Frame::cx, Frame::cy; 
float Frame::fx, Frame::fy;
float Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY; 
float Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv;
float Frame::mfGridElementHeightInv;

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
      , mN(frame.mN)
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
      , mvInvLevelSigma2(frame.mvInvLevelSigma2) {
  grid_ = frame.GetGrid();
  if (!frame.mTcw.empty()) {
    SetPose(frame.mTcw);
  }
}

Frame::Frame(const cv::Mat& imLeft, 
             const cv::Mat& imRight, 
             const double timeStamp, 
             const std::shared_ptr<ORBextractor>& extractorLeft, 
             const std::shared_ptr<ORBextractor>& extractorRight, 
             const std::shared_ptr<OrbVocabulary>& voc, 
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
    , mpReferenceKF(nullptr) {
  // Frame ID
  mnId = nNextId++;

  // Scale Level Info
  mnScaleLevels = mpORBextractorLeft->GetLevels();
  mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
  mfLogScaleFactor = std::log(mfScaleFactor);
  mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
  mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
  mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
  mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

  // ORB extraction
  std::thread thread_left(&Frame::ExtractORB, this, 0, imLeft);
  std::thread thread_right(&Frame::ExtractORB, this, 1, imRight);
  thread_left.join();
  thread_right.join();

  mN = mvKeys.size();
  if(mvKeys.empty()) {
    return;
  }

  UndistortKeyPoints();
  ComputeStereoMatches();

  mvpMapPoints = std::vector<MapPoint*>(mN, nullptr);    
  mvbOutlier = std::vector<bool>(mN, false); // vector of bools is bad

  // This is done only for the first Frame (or after a change in the calibration)
  if (mbInitialComputations)  {
    ComputeImageBounds(imLeft);

    mfGridElementWidthInv = static_cast<float>(grid_cols)/(mnMaxX-mnMinX);
    mfGridElementHeightInv = static_cast<float>(grid_rows)/(mnMaxY-mnMinY);

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


Frame::Frame(const cv::Mat& imGray, 
             const cv::Mat& imDepth, 
             const double timeStamp, 
             const std::shared_ptr<ORBextractor>& extractor,
             const std::shared_ptr<OrbVocabulary>& voc, 
             cv::Mat& K, 
             cv::Mat& distCoef, 
             const float bf, 
             const float thDepth)
      : mpORBvocabulary(voc)
      , mpORBextractorLeft(extractor)
      , mpORBextractorRight(nullptr)
      , mTimeStamp(timeStamp)
      , mK(K.clone())
      , mDistCoef(distCoef.clone())
      , mbf(bf)
      , mThDepth(thDepth) {
  // Frame ID
  mnId = nNextId++;

  // Scale Level Info
  mnScaleLevels = mpORBextractorLeft->GetLevels();
  mfScaleFactor = mpORBextractorLeft->GetScaleFactor();    
  mfLogScaleFactor = std::log(mfScaleFactor);
  mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
  mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
  mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
  mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

  // ORB extraction
  ExtractORB(0,imGray);

  mN = mvKeys.size();

  if(mvKeys.empty()) {
    return;
  }

  UndistortKeyPoints();

  ComputeStereoFromRGBD(imDepth);

  mvpMapPoints = std::vector<MapPoint*>(mN, nullptr);
  mvbOutlier = std::vector<bool>(mN, false);

  // This is done only for the first Frame (or after a change in the calibration)
  if (mbInitialComputations) {
    ComputeImageBounds(imGray);

    mfGridElementWidthInv = static_cast<float>(grid_cols)/(mnMaxX-mnMinX);
    mfGridElementHeightInv = static_cast<float>(grid_rows)/(mnMaxY-mnMinY);

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


Frame::Frame(const cv::Mat& imGray, 
             const double timeStamp, 
             const std::shared_ptr<ORBextractor>& extractor,
             const std::shared_ptr<OrbVocabulary>& voc, 
             cv::Mat& K, 
             cv::Mat& distCoef, 
             const float bf, 
             const float thDepth)
      : mpORBvocabulary(voc)
      , mpORBextractorLeft(extractor)
      , mpORBextractorRight(nullptr)
      , mTimeStamp(timeStamp)
      , mK(K.clone())
      , mDistCoef(distCoef.clone())
      , mbf(bf)
      , mThDepth(thDepth) {
    // Frame ID
  mnId = nNextId++;

  // Scale Level Info
  mnScaleLevels = mpORBextractorLeft->GetLevels();
  mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
  mfLogScaleFactor = std::log(mfScaleFactor);
  mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
  mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
  mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
  mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

  // ORB extraction
  ExtractORB(0,imGray);

  mN = mvKeys.size();

  if(mvKeys.empty()) {
    return;
  }

  UndistortKeyPoints();

  // Set no stereo information
  mvuRight = std::vector<float>(mN,-1);
  mvDepth = std::vector<float>(mN,-1);

  mvpMapPoints = std::vector<MapPoint*>(mN, nullptr);
  mvbOutlier = std::vector<bool>(mN, false);

  // This is done only for the first Frame (or after a change in the calibration)
  if(mbInitialComputations) {
    ComputeImageBounds(imGray);

    mfGridElementWidthInv = static_cast<float>(grid_cols)/(mnMaxX-mnMinX);
    mfGridElementHeightInv = static_cast<float>(grid_rows)/(mnMaxY-mnMinY);

    fx = K.at<float>(0,0);
    fy = K.at<float>(1,1);
    cx = K.at<float>(0,2);
    cy = K.at<float>(1,2);
    invfx = 1.0f/fx;
    invfy = 1.0f/fy;

    mbInitialComputations = false;
  }

  mb = mbf/fx;

  AssignFeaturesToGrid();
}


void Frame::AssignFeaturesToGrid() {
  int nReserve = 0.5f * mN / (grid_cols * grid_rows);
  for(unsigned int i=0; i < grid_cols; ++i) {
    for (unsigned int j=0; j < grid_rows; ++j) {
      grid_[i][j].reserve(nReserve);
    }
  }
  for (int i=0; i < mN; ++i) {
    const cv::KeyPoint& kp = mvKeysUn[i];
    int nGridPosX, nGridPosY;
    if(PosInGrid(kp, nGridPosX, nGridPosY)) {
      grid_[nGridPosX][nGridPosY].push_back(i);
    }
  }
}

void Frame::ExtractORB(int flag, const cv::Mat& im) {
  if(flag==0) {
    (*mpORBextractorLeft)(im, cv::Mat(), mvKeys, mDescriptors); // TODO unclear syntax, why do we need to overload the () operator??
  } else {
    (*mpORBextractorRight)(im, cv::Mat(), mvKeysRight, mDescriptorsRight);
  }    
}

void Frame::ComputeBoW() {
  if(mBowVec.empty()) {
    std::vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
    mpORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
  }
}

void Frame::SetPose(const cv::Mat& Tcw) {
  mTcw = Tcw.clone();
  UpdatePoseMatrices();
}

void Frame::UpdatePoseMatrices() { 
  mRcw = mTcw.rowRange(0,3).colRange(0,3);
  mRwc = mRcw.t();
  mtcw = mTcw.rowRange(0,3).col(3);
  mOw = -mRcw.t() * mtcw;
}

bool Frame::isInFrustum(MapPoint* pMP, float viewingCosLimit) {
  pMP->track_is_in_view = false;

  // 3D in absolute coordinates
  cv::Mat P = pMP->GetWorldPos(); 

  // 3D in camera coordinates
  const cv::Mat Pc = mRcw*P+mtcw;
  const float PcX = Pc.at<float>(0);
  const float PcY = Pc.at<float>(1);
  const float PcZ = Pc.at<float>(2);

  // Check positive depth
  if( PcZ < 0.0f) {
    return false;
  }

  // Project in image and check it is not outside
  const float invz = 1.0f / PcZ;
  const float u = fx * PcX * invz + cx;
  const float v = fy * PcY * invz + cy;

  if( u < mnMinX || u > mnMaxX ) {
    return false;
  }
  if( v < mnMinY || v > mnMaxY ){
    return false;
  }

  // Check distance is in the scale invariance region of the MapPoint
  const float maxDistance = pMP->GetMaxDistanceInvariance();
  const float minDistance = pMP->GetMinDistanceInvariance();
  const cv::Mat PO = P - mOw;
  const float dist = cv::norm(PO);

  if( dist < minDistance || dist > maxDistance) {
    return false;
  }

  // Check viewing angle
  cv::Mat Pn = pMP->GetNormal();

  const float viewCos = PO.dot(Pn) / dist;

  if(viewCos < viewingCosLimit) {
    return false;
  }

  // Predict scale in the image
  const int nPredictedLevel = pMP->PredictScale(dist, this);

  // Data used by the tracking
  pMP->track_is_in_view = true;
  pMP->track_projected_x = u;
  pMP->track_projected_x_right = u - mbf * invz;
  pMP->track_projected_y = v;
  pMP->track_scale_level= nPredictedLevel;
  pMP->track_view_cos = viewCos;

  return true;
}

bool Frame::PosInGrid(const cv::KeyPoint& kp, int& posX, int& posY) {
  posX = round( (kp.pt.x - mnMinX) * mfGridElementWidthInv);
  posY = round( (kp.pt.y - mnMinY) * mfGridElementHeightInv);

  //Keypoint's coordinates are undistorted, which could cause to go out of the image
  if( posX < 0 || posX >= grid_cols || posY < 0 || posY >= grid_rows ) {
    return false;
  }
  return true;
}

std::vector<size_t> Frame::GetFeaturesInArea(const float x, 
                                             const float y, 
                                             const float r, 
                                             const int minLevel, 
                                             const int maxLevel) const {
  std::vector<size_t> vIndices;
  vIndices.reserve(mN);

  const int nMinCellX = std::max(0, 
                                  static_cast<int>(std::floor((x-mnMinX-r) * mfGridElementWidthInv)));
  const int nMaxCellX = std::min(static_cast<int>(grid_cols-1),
                                  static_cast<int>(std::ceil((x-mnMinX+r) * mfGridElementWidthInv)));
  if( nMaxCellX < 0 || nMinCellX >= grid_cols ) {
    return vIndices;
  }

  const int nMinCellY = std::max(0,
                                  static_cast<int>(std::floor((y-mnMinY-r) * mfGridElementHeightInv)));
  const int nMaxCellY = std::min(static_cast<int>(grid_rows-1),
                                  static_cast<int>(std::ceil((y-mnMinY+r) * mfGridElementHeightInv)));
  if(nMaxCellY < 0 || nMinCellY >= grid_rows) {
    return vIndices;
  }

  const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);

  for(int ix = nMinCellX; ix <= nMaxCellX; ++ix) {
    for(int iy = nMinCellY; iy <= nMaxCellY; ++iy) {
      const std::vector<size_t>& vCell = grid_[ix][iy]; // TODO needless copying? changed to ref
      if(vCell.empty()) {
        continue;
      }

      for(size_t j=0; j < vCell.size(); ++j) {
        const cv::KeyPoint& kpUn = mvKeysUn[vCell[j]];
        if(bCheckLevels) {
          if(kpUn.octave < minLevel) {
            continue;
          }
          if(maxLevel>=0) {
            if(kpUn.octave > maxLevel) {
              continue;
            }
          }
        }

        const float distx = kpUn.pt.x - x;
        const float disty = kpUn.pt.y - y;

        if(std::fabs(distx) < r && std::fabs(disty) < r)
          vIndices.push_back(vCell[j]);
      }
    }
  }
  return vIndices;
}


void Frame::ComputeStereoMatches() {
  mvuRight = std::vector<float>(mN,-1.0f);
  mvDepth = std::vector<float>(mN,-1.0f);

  const int thOrbDist = (OrbMatcher::TH_HIGH + OrbMatcher::TH_LOW)/2;

  const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

  //Assign keypoints to row table
  std::vector<std::vector<size_t>> vRowIndices(nRows, std::vector<size_t>());

  for(int i=0; i < nRows; i++) {
    vRowIndices[i].reserve(200);
  }

  const int Nr = mvKeysRight.size();

  for(int iR=0; iR < Nr; iR++) {
    const cv::KeyPoint& kp = mvKeysRight[iR];
    const float kpY = kp.pt.y;
    const float r = 2.0f * mvScaleFactors[mvKeysRight[iR].octave];
    const int maxr = std::ceil(kpY + r);
    const int minr = std::floor(kpY - r);

    for(int yi=minr; yi <= maxr; ++yi) {
      vRowIndices[yi].push_back(iR);
    }
  }

  // Set limits for search
  const float minZ = mb;
  const float minD = 0;
  const float maxD = mbf / minZ;

  // For each left keypoint search a match in the right image
  std::vector<std::pair<int,int>> vDistIdx;
  vDistIdx.reserve(mN);

  for (int iL = 0; iL < mN; ++iL) {
    const cv::KeyPoint& kpL = mvKeys[iL];
    const int levelL = kpL.octave;
    const float vL = kpL.pt.y;
    const float uL = kpL.pt.x;

    const std::vector<size_t>& vCandidates = vRowIndices[vL];

    if(vCandidates.empty()) {
      continue;
    }

    const float minU = uL-maxD;
    const float maxU = uL-minD;

    if(maxU < 0) { 
      continue;
    }

    int bestDist = OrbMatcher::TH_HIGH;
    size_t bestIdxR = 0;

    const cv::Mat& dL = mDescriptors.row(iL);

    // Compare descriptor to right keypoints
    for(size_t iC=0; iC < vCandidates.size(); ++iC) {
      const size_t iR = vCandidates[iC];
      const cv::KeyPoint& kpR = mvKeysRight[iR];

      if(kpR.octave < levelL-1 || kpR.octave > levelL+1) {
        continue;
      }

      const float uR = kpR.pt.x;

      if(uR >= minU && uR <= maxU) {
        const cv::Mat& dR = mDescriptorsRight.row(iR);
        const int dist = OrbMatcher::DescriptorDistance(dL,dR);

        if(dist < bestDist) {
          bestDist = dist;
          bestIdxR = iR;
        }
      }
    }

    // Subpixel match by correlation
    if(bestDist < thOrbDist) {
      // coordinates in image pyramid at keypoint scale
      const float uR0 = mvKeysRight[bestIdxR].pt.x;
      const float scaleFactor = mvInvScaleFactors[kpL.octave];
      const float scaleduL = std::round(kpL.pt.x * scaleFactor);
      const float scaledvL = std::round(kpL.pt.y * scaleFactor);
      const float scaleduR0 = std::round(uR0 * scaleFactor);

      // sliding window search
      const int w = 5;
      cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
      IL.convertTo(IL, CV_32F);
      IL = IL - IL.at<float>(w,w) * cv::Mat::ones(IL.rows,IL.cols,CV_32F);

      int bestDist = INT_MAX; // TODO replace with numeric limits
      int bestincR = 0;
      const int L = 5;
      std::vector<float> vDists;
      vDists.resize(2*L+1);

      const float iniu = scaleduR0+L-w; // TODO don't get this part 
      const float endu = scaleduR0+L+w+1;
      if(iniu < 0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols) {
        continue;
      }
          

      for(int incR=-L; incR  <= L; ++incR) {
        cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
        IR.convertTo(IR,CV_32F);
        IR = IR - IR.at<float>(w,w) * cv::Mat::ones(IR.rows,IR.cols,CV_32F);

        float dist = cv::norm(IL,IR,cv::NORM_L1);
        if(dist<bestDist) {
          bestDist =  dist;
          bestincR = incR;
        }

        vDists[L+incR] = dist;
      }

      if(bestincR==-L || bestincR==L) {
        continue;
      }

      // Sub-pixel match (Parabola fitting)
      const float dist1 = vDists[L+bestincR-1];
      const float dist2 = vDists[L+bestincR];
      const float dist3 = vDists[L+bestincR+1];

      const float deltaR = (dist1-dist3) / (2.0f * (dist1+dist3-2.0f*dist2));

      if(deltaR<-1 || deltaR>1) {
        continue;
      }

      // Re-scaled coordinate
      float bestuR = mvScaleFactors[kpL.octave]*(static_cast<float>(scaleduR0) + static_cast<float>(bestincR) + deltaR);

      float disparity = (uL-bestuR);

      if(disparity >= minD && disparity < maxD) {
        if(disparity <= 0) {
          disparity = 0.01f;
          bestuR = uL - 0.01f;
        }
        mvDepth[iL] = mbf / disparity;
        mvuRight[iL] = bestuR;
        vDistIdx.push_back(std::pair<int,int>(bestDist,iL));
      }
    }
  }

  sort(vDistIdx.begin(),vDistIdx.end());
  const float median = vDistIdx[vDistIdx.size()/2].first;
  const float thDist = 1.5f*1.4f*median; // TODO Where do these numbers come from??

  for(int i=vDistIdx.size()-1; i >= 0; --i) {
    if(vDistIdx[i].first<thDist) {
      break;
    } else {
      mvuRight[vDistIdx[i].second]=-1;
      mvDepth[vDistIdx[i].second]=-1;
    }
  }
}

void Frame::ComputeStereoFromRGBD(const cv::Mat& imDepth) {
  mvuRight = std::vector<float>(mN,-1);
  mvDepth = std::vector<float>(mN,-1);

  for (int i = 0; i < mN; ++i) {
    const cv::KeyPoint& kp = mvKeys[i];
    const cv::KeyPoint& kpU = mvKeysUn[i];

    const float v = kp.pt.y;
    const float u = kp.pt.x;

    const float d = imDepth.at<float>(v,u);

    if (d > 0) {
      mvDepth[i] = d;
      mvuRight[i] = kpU.pt.x - mbf/d;
    }
  }
}

cv::Mat Frame::UnprojectStereo(const int i) {
  const float z = mvDepth[i];
  if(z > 0) {
    const float u = mvKeysUn[i].pt.x;
    const float v = mvKeysUn[i].pt.y;
    const float x = (u - cx) * z * invfx;
    const float y = (v - cy) * z * invfy;
    cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
    return mRwc*x3Dc+mOw;
  } else {
    return cv::Mat();
  }
}


void Frame::UndistortKeyPoints() {

  if(mDistCoef.at<float>(0) == 0.0f) {
    mvKeysUn=mvKeys;
    return;
  }

  // Fill matrix with points
  cv::Mat mat(mN,2,CV_32F);
  for(int i=0; i < mN; ++i) {
    mat.at<float>(i,0) = mvKeys[i].pt.x;
    mat.at<float>(i,1) = mvKeys[i].pt.y;
  }

  // Undistort points
  mat = mat.reshape(2);
  cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
  mat = mat.reshape(1);

  // Fill undistorted keypoint vector
  mvKeysUn.resize(mN);
  for (int i = 0; i < mN; ++i) {
    cv::KeyPoint kp = mvKeys[i];
    kp.pt.x = mat.at<float>(i,0);
    kp.pt.y = mat.at<float>(i,1);
    mvKeysUn[i] = kp;
  }
}


void Frame::ComputeImageBounds(const cv::Mat& imLeft) {

  if(mDistCoef.at<float>(0) != 0.0) {
    cv::Mat mat(4,2,CV_32F);
    mat.at<float>(0,0) = 0.0; 
    mat.at<float>(0,1)= 0.0;
    mat.at<float>(1,0) = imLeft.cols; 
    mat.at<float>(1,1) = 0.0;
    mat.at<float>(2,0) = 0.0; 
    mat.at<float>(2,1) = imLeft.rows;
    mat.at<float>(3,0) = imLeft.cols; 
    mat.at<float>(3,1) = imLeft.rows;

    // Undistort corners
    mat = mat.reshape(2);
    cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
    mat = mat.reshape(1);

    mnMinX = std::min(mat.at<float>(0,0), mat.at<float>(2,0));
    mnMaxX = std::max(mat.at<float>(1,0), mat.at<float>(3,0));
    mnMinY = std::min(mat.at<float>(0,1), mat.at<float>(1,1));
    mnMaxY = std::max(mat.at<float>(2,1), mat.at<float>(3,1));
  
  } else {
    mnMinX = 0.0f;
    mnMaxX = imLeft.cols;
    mnMinY = 0.0f;
    mnMaxY = imLeft.rows;
  }
}

