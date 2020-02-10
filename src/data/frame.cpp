#include "data/frame.h"

#include <thread>

#include "util/converter.h"
#include "orb_features/orb_matcher.h"

// Define static variables with initial values
long unsigned int Frame::next_id_ = 0;

bool Frame::mbInitialComputations = true;

float Frame::cx_, Frame::cy_; 
float Frame::fx_, Frame::fy_;
float Frame::invfx_, Frame::invfy_;

float Frame::min_x_, Frame::min_y_; 
float Frame::max_x_, Frame::max_y_;

float Frame::grid_element_width_;
float Frame::grid_element_height_;

Frame::Frame(const Frame& frame)
      : mnScaleLevels(frame.mnScaleLevels)
      , mfScaleFactor(frame.mfScaleFactor)
      , mfLogScaleFactor(frame.mfLogScaleFactor)
      , mvScaleFactors(frame.mvScaleFactors)
      , mvInvScaleFactors(frame.mvInvScaleFactors)
      , mvLevelSigma2(frame.mvLevelSigma2)
      , mvInvLevelSigma2(frame.mvInvLevelSigma2)
      , mnId(frame.Id())
      , calib_mat_(frame.calib_mat_.clone())
      , dist_coeff_(frame.dist_coeff_.clone())
      , baseline_fx_(frame.GetBaselineFx())
      , baseline_(frame.GetBaseline())
      , thresh_depth_(frame.GetDepthThrehold())
      , orb_vocabulary_(frame.GetVocabulary())
      , left_orb_extractor_(frame.GetLeftOrbExtractor())
      , right_orb_extractor_(frame.GetRightOrbExtractor())
      , timestamp_(frame.GetTimestamp())
      , num_keypoints_(frame.num_keypoints_)
      , keypoints_(frame.GetKeys())
      , right_keypoints_(frame.GetRightKeys())
      , undistorted_keypoints_(frame.GetUndistortedKeys())
      , stereo_coords_(frame.StereoCoordRight())
      , depths_(frame.StereoDepth()) 
      , bow_vec_(frame.GetBowVector())
      , feature_vec_(frame.GetFeatureVector())
      , descriptors_(frame.GetDescriptors().clone())
      , right_descriptors_(frame.GetRightDescriptors().clone())
      , map_points_(frame.GetMapPoints())
      , is_outlier_(frame.GetOutliers())
      , reference_keyframe_(frame.GetReferenceKeyframe()) {

  grid_ = frame.GetGrid();
  if (!frame.GetPose().empty()) {
    SetPose(frame.GetPose());
  }
}

Frame::Frame(const cv::Mat& imLeft, 
             const cv::Mat& imRight, 
             const double timestamp, 
             const std::shared_ptr<ORBextractor>& extractorLeft, 
             const std::shared_ptr<ORBextractor>& extractorRight, 
             const std::shared_ptr<OrbVocabulary>& voc, 
             cv::Mat& K, 
             cv::Mat& dist_coeff, 
             const float bf, 
             const float thDepth)
    : calib_mat_(K.clone())
    , dist_coeff_(dist_coeff.clone())
    , baseline_fx_(bf)
    , thresh_depth_(thDepth)
    , orb_vocabulary_(voc)
    , left_orb_extractor_(extractorLeft)
    , right_orb_extractor_(extractorRight)
    , timestamp_(timestamp)
    , reference_keyframe_(nullptr)  {
  // Frame ID
  mnId = next_id_++;

  // Scale Level Info
  mnScaleLevels = left_orb_extractor_->GetLevels();
  mfScaleFactor = left_orb_extractor_->GetScaleFactor();
  mfLogScaleFactor = std::log(mfScaleFactor);
  mvScaleFactors = left_orb_extractor_->GetScaleFactors();
  mvInvScaleFactors = left_orb_extractor_->GetInverseScaleFactors();
  mvLevelSigma2 = left_orb_extractor_->GetScaleSigmaSquares();
  mvInvLevelSigma2 = left_orb_extractor_->GetInverseScaleSigmaSquares();

  // ORB extraction
  std::thread thread_left(&Frame::ExtractORB, this, 0, imLeft);
  std::thread thread_right(&Frame::ExtractORB, this, 1, imRight);
  thread_left.join();
  thread_right.join();

  num_keypoints_ = keypoints_.size();
  if(keypoints_.empty()) {
    return;
  }

  UndistortKeyPoints();
  ComputeStereoMatches();

  map_points_ = std::vector<MapPoint*>(num_keypoints_, nullptr);    
  is_outlier_ = std::deque<bool>(num_keypoints_, false);

  // This is done only for the first Frame (or after a change in the calibration)
  if (mbInitialComputations)  {
    MakeInitialComputations(imLeft, K);
    mbInitialComputations = false;
  }

  baseline_ = baseline_fx_ / fx_;
  
  AssignFeaturesToGrid();
}


Frame::Frame(const cv::Mat& imGray, 
             const cv::Mat& imDepth, 
             const double timestamp, 
             const std::shared_ptr<ORBextractor>& extractor,
             const std::shared_ptr<OrbVocabulary>& voc, 
             cv::Mat& K, 
             cv::Mat& dist_coeff, 
             const float bf, 
             const float thDepth)
      : calib_mat_(K.clone())
      , dist_coeff_(dist_coeff.clone())
      , baseline_fx_(bf)
      , thresh_depth_(thDepth) 
      , orb_vocabulary_(voc)
      , left_orb_extractor_(extractor)
      , right_orb_extractor_(nullptr) 
      , timestamp_(timestamp) {
  // Frame ID
  mnId = next_id_++;

  // Scale Level Info
  mnScaleLevels = left_orb_extractor_->GetLevels();
  mfScaleFactor = left_orb_extractor_->GetScaleFactor();    
  mfLogScaleFactor = std::log(mfScaleFactor);
  mvScaleFactors = left_orb_extractor_->GetScaleFactors();
  mvInvScaleFactors = left_orb_extractor_->GetInverseScaleFactors();
  mvLevelSigma2 = left_orb_extractor_->GetScaleSigmaSquares();
  mvInvLevelSigma2 = left_orb_extractor_->GetInverseScaleSigmaSquares();

  // ORB extraction
  ExtractORB(0,imGray);

  num_keypoints_ = keypoints_.size();
  if(keypoints_.empty()) {
    return;
  }

  UndistortKeyPoints();

  ComputeStereoFromRGBD(imDepth);

  map_points_ = std::vector<MapPoint*>(num_keypoints_, nullptr);
  is_outlier_ = std::deque<bool>(num_keypoints_, false);

  // This is done only for the first Frame (or after a change in the calibration)
  if (mbInitialComputations) {
    MakeInitialComputations(imGray, K);
    mbInitialComputations=false;
  }

  baseline_ = baseline_fx_ / fx_;

  AssignFeaturesToGrid();
}


Frame::Frame(const cv::Mat& imGray, 
             const double timeStamp, 
             const std::shared_ptr<ORBextractor>& extractor,
             const std::shared_ptr<OrbVocabulary>& voc, 
             cv::Mat& K, 
             cv::Mat& dist_coeff, 
             const float bf, 
             const float th_depth)
      : calib_mat_(K.clone())
      , dist_coeff_(dist_coeff.clone())
      , baseline_fx_(bf)
      , thresh_depth_(th_depth)
      , orb_vocabulary_(voc)
      , left_orb_extractor_(extractor)
      , right_orb_extractor_(nullptr) 
      , timestamp_(timeStamp){
    // Frame ID
  mnId = next_id_++;

  // Scale Level Info
  mnScaleLevels = left_orb_extractor_->GetLevels();
  mfScaleFactor = left_orb_extractor_->GetScaleFactor();
  mfLogScaleFactor = std::log(mfScaleFactor);
  mvScaleFactors = left_orb_extractor_->GetScaleFactors();
  mvInvScaleFactors = left_orb_extractor_->GetInverseScaleFactors();
  mvLevelSigma2 = left_orb_extractor_->GetScaleSigmaSquares();
  mvInvLevelSigma2 = left_orb_extractor_->GetInverseScaleSigmaSquares();

  // ORB extraction
  ExtractORB(0,imGray);

  num_keypoints_ = keypoints_.size();
  if(keypoints_.empty()) {
    return;
  }

  UndistortKeyPoints();

  // Set no stereo information
  stereo_coords_ = std::vector<float>(num_keypoints_, -1.0f);
  depths_ = std::vector<float>(num_keypoints_, -1.0f);

  map_points_ = std::vector<MapPoint*>(num_keypoints_, nullptr);
  is_outlier_ = std::deque<bool>(num_keypoints_, false);

  // This is done only for the first Frame (or after a change in the calibration)
  if(mbInitialComputations) {
    MakeInitialComputations(imGray, K);
    mbInitialComputations = false;
  }

  baseline_ = baseline_fx_ / fx_;

  AssignFeaturesToGrid();
}

void Frame::MakeInitialComputations(const cv::Mat& image, cv::Mat& calibration_mat) {
  ComputeImageBounds(image);
  grid_element_width_ = static_cast<float>(max_x_-min_x_)/(grid_cols);
  grid_element_height_ = static_cast<float>(max_y_-min_y_)/(grid_rows);

  fx_ = calibration_mat.at<float>(0,0);
  fy_ = calibration_mat.at<float>(1,1);
  cx_ = calibration_mat.at<float>(0,2);
  cy_ = calibration_mat.at<float>(1,2);
  invfx_ = 1.0f / fx_;
  invfy_ = 1.0f / fy_;
}

void Frame::AssignFeaturesToGrid() {
  int nReserve = 0.5f * num_keypoints_ / (grid_cols * grid_rows);
  for(unsigned int i=0; i < grid_cols; ++i) {
    for (unsigned int j=0; j < grid_rows; ++j) {
      grid_[i][j].reserve(nReserve);
    }
  }
  for (int i=0; i < num_keypoints_; ++i) {
    const cv::KeyPoint& kp = undistorted_keypoints_[i];
    int nGridPosX, nGridPosY;
    if (PosInGrid(kp, nGridPosX, nGridPosY)) {
      grid_[nGridPosX][nGridPosY].push_back(i);
    }
  }
}

void Frame::ExtractORB(int flag, const cv::Mat& im) {
  if (flag == 0) {
    left_orb_extractor_->Compute(im, cv::Mat(), keypoints_, descriptors_);
  } else {
    right_orb_extractor_->Compute(im, cv::Mat(), right_keypoints_, right_descriptors_);
  }    
}

void Frame::ComputeBoW() {
  if (bow_vec_.empty()) {
    std::vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(descriptors_);
    orb_vocabulary_->transform(vCurrentDesc, bow_vec_, feature_vec_, 4);
  }
}

void Frame::SetPose(const cv::Mat& Tcw) {
  mTcw = Tcw.clone();
  UpdatePoseMatrices();
}

void Frame::UpdatePoseMatrices() { 
  Rcw_ = mTcw.rowRange(0,3).colRange(0,3);
  Rwc_ = Rcw_.t();
  tcw_ = mTcw.rowRange(0,3).col(3);
  Ow_ = -Rcw_.t() * tcw_;
}

bool Frame::isInFrustum(MapPoint* pMP, float viewingCosLimit) {
  pMP->track_is_in_view = false;

  // 3D in absolute coordinates
  cv::Mat P = pMP->GetWorldPos(); 

  // 3D in camera coordinates
  const cv::Mat Pc = Rcw_*P + tcw_;
  const float PcX = Pc.at<float>(0);
  const float PcY = Pc.at<float>(1);
  const float PcZ = Pc.at<float>(2);

  // Check positive depth
  if( PcZ < 0.0f) {
    return false;
  }

  // Project in image and check it is not outside
  const float invz = 1.0f / PcZ;
  const float u = fx_ * PcX * invz + cx_;
  const float v = fy_ * PcY * invz + cy_;

  if( u < min_x_ || u > max_x_ ) {
    return false;
  }
  if( v < min_y_ || v > max_y_ ){
    return false;
  }

  // Check distance is in the scale invariance region of the MapPoint
  const float maxDistance = pMP->GetMaxDistanceInvariance();
  const float minDistance = pMP->GetMinDistanceInvariance();
  const cv::Mat PO = P - Ow_;
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
  pMP->track_projected_x_right = u - baseline_fx_ * invz;
  pMP->track_projected_y = v;
  pMP->track_scale_level= nPredictedLevel;
  pMP->track_view_cos = viewCos;

  return true;
}

bool Frame::PosInGrid(const cv::KeyPoint& kp, int& posX, int& posY) {
  posX = round( (kp.pt.x - min_x_) / grid_element_width_);
  posY = round( (kp.pt.y - min_y_) / grid_element_height_);

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
  vIndices.reserve(num_keypoints_);

  const int nMinCellX = std::max(0, 
                                  static_cast<int>(std::floor((x-min_x_-r) / grid_element_width_)));
  const int nMaxCellX = std::min(static_cast<int>(grid_cols-1),
                                  static_cast<int>(std::ceil((x-min_x_+r) / grid_element_width_)));
  if( nMaxCellX < 0 || nMinCellX >= grid_cols ) {
    return vIndices;
  }

  const int nMinCellY = std::max(0,
                                  static_cast<int>(std::floor((y-min_y_-r) / grid_element_height_)));
  const int nMaxCellY = std::min(static_cast<int>(grid_rows-1),
                                  static_cast<int>(std::ceil((y-min_y_+r) / grid_element_height_)));
  if(nMaxCellY < 0 || nMinCellY >= grid_rows) {
    return vIndices;
  }

  const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);

  for(int ix = nMinCellX; ix <= nMaxCellX; ++ix) {
    for(int iy = nMinCellY; iy <= nMaxCellY; ++iy) {
      const std::vector<size_t>& vCell = grid_[ix][iy];
      if(vCell.empty()) {
        continue;
      }

      for(size_t j=0; j < vCell.size(); ++j) {
        const cv::KeyPoint& kpUn = undistorted_keypoints_[vCell[j]];
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
  stereo_coords_ = std::vector<float>(num_keypoints_,-1.0f);
  depths_ = std::vector<float>(num_keypoints_,-1.0f);

  const int thOrbDist = (OrbMatcher::TH_HIGH + OrbMatcher::TH_LOW)/2;

  const int nRows = left_orb_extractor_->GetImagePyramid()[0].rows;

  //Assign keypoints to row table
  std::vector<std::vector<size_t>> vRowIndices(nRows, std::vector<size_t>());

  for(int i=0; i < nRows; i++) {
    vRowIndices[i].reserve(200);
  }

  const int Nr = right_keypoints_.size();

  for(int iR=0; iR < Nr; iR++) {
    const cv::KeyPoint& kp = right_keypoints_[iR];
    const float kpY = kp.pt.y;
    const float r = 2.0f * mvScaleFactors[right_keypoints_[iR].octave];
    const int maxr = std::ceil(kpY + r);
    const int minr = std::floor(kpY - r);

    for(int yi=minr; yi <= maxr; ++yi) {
      vRowIndices[yi].push_back(iR);
    }
  }

  // Set limits for search
  const float minZ = baseline_;
  const float minD = 0;
  const float maxD = baseline_fx_ / minZ;

  // For each left keypoint search a match in the right image
  std::vector<std::pair<int,int>> vDistIdx;
  vDistIdx.reserve(num_keypoints_);

  for (int iL = 0; iL < num_keypoints_; ++iL) {
    const cv::KeyPoint& kpL = keypoints_[iL];
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

    const cv::Mat& dL = descriptors_.row(iL);

    // Compare descriptor to right keypoints
    for(size_t iC=0; iC < vCandidates.size(); ++iC) {
      const size_t iR = vCandidates[iC];
      const cv::KeyPoint& kpR = right_keypoints_[iR];

      if(kpR.octave < levelL-1 || kpR.octave > levelL+1) {
        continue;
      }

      const float uR = kpR.pt.x;

      if(uR >= minU && uR <= maxU) {
        const cv::Mat& dR = right_descriptors_.row(iR);
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
      const float uR0 = right_keypoints_[bestIdxR].pt.x;
      const float scaleFactor = mvInvScaleFactors[kpL.octave];
      const float scaleduL = std::round(kpL.pt.x * scaleFactor);
      const float scaledvL = std::round(kpL.pt.y * scaleFactor);
      const float scaleduR0 = std::round(uR0 * scaleFactor);

      // sliding window search
      const int w = 5;
      cv::Mat IL = left_orb_extractor_->GetImagePyramid()[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
      IL.convertTo(IL, CV_32F);
      IL = IL - IL.at<float>(w,w) * cv::Mat::ones(IL.rows,IL.cols,CV_32F);

      int bestDist = std::numeric_limits<int>::max();
      int bestincR = 0;
      const int L = 5;
      std::vector<float> vDists;
      vDists.resize(2*L+1);

      const float iniu = scaleduR0+L-w;
      const float endu = scaleduR0+L+w+1;
      if (iniu < 0 
          || endu >= right_orb_extractor_->GetImagePyramid()[kpL.octave].cols) {
        continue;
      }
          

      for (int incR=-L; incR  <= L; ++incR) {
        cv::Mat IR = right_orb_extractor_->GetImagePyramid()[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
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
        depths_[iL] = baseline_fx_ / disparity;
        stereo_coords_[iL] = bestuR;
        vDistIdx.push_back(std::pair<int,int>(bestDist,iL));
      }
    }
  }

  sort(vDistIdx.begin(),vDistIdx.end());
  const float median = vDistIdx[vDistIdx.size()/2].first;
  const float thDist = 1.5f*1.4f*median;

  for(int i=vDistIdx.size()-1; i >= 0; --i) {
    if(vDistIdx[i].first<thDist) {
      break;
    } else {
      stereo_coords_[vDistIdx[i].second] = -1.0f;
      depths_[vDistIdx[i].second] = -1.0f;
    }
  }
}

void Frame::ComputeStereoFromRGBD(const cv::Mat& imDepth) {
  stereo_coords_ = std::vector<float>(num_keypoints_, -1.0f);
  depths_ = std::vector<float>(num_keypoints_, -1.0f);

  for (int i = 0; i < num_keypoints_; ++i) {
    const cv::KeyPoint& kp = keypoints_[i];
    const cv::KeyPoint& kpU = undistorted_keypoints_[i];

    const float v = kp.pt.y;
    const float u = kp.pt.x;

    const float d = imDepth.at<float>(v,u);

    if (d > 0) {
      depths_[i] = d;
      stereo_coords_[i] = kpU.pt.x - baseline_fx_ / d;
    }
  }
}

cv::Mat Frame::UnprojectStereo(const int i) {
  const float z = depths_[i];
  if(z > 0) {
    const float u = undistorted_keypoints_[i].pt.x;
    const float v = undistorted_keypoints_[i].pt.y;
    const float x = (u - cx_) * z * invfx_;
    const float y = (v - cy_) * z * invfy_;
    cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
    return Rwc_ * x3Dc + Ow_;
  } else {
    return cv::Mat();
  }
}


void Frame::UndistortKeyPoints() {

  if (dist_coeff_.at<float>(0) == 0.0f) {
    undistorted_keypoints_=keypoints_;
    return;
  }

  // Fill matrix with points
  cv::Mat mat(num_keypoints_,2,CV_32F);
  for(int i=0; i < num_keypoints_; ++i) {
    mat.at<float>(i,0) = keypoints_[i].pt.x;
    mat.at<float>(i,1) = keypoints_[i].pt.y;
  }

  // Undistort points
  mat = mat.reshape(2);
  cv::undistortPoints(mat, mat, calib_mat_, dist_coeff_, cv::Mat(), calib_mat_);
  mat = mat.reshape(1);

  // Fill undistorted keypoint vector
  undistorted_keypoints_.resize(num_keypoints_);
  for (int i = 0; i < num_keypoints_; ++i) {
    cv::KeyPoint kp = keypoints_[i];
    kp.pt.x = mat.at<float>(i,0);
    kp.pt.y = mat.at<float>(i,1);
    undistorted_keypoints_[i] = kp;
  }
}


void Frame::ComputeImageBounds(const cv::Mat& imLeft) {

  if (dist_coeff_.at<float>(0) != 0.0) {
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
    cv::undistortPoints(mat, mat, calib_mat_, dist_coeff_, cv::Mat(), calib_mat_);
    mat = mat.reshape(1);

    min_x_ = std::min(mat.at<float>(0,0), mat.at<float>(2,0));
    max_x_ = std::max(mat.at<float>(1,0), mat.at<float>(3,0));
    min_y_ = std::min(mat.at<float>(0,1), mat.at<float>(1,1));
    max_y_ = std::max(mat.at<float>(2,1), mat.at<float>(3,1));
  
  } else {
    min_x_ = 0.0f;
    max_x_ = imLeft.cols;
    min_y_ = 0.0f;
    max_y_ = imLeft.rows;
  }
}

