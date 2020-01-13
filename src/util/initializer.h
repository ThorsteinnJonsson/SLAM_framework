#ifndef SRC_INITIALIZER_H
#define SRC_INITIALIZER_H

#include<opencv2/opencv.hpp>
#include "data/frame.h"



// THIS IS THE INITIALIZER FOR MONOCULAR SLAM. NOT USED IN THE STEREO OR RGBD CASE.
class Initializer {
typedef std::pair<int,int> Match;
public:
  // Fix the reference frame
  Initializer(const Frame& ReferenceFrame, 
              float sigma = 1.0, 
              int iterations = 200);

  // Computes in parallel a fundamental matrix and a homography
  // Selects a model and tries to recover the motion and the structure from motion
  bool Initialize(const Frame& CurrentFrame, 
                  const std::vector<int>& vMatches12,
                  cv::Mat& R21, 
                  cv::Mat& t21, 
                  std::vector<cv::Point3f>& vP3D, 
                  std::deque<bool>& vbTriangulated);

private:

  void FindHomography(std::deque<bool>& vbMatchesInliers, 
                      float& score, 
                      cv::Mat& H21);
  void FindFundamental(std::deque<bool>& vbInliers, 
                       float& score, 
                       cv::Mat& F21);

  cv::Mat ComputeH21(const std::vector<cv::Point2f>& vP1, 
                     const std::vector<cv::Point2f>& vP2);
  cv::Mat ComputeF21(const std::vector<cv::Point2f>& vP1, 
                     const std::vector<cv::Point2f>& vP2);

  float CheckHomography(const cv::Mat& H21, 
                        const cv::Mat& H12, 
                        std::deque<bool>& vbMatchesInliers, 
                        const float sigma);

  float CheckFundamental(const cv::Mat& F21, 
                         std::deque<bool>& vbMatchesInliers, 
                         const float sigma);

  bool ReconstructF(std::deque<bool>& vbMatchesInliers, 
                    cv::Mat& F21, 
                    cv::Mat& K,
                    cv::Mat& R21, 
                    cv::Mat& t21, 
                    std::vector<cv::Point3f>& vP3D, 
                    std::deque<bool>& vbTriangulated, 
                    float minParallax, 
                    int minTriangulated);

  bool ReconstructH(std::deque<bool> &vbMatchesInliers, 
                    cv::Mat& H21, 
                    cv::Mat& K,
                    cv::Mat& R21, 
                    cv::Mat& t21, 
                    std::vector<cv::Point3f>& vP3D, 
                    std::deque<bool>& vbTriangulated, 
                    float minParallax, 
                    int minTriangulated);

  void Triangulate(const cv::KeyPoint& kp1, 
                   const cv::KeyPoint& kp2, 
                   const cv::Mat& P1, 
                   const cv::Mat& P2, 
                   cv::Mat& x3D);

  void Normalize(const std::vector<cv::KeyPoint>& vKeys, 
                 std::vector<cv::Point2f>& vNormalizedPoints, 
                 cv::Mat& T);

  int CheckRT(const cv::Mat& R, 
              const cv::Mat& t, 
              const std::vector<cv::KeyPoint>& vKeys1, 
              const std::vector<cv::KeyPoint>& vKeys2,
              const std::vector<Match>& vMatches12, 
              std::deque<bool>& vbInliers,
              const cv::Mat& K, 
              std::vector<cv::Point3f>& vP3D, 
              float th2, 
              std::deque<bool>& vbGood, 
              float& parallax);

  void DecomposeE(const cv::Mat& E, 
                  cv::Mat& R1, 
                  cv::Mat& R2, 
                  cv::Mat& t);


  // Keypoints from Reference Frame (Frame 1)
  std::vector<cv::KeyPoint> mvKeys1;

  // Keypoints from Current Frame (Frame 2)
  std::vector<cv::KeyPoint> mvKeys2;

  // Current Matches from Reference to Current
  std::vector<Match> mvMatches12;
  std::deque<bool> mvbMatched1;

  // Calibration
  cv::Mat mK;

  // Standard Deviation and Variance
  float mSigma, mSigma2;

  // Ransac max iterations
  int mMaxIterations;

  // Ransac sets
  std::vector<std::vector<size_t>> mvSets;   

};

#endif // SRC_INITIALIZER_H
