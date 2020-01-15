#ifndef ORB_EXTRACTOR_H_
#define ORB_EXTRACTOR_H_

#include <vector>
#include <list>
#include <opencv/cv.h>


class ExtractorNode
{
public:
  ExtractorNode():bNoMore(false){}

  void DivideNode(ExtractorNode& n1, 
                  ExtractorNode& n2, 
                  ExtractorNode& n3, 
                  ExtractorNode& n4);

  std::vector<cv::KeyPoint> vKeys;
  cv::Point2i UL, UR, BL, BR;
  std::list<ExtractorNode>::iterator lit;
  bool bNoMore;
};

class ORBextractor {
public:
  
  enum {HARRIS_SCORE=0, FAST_SCORE=1 };

  ORBextractor(int nfeatures, 
                float scaleFactor, 
                int nlevels,
                int iniThFAST, 
                int minThFAST);

  ~ORBextractor(){}

  // Compute the ORB features and descriptors on an image.
  // ORB are dispersed on the image using an octree.
  // Mask is ignored in the current implementation.
  void Compute(cv::InputArray image, 
                cv::InputArray mask,
                std::vector<cv::KeyPoint>& keypoints,
                cv::OutputArray descriptors);

  int GetLevels() const { return nlevels; }

  float GetScaleFactor() const { return scaleFactor; }

  std::vector<float> GetScaleFactors() const { return mvScaleFactor; }

  std::vector<float> GetInverseScaleFactors() const { return mvInvScaleFactor; }

  std::vector<float> GetScaleSigmaSquares() const { return mvLevelSigma2; }

  std::vector<float> GetInverseScaleSigmaSquares() const { return mvInvLevelSigma2; }

  const std::vector<cv::Mat>& GetImagePyramid() { return mvImagePyramid; }

protected:

  void ComputePyramid(cv::Mat image);

  void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint>>& allKeypoints);

  std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, 
                                              const int& minX,
                                              const int& maxX, 
                                              const int& minY, 
                                              const int& maxY, 
                                              const int& nFeatures, 
                                              const int& level);

  void ComputeKeyPointsOld(std::vector<std::vector<cv::KeyPoint>>& allKeypoints);
  
  std::vector<cv::Point> pattern;

  int nfeatures;
  double scaleFactor;
  int nlevels;
  int iniThFAST;
  int minThFAST;

  std::vector<int> mnFeaturesPerLevel;

  std::vector<int> umax;

  std::vector<float> mvScaleFactor;
  std::vector<float> mvInvScaleFactor;    
  std::vector<float> mvLevelSigma2;
  std::vector<float> mvInvLevelSigma2;
  std::vector<cv::Mat> mvImagePyramid;
};


#endif // ORB_EXTRACTOR_H_