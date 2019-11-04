#ifndef SRC_ORB_MATCHER_H_
#define SRC_ORB_MATCHER_H_

#include<opencv2/core/core.hpp>

class OrbMatcher{
public:
  OrbMatcher();
  ~OrbMatcher();

  static int DescriptorDistance(const cv::Mat& a, const cv::Mat& b) {return -1; } //TODO

private:

};

#endif // SRC_ORB_MATCHER_H_