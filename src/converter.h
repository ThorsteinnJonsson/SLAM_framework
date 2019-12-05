#ifndef CONVERTER_H_
#define CONVERTER_H_

class Converter {
public:
  Converter();
  ~Converter();

  static std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors) { return std::vector<cv::Mat>(); } //TODO

  static g2o::SE3Quat toSE3Quat(const cv::Mat &cvT);
  static g2o::SE3Quat toSE3Quat(const g2o::Sim3 &gSim3);

  static cv::Mat toCvMat(const g2o::SE3Quat &SE3);
  static cv::Mat toCvMat(const g2o::Sim3 &Sim3);
  static cv::Mat toCvMat(const Eigen::Matrix<double,4,4> &m);
  static cv::Mat toCvMat(const Eigen::Matrix3d &m);
  static cv::Mat toCvMat(const Eigen::Matrix<double,3,1> &m);
  static cv::Mat toCvSE3(const Eigen::Matrix<double,3,3> &R, const Eigen::Matrix<double,3,1> &t);

  static Eigen::Matrix<double,3,1> toVector3d(const cv::Mat &cvVector);
  static Eigen::Matrix<double,3,1> toVector3d(const cv::Point3f &cvPoint);
  static Eigen::Matrix<double,3,3> toMatrix3d(const cv::Mat &cvMat3);

private:

};

#endif // CONVERTER_H_