#ifndef CONVERTER_H_
#define CONVERTER_H_

class Converter {
public:
  Converter();
  ~Converter();

  static std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors) { return std::vector<cv::Mat>(); } //TODO

private:

};

#endif // CONVERTER_H_