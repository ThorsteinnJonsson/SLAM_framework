
#include <iostream>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include "slam.h"


void LoadKittiImages(const std::string& kitti_path, 
                     std::vector<std::string>& left_image_paths,
                     std::vector<std::string>& right_image_paths, 
                     std::vector<double>& timestamps) {
    std::string time_file = kitti_path + "/times.txt";
    std::ifstream time_stream;
    time_stream.open(time_file);
    while(!time_stream.eof()) {
        std::string s;
        getline(time_stream,s);
        if(!s.empty()) {
            double t = std::stod(s);
            timestamps.push_back(t);
        }
    }
    time_stream.close();

    std::string left_folder = kitti_path + "/image_0/";
    std::string right_folder = kitti_path + "/image_1/";

    const int num_files = timestamps.size();
    left_image_paths.reserve(num_files);
    right_image_paths.reserve(num_files);

    for(int i = 0; i < num_files; ++i) {
        std::stringstream ss;
        ss << std::setfill('0') << std::setw(6) << i;
        left_image_paths.push_back(left_folder + ss.str() + ".png");
        right_image_paths.push_back(right_folder + ss.str() + ".png");
    }
}

int main(int argc, char **argv) {
  // Input TODO: change to actual input 
  std::string kitti_path = "/home/steini/Dev/stereo_slam/dataset/01";

  // Get paths for images
  std::vector<std::string> left_image_paths;
  std::vector<std::string> right_image_paths;
  std::vector<double> timestamps;
  LoadKittiImages(kitti_path, left_image_paths, right_image_paths, timestamps);


  // Set up SLAM system
  StereoSlam slam;

  // Main loop
  int num_frames = timestamps.size();
  for (int frame_id = 0; frame_id < num_frames; ++frame_id) {

    cv::Mat l_image = cv::imread(left_image_paths[frame_id], CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat r_image = cv::imread(right_image_paths[frame_id], CV_LOAD_IMAGE_UNCHANGED);
    double timestamp = timestamps[frame_id];
    if (l_image.empty() || r_image.empty()) {
      std::cerr << "Failed to load image pair:\n" 
                << left_image_paths[frame_id] << "\n" 
                << right_image_paths[frame_id] << std::endl;
      return -1;
    }

    // Do SLAM stuff
    // TODO


  }


  std::cout << "Done\n";


}