
#include <iostream>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include "slam_system.h"


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

// void WriteResultsToFile(const std::vector<std::array<float,3>>& positions) {
//   std::ofstream myfile;
//   myfile.open("tmp/positions.txt");
//   for (const auto& pos : positions) {
//     myfile << pos[0] << " " << pos[1] << " " << pos[2] << std::endl;
//   }
//   myfile.close();
// }

int main(int argc, char **argv) {
  // Input TODO: change to actual input 
  std::string kitti_path = "/home/steini/Dev/stereo_slam/dataset/03";

  // Get paths for images
  std::vector<std::string> left_image_paths;
  std::vector<std::string> right_image_paths;
  std::vector<double> timestamps;
  LoadKittiImages(kitti_path, left_image_paths, right_image_paths, timestamps);


  // Set up SLAM system
  std::string vocab_filename = "vocabulary/ORBvoc.txt";
  std::string config_file = "";
  SENSOR_TYPE sensor = SENSOR_TYPE::STEREO;
  SlamSystem slam_system(vocab_filename, config_file, sensor);
  
  // For tracking statistics
  std::vector<double> tracked_times(timestamps.size(), -1.0);
  std::vector<std::array<float,3>> positions;

  // Main loop
  for (size_t frame_id = 0; frame_id < timestamps.size(); ++frame_id) {
  // for (size_t frame_id = 0; frame_id < 100; ++frame_id) {

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
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    
    cv::Mat pose = slam_system.TrackStereo(l_image,r_image,timestamp);
    std::cout << "Finished SLAM on frame " << frame_id << std::endl;
    // printf(" - Pos: %.2f, %.2f, %.2f\n", pose.at<float>(0,3), pose.at<float>(1,3), pose.at<float>(2,3));
    positions.push_back({pose.at<float>(0,3),
                         pose.at<float>(1,3),
                         pose.at<float>(2,3)});

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

    double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    tracked_times[frame_id] = ttrack;

    // Wait to load the next frame
    double T = 0;
    if (frame_id < timestamps.size()-1) {
      T = tracked_times[frame_id+1] - timestamp;
    } else if ( frame_id > 0) {
      T = timestamp - tracked_times[frame_id-1];
    }

    if (ttrack < T) {
      usleep((T-ttrack) * 1e6);
    }
  }

  slam_system.Shutdown();

  std::cout << "Done\n";
  
  // TODO print post-processing stuff
  // WriteResultsToFile(positions);
  slam_system.SaveTrajectoryKITTI("tmp/positions.txt");

}