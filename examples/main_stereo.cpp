// Main

#include "slam_system.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <iostream>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>

#include <ros/ros.h>

void LoadKittiImages(const std::string& kitti_path, 
                     std::vector<std::string>& left_image_paths,
                     std::vector<std::string>& right_image_paths, 
                     std::vector<double>& timestamps) {
  std::string time_file = kitti_path + "/times.txt";
  std::ifstream time_stream;
  time_stream.open(time_file);
  if (!time_stream.is_open()) {
    std::cerr << "Error opening " << time_file << "\n";
  }
  while (time_stream.good()) {
    std::string s;
    getline(time_stream,s);
    if (!s.empty()) {
      double t = std::stod(s);
      timestamps.push_back(t);
    }
  }
  time_stream.close();

  std::string left_folder = kitti_path + "/image_2/";
  std::string right_folder = kitti_path + "/image_3/";

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

bool ParseArguments(int argc, char** argv,
                    std::string& config_path,
                    std::string& dataset_path) {
  if (argc < 1){
		std::cerr << "No path specified for config json-file.\n";
		return false;
	}
  config_path = argv[1];
  std::cout << "Config path (JSON): " << config_path << "\n";

  if (argc < 2) {
    std::cerr << "No path specified for dataset-folder.\n";
    return false;
  }
  dataset_path = argv[2];
  std::cout << "Dataset path: " << dataset_path << "\n";
  
  return true;
}

int main(int argc, char** argv) {

  std::string config_path;
  std::string kitti_path;
  if (!ParseArguments(argc, argv, config_path, kitti_path)) {
    return 0;
  }

  std::cout << "\nSTEREO SLAM EXAMPLE\n";

  ros::init(argc, argv, "stereo_slam");

  // Get paths for images
  std::vector<std::string> left_image_paths;
  std::vector<std::string> right_image_paths;
  std::vector<double> timestamps;
  LoadKittiImages(kitti_path, left_image_paths, right_image_paths, timestamps);
  if (left_image_paths.empty()) {
    std::cerr << "Could not load images\n";
    return -1;
  }

  // Set up SLAM system
  SENSOR_TYPE sensor = SENSOR_TYPE::STEREO;
  SlamSystem slam_system(config_path, sensor);
  
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
    if (frame_id % 100 == 0) {
      std::cout << "Finished SLAM on frame " << frame_id << std::endl;
    }
    
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
  
  slam_system.SaveTrajectoryKITTI("tmp/positions.txt");

}