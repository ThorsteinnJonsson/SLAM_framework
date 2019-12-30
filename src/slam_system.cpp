#include "slam_system.h"

#include <unistd.h> // usleep

// #pragma GCC optimize ("O0")

SlamSystem::SlamSystem(const std::string& vocabulary_path, 
                       const std::string& settings_path, 
                       const SENSOR_TYPE sensor) 
        : sensor_type_(sensor) 
        , activate_localization_mode_(false)
        , deactivate_localization_mode_(false) {
    
  // Output welcome message
  std::cout << "\n" << "Starting SLAM system." << "\n";

  std::cout << "Input sensor was set to: ";
  if (sensor_type_ == SENSOR_TYPE::MONOCULAR) {
    std::cout << "Monocular" << "\n";
  } else if (sensor_type_ == SENSOR_TYPE::STEREO) {
    std::cout << "Stereo" << "\n";
  } else if (sensor_type_ == SENSOR_TYPE::RGBD) {
    std::cout << "RGB-D" << "\n";
  }

  //Load ORB Vocabulary
  std::cout << "\n" << "Loading ORB Vocabulary. This could take a while..." << "\n";

  orb_vocabulary_ = std::make_shared<OrbVocabulary>();
  bool load_sucessful = orb_vocabulary_->loadFromTextFile(vocabulary_path);
  if (!load_sucessful) {
      std::cerr << "Wrong path to vocabulary. " << "\n";
      std::cerr << "Failed to open: " << vocabulary_path << "\n";
      exit(-1);
  }
  std::cout << "Vocabulary loaded!" << "\n" << "\n";

  //Create KeyFrame Database
  keyframe_database_ = std::make_shared<KeyframeDatabase>(orb_vocabulary_);

  //Create the Map
  map_ = std::make_shared<Map>();

  //Initialize the Tracking thread
  //(it will live in the main thread of execution, the one that called this constructor)
  tracker_ = std::make_shared<Tracker>(orb_vocabulary_, 
                                       map_, 
                                       keyframe_database_, 
                                       settings_path, 
                                       sensor_type_);

  //Initialize the Local Mapping thread and launch
  local_mapper_ = std::make_shared<LocalMapper>(map_, 
                                                sensor_type_==SENSOR_TYPE::MONOCULAR);
  local_mapping_thread_.reset(new std::thread(&LocalMapper::Run, local_mapper_));

  //Initialize the Loop Closing thread and launch
  loop_closer_ = std::make_shared<LoopCloser>(map_, 
                                              keyframe_database_, 
                                              orb_vocabulary_, 
                                              sensor_type_ != SENSOR_TYPE::MONOCULAR);
  loop_closing_thread_.reset(new std::thread(&LoopCloser::Run, loop_closer_));

  
  if (ros_output_enabled) {
    ros_publisher_ = std::make_shared<RosPublisher>(map_, tracker_);
    ros_pub_thread_.reset(new std::thread(&RosPublisher::Run, ros_publisher_));
  }

  //Set pointers between threads
  tracker_->SetLocalMapper(local_mapper_);
  tracker_->SetLoopCloser(loop_closer_);

  local_mapper_->SetLoopCloser(loop_closer_);

  loop_closer_->SetLocalMapper(local_mapper_);

}

SlamSystem::~SlamSystem() {

}

cv::Mat SlamSystem::TrackStereo(const cv::Mat& imLeft, 
                                const cv::Mat& imRight, 
                                const double timestamp) {
  if (sensor_type_ != SENSOR_TYPE::STEREO) {
    std::cerr << "ERROR: you called TrackStereo but input sensor was not set to STEREO.\n";
    exit(-1);
  }   

  // Check mode change
  {
    std::unique_lock<std::mutex> lock(mode_mutex_);
    if (activate_localization_mode_) {
      local_mapper_->RequestStop();

      // Wait until Local Mapping has effectively stopped
      while (!local_mapper_->IsStopped()) {
        usleep(1000);
      }

      tracker_->InformOnlyTracking(true);
      activate_localization_mode_ = false;
    }
    if (deactivate_localization_mode_) {
      tracker_->InformOnlyTracking(false);
      local_mapper_->Release();
      deactivate_localization_mode_ = false;
    }
  }

  if (tracker_->NeedSystemReset()) {
    tracker_->Reset();
  }

  cv::Mat Tcw = tracker_->GrabImageStereo(imLeft, imRight, timestamp);

  std::unique_lock<std::mutex> lock2(state_mutex_);
  tracking_state_ = tracker_->GetState();
  tracked_map_points_ = tracker_->GetCurrentFrame().mvpMapPoints;
  tracked_keypoints_un_ = tracker_->GetCurrentFrame().mvKeysUn;
  return Tcw;
}

cv::Mat SlamSystem::TrackRGBD(const cv::Mat& im, 
                              const cv::Mat& depthmap, 
                              const double timestamp) {
  if(sensor_type_!=RGBD) {
    std::cerr << "ERROR: you called TrackRGBD but input sensor was not set to RGBD.\n";
    exit(-1);
  }    

  // Check mode change
  {
    std::unique_lock<std::mutex> lock(mode_mutex_);
    if (activate_localization_mode_) {
      local_mapper_->RequestStop();

      // Wait until Local Mapping has effectively stopped
      while (!local_mapper_->IsStopped()) {
        usleep(1000);
      }

      tracker_->InformOnlyTracking(true);
      activate_localization_mode_ = false;
    }

    if(deactivate_localization_mode_) {
      tracker_->InformOnlyTracking(false);
      local_mapper_->Release();
      deactivate_localization_mode_ = false;
    }
  }

  if (tracker_->NeedSystemReset()) {
    tracker_->Reset();
  }

  cv::Mat Tcw = tracker_->GrabImageRGBD(im, depthmap, timestamp);

  std::unique_lock<std::mutex> lock2(state_mutex_);
  tracking_state_ = tracker_->GetState();
  tracked_map_points_ = tracker_->GetCurrentFrame().mvpMapPoints;
  tracked_keypoints_un_ = tracker_->GetCurrentFrame().mvKeysUn;
  return Tcw;
}

cv::Mat SlamSystem::TrackMonocular(const cv::Mat& im, 
                                   const double timestamp) {
  if (sensor_type_!=MONOCULAR) {
    std::cerr << "ERROR: you called TrackMonocular but input sensor was not set to Monocular.\n";
    exit(-1);
  }

  // Check mode change
  {
    std::unique_lock<std::mutex> lock(mode_mutex_);
    if (activate_localization_mode_) {
      local_mapper_->RequestStop();

      // Wait until Local Mapping has effectively stopped
      while (!local_mapper_->IsStopped()) {
        usleep(1000);
      }

      tracker_->InformOnlyTracking(true);
      activate_localization_mode_ = false;
    }

    if(deactivate_localization_mode_) {
      tracker_->InformOnlyTracking(false);
      local_mapper_->Release();
      deactivate_localization_mode_ = false;
    }
  }

  if (tracker_->NeedSystemReset()) {
    tracker_->Reset();
  }

  cv::Mat Tcw = tracker_->GrabImageMonocular(im, timestamp);

  std::unique_lock<std::mutex> lock2(state_mutex_);
  tracking_state_ = tracker_->GetState();
  tracked_map_points_ = tracker_->GetCurrentFrame().mvpMapPoints;
  tracked_keypoints_un_ = tracker_->GetCurrentFrame().mvKeysUn;
  return Tcw;
}

void SlamSystem::ActivateLocalizationMode() {
  std::unique_lock<std::mutex> lock(mode_mutex_);
  activate_localization_mode_ = true;
}

void SlamSystem::DeactivateLocalizationMode() {
  std::unique_lock<std::mutex> lock(mode_mutex_);
  deactivate_localization_mode_ = true;
}

void SlamSystem::Shutdown() {
  local_mapper_->RequestFinish();
  loop_closer_->RequestFinish();

  // Wait until all thread have effectively stopped
  while (!local_mapper_->IsFinished()  || 
         !loop_closer_->IsFinished()   ||
          loop_closer_->IsRunningGBA()) {
    usleep(5000);
  }
  
  local_mapping_thread_->join();
  loop_closing_thread_->join();
  
  if (ros_publisher_) {
    ros_publisher_->RequestFinish();
    while (!ros_publisher_->IsFinished()) {
      usleep(1000);
    }
    ros_pub_thread_->join();
  }
}

TrackingState SlamSystem::GetTrackingState() const {
  std::unique_lock<std::mutex> lock(state_mutex_);
  return tracking_state_;
}

std::vector<MapPoint*> SlamSystem::GetTrackedMapPoints() const {
  std::unique_lock<std::mutex> lock(state_mutex_);
  return tracked_map_points_;
}

std::vector<cv::KeyPoint> SlamSystem::GetTrackedKeyPointsUn() const {
  std::unique_lock<std::mutex> lock(state_mutex_);
  return tracked_keypoints_un_;
}

void SlamSystem::SaveTrajectoryKITTI(const std::string& filename) const {
  std::cout << "\nSaving camera trajectory to " << filename << " ..." << std::endl;
  if (sensor_type_ == MONOCULAR) {
    std::cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular.\n";
    return;
  }

  std::vector<KeyFrame*> vpKFs = map_->GetAllKeyFrames();
  std::sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

  // Transform all keyframes so that the first keyframe is at the origin.
  // After a loop closure the first keyframe might not be at the origin.
  cv::Mat T_world_origin = vpKFs[0]->GetPoseInverse();

  std::ofstream f;
  f.open(filename.c_str());
  f << std::fixed;

  // Frame pose is stored relative to its reference keyframe 
  // (which is optimized by BA and pose graph).
  // We need to get first the keyframe pose and then concatenate the relative transformation.
  // Frames not localized (tracking failure) are not saved.

  // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
  // which is true when tracking failed (lbL).
  auto lRit = tracker_->GetReferenceKeyframes().begin();
  for (auto lit = tracker_->GetRelativeFramePoses().begin();
            lit != tracker_->GetRelativeFramePoses().end();
            ++lit, ++lRit) {
    KeyFrame* pKF = *lRit;

    cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);
    while(pKF->isBad()) {
      Trw = Trw * pKF->mTcp;
      pKF = pKF->GetParent();
    }

    Trw = Trw * pKF->GetPose() * T_world_origin;

    cv::Mat Tcw = (*lit) * Trw;
    cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
    cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

    f << std::setprecision(9) 
      <<  Rwc.at<float>(0,0) << " " << Rwc.at<float>(0,1)  << " " << Rwc.at<float>(0,2) << " "  << twc.at<float>(0) << " " <<
          Rwc.at<float>(1,0) << " " << Rwc.at<float>(1,1)  << " " << Rwc.at<float>(1,2) << " "  << twc.at<float>(1) << " " <<
          Rwc.at<float>(2,0) << " " << Rwc.at<float>(2,1)  << " " << Rwc.at<float>(2,2) << " "  << twc.at<float>(2) << std::endl;
  }
  f.close();
  std::cout << "\nTrajectory saved!\n";
}

void SlamSystem::SaveKeyFrameTrajectory(const std::string& filename) const {
  std::cout << "\nSaving keyframe trajectory to " << filename << " ...\n";

  std::vector<KeyFrame*> keyframes = map_->GetAllKeyFrames();
  std::sort(keyframes.begin(), keyframes.end(), KeyFrame::lId);

  // Transform all keyframes so that the first keyframe is at the origin.
  // After a loop closure the first keyframe might not be at the origin.
  //cv::Mat Two = keyframes[0]->GetPoseInverse();

  std::ofstream f;
  f.open(filename.c_str());
  f << std::fixed;

  for (size_t i = 0; i < keyframes.size(); ++i) {
    KeyFrame* pKF = keyframes[i];

    // pKF->SetPose(pKF->GetPose()*Two);

    if(pKF->isBad()) {
      continue;
    }

    cv::Mat R = pKF->GetRotation().t();
    cv::Mat t = pKF->GetCameraCenter();
    f << std::setprecision(9) 
      <<  R.at<float>(0,0) << " " << R.at<float>(0,1)  << " " << R.at<float>(0,2) << " "  << t.at<float>(0) << " " <<
          R.at<float>(1,0) << " " << R.at<float>(1,1)  << " " << R.at<float>(1,2) << " "  << t.at<float>(1) << " " <<
          R.at<float>(2,0) << " " << R.at<float>(2,1)  << " " << R.at<float>(2,2) << " "  << t.at<float>(2) << std::endl;
  }

  f.close();
  std::cout << "\nTrajectory saved!\n";
}