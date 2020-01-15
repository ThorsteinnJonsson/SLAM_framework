#include "ros_publisher.h"

#include <opencv2/core/eigen.hpp>
#include <sensor_msgs/point_cloud2_iterator.h>

#include <thread>
#include <unistd.h>

RosPublisher::RosPublisher(const std::shared_ptr<Map>& map,
                           const std::shared_ptr<Tracker>& tracker)
      : map_(map)
      , tracker_(tracker) {
  camera_path_pub_ = nh_.advertise<nav_msgs::Path>("camera_path", 1);
  all_pointcloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("point_cloud_all", 1);
  ref_pointcloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("point_cloud_ref", 1);
}

void RosPublisher::Run() {
  
  std::thread camera_path_thread(&RosPublisher::PublishCameraPath, this);
  std::thread pointcloud_thread(&RosPublisher::PublishPointClouds, this);

  camera_path_thread.join();
  pointcloud_thread.join();
  SetFinish();
}


bool RosPublisher::CheckFinish() {
  std::unique_lock<std::mutex> lock(finished_mutex_);
  return finish_requested_;
}

void RosPublisher::RequestFinish() {
  std::unique_lock<std::mutex> lock(finished_mutex_);
  finish_requested_ = true;
}

bool RosPublisher::IsFinished() {
  std::unique_lock<std::mutex> lock(finished_mutex_);
  return is_finished_;
}

void RosPublisher::SetFinish() {
  std::unique_lock<std::mutex> lock(finished_mutex_);
  is_finished_ = true;
}

void RosPublisher::PublishCameraPath() {
  while (!CheckFinish()) {
    nav_msgs::Path camera_path = GetCameraTrajectory();
    camera_path_pub_.publish(camera_path);
    usleep(1000);
  }
}

void RosPublisher::PublishPointClouds() {
  while (!CheckFinish()) {
    sensor_msgs::PointCloud2 all_points_msg = GetMapPoints();
    sensor_msgs::PointCloud2 ref_points_msg = GetRefMapPoints();
    all_pointcloud_pub_.publish(all_points_msg);
    ref_pointcloud_pub_.publish(ref_points_msg);
    usleep(1000);
  }
}

nav_msgs::Path RosPublisher::GetCameraTrajectory() {
  nav_msgs::Path camera_path;
  camera_path.header.frame_id = "camera";
  camera_path.header.stamp = ros::Time::now();

  // Get current trajectory
  std::vector<cv::Mat> current_trajectory;
  std::vector<KeyFrame*> keyframes = map_->GetAllKeyFrames();
  if (keyframes.empty()) {
    return camera_path;
  }
  // std::sort(keyframes.begin(),keyframes.end(),KeyFrame::lId); // TODO enough to just get the first?
  std::nth_element(keyframes.begin(), keyframes.begin(), keyframes.end(), KeyFrame::lId);
  cv::Mat T_world_to_origin = keyframes[0]->GetPoseInverse();

  auto ref_keyframe_it = tracker_->GetReferenceKeyframes().begin();
  for (auto frame_pose = tracker_->GetRelativeFramePoses().begin();
            frame_pose != tracker_->GetRelativeFramePoses().end();
            ++frame_pose, ++ref_keyframe_it){
    KeyFrame* ref_keyframe = *ref_keyframe_it;
    cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);

    while (ref_keyframe->isBad()) {
      Trw = Trw*ref_keyframe->mTcp;
      ref_keyframe = ref_keyframe->GetParent();
    }

    //Trw = Trw*ref_keyframe->GetPose()*Two;  // keep the first frame on the origin
	  Trw = Trw * ref_keyframe->GetPose();

    cv::Mat Tcw = (*frame_pose)*Trw;
    cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
    cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

	  current_trajectory.push_back(Tcw.inv());
  }


  geometry_msgs::PoseStamped cam_pose;

  for (const cv::Mat& pose : current_trajectory) {
    Eigen::Matrix4f eig_pose;
    cv::cv2eigen(pose, eig_pose);
    
    cam_pose.pose.position.x = eig_pose(2,3);
    cam_pose.pose.position.y = -eig_pose(0,3);
    cam_pose.pose.position.z = -eig_pose(1,3);
    Eigen::Quaternionf q(eig_pose.block<3,3>(0,0));	      
    cam_pose.pose.orientation.x = q.z();
    cam_pose.pose.orientation.y = q.x();
    cam_pose.pose.orientation.z = q.y();
    cam_pose.pose.orientation.w = q.w();
    
    camera_path.poses.push_back(cam_pose);	     
  }
  return camera_path;
}

sensor_msgs::PointCloud2 RosPublisher::GetMapPoints() {
  sensor_msgs::PointCloud2 all_points_message;
  all_points_message.header.frame_id = "camera";
  all_points_message.header.stamp = ros::Time::now();
  all_points_message.height = 1;
  all_points_message.width = 0;
  all_points_message.is_bigendian = false;
  all_points_message.is_dense = false;

  const std::vector<MapPoint*>& map_points = map_->GetAllMapPoints();

  if (!map_points.empty()) {
    std::vector<Eigen::Vector4f> points;
    points.reserve(map_points.size());
    for (MapPoint* point : map_points) {
      if (point->isBad()) {
        continue;
      }
      cv::Mat pos = point->GetWorldPos();
      points.emplace_back(pos.at<float>(0),
                          pos.at<float>(1),
                          pos.at<float>(2),
                          1.0f);
    }
    points.shrink_to_fit();

    all_points_message.width = points.size();
    
    sensor_msgs::PointCloud2Modifier modifier(all_points_message);
    modifier.setPointCloud2FieldsByString(1,"xyz");
    modifier.resize(points.size());

    sensor_msgs::PointCloud2Iterator<float> out_x(all_points_message, "x");
    sensor_msgs::PointCloud2Iterator<float> out_y(all_points_message, "y");
    sensor_msgs::PointCloud2Iterator<float> out_z(all_points_message, "z");
    for (const auto& p : points) {
      *out_x = p.z();
      *out_y = -p.x();
      *out_z = -p.y();
      ++out_x;
      ++out_y;
      ++out_z;
    }
  }
  return all_points_message;
}

sensor_msgs::PointCloud2 RosPublisher::GetRefMapPoints() {
  sensor_msgs::PointCloud2 ref_points_message;
  ref_points_message.header.frame_id = "camera";
  ref_points_message.header.stamp = ros::Time::now();
  ref_points_message.height = 1;
  ref_points_message.width = 0;
  ref_points_message.is_bigendian = false;
  ref_points_message.is_dense = false;

  const std::vector<MapPoint*>& ref_points = map_->GetReferenceMapPoints();
  std::vector<Eigen::Vector4f> points;
  points.reserve(ref_points.size());
  for (MapPoint* point : ref_points) {
    if (point->isBad()) {
      continue;
    }
    cv::Mat pos = point->GetWorldPos();
    points.emplace_back(pos.at<float>(0),
                        pos.at<float>(1),
                        pos.at<float>(2),
                        1.0f);
  }
  points.shrink_to_fit();
  
  ref_points_message.width = points.size();
  
  if (!points.empty()) {
    sensor_msgs::PointCloud2Modifier modifier(ref_points_message);
    modifier.setPointCloud2FieldsByString(1,"xyz");
    modifier.resize(points.size());

    sensor_msgs::PointCloud2Iterator<float> out_x(ref_points_message, "x");
    sensor_msgs::PointCloud2Iterator<float> out_y(ref_points_message, "y");
    sensor_msgs::PointCloud2Iterator<float> out_z(ref_points_message, "z");
    for (const auto& p : points) {
      *out_x = p.z();
      *out_y = -p.x();
      *out_z = -p.y();
      ++out_x;
      ++out_y;
      ++out_z;
    }
  }
  
  return ref_points_message;
}
