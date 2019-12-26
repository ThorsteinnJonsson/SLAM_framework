#ifndef SRC_ROS_PUBLISHER_
#define SRC_ROS_PUBLISHER_

#include "core/tracker.h"
#include "data/map.h"
#include "data/keyframe.h"

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud2.h>

#include <mutex>
#include <memory>

// Forward declarations
class Tracker;

class RosPublisher {
public:
  RosPublisher(const std::shared_ptr<Map>& map,
               const std::shared_ptr<Tracker>& tracker);
  ~RosPublisher() {}

  void Run();
  
  void RequestFinish();
  bool IsFinished();

private:
  bool CheckFinish();
  void SetFinish();

  void PublishCameraPath();
  void PublishPointClouds();

  nav_msgs::Path GetCameraTrajectory();
  sensor_msgs::PointCloud2 GetMapPoints();
  sensor_msgs::PointCloud2 GetRefMapPoints();

private:
  ros::NodeHandle nh_;

  ros::Publisher camera_path_pub_;
  ros::Publisher all_pointcloud_pub_;
  ros::Publisher ref_pointcloud_pub_; 

  std::shared_ptr<Map> map_;
  std::shared_ptr<Tracker> tracker_;

  std::mutex finished_mutex_;
  bool finish_requested_ = false;
  bool is_finished_ = false;
};


#endif // SRC_ROS_PUBLISHER_