#ifndef SRC_OPTIMIZER_H_
#define SRC_OPTIMIZER_H_

#include "core/loop_closer.h"
#include "data/frame.h"
#include "data/keyframe.h"
#include "data/map_point.h"
#include "data/map.h"


#include "g2o/types/types_seven_dof_expmap.h"

class Optimizer {
public:

  static void GlobalBundleAdjustemnt(const std::shared_ptr<Map>& map, 
                                     const int num_iter=5, 
                                     bool *stop_flag=nullptr,
                                     const unsigned long loop_kf_index=0, 
                                     const bool is_robust=true);

  static void LocalBundleAdjustment(KeyFrame* keyframe, 
                                    bool* stop_flag, 
                                    const std::shared_ptr<Map>& map);

  static int PoseOptimization(Frame& frame);
  
  // if fix_scale is true, 6DoF optimization (stereo,rgbd), 7DoF otherwise (mono)
  static void OptimizeEssentialGraph(const std::shared_ptr<Map>& pMap, 
                                     KeyFrame* pLoopKF, 
                                     KeyFrame* pCurKF,
                                     const LoopCloser::KeyFrameAndPose& non_corrected_sim3,
                                     const LoopCloser::KeyFrameAndPose& corrected_sim3,
                                     const std::map<KeyFrame*,std::set<KeyFrame*>>& LoopConnections,
                                     const bool fix_scale);

  // if bFixScale is true, optimize SE3 (stereo,rgbd), Sim3 otherwise (mono)
  static int OptimizeSim3(KeyFrame* pKF1, 
                          KeyFrame* pKF2, 
                          std::vector<MapPoint*>& vpMatches1,
                          g2o::Sim3& g2oS12, 
                          const float th2, 
                          const bool bFixScale);

private:
  static void BundleAdjustment(const std::vector<KeyFrame*>& keyframes, 
                               const std::vector<MapPoint*>& map_points,
                               const int num_iter=5, 
                               bool* stop_flag=nullptr, 
                               const unsigned long loop_kf_index=0,
                               const bool is_robust=true);
};

#endif // SRC_OPTIMIZER_H_