##################
## SLAM library ##
##################
add_library(slam_framework SHARED
            
            slam_system.cpp
            
            core/tracker.cpp
            core/local_mapper.cpp
            core/loop_closer.cpp
            
            data/map.cpp
            data/map_point.cpp
            data/frame.cpp
            data/keyframe.cpp
            data/keyframe_database.cpp

            solvers/sim3solver.cpp
            solvers/pnp_solver.cpp

            optimizer/optimizer.cpp

            orb_features/orb_extractor.cpp
            orb_features/orb_matcher.cpp

            util/initializer.cpp
            util/converter.cpp

            ros/ros_publisher.cpp
)

add_dependencies(slam_framework ${catkin_EXPORTED_TARGETS})

target_include_directories(slam_framework PUBLIC
                           ${PROJECT_SOURCE_DIR}/src
                           ${PROJECT_SOURCE_DIR}
                           ${EIGEN3_INCLUDE_DIR}
                           ${catkin_INCLUDE_DIRS}
                           ${PROJECT_SOURCE_DIR}/third_party/nlohmann
)

target_link_libraries(slam_framework PUBLIC
                      ${OpenCV_LIBS}
                      ${catkin_LIBRARIES}
                      DBoW2
                      g2o
)