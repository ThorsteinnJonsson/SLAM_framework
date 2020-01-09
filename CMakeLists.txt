cmake_minimum_required(VERSION 3.0)
project(stone_slam)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

# 
if (CMAKE_BUILD_TYPE STREQUAL "Release")
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")
elseif (CMAKE_BUILD_TYPE STREQUAL "Debug")
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O0 ")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O0 ")
else ()
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O0 ")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O0 ")
endif()

##########################################
## Check C++11 or C++0x support         ##
##########################################
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


##########################################
## Third party libraries                ##
##########################################
# External
find_package(OpenCV 3.0 REQUIRED)
find_package(Eigen3 REQUIRED)
MESSAGE("Found OpenCV: ${OpenCV_FOUND}") 
MESSAGE("Found Eigen: ${EIGEN3_FOUND}") 

# Included in this repo
add_subdirectory(third_party)

# ROS (catkin)
find_package(catkin REQUIRED COMPONENTS 
      roscpp
      rospy 
      rosbag
      tf
      std_msgs 
      sensor_msgs
      geometry_msgs
      nav_msgs
      message_filters 
      message_generation 
)
MESSAGE("Found ROS: ${roscpp_FOUND}") 

catkin_package(
 CATKIN_DEPENDS roscpp 
                sensor_msgs 
                std_msgs 
                message_filters 
                message_runtime
)

##########################################
## Library config                       ##
##########################################
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/lib)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/lib)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

##########################################
## Internal libraries                   ##
##########################################
add_subdirectory(src)
# add_library(stone_slam SHARED
            
#             src/slam_system.cpp
            
#             src/core/tracker.cpp
#             src/core/local_mapper.cpp
#             src/core/loop_closer.cpp
            
#             src/data/map.cpp
#             src/data/map_point.cpp
#             src/data/frame.cpp
#             src/data/keyframe.cpp
#             src/data/keyframe_database.cpp

#             src/solvers/sim3solver.cpp
#             src/solvers/pnp_solver.cpp

#             src/optimizer/optimizer.cpp

#             src/orb_features/orb_extractor.cpp
#             src/orb_features/orb_matcher.cpp

#             src/util/initializer.cpp
#             src/util/converter.cpp

#             src/ros/ros_publisher.cpp
# )

# add_dependencies(stone_slam ${catkin_EXPORTED_TARGETS})

# target_include_directories(stone_slam PUBLIC
#                            ${PROJECT_SOURCE_DIR}/src
#                            ${PROJECT_SOURCE_DIR}
#                            ${EIGEN3_INCLUDE_DIR}
#                            ${catkin_INCLUDE_DIRS}
#                            ${PROJECT_SOURCE_DIR}/third_party/nlohmann
# )

# target_link_libraries(stone_slam PUBLIC
#                       ${OpenCV_LIBS}
#                       ${catkin_LIBRARIES}
#                       DBoW2
#                       g2o
# )


##########################################
## Executables                          ##
##########################################
## STEREO
add_executable(main_stereo examples/main_stereo.cpp)

target_link_libraries(main_stereo 
                      ${OpenCV_LIBS}
                      ${EIGEN3_LIBS}
                      stone_slam
)

## Monocular
add_executable(main_mono examples/main_mono.cpp)

target_link_libraries(main_mono 
                      ${OpenCV_LIBS}
                      ${EIGEN3_LIBS}
                      stone_slam
)
