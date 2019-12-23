cmake_minimum_required(VERSION 3.0)
project(stereo_slam)

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
## Library config                       ##
##########################################
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/lib)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/lib)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

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

##########################################
## Internal libraries                   ##
##########################################
add_library(stereo_slam SHARED
            src/slam_system.cpp
            src/tracker.cpp
            src/local_mapper.cpp
            src/loop_closer.cpp 
            src/orb_extractor.cpp
            src/orb_matcher.cpp
            src/data/map_point.cpp
            src/data/keyframe.cpp
            src/data/frame.cpp
            src/map.cpp
            src/optimizer.cpp
            src/pnp_solver.cpp
            src/keyframe_database.cpp
            src/sim3solver.cpp
            src/initializer.cpp
            src/util/converter.cpp
)

target_include_directories(stereo_slam PUBLIC
                           ${PROJECT_SOURCE_DIR}/src
                           ${PROJECT_SOURCE_DIR}
                           ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(stereo_slam PUBLIC
                      ${OpenCV_LIBS}
                      DBoW2
                      g2o
)


##########################################
## Executables                          ##
##########################################
add_executable(main_exe main.cpp)

# target_include_directories(main_exe PUBLIC 
#                            # ${OpenCV_INCLUDE_DIRS}
# )

target_link_libraries(main_exe 
                      ${OpenCV_LIBS}
                      ${EIGEN3_LIBS}
                      stereo_slam
)

