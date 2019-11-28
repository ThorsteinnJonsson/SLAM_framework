cmake_minimum_required(VERSION 3.0)
project(stereo_slam)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Debug)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

# Check C++11 or C++0x support
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

find_package(OpenCV QUIET)
# find_package(OpenCV 3.0 QUIET)
# if(NOT OpenCV_FOUND)
#    find_package(OpenCV 2.4.3 QUIET)
#    if(NOT OpenCV_FOUND)
#       message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
#    endif()
# endif()
find_package(Eigen3 REQUIRED)
MESSAGE("Found OpenCV: ${OpenCV_FOUND}") 
MESSAGE("Found Eigen: ${EIGEN3_FOUND}") 

##########################################
## Library config                       ##
##########################################
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/lib) #TODO change to build/lib

##########################################
## Third party libraries                ##
##########################################
add_subdirectory(third_party)

##########################################
## Include Dirs                         ##
##########################################
# TODO Not the best way to do this, move to target_include... later 
include_directories(${PROJECT_SOURCE_DIR}
                    ${PROJECT_SOURCE_DIR}/src
                    ${EIGEN3_INCLUDE_DIR}
)

##########################################
## Libraries                            ##
##########################################
add_library(stereo_slam SHARED
            src/slam_system.cpp
            src/tracker.cpp
            # src/local_mapper.cpp
            # src/loop_closer.cpp 
            src/orb_extractor.cpp
            src/orb_matcher.cpp
            # src/converter.cpp
            src/map_point.cpp
            src/keyframe.cpp
            src/map.cpp
            # src/optimizer.cpp
            src/pnp_solver.cpp
            src/frame.cpp
            # src/keyframe_database.cpp
)

target_include_directories(stereo_slam PUBLIC
                           ${PROJECT_SOURCE_DIR}/src
                           ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(stereo_slam PUBLIC
                      ${OpenCV_LIBS}
                      ${EIGEN3_LIBS}
                      DBoW2
                     #  g20
)


##########################################
## Executables                          ##
##########################################
add_executable(main_exe main.cpp)
target_include_directories(main_exe PUBLIC 
                           ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(main_exe ${OpenCV_LIBS}
                               ${EIGEN3_LIBS}
                               stereo_slam
)

