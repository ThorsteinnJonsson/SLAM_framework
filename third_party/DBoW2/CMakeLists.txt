project(DBoW2)

set(SRCS_DBOW2
  DBoW2/BowVector.cpp
  DBoW2/FORB.cpp      
  DBoW2/FeatureVector.cpp
  DBoW2/ScoringObject.cpp)

set(SRCS_DUTILS
  DUtils/Random.cpp
  DUtils/Timestamp.cpp)

add_library(DBoW2 SHARED 
            ${SRCS_DBOW2} 
            ${SRCS_DUTILS}
)
            
target_link_libraries(DBoW2 
                      ${OpenCV_LIBS}
)
                      
target_include_directories(DBoW2 PUBLIC
                           ${CMAKE_CURRENT_SOURCE_DIR}
                           ${OpenCV_INCLUDE_DIRS}
)


