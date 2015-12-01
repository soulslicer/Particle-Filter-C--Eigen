cmake_minimum_required(VERSION 2.8.3)
project(eigen_testing)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(GSL REQUIRED)

find_package(catkin REQUIRED PCL REQUIRED COMPONENTS
  pcl_conversions
  pcl_ros
  roscpp
  sensor_msgs
  PCL
)

catkin_package(
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${GSL_INCLUDE_DIRS}
)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

file(GLOB eigen_testor_SRC
    "src/*.h"
    "src/*.cpp"
)
add_executable(eigen_testor ${eigen_testor_SRC})

target_link_libraries(
  eigen_testor
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${GSL_LIBRARIES}
)
target_link_libraries(eigen_testor /usr/local/lib/libyaml-cpp.a)
#target_link_libraries(eigen_testor /usr/lib/x86_64-linux-gnu/libyaml-cpp.so.0.5.1)
