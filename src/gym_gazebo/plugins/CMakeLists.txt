cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

find_package(gazebo REQUIRED)
find_package(roscpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(gym_gazebo REQUIRED)
include_directories(${GAZEBO_INCLUDE_DIRS})
include_directories(${roscpp_INCLUDE_DIRS})
include_directories(${std_msgs_INCLUDE_DIRS})
include_directories(${gym_gazebo_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")

add_library(hello_world SHARED hello_world.cpp)
target_link_libraries(hello_world ${GAZEBO_LIBRARIES})

add_library(model_push SHARED model_push.cpp)
target_link_libraries(model_push ${GAZEBO_LIBRARIES})

add_library(world_edit SHARED world_edit.cpp)
target_link_libraries(world_edit ${GAZEBO_LIBRARIES})

add_library(step SHARED step.cpp)
target_link_libraries(step ${GAZEBO_LIBRARIES} ${roscpp_LIBRARIES})
