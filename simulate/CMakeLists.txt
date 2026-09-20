cmake_minimum_required(VERSION 3.16)
project(unitree_mujoco)

if (CMAKE_BUILD_TYPE MATCHES "Debug")
  message("Debug mode")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g -O0 -fPIC")
else()
  message("Release mode")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -DNDEBUG -fPIC")
endif()

list(APPEND CMAKE_PREFIX_PATH "/opt/unitree_robotics/lib/cmake")
find_package(unitree_sdk2 REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

# unitree_sdk2 ships a matched libddsc/libddscxx pair.  A sourced ROS
# installation can put a different CycloneDDS build first in LD_LIBRARY_PATH,
# which mixes the two at runtime and trips an assertion in dds_write.  Record
# the SDK library directory as DT_RPATH (rather than the lower-priority
# DT_RUNPATH) so the pair used at link time is also used at runtime.
get_target_property(UNITREE_DDS_LIBRARY ddsc IMPORTED_LOCATION)
get_filename_component(UNITREE_SDK_LIBRARY_DIR "${UNITREE_DDS_LIBRARY}" DIRECTORY)


include_directories(
  mujoco/include
  mujoco/simulate
  src/lodepng
)
link_directories(mujoco/lib)

file(GLOB SIM_SRC
  src/joystick/joystick.cc

  # exclude mujoco/simulate/main.cc
  mujoco/simulate/glfw_*.cc
  mujoco/simulate/platform_*.cc
  mujoco/simulate/simulate.cc
  src/lodepng/lodepng.cpp
)

link_libraries(
  pthread
  mujoco
  glfw
  yaml-cpp
  unitree_sdk2
  boost_program_options
  fmt
)

add_executable(unitree_mujoco ${SIM_SRC} src/main.cc)

add_executable(jstest src/joystick/jstest.cc src/joystick/joystick.cc)

foreach(target unitree_mujoco jstest)
  set_target_properties(${target} PROPERTIES
    BUILD_RPATH "${UNITREE_SDK_LIBRARY_DIR}"
  )
  target_link_options(${target} PRIVATE "LINKER:--disable-new-dtags")
endforeach()
