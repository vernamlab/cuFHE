cmake_minimum_required(VERSION 3.16)
project(cuFHE LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
find_package(CUDA)
# set(CMAKE_CUDA_FLAGS "-std=c++17 --ptxas-options=-v -arch=sm_61")
set(CMAKE_CUDA_FLAGS "-std=c++17 --ptxas-options=-v -arch=sm_80")

# For OpenMP
find_package(OpenMP)
if(OpenMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

OPTION(USE_80BIT_SECURITY "Use 80bit security parameter(faster)" OFF)
OPTION(ENABLE_TEST "Build tests" OFF)
OPTION(ENABLE_TUTORIAL "Build tutorial" OFF)

add_subdirectory(src)
add_subdirectory(thirdparties/TFHEpp)

if(ENABLE_TEST)
    add_subdirectory(test)
endif()