cmake_minimum_required(VERSION 3.16)
project(cuFHE LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
find_package(CUDA)
# set(CMAKE_CUDA_FLAGS "-std=c++17 --ptxas-options=-v -arch=sm_61")
set(CMAKE_CUDA_FLAGS "-std=c++17 --ptxas-options=-v -arch=sm_80")

OPTION(USE_80BIT_SECURITY "Use 80bit security parameter(faster)" OFF)
OPTION(USE_CGGI19 "Use the parameter set proposed in CGGI19" OFF)

if(USE_80BIT_SECURITY)
    add_compile_definitions(USE_80BIT_SECURITY)
elseif(USE_CGGI19)
    add_compile_definitions(USE_CGGI19)
endif()

# For OpenMP
find_package(OpenMP)
if(OpenMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

OPTION(ENABLE_TEST "Build tests" OFF)
OPTION(ENABLE_TUTORIAL "Build tutorial" OFF)

add_subdirectory(src)
add_subdirectory(thirdparties/TFHEpp)

if(ENABLE_TEST)
    add_subdirectory(test)
endif()