cmake_minimum_required(VERSION 3.10)
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

add_compile_definitions(USE_80BIT_SECURITY)

add_subdirectory(src)
option(USE_80BIT_SECURITY ON)
add_subdirectory(thirdparties/TFHEpp)

if(ENABLE_TEST)
    add_subdirectory(test)
endif()