cmake_minimum_required(VERSION 3.10)
project(cuFHE LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
find_package(CUDA)
# set(CMAKE_CUDA_FLAGS "-std=c++17 --ptxas-options=-v -arch=sm_61")
set(CMAKE_CUDA_FLAGS "-std=c++17 --ptxas-options=-v -arch=sm_80")

add_compile_definitions(USE_80BIT_SECURITY)

add_subdirectory(lib)

if(ENABLE_TEST)
    add_subdirectory(test)
endif()