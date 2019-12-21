cmake_minimum_required(VERSION 3.10)
project(cuFHE LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 14)
find_package(CUDA)

add_subdirectory(lib)
