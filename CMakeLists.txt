cmake_minimum_required(VERSION 3.10)
project(AUGRAN)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -march=native -fopenmp -fpic -mavx2 -mavx -msse -finline-functions -w")

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/hnswlib)

add_executable(build build.cc)
add_executable(update update.cc)
add_executable(query query.cc)

target_link_libraries(build rt)
target_link_libraries(update rt)
target_link_libraries(query rt)