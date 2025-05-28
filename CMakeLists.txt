cmake_minimum_required(VERSION 3.30)
project(HyperSpheres)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(HyperSpheres main.cpp
        HyperCircle.cpp
        HyperCircle.h
        Point.h
        Utils.h)


find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(HyperSpheres PRIVATE OpenMP::OpenMP_CXX)
endif()