cmake_minimum_required(VERSION 3.0.0)
project(ELSED)

set(CMAKE_BUILD_TYPE "Release")

find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB_RECURSE LIB_SOURCES "src/*.cpp" "src/*.h")
include_directories(src/include)
add_library(elsed ${LIB_SOURCES})
target_link_libraries(elsed ${OpenCV_LIBS})

add_executable(elsed_main src/main.cpp)
target_link_libraries(elsed_main elsed ${OpenCV_LIBS})
