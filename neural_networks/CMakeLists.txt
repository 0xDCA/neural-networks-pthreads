cmake_minimum_required(VERSION 3.5.1)
project(neural_networks)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -O3 -pthread")

include_directories(
    ${CMAKE_SOURCE_DIR}/lib/eigen
    ${CMAKE_SOURCE_DIR}/lib
)

set(SOURCE_FILES main.cpp FeedforwardNeuralNetwork.cpp FeedforwardNeuralNetwork.h
  common/util.h common/util.cpp common/TrainSettings.h)
add_executable(neural_networks ${SOURCE_FILES})
