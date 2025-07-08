cmake_minimum_required(VERSION 3.30)
project(Neural_Network)

set(CMAKE_CXX_STANDARD 20)

add_executable(Neural_Network main.cpp MINST_Loader.cpp)
