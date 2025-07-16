cmake_minimum_required(VERSION 3.30)

project(Neural_Network)

set(CMAKE_CXX_STANDARD 20)

# Dodaj wszystkie pliki .cpp i .cu
add_executable(Neural_Network
        main.cpp
        MINST_Loader.cpp
        Layer.cpp
        Arena.cpp
        softmax_cross_entropy.cpp
)

