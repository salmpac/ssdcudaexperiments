cmake_minimum_required(VERSION 3.10)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(cudaproject LANGUAGES CXX CUDA)

find_package(GDAL REQUIRED)

add_executable(cudaproject gdal_stuff.cu interpolations.cu main.cu)

target_link_libraries(cudaproject ${GDAL_LIBRARIES})

include_directories(${GDAL_INCLUDE_DIRS})

set_property(TARGET cudaproject PROPERTY CXX_STANDARD_REQUIRED ON)
