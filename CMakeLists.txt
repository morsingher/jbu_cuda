cmake_minimum_required (VERSION 3.5)
project (upsampling)

set(CMAKE_CXX_FLAGS "-O3 -ffast-math -march=native -mtune=native")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(THREADS_PREFER_PTHREAD_FLAG ON)

set(CUDA_NVCC_FLAGS "-O3 --use_fast_math -gencode arch=compute_62,code=sm_62")

find_package(CUDA 10.2 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS}
                    src/utils
                    src/jbu)

cuda_add_executable(upsampling src/utils/io_utils.cc
                               src/jbu/upsampling.cc
                               src/jbu/upsampling.cu
                               src/main.cc)

target_link_libraries(upsampling ${OpenCV_LIBS})
