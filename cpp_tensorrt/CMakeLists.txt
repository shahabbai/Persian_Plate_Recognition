cmake_minimum_required(VERSION 3.15)
project(PersianPlateRecognition LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# CUDA architectures: honor an explicit -DCMAKE_CUDA_ARCHITECTURES=... and
# otherwise build for common modern GPUs (Turing/Ampere/Ada). The PTX of the
# highest arch lets newer GPUs JIT at first launch.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75;86;89)
endif()

# TensorRT
set(TENSORRT_ROOT_DIR "" CACHE PATH "Path to TensorRT installation")

# Common install locations checked when TENSORRT_ROOT_DIR is not provided
file(GLOB TENSORRT_ROOT_CANDIDATES
    ${TENSORRT_ROOT_DIR}/lib
    /opt/tensorrt*
    /opt/TensorRT*
    /usr/local/tensorrt*
    /usr/local/TensorRT*
)

find_library(NVINFER_LIB nvinfer
    HINTS ${TENSORRT_ROOT_CANDIDATES}
    PATH_SUFFIXES lib lib64
)
find_library(NVONNXPARSER_LIB nvonnxparser
    HINTS ${TENSORRT_ROOT_CANDIDATES}
    PATH_SUFFIXES lib lib64
)
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin
    HINTS ${TENSORRT_ROOT_CANDIDATES}
    PATH_SUFFIXES lib lib64
)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT_DIR} ${TENSORRT_ROOT_CANDIDATES}
    PATH_SUFFIXES include
)

if(NOT NVINFER_LIB OR NOT NVONNXPARSER_LIB OR NOT NVINFER_PLUGIN_LIB
   OR NOT TENSORRT_INCLUDE_DIR)
    message(FATAL_ERROR
        "TensorRT not found. Re-run cmake with: -DTENSORRT_ROOT_DIR=/path/to/TensorRT-x.x.x")
endif()

set(TENSORRT_LIBRARIES ${NVINFER_LIB} ${NVONNXPARSER_LIB} ${NVINFER_PLUGIN_LIB})
set(TENSORRT_INCLUDE_DIRS ${TENSORRT_INCLUDE_DIR})

# Library directory of the found TensorRT installation
get_filename_component(TENSORRT_LIB_DIRS "${NVINFER_LIB}" DIRECTORY)

message(STATUS "TensorRT include dir: ${TENSORRT_INCLUDE_DIRS}")
message(STATUS "TensorRT lib dir     : ${TENSORRT_LIB_DIRS}")

# Make sure the built binaries can locate the TensorRT shared libraries at
# runtime, even when ${TENSORRT_LIB_DIRS} is not in LD_LIBRARY_PATH.
set(CMAKE_BUILD_RPATH "${TENSORRT_LIB_DIRS}")
set(CMAKE_INSTALL_RPATH "${TENSORRT_LIB_DIRS}")

add_definitions(-DUSE_TENSORRT)

# ---------------------------------------------------------------------------
# YOLO26 TensorRT inference library (from the yolo26 C++ inference project):
# end-to-end (NMS-free) YOLO26 head with full CUDA pre/post-processing.
# ---------------------------------------------------------------------------
add_library(yolo26_inference STATIC
    src/yolo26_base.cpp
    src/yolo26_tensorrt.cpp
    src/cuda_preprocessing.cu
    src/cuda_postprocessing.cu
)

target_include_directories(yolo26_inference PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIRS}
)

target_link_libraries(yolo26_inference PUBLIC
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBRARIES}
)

# ---------------------------------------------------------------------------
# Persian plate recognition application (two-stage pipeline)
# ---------------------------------------------------------------------------
add_executable(persian_plate_recognizer
    src/main.cpp
    src/plate_pipeline.cpp
)

target_link_libraries(persian_plate_recognizer PRIVATE yolo26_inference)
