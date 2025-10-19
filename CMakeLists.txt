cmake_minimum_required(VERSION 3.18)
project(gpupy LANGUAGES CXX CUDA)

# ============================================================
# Basic configuration
# ============================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 打印调试信息
message(STATUS "Building gpupy with CUDA support")

# ============================================================
# Find Python and pybind11
# ============================================================
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# ============================================================
# CUDA configuration
# ============================================================
enable_language(CUDA)
find_package(CUDA REQUIRED)

# 设置 CUDA 架构 (根据你的 GPU 调整)
# 常见：86 (Ampere), 89 (Ada), 90 (Hopper)
set(CMAKE_CUDA_ARCHITECTURES 86)

# 编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xcompiler -fPIC")

# ============================================================
# Include directories
# ============================================================
include_directories(${PROJECT_SOURCE_DIR}/src)

# ============================================================
# Source files
# ============================================================
set(SRC_FILES
    src/gpu_memory.cpp
    src/gpu_array.cpp
    src/cuda_kernels.cu
    src/bindings.cpp
)

# ============================================================
# Build shared library for Python import
# ============================================================
pybind11_add_module(gpupy ${SRC_FILES})

# 链接 CUDA 库
target_link_libraries(gpupy PRIVATE CUDA::cudart)

# ============================================================
# Compile options for C++
# ============================================================
target_compile_options(gpupy PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
)

# ============================================================
# Output settings
# ============================================================
set_target_properties(gpupy PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/gpupy
)

message(STATUS "gpupy will be built into: ${PROJECT_SOURCE_DIR}/gpupy")
