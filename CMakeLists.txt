cmake_minimum_required(VERSION 3.18)
project(gpupy LANGUAGES CXX CUDA)

# 要求 C++17（现代 pybind11 与 CUDA 都推荐）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 启用 CUDA 支持
enable_language(CUDA)

# 选择 Release 模式
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 找到 pybind11（需要先 pip install pybind11）
find_package(pybind11 REQUIRED)

# 输出消息（调试用）
message(STATUS "Using pybind11: ${pybind11_VERSION}")
message(STATUS "CUDA toolkit found at: ${CMAKE_CUDA_COMPILER}")

# 设置源文件
set(SOURCES
    src/gpu_memory.cpp
    src/gpu_array.cpp
    src/cuda_kernels.cu
    src/bindings.cpp
)

# 创建 Python 模块（类似 setup.py build_ext）
pybind11_add_module(gpupy_backend ${SOURCES})

# 编译选项
target_compile_options(gpupy_backend PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --extended-lambda
        -lineinfo
        -Xcompiler=-fPIC
    >
)

# 链接 CUDA 库
find_package(CUDAToolkit REQUIRED)
target_link_libraries(gpupy_backend PRIVATE CUDA::cudart)

# 输出路径（将生成的 .so 放到 Python 包中）
set_target_properties(gpupy_backend PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/gpupy
)
