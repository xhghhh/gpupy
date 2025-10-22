cmake_minimum_required(VERSION 3.18)
project(GemmCutlass LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)
find_package(CUDAToolkit REQUIRED)

set(CUTLASS_ENABLE_EXAMPLES OFF CACHE BOOL "Enable CUTLASS examples")
set(CUTLASS_ENABLE_TESTS OFF CACHE BOOL "Enable CUTLASS tests")
set(CUTLASS_ENABLE_TOOLS OFF CACHE BOOL "Enable CUTLASS tools")

add_subdirectory(3rdparty/cutlass)

add_executable(gemm_example gemm.cu)

set_target_properties(gemm_example PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(gemm_example PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/cutlass/tools/util/include
)

target_link_libraries(gemm_example PRIVATE
    CUDA::cudart
    CUDA::cublas
)

if(NOT DEFINED CUDA_ARCHITECTURES)
    message(FATAL_ERROR "Must specify CUDA architectures via command line! Use: cmake -DCUDA_ARCHITECTURES=\"75;80;86\" ...")
endif()
set_property(TARGET gemm_example PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})

target_compile_definitions(gemm_example PRIVATE DEBUG)

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CUDA_FLAGS_DEBUG "-O3")
