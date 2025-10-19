// 封装 CUDA 内存分配与释放，作为 RAII 管理的 GPU 内存对象

#pragma once
#include <cuda_runtime.h>
#include <cstddef>   // for size_t
#include <stdexcept> // for exceptions

// GPU 显存管理类
class GPUMemory {
private:
    void* d_ptr;     // 指向 GPU 显存的指针
    size_t nbytes;   // 显存大小（字节数）

public:
    // 构造函数：分配 nbytes 字节的 GPU 显存
    explicit GPUMemory(size_t nbytes);

    // 禁止拷贝（防止重复释放）
    GPUMemory(const GPUMemory&) = delete;
    GPUMemory& operator=(const GPUMemory&) = delete;

    // 允许移动（资源转移）
    GPUMemory(GPUMemory&& other) noexcept;
    GPUMemory& operator=(GPUMemory&& other) noexcept;

    // 析构函数：自动释放显存
    ~GPUMemory();

    // 从 CPU 复制数据到 GPU
    void copy_from_host(const void* src, size_t nbytes);

    // 从 GPU 复制数据到 CPU
    void copy_to_host(void* dst, size_t nbytes) const;

    // 获取 GPU 指针
    void* data() { return d_ptr; }

    // 获取大小
    size_t size() const { return nbytes; }
};
