// 封装一个可操作的 GPU 数组（类似 CuPy 的 ndarray）

#pragma once
#include "gpu_memory.h"
#include <vector>
#include <cstddef>
#include <string>

class GPUArray {
private:
    GPUMemory buffer;            // 底层 GPU 显存封装
    std::vector<int> shape;      // 数组形状
    size_t numel;                // 元素数量

public:
    // 构造函数
    GPUArray(const std::vector<int>& shape);
    GPUArray(const std::vector<float>& host_data, const std::vector<int>& shape);

    // 禁止拷贝
    GPUArray(const GPUArray&) = delete;
    GPUArray& operator=(const GPUArray&) = delete;

    // 允许移动
    GPUArray(GPUArray&& other) noexcept;
    GPUArray& operator=(GPUArray&& other) noexcept;

    // 数据同步
    void copy_from_host(const std::vector<float>& host_data);
    std::vector<float> copy_to_host() const;

    // 基本运算
    GPUArray add(const GPUArray& other) const;
    GPUArray mul(const GPUArray& other) const;
    float dot(const GPUArray& other) const;

    // 获取信息
    const std::vector<int>& get_shape() const { return shape; }
    size_t size() const { return numel; }
    void* data() { return buffer.data(); }
    const void* data() const { return buffer.data(); }

    // 打印调试信息
    std::string info() const;
};
