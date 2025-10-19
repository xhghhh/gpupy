// src/gpu_array.cpp
#include "gpu_array.h"
#include "cuda_kernels.h" // CUDA kernel declarations
#include <iostream>
#include <numeric>

GPUArray::GPUArray(const std::vector<int>& shape)
    : shape(shape) {
    numel = 1;
    for (int d : shape) numel *= d;
    buffer = GPUMemory(numel * sizeof(float));
}

GPUArray::GPUArray(const std::vector<float>& host_data, const std::vector<int>& shape)
    : GPUArray(shape) {
    if (host_data.size() != numel)
        throw std::runtime_error("Host data size mismatch with shape");
    copy_from_host(host_data);
}

GPUArray::GPUArray(GPUArray&& other) noexcept
    : buffer(std::move(other.buffer)), shape(std::move(other.shape)), numel(other.numel) {}

GPUArray& GPUArray::operator=(GPUArray&& other) noexcept {
    if (this != &other) {
        buffer = std::move(other.buffer);
        shape = std::move(other.shape);
        numel = other.numel;
    }
    return *this;
}

void GPUArray::copy_from_host(const std::vector<float>& host_data) {
    if (host_data.size() != numel)
        throw std::runtime_error("Size mismatch in copy_from_host");
    buffer.copy_from_host(host_data.data(), numel * sizeof(float));
}

std::vector<float> GPUArray::copy_to_host() const {
    std::vector<float> host_data(numel);
    buffer.copy_to_host(host_data.data(), numel * sizeof(float));
    return host_data;
}

GPUArray GPUArray::add(const GPUArray& other) const {
    if (other.numel != numel)
        throw std::runtime_error("Shape mismatch in add()");
    GPUArray result(shape);
    launch_add_kernel((float*)data(), (float*)other.data(), (float*)result.data(), numel);
    return result;
}

GPUArray GPUArray::mul(const GPUArray& other) const {
    if (other.numel != numel)
        throw std::runtime_error("Shape mismatch in mul()");
    GPUArray result(shape);
    launch_mul_kernel((float*)data(), (float*)other.data(), (float*)result.data(), numel);
    return result;
}

float GPUArray::dot(const GPUArray& other) const {
    if (other.numel != numel)
        throw std::runtime_error("Shape mismatch in dot()");
    return launch_dot_kernel((float*)data(), (float*)other.data(), numel);
}

std::string GPUArray::info() const {
    std::string s = "GPUArray(shape=[";
    for (size_t i = 0; i < shape.size(); ++i) {
        s += std::to_string(shape[i]);
        if (i + 1 < shape.size()) s += ", ";
    }
    s += "], numel=" + std::to_string(numel) + ")";
    return s;
}
