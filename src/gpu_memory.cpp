// src/gpu_memory.cpp
#include "gpu_memory.h"
#include <iostream>

GPUMemory::GPUMemory(size_t nbytes) : d_ptr(nullptr), nbytes(nbytes) {
    if (nbytes == 0) return;

    cudaError_t err = cudaMalloc(&d_ptr, nbytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
    }
}

GPUMemory::~GPUMemory() {
    if (d_ptr) {
        cudaFree(d_ptr);
        d_ptr = nullptr;
    }
}

GPUMemory::GPUMemory(GPUMemory&& other) noexcept
    : d_ptr(other.d_ptr), nbytes(other.nbytes) {
    other.d_ptr = nullptr;
    other.nbytes = 0;
}

GPUMemory& GPUMemory::operator=(GPUMemory&& other) noexcept {
    if (this != &other) {
        if (d_ptr) cudaFree(d_ptr);
        d_ptr = other.d_ptr;
        nbytes = other.nbytes;
        other.d_ptr = nullptr;
        other.nbytes = 0;
    }
    return *this;
}

void GPUMemory::copy_from_host(const void* src, size_t nbytes) {
    if (nbytes > this->nbytes)
        throw std::runtime_error("copy_from_host: size exceeds allocation");
    cudaMemcpy(d_ptr, src, nbytes, cudaMemcpyHostToDevice);
}

void GPUMemory::copy_to_host(void* dst, size_t nbytes) const {
    if (nbytes > this->nbytes)
        throw std::runtime_error("copy_to_host: size exceeds allocation");
    cudaMemcpy(dst, d_ptr, nbytes, cudaMemcpyDeviceToHost);
}
