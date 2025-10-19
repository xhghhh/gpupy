# gpupy: A Minimal CuPy-like GPU Array Library

## 📘 Overview

**gpupy** is a minimal GPU-accelerated array computation library implemented in **C++**, **CUDA**, and **Python**.  
It is designed as an educational and experimental project to reproduce the **core functionalities of CuPy**, including:

- GPU tensor storage and management  
- Basic GPU operators (`add`, `mul`, `dot`)  
- CPU–GPU data transfer  
- NumPy-compatible Python API  
- Correctness and performance benchmarking  

This project demonstrates how a Python interface can be bound to CUDA code for high-performance numerical computing.

---

## 🧩 Project Structure

```

gpupy/
├── README.md                 # Project introduction and usage guide
├── CMakeLists.txt            # Build configuration (pybind11 + CUDA)
│
├── src/                      # C++ / CUDA backend
│   ├── gpu_memory.h/.cpp     # GPU memory management
│   ├── gpu_array.h/.cpp      # Core GPU array class
│   ├── cuda_kernels.cu       # CUDA kernels (add, mul, dot)
│   └── bindings.cpp          # pybind11 bindings
│
├── gpupy/                    # Python frontend (NumPy-like API)
│   └── **init**.py
│
└── tests/                    # Unit and performance tests
├── test_add.py
├── test_mul.py
├── test_dot.py
└── test_perf.py

````

---

## ⚙️ Build Instructions

### 1. Requirements

- **CUDA Toolkit** ≥ 11.0  
- **CMake** ≥ 3.18  
- **Python** ≥ 3.8  
- **pybind11** (for Python–C++ bindings)  
- **NumPy** (for testing and validation)

Install dependencies:
```bash
pip install pybind11 numpy
````

### 2. Build

From the project root (`gpupy/`):

```bash
mkdir build && cd build
cmake .. -DPYBIND11_FINDPYTHON=ON
make -j$(nproc)
```

This will generate a shared library (e.g. `gpupy.cpython-*.so`) that can be directly imported in Python.

---

## 🧪 Usage Example

```python
import numpy as np
import gpupy

# Create arrays
a = np.random.rand(3, 3).astype(np.float32)
b = np.random.rand(3, 3).astype(np.float32)

ga = gpupy.array(a)
gb = gpupy.array(b)

# GPU computations
gc = ga.add(gb)
gd = ga.dot(gb)

# Move back to CPU (NumPy)
print(gpupy.asnumpy(gc))
```

---

## ✅ Testing

Run all correctness tests:

```bash
pytest tests/
```

Run a performance comparison with NumPy:

```bash
python tests/test_perf.py
```

Expected results:

* GPU speedup over NumPy (≥10× for large matrices)
* Numerical results within `1e-6` tolerance

---

## 🧠 Implementation Notes

### Core Modules

| Module         | Function                                       |
| -------------- | ---------------------------------------------- |
| `gpu_memory`   | CUDA memory allocation, copy, and free         |
| `gpu_array`    | Class wrapping GPU pointer and shape info      |
| `cuda_kernels` | CUDA kernels for add, mul, dot                 |
| `bindings`     | pybind11 bridge to expose `GpuArray` in Python |

### Data Flow

```
NumPy array → gpupy.array() → GPU memory (cudaMalloc)
→ CUDA kernels (add/mul/dot) → GPU result
→ gpupy.asnumpy() → back to CPU
```

---

## 📈 Validation Goals

| Criterion   | Requirement                                  |
| ----------- | -------------------------------------------- |
| Correctness | NumPy-equivalent results, error < 1e-6       |
| Performance | GPU version ≥10× faster for 1024×1024 matmul |
| Stability   | No memory leaks or crashes after 100 runs    |
| Usability   | Clean Python API similar to CuPy             |

---

## 🚀 Future Extensions

* Support for CUDA Streams and async execution
* Automatic broadcasting
* cuBLAS/cuFFT integration
* Autograd engine for differentiation
* Unified `device='cuda'|'cpu'` interface

---

## 📚 References

* [CuPy Official Documentation](https://docs.cupy.dev)
* [PyBind11 Documentation](https://pybind11.readthedocs.io)
* [CUDA C Programming Guide (NVIDIA)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
