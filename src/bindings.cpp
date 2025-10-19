// C++ → Python 绑定层，通过 pybind11 暴露 API

// src/bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // 支持 std::vector <-> list
#include "gpu_array.h"

namespace py = pybind11;

PYBIND11_MODULE(gpupy, m) {
    m.doc() = "A minimal CuPy-like library implemented in C++/CUDA";

    // 绑定 GPUArray 类
    py::class_<GPUArray>(m, "GPUArray")
        // 构造函数：从 shape 构造空数组
        .def(py::init<const std::vector<int>&>(), py::arg("shape"))
        // 构造函数：从 host 数据构造
        .def(py::init<const std::vector<float>&, const std::vector<int>&>(),
             py::arg("data"), py::arg("shape"))
        // 方法绑定
        .def("to_host", &GPUArray::copy_to_host)
        .def("add", &GPUArray::add)
        .def("mul", &GPUArray::mul)
        .def("dot", &GPUArray::dot)
        .def("info", &GPUArray::info)
        .def("__repr__", &GPUArray::info)
        // 支持 Python 运算符 +
        .def("__add__", &GPUArray::add)
        .def("__mul__", &GPUArray::mul);

    // 绑定一个方便的函数：gpupy.array(...)
    m.def("array", [](const std::vector<float>& data) {
        std::vector<int> shape = { static_cast<int>(data.size()) };
        return GPUArray(data, shape);
    }, "Create a 1D GPUArray from Python list");

    // 打印版本
    m.attr("__version__") = "0.1.0";
}
