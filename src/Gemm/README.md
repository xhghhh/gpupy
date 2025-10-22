# 构建好第三方库
```bash
cmake -B build -G Ninja -DCUTLASS_NVCC_ARCHS=${YOUR_GPU_ARCH} -DCUTLASS_ENABLE_TESTS=OFF -DCUTLASS_UNITY_BUILD_ENABLED=ON 
cmake --build build -j
# eg. For Ampere GPU(A100), the GPU ARCHS is 80
# 更多请查阅：https://docs.nvidia.com/cutlass/media/docs/cpp/quickstart.html#building-for-multiple-architectures
```

# 构建本项目
```bash
cmake -B build -G Ninja -DCUDA_ARCHITECTURES=${YOUR_GPU_ARCH}
cmake --build build -j
```
构建成功后输入命令:  `./build/gemm_example`
预期结果:
```bash
Found xxx CUDA devices
Device name: NVIDIA xxx PCIe
Compute capability: 8.0
```
# 调试 CUDA 程序(!!!!!)
程序调试是
1. 安装 cuda-gdb：查看自己是否安装完毕
```bash
which cuda-gdb
# /usr/local/cuda/bin/cuda-gdb
```
2. 安装 Vscode 插件（默认你是在 Vscode 开发）
- Vscode 搜索：`Nsight Visual Studio Code Edition`
- 配置 `.vscode/launch.json`
```bash
"configurations": [
		{
			"name": "GEMM",
			"type": "cuda-gdb",
			"request": "launch",
			"program": "${workspaceFolder}/MLSys/Gemm/build/gemm_example",
			"cwd": "${workspaceFolder}/MLSys/Gemm",
			"environment": [],
			"stopAtEntry": false,
			"setupCommands": [
				{
					"description": "启用 gdb 的整齐打印",
					"text": "-enable-pretty-printing",
					"ignoreFailures": true
				}
			]
		}
	]
```
- 开始调试

# 本章节目标
矩阵乘法是机器学习中最基础且最重要的计算操作之一。本章节的主要目标是：
1. **理解GEMM基础**：掌握通用矩阵乘法（General Matrix Multiply, GEMM）的基本实现
2. **CUDA并行编程**：学习使用CUDA实现高性能的矩阵乘法
3. **性能优化**：通过多种优化技术提高CUDA GEMM的加速比
4. **性能分析**：学会测量和分析GPU程序的性能

## 要求
1. **基础实现**：使用CUDA实现一个基本的矩阵乘法kernel
2. **优化实现**：尽可能利用可行的优化提高加速比：
3. **性能测试**：对比不同实现的性能，计算相对于基准(baseline)的加速比

## 评估标准(这很重要～)
- **正确性**：实现结果与参考 Baseline 实现一致
- **性能**：CUDA 实现相对于 Baseline 的加速比
- **代码质量**：代码结构清晰，注释完善
- **优化程度**：尝试多种优化技术的效果

## 加速比计算
```
加速比 = (你的程序)执行时间 / Naive CUDA实现(Baseline)执行时间
```
## GEMM
对于 `C[M,N] = A[M,K] * B[K,N]`的主要计算步骤:
```c
void gemm_example(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}
```

### 优化方向
1. **使用共享内存**：减少全局内存访问次数
2. **线程块tiling**：提高数据重用
3. **向量化访问**：使用float4等向量类型
4. **循环展开**：减少循环开销
5. **预取数据**：隐藏内存延迟
6. **.......**

## 性能分析工具
如何知道你的程序优化空间/瓶颈在哪是一个非常重要的问题，他能更好的帮助你优化你的 CUDA 程序
### NVIDIA Profiler
```bash
# 使用nsight compute分析kernel性能
# 可以查看内存带宽利用率、计算吞吐量等关键指标
ncu --target-processes all --export profile ./build/gemm_example
```
