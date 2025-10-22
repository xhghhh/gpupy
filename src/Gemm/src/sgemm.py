import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.cpp_extension import load


os.environ['TORCH_CUDA_ARCH_LIST'] = "8.9"
torch.set_grad_enabled(False)

device = torch.device('cuda')

# 1. 编译并加载 CUDA kernels

lib = load(
    name='gemmopt_lib',
    sources=['sgemm_test.cu'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    extra_cflags=['-std=c++17'],
)

# 2. 要测试的矩阵规模列表
sizes = [256, 512, 1024,2048, 4096]

# 3. 要测试的 kernel 列表
kernels = {
    'naive_sgemm_test': lib.naive_sgemm_test,
    'sgemmopt_test': lib.sgemmopt_test,
    'gemm_bank_free_test': lib.gemm_bank_free_test,
    'gemm_block_swizzle_test': lib.gemm_block_swizzle_test,
    'gemm_cp_sync_test': lib.gemm_cp_sync_test,
    'sgemm_wmma_test_2stages': lambda A, B, C: lib.sgemm_wmma_test(A, B, C, 2, False, 256),
    'sgemm_wmma_test_3stages': lambda A, B, C: lib.sgemm_wmma_test(A, B, C, 3, False, 256),
    'sgemm_wmma_test_4stages': lambda A, B, C: lib.sgemm_wmma_test(A, B, C, 4, False, 256),
    'sgemm_wmma_test_2stages_swizzle': lambda A, B, C: lib.sgemm_wmma_test(A, B, C, 2, True, 256),
}

# 4. 测试函数：给定 kernel 名称、矩阵大小，返回执行时间（ms）和带宽（GB/s）
def benchmark_kernel(fn, M, repeat=1000,tol=1e-5):
    K = M
    N = M
    # 分配输入/输出矩阵（单精度）
    A = torch.randn((M, M), dtype=torch.float32, device=device)
    B = torch.randn((M, M), dtype=torch.float32, device=device)
    C = torch.empty((M, M), dtype=torch.float32, device=device)


    # 计算参考结果
    C_ref = torch.matmul(A, B)
    # 预热
    fn(A, B, C)
    torch.cuda.synchronize()

    fn(A, B, C)
    torch.cuda.synchronize()
    diff = (C - C_ref).abs()
    max_err = diff.max().item()
    rel_err = (diff / C_ref.abs().clamp_min(1e-6)).max().item()
    correct = (max_err < tol) and (rel_err < tol)
    status = 'OK' if correct else f'ERR max {max_err:.3e}, rel {rel_err:.3e}'

    # 计时
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn(A, B, C)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / repeat  # 平均每次耗时，ms

    flopsPerMatrixMul = 2.0 * M * N * K
    Gflop_s = flopsPerMatrixMul / (ms * 1e-3) / 1e9  # 每秒 Gflop
    return ms, Gflop_s

# 5. 运行所有测试，收集数据
results = { name: {'bw': []} for name in kernels }
for M in sizes:
    for name, fn in kernels.items():
        ms, bw = benchmark_kernel(fn, M)
        print(f"{name:10s}  M={M:4d}  time={ms:7.3f} ms  cf={bw:6.1f} GFlop/s")
        results[name]['bw'].append(bw)

# 6. 画柱状图
x = np.arange(len(sizes))
num_bars = len(results)
bar_width = 0.18  # 每个柱子的宽度
group_gap = 0.25  # 组与组之间的间隔

plt.figure(figsize=(10, 6))

for i, (name, data) in enumerate(results.items()):
    plt.bar(
        x * (num_bars * bar_width + group_gap) + i * bar_width,
        data['bw'],
        bar_width,
        label=name
    )

plt.xticks(
    x * (num_bars * bar_width + group_gap) + (num_bars / 2 - 0.5) * bar_width,
    [str(M) for M in sizes]
)
plt.xlabel('Matrix size M=N=K')
plt.ylabel('Computing performance (Flops/s)')
plt.title('Computing Performance Comparison')
plt.legend()
plt.tight_layout()
plt.savefig("computing_performance_comparison.png", dpi=300)  
plt.show()

