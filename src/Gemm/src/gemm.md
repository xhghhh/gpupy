# Gemm

## 结果正确性

终端的测试结果如下
>Found 1 CUDA devices
Device name: NVIDIA GeForce RTX 4050 Laptop GPU
Compute capability: 8.9
>
>--- 测试矩阵大小: 256x256 x 256x256 ---
CPU参考时间: 9 ms
Baseline GEMM [256x256] x [256x256] = [256x256]
执行时间: 1.08314 ms
性能: 14.009 GFLOPS
结果验证: **通过**
My GEMM [256x256] x [256x256] = [256x256]
执行时间: 0.079904 ms
性能: 555.096 GFLOPS
结果验证: **通过**
加速比 = 13.5555
>
>--- 测试矩阵大小: 512x512 x 512x512 ---
CPU参考时间: 71 ms
Baseline GEMM [512x512] x [512x512] = [512x512]
执行时间: 0.44336 ms
性能: 605.457 GFLOPS
结果验证: **通过**
My GEMM [512x512] x [512x512] = [512x512]
执行时间: 0.084256 ms
性能: 3185.95 GFLOPS
结果验证: **通过**
加速比 = 5.26206
>
>--- 测试矩阵大小: 1024x1024 x 1024x1024 ---
CPU参考时间: 2726 ms
Baseline GEMM [1024x1024] x [1024x1024] = [1024x1024]
执行时间: 3.33952 ms
性能: 643.052 GFLOPS
结果验证: 通过
My GEMM [1024x1024] x [1024x1024] = [1024x1024]
执行时间: 0.381088 ms
性能: 5635.14 GFLOPS
结果验证: **通过**
加速比 = 8.76312
>
>--- 测试矩阵大小: 2048x2048 x 2048x2048 ---
CPU参考时间: 104209 ms
Baseline GEMM [2048x2048] x [2048x2048] = [2048x2048]
执行时间: 228.091 ms
性能: 75.3203 GFLOPS
结果验证: **通过**
My GEMM [2048x2048] x [2048x2048] = [2048x2048]
执行时间: 2.25299 ms
性能: 7625.36 GFLOPS
结果验证: **通过**
加速比 = 101.239
>
>=== 测试完成 ===

可以看到，优化kernal的结果是正确的，并且加速比随矩阵大小逐渐增大，在M=N=K=2048时，甚至达到了上百倍的加速比

## 不同优化版本数据分析

在上面的测试数据中，优化后的kernal性能已经非常好了，代码里的kernal主要使用的优化手段有
1. **使用shared memory**：减少全局内存访问次数
2. **线程块tiling**：提高数据重用
3. **转置store A矩阵**：便于合并访存，减少访存事务
4. **向量化访问**：使用float4向量类型
5. **循环展开**：减少循环开销，#pragma unroll
6. **预取数据**：double-buffer，隐藏内存延迟
7. **free bank conflict**：shared-memory swizzle，优化访存

此外我还探索了其它的一些方法，比如异步指令cp.async，block swizzle等，测试效果如下图：
![performerce_img](https://github.com/cackio/image_pic/blob/main/img/202507210856259.png?raw=true)
对比于naive实现，其他几个优化kernal要快很多，尤其是在大矩阵上。图中sgemmopt是使用了优化方法1,2,3,4,5,6，而gemm_bank_free是在sgemmopt的基础上进行了shared memory swizzle，解决了bank conflict冲突，用ncu测了一下，l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum：0，可以看到ld shared memory是没有bank冲突的，从图中看到的优化效果也很明显。

Ampere架构引入了异步拷贝指令cp.async，它直接从全局内存加载数据到SM共享内存中，消除了对中间寄存器文件使用的需要。gemm_cp_sync是在gemm_bank_free的基础上使用异步指令。我在gemm_cp_sync代码中将B矩阵的load shared memory用cp.async实现，A矩阵还是使用寄存器加载数据，因为Ampere架构下的cp.async并不支持直接转置。然后测试得到的结果和原来的gemm_bank_free性能相差无几，用ncu测这两个kernal的数据也差不多，Memory Workload Analysis数据基本一样，究其原因，我认为是流水的深度不够，kernal中使用的是double buffer，在multi-stage策略下cp.async才能很好发挥其作用。

block swizzle也是经常被提到的优化策略，其目的是更好地利用L2 cache，尽量的提高L2 Hit率。在gemm_block_swizzle代码中，我使用的是蛇形swizzle，如图：
<p align="center">
  <img src="https://github.com/cackio/image_pic/blob/main/img/202507211047955.png?raw=true" alt="swizzle_img">
</p>
但从性能图可以看出swizzle后的kernal性能几乎没有提升，然后我用ncu分别测试M=N=K=2048下gemm_bank_free和gemm_block_swizzle这两个kernal，前者的L2 hit率甚至达到了96.36%，而gemm_block_swizzle只高了百分之零点几，几乎没有提升，我认为是在这个矩阵大小下L2 cache足够大，能很好的放下缓存数据，所以block swizzle起不了明显的作用，然后我又在M=N=K=4096的矩阵下测试这两个kernal，两者的L2 hit率明显降低，前者的L2 hit率为50.27%，后者只高了4%左右，swizzle的效果仍然不明显，对此我认为一个重要的原因是block的发射行为难以预测，虽然目前一个共性的认知是block发射顺序是按照x->y->z的顺序来发射的，但随着部分block的结束，有SM可以容纳新的block，SM和block的映射关系是不确定的，或许本来发射顺序靠后的block先被某个SM发射，从而无法利用L2 cache，至少在我的架构和代码上，这种swizzle策略效果甚微。

## 各种优化技术原理

### 1.使用shared memory

对于naive的gemm实现，如果使用mn个线程，每个线程需要读取矩阵A的一行与矩阵B的一列，而后将计算结果写回至矩阵C中，完成计算一共需要从global memory中进行2mnk次读操作和mn次写操作。而从global memory取数非常耗时的，大量的访存操作使得性能无法提高，因此使用shared memory是必要的，计算单元进行计算时可以直接从shared memory中取数，大大减少了访存所需要的时延。

### 2.线程块tiling

分块策略如图：
![tiling](https://github.com/cackio/image_pic/blob/main/img/202507211142327.png?raw=true)
首先将m×n的C矩阵分为多维度的bm×bn小块，将A矩阵和B矩阵分别分为多维度的bm×bk、bk×bn的小块。令M=m/bm，N=n/bn，K=k/bk，那么就需开启M×N个block，每个block负责C矩阵的一个bm×bn小块，并进行K次的迭代。那么分块策略下，一共需要从下global memory访问M×N×K×(bm×bk+bk×bn)个数，也就是m×n×k×(1/bm+1/bn)个数，相比于naive的实现减少为原来的1/2×(1/bm+1/bn)极大地减少了对global memory的访存量。

### 3.转置store A矩阵

转置store A矩阵主要目的是连续访存，从全局内存load bm×bk的A块到shared memory，如果不转置，计算时load shared memory取的是tile的一列，内存不连续，无法用float4向量化访存，并且会造成更多的bank conflict。所以转置store A矩阵也是很有必要的。

### 4.向量化访问

float4向量化访存能减少指令数，一次float4加载等同于4次 float 加载，编译器生成的汇编里LD.128/ST.128代替多个LD.32/ST.128，而GPU每个周期能发射的load/store指令数有限，float4向量化能让总吞吐更高。

除此之外float4向量化访存还有利于合并访存，比如从全局内存load bk×bn的B块到shared memory，代码中该tile的形状为8×128,一个block中的16×16个thread会load这个tile，一个thread负责load一个float4，一个warp就负责一行连续的128个float数，这样的话可以合并成4个128B 的transaction，大大减少了transaction的数量，大幅降低总线开销。

### 5.循环展开

循环展开主要是用#pragma unroll这个指令通知编译器对后面紧跟的循环进行展开。循环代码往往会有跳转指令，而跳转指令会带来分支预测、流水线清空等延迟，循环里面指令较少的时候，循环展开可以有效地减少分支跳转的频率，以及提高计算指令的占比，从而能更好地提高指令ILP，但循环展开的次数也不能太多，不然会导致指令过多，指令cache压力过大，反而会出现性能下降的问题。

循环展开指令后，编译器还能在更大的指令窗口内做指令重排，充分利用 GPU 的超标量能力和多条发射宽度。

### 6.预取数据

数据预取的一个重要作用就是减少数据依赖，如果不使用数据预取，一个线程要先发射load指令，并等待load完成后，才能发射计算指令，因为计算指令是依赖于load的数据的，而采用数据预取后，一个线程发射load指令后，可以马上发射计算指令，因为计算指令所依赖的数据已经取好了，虽然下发指令有前后顺序，但这时硬件内存单元和计算单元可以同时执行，很好地隐藏了内存访问延迟，大大提高吞吐率，提升了kernal性能。

### 7.free bank conflict

解决bank conflict是kernal性能优化的重点内容，从上面的性能图里可以看出优化效果是很显著的。

和单个float访存不同，以float4向量化访问共享内存，**warp的访存会被拆成多个 transaction进行**，并以Half-Warp为单位来计算，对于每个Half-Warp而言，除非触发广播机制，这个Half-Warp中有多少个活跃的Quarter-Warp就需要多少个 memory transaction，如果触发广播机制那么两个Quarter-Warp中的 transaction就可以被合并成一个，触发广播机制只需满足以下条件中的至少一个：
> * 对于Warp内所有活跃的第 i 号线程，第 i xor 1号线程不活跃或者访存地址和其一致；
> * 对于Warp内所有活跃的第 i 号线程，第 i xor 2号线程不活跃或者访存地址和其一致；

每个quarter warp产生一次memory transaction。所以每个warp每次请求，默认会有4次memory transaction。（没有 bank conflict 的情况下）。

sgemmopt的实现里，从共享内存load矩阵时会产生两路bank conflict。就B矩阵而言，两个Half-Warp会取Bs同一行128个float数据，示意图如下：
![bank](https://github.com/cackio/image_pic/blob/main/img/202507212053703.png?raw=true)
(只画了一个Quarter-warp，并未全部画完，怕线太多影响效果)
以一个Quarter-Warp为例，可以看到发生了两路的bank conflict，原本一个quarter warp需要一次memory transaction，因为bank conflict变成2个memory transaction，然后在一个half warp内也没能满足合并规则，一个half warp就是4个memory transaction，一个warp就是8个memory transaction。

然后再看gemm_bank_free里的swizzle方式，8×8的读取变成4个4×4的读取，从而避免bank冲突，并且warp内的线程以z-order方式排列。
![swizzle](https://github.com/cackio/image_pic/blob/main/img/202507212138761.png?raw=true)
以取A shared memory为例，bank 示意图如下：
![bank2](https://github.com/cackio/image_pic/blob/main/img/202507212214714.png?raw=true)
(只画了一个half-warp，并未全部画完)
可以看到并没有bank冲突，并且z-order排列下满足transaction合并规则，原本一个half-warp需要两个transaction，现在合并为一个transaction，一个warp就只需要2个transaction，可以看到相比于原本的kernal，大大减少了访存事务。

## Tensor Core

Tensor Core（张量核心）是NVIDIA在其GPU架构中新增的一类专用计算单元，专门为高吞吐的矩阵乘加MMA操作设计。我电脑的GPU是Ada架构，虽然tensor核心的MMA/WMMA指令不直接接受fp32输入矩阵，但可以通过转换为tf32进行计算，并且计算结果的精度足以保证大部分元素的误差在可接受范围内，至少在epsilon=1e-6时，完全可以通过正确性测试。下面介绍一下使用wmma的sgemm的一些优化策略。

先给出性能测试图：
![wmma](https://github.com/cackio/image_pic/blob/main/img/202507251852414.png?raw=true)

### WMMA指令

使用wmma::fragment,wmma::load_matrix_sync,wmma::mma_sync等指令，充分利用Ada架构的tensor core进行高吞吐量的矩阵乘加运算。根据NV的文档，支持tf32的wmma形状为m16n16k8，然后我还好奇sass层面对应的mma是什么形状，反汇编代码后看到形状是m16n8k4，但为什么是这个形状呢，又去查了下文档才知道，与tf32对应mma形状就是m16n8k4，这个是由编译器来做的，CUDA编程层面只需关注wmma API就行。

### 多阶段流水multi-stage 与 异步指令cp.async

通过 cp.async.cg.shared.global 指令实现全局内存到共享内存的异步数据运。使用多阶段流水线（K_STAGE），将数据搬运和计算重叠，隐藏内存访问延迟。
整个过程可简述为预热阶段先搬运 K_STAGE-1 组数据，主循环每次推进一组，最后收尾阶段处理剩余数据。

值得一提的是stage=3时性能较差，我问了几个大模型其中的原因，都提到了bank 冲突和代码中```%3```的指令开销，我是更偏向于后者，我觉得提升流水深度并不会导致bank冲突，我用ncu分别测了两种情况，得到的bank 冲突情况是完全一样的，那很有可能是代码中```%3```的指令开销。
```c++
    int smem_sel = (k + 1) % K_STAGE; 
    int smem_sel_next = k % K_STAGE;  
```
这是代码里的一部分，相比于位运算（AND/SHR）或乘加（IMAD）等指令，PTX 里的整型除法（DIV）和取余（REM）开销非常巨大，当K_STAGE=2or4时，编译器会用位与运算(&1/&3)进行很好的优化；当K_STAGE=3时，编译器其实也进行了优化，没有使用div（除法）和 rem（取余）指令，但优化后的指令还是会比前者的更复杂，我将两者这部分的ptx码进行了比对：
```
//stage-2
shl.b32 	%r71, %r155, 12;
	and.b32  	%r72, %r71, 4096;
	add.s32 	%r74, %r56, %r72;
	add.s32 	%r76, %r74, %r55;
	shl.b32 	%r77, %r6, 2;
	add.s32 	%r69, %r76, %r77;
	mul.wide.s32 	%rd12, %r153, 4;
	add.s64 	%rd10, %rd3, %rd12;
```

```
//stage3
add.s32 	%r13, %r185, 1;
	mul.wide.u32 	%rd16, %r13, -1431655765;
	shr.u64 	%rd17, %rd16, 33;
	cvt.u32.u64 	%r64, %rd17;
	mul.lo.s32 	%r65, %r64, 3;
	sub.s32 	%r66, %r13, %r65;
	shl.b32 	%r67, %r185, 3;
	add.s32 	%r68, %r11, %r67;
	add.s32 	%r69, %r67, %r2;
	mad.lo.s32 	%r70, %r69, %r14, %r10;
	mul.wide.u32 	%rd18, %r185, -1431655765;
	shr.u64 	%rd19, %rd18, 33;
	cvt.u32.u64 	%r71, %rd19;
	mul.lo.s32 	%r72, %r71, 3;
	sub.s32 	%r73, %r185, %r72;
	shl.b32 	%r74, %r73, 12;
	add.s32 	%r76, %r48, %r74;
	add.s32 	%r78, %r76, %r47;
	shl.b32 	%r79, %r6, 2;
	add.s32 	%r62, %r78, %r79;
	mul.wide.s32 	%rd20, %r68, 4;
	add.s64 	%rd14, %rd3, %rd20;
```
可以看到stage3这部分的指令明显要比stage2要多，随着迭代的进行，性能自然也就比stage2要差一点。

### block swizzle

block swizzle已经提过了，不再赘述，代码中的swizzle策略是把原本连续的块编号打散到不同的z层，从性能图中可以看出这样的swizzle方式能一定程度优化kernal性能的。代码：
```c++
    \\no swizzle
    dim3 block(NUM_THREADS);                                                
    dim3 grid(div_ceil(N, BN), div_ceil(M, BM)); 

    \\swizzle
    const int N_SWIZZLE = (N+stride -1) / stride;
    dim3 block(NUM_THREADS);
    dim3 grid((div_ceil(N, BN) + N_SWIZZLE - 1) / N_SWIZZLE, div_ceil(M, BM),  N_SWIZZLE);
```

以M=N=K=1024的矩阵，blocksize=128*128为例，swizzle示意图如下：

![block_swizzle](https://github.com/cackio/image_pic/blob/main/img/202507251504365.png?raw=true)

测试代码见同一文件夹下sgemm_test.cu和sgemm.py。

若有不对的地方，还请佬指正。