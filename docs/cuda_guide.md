# CUDA 学习指南：配合第 2–3 周

> 目标：读完你应该能 ① 说清 GPU 和 CPU 在硬件上差在哪；② 看懂 ggml 的 CUDA 后端把 whisper 的每个矩阵乘交给了哪个 kernel；③ 解释第一轮 GPU 数据里"encode 快几百倍、decode 只快几倍、量化在 GPU 上几乎没用"这三件事；④ 第 2 周写评测脚本时不踩 GPU 计时的坑。
>
> 标注约定和 [chunk1.md](chunk1.md) 一样：✅ = 已经从源码或实测数据里确认；🔍 = 合理的假设，第 3 周用 Nsight 验证。
>
> 本文的 GPU 数据都在这台台式机上测：RTX 5080 + Ryzen 7 9700X，CUDA 13.4，驱动 621.01。CPU 对比数据来自笔记本 4600H 的第 1 周结果（[results/week1.md](../results/week1.md)，6 线程）。

---

## 目录

0. [一句话总结](#0-一句话总结)
1. [这台机器的 CUDA 环境](#1-这台机器的-cuda-环境)
2. [GPU 硬件模型：用你的 5080 讲](#2-gpu-硬件模型用你的-5080-讲)
3. [CUDA 编程模型](#3-cuda-编程模型)
4. [GPU 上的 Roofline](#4-gpu-上的-roofline)
5. [ggml 的 CUDA 后端：whisper.cpp 怎么用 GPU](#5-ggml-的-cuda-后端whispercpp-怎么用-gpu)
6. [第一轮 GPU 数据和分析](#6-第一轮-gpu-数据和分析)
7. [GPU 计时的坑（写第 2 周评测脚本前必读）](#7-gpu-计时的坑写第-2-周评测脚本前必读)
8. [工具](#8-工具)
9. [动手练习](#9-动手练习)
10. [学习路线和资料](#10-学习路线和资料)
11. [踩过的坑](#11-踩过的坑)
12. [术语表](#12-术语表)
13. [自测题](#13-自测题)
14. [和下一周的关系](#14-和下一周的关系)

---

## 0. 一句话总结

GPU 用几万个同时在跑的线程把 encoder 的大矩阵乘变成了"几毫秒的事"（base 899 ms → 3.7 ms），但 decoder 每个 token 只做很小的矩阵乘，GPU 大部分时间花在**启动 kernel 和同步**上，既没吃满算力也没吃满带宽，所以只快 3–6 倍，量化省下的那点带宽也体现不出来。**CPU 上的两种瓶颈（算力 / 带宽）之外，GPU 上还有第三种：开销（overhead / latency）。**

---

## 1. 这台机器的 CUDA 环境

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5080，Blackwell 架构，compute capability 12.0（`sm_120`） |
| 驱动 | 621.01，支持到 CUDA 13.5 |
| CUDA Toolkit | 13.4，装在 `D:\NVIDIACUDA\NVIDIA_Toolkit\CUDA\v13.4`（`CUDA_PATH` 已指向这里） |
| Nsight | Nsight Systems 2026.3.2、Nsight Compute 2026.3.0，在 `C:\Program Files\NVIDIA Corporation\` |
| 编译 | `third_party/whisper.cpp/build-cuda/`，`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120` |
| CPU 版 | `third_party/whisper.cpp/build/`，两个 build 目录互不影响 |

配置命令（新开的终端里，CUDA 的环境变量才会生效）：

```bash
cmake -S third_party/whisper.cpp -B third_party/whisper.cpp/build-cuda -G "Visual Studio 17 2022" -A x64 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build third_party/whisper.cpp/build-cuda --config RelWithDebInfo -j 16
```

为什么指定 `CMAKE_CUDA_ARCHITECTURES=120`：不指定的话 ggml 会为好几代 GPU 都编一份 kernel，编译慢很多。配置日志里写着 `Replacing 120 in CMAKE_CUDA_ARCHITECTURES with 120a` ✅：`120a` 是"架构专属"版本，能用 Blackwell 特有的指令（比如 `system_info` 里的 `BLACKWELL_NATIVE_FP4 = 1`），代价是编出来的二进制只能在 cc 12.0 的卡上跑。

怎么确认程序真的用了 GPU（`whisper-cli` 的日志）✅：

```
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 16275 MiB):
  Device 0: NVIDIA GeForce RTX 5080, compute capability 12.0, VMM: yes, VRAM: 16275 MiB
whisper_model_load:        CUDA0 total size =    54.71 MB     ← base-q5_0 的权重被整个搬进了显存
whisper_backend_init_gpu: using CUDA0 backend
system_info: ... CUDA : ARCHS = 1200 | ... | CPU : ... AVX512 = 1 ...
```

---

## 2. GPU 硬件模型：用你的 5080 讲

### 2.1 实测参数（✅ 来自 [`scripts/cuda/devq.cu`](../scripts/cuda/devq.cu)，调用 `cudaGetDeviceProperties`）

| 参数 | 值 | 含义 |
|---|--:|---|
| SM 数 | 84 | Streaming Multiprocessor，GPU 的"核"。每个 SM 独立调度自己的线程 |
| 每 SM 最多线程 | 1536 | = 48 个 warp 同时驻留 |
| 每 block 最多线程 | 1024 | |
| warp 大小 | 32 | 32 个线程一组，执行同一条指令 |
| 每 SM 寄存器 | 65536 个 32 位 | = 256 KB，比 L1 还大 |
| 每 SM 共享内存 | 100 KB | 一个 block 最多申请 99 KB |
| L2 | 64 MB | 所有 SM 共享 |
| 显存 | 16 GB GDDR7，256 位总线 | |
| 显存时钟 | 15001 MHz | |
| SM 时钟（最大） | 2655 MHz | |

推算出来的数：

- **CUDA core**：Blackwell 每个 SM 128 个 FP32 单元 → 84 × 128 = **10752**（和官方规格一致）
- **FP32 峰值**：10752 × 2（FMA 算 2 FLOP）× 2.655 GHz ≈ **57 TFLOPS**
- **显存理论带宽**：256 bit ÷ 8 × 15001 MHz × 2（每个时钟传两次）≈ **960 GB/s**
- **实测拷贝带宽**：703 GB/s（1 GiB 读 + 1 GiB 写，约为理论值的 73%；另一次运行是 724，下文统一按 703 算）✅
- **PCIe 实测**（锁页内存）：主机→显卡 51.6 GB/s，显卡→主机 56.1 GB/s ✅。超过了 PCIe 4.0 x16 的上限（约 32 GB/s），说明跑的是 **PCIe 5.0 x16**
- **Tensor Core**：专门做小矩阵乘加的单元，FP16 / INT8 / FP4 的吞吐比 FP32 CUDA core 高好几倍。具体峰值以 NVIDIA Blackwell 白皮书为准，第 3 周用 cuBLAS 实测（练习 9.5）🔍

### 2.2 和 CPU 放在一起比

| | 4600H（笔记本） | 5080 | 比值 |
|---|--:|--:|--:|
| "核" | 6 核 12 线程 | 84 SM，最多 129024 个线程同时驻留 | |
| FP32 峰值 | 约 576 GFLOPS（AVX2，见 chunk1 8.2 节） | 约 57 TFLOPS（不算 Tensor Core） | 约 100× |
| 内存带宽（理论） | 51.2 GB/s | 960 GB/s | 约 19× |
| 最后一级缓存 | L3 8 MB | L2 64 MB | 8× |
| Ridge point（FP32） | 约 19 FLOP/byte | 约 59 FLOP/byte | |

**要记住的一点**：算力差 100 倍，带宽只差 19 倍。所以**计算瓶颈的部分（encoder）在 GPU 上收益最大，带宽瓶颈的部分（decoder）收益小得多**，这是第 6 节所有数据的出发点。

### 2.3 CPU 和 GPU 的设计思路完全不同

- **CPU 追求单线程快**：大缓存、乱序执行、分支预测，目的是让**一个**线程尽快跑完。等内存的时候靠缓存和预取来躲。
- **GPU 追求吞吐量**：每个 SM 驻留几十个 warp，一个 warp 在等内存，调度器马上切到另一个已经准备好的 warp（**切换零开销**，因为每个线程的寄存器一直在寄存器堆里）。GPU 不是"躲开"内存延迟，而是**用足够多的并行把延迟藏起来**（latency hiding）。

推论：**GPU 需要足够多的活才能跑满。** 活太少（比如 decoder 一次只处理 1 个 token），大部分 SM 是空闲的，延迟也藏不住。

### 2.4 内存层级（从快到慢）

```
寄存器        每线程私有，最快，256 KB / SM
  ▼
共享内存 / L1  每 SM 一块，block 内的线程共享，程序员手动管理（__shared__）
  ▼
L2            64 MB，所有 SM 共享，硬件自动管理
  ▼
显存 (VRAM)    16 GB，703 GB/s 实测
  ▼
PCIe 5.0      约 52–56 GB/s，比显存慢 13 倍以上   ← 主机内存在这一头
```

**重点：PCIe 是整条链路里最窄的一段。** 所以 whisper.cpp 在加载时就把所有权重一次性搬进显存（`CUDA0 total size = 54.71 MB`），推理过程中只来回传很小的数据（音频特征、logits）。如果每次推理都要重新传权重，GPU 再快也没用。

---

## 3. CUDA 编程模型

### 3.1 Host 和 Device

- **Host** = CPU 和主机内存，**Device** = GPU 和显存。两边的内存是分开的，指针不能混用。
- 流程：host 分配显存（`cudaMalloc`）→ 把数据拷过去（`cudaMemcpy`）→ 启动 kernel → 把结果拷回来。

### 3.2 Kernel、线程、block、grid

```cpp
__global__ void add(const float * a, const float * b, float * c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // 我是第几个线程
    if (i < n) c[i] = a[i] + b[i];
}

add<<<(n + 255) / 256, 256>>>(a, b, c, n);           // grid 有 ceil(n/256) 个 block，每个 block 256 个线程
```

- `__global__`：kernel，在 GPU 上跑、由 CPU 发起
- 写 kernel 时只写"**一个线程**做什么"，GPU 同时启动成千上万份
- **thread → warp（32 个）→ block（最多 1024 个，同一 block 一定在同一个 SM 上，可以共享 `__shared__` 内存、用 `__syncthreads()` 同步）→ grid（整个 kernel）**
- block 之间没有执行顺序保证，也不能互相等待

### 3.3 Warp 和 SIMT

一个 warp 的 32 个线程**同时执行同一条指令**（SIMT，Single Instruction Multiple Threads），对应 CPU 的 SIMD（AVX2 一条指令处理 8 个 float）。区别是 SIMT 的每个"通道"看起来像一个独立线程，能有自己的分支：

- **分支发散（divergence）**：同一 warp 里一半线程走 `if`、一半走 `else`，硬件只能先跑完一边再跑另一边，效率减半
- **合并访存（coalescing）**：一个 warp 的 32 个线程读**连续**的 32 个 float（128 字节），硬件合成一次内存事务；如果读的地址是分散的，就要拆成很多次事务，带宽大打折扣。**这是写 CUDA kernel 时最常见的性能问题。**

### 3.4 Occupancy（占用率）

每个 SM 最多驻留 48 个 warp，但实际能驻留多少，取决于每个线程用了多少寄存器、每个 block 用了多少共享内存。例：一个线程用 128 个寄存器 → 一个 warp 用 128 × 32 = 4096 个 → 65536 ÷ 4096 = 最多 16 个 warp，占用率 16/48 = 33%。

占用率低 → 能用来藏延迟的 warp 少。但**占用率不是越高越好**：矩阵乘 kernel 常常故意用很多寄存器做 tiling（分块），用低占用率换高数据复用。

### 3.5 Stream、异步和同步

- **kernel 启动是异步的**：`add<<<...>>>()` 只是把任务放进队列就返回，CPU 继续往下跑，GPU 在后台执行
- **stream**：一个任务队列，同一 stream 里的任务按顺序执行，不同 stream 可以并行
- **同步**：`cudaStreamSynchronize` / `cudaDeviceSynchronize` 让 CPU 等 GPU 做完；`cudaMemcpy`（非 Async 版本）也会隐式同步
- **后果**：如果在 kernel 启动后立刻用 CPU 计时，量到的只是"放进队列"的时间（几微秒），不是真正的执行时间。**这是 GPU 计时的第一大坑，见第 7 节。**

### 3.6 Kernel 启动开销

每次启动一个 kernel，CPU 端（驱动）和 GPU 端都有固定开销，量级是**几微秒**。kernel 本身如果只跑几微秒（很小的矩阵乘、逐元素加法），开销就和计算本身一样大了。一个 transformer 层要跑几十个 kernel，decoder 每生成一个 token 要跑整个 decoder → 几百个 kernel → 光开销就可能有 1 ms 左右 🔍。

**CUDA Graphs** 是解决办法：把一串 kernel 录制成一张图，之后一次提交，省掉大部分启动开销。ggml 支持，但 whisper.cpp 默认没打开（见 5.5 节）。

---

## 4. GPU 上的 Roofline

（概念回顾见 [chunk1.md 第 8 节](chunk1.md#8-核心概念计算瓶颈-vs-带宽瓶颈roofline)。）

### 4.1 5080 的屋顶

```
可达性能 = min(峰值算力, 算术强度 × 带宽)

FP32 CUDA core: 57 TFLOPS     ridge point = 57 T / 960 G ≈ 59 FLOP/byte（按实测 703 GB/s 算约 81）
Tensor Core:    高好几倍       ridge point 也跟着变高好几倍
```

| | 算术强度 | 4600H（ridge 约 19） | 5080 FP32（ridge 约 59） |
|---|--:|---|---|
| Encoder | 约 1500 | 计算瓶颈 | 计算瓶颈 |
| Decoder 单 token | 约 1 | 带宽瓶颈 | 带宽瓶颈（理论上） |
| Decoder 5 beam | 约 5 | 带宽瓶颈 | 带宽瓶颈（理论上） |
| Prompt 256 token | 约 256 | 计算瓶颈 | 计算瓶颈 |

### 4.2 Roofline 没画出来的第三种瓶颈

Roofline 假设"活足够多，硬件能跑满"。但 decoder 单 token 的矩阵乘很小：base 的一个 512 × 512 矩阵乘 1 个向量只有 0.5 MFLOP、读 0.5 MB 权重，在 703 GB/s 下只需要 **0.7 微秒**，还没有一次 kernel 启动开销长。这时候时间由**启动开销、同步、CPU 端的调度**决定，叫 **overhead-bound** 或 **latency-bound**。第 6.3 节的数据就是这种情况。

---

## 5. ggml 的 CUDA 后端：whisper.cpp 怎么用 GPU

### 5.1 整体分工（✅ 来自源码）

| 步骤 | 在哪跑 | 依据 |
|---|---|---|
| 读 wav、算 log-mel 频谱图 | **CPU**（多线程） | `log_mel_spectrogram()`，[`src/whisper.cpp`](../third_party/whisper.cpp/src/whisper.cpp) |
| 加载权重 | 读盘后一次性拷进显存 | 日志 `CUDA0 total size` |
| Encoder、cross KV、Decoder 的计算图 | **GPU** | ggml backend scheduler 把计算图交给 CUDA 后端 |
| logits 拷回主机 | 每个 token 一次，51865 个 float ≈ 207 KB | `ggml_backend_tensor_get(logits, ...)`，`src/whisper.cpp:3023` |
| 采样 / beam search | **CPU** | |

所以 decode 的每一步都有一次 **GPU → CPU 同步**：CPU 必须拿到 logits 才能选下一个 token，才能发起下一步计算。数据量本身很小（207 KB ÷ 56 GB/s ≈ 4 微秒），但同步意味着 GPU 必须先把队列清空、等 CPU 下一个指令，流水线断开。

### 5.2 计算图和 backend scheduler

ggml 先在 CPU 上把整个 encoder / decoder 描述成一张**计算图**（每个节点是一个算子：`mul_mat`、`add`、`soft_max`……），然后交给 **backend scheduler**（[`ggml/src/ggml-backend.cpp`](../third_party/whisper.cpp/ggml/src/ggml-backend.cpp)）：

- 每个节点分配到某个后端（CUDA 或 CPU），尽量都放 GPU
- 相邻、同后端的节点合成一个 **split**；跨后端的边界上插入数据拷贝
- `ggml_backend_sched_graph_compute()` = 异步执行 + `ggml_backend_sched_synchronize()` ✅（`ggml-backend.cpp:2112`）。所以 `whisper_encode()` / `whisper_decode()` 返回时，GPU 一定已经算完了，`whisper_print_timings` 的时间是可信的。

CUDA 后端对每个节点调用对应的 kernel，入口在 [`ggml/src/ggml-cuda/ggml-cuda.cu`](../third_party/whisper.cpp/ggml/src/ggml-cuda/ggml-cuda.cu)，具体的 kernel 分在 160 多个 `.cu` / `.cuh` 文件里，每个算子一个文件。

### 5.3 矩阵乘的分派：同一个 `mul_mat`，五种实现（✅ 来自 `ggml_cuda_mul_mat()`，`ggml-cuda.cu` 约 2015–2067 行）

这是本文最重要的一节：**GPU 上跑哪个 kernel，取决于权重的类型和 batch 大小**（`ne11` = 一次乘多少个向量）。按代码里的判断顺序：

| 顺序 | 实现 | 文件 | 什么时候用 | 怎么算 |
|--:|---|---|---|---|
| 1 | **MMVF** | `mmvf.cu` | 权重是 f16/f32，batch 很小。5080 这类有 Ampere 以上 Tensor Core 的卡，f16 只在 batch = 1 时用 | 矩阵 × 向量，CUDA core 做浮点乘加 |
| 2 | **MMF** | `mmf.cu` | 权重是浮点（不是量化），batch ≤ 16，矩阵行数要能被 32 整除 | Tensor Core 浮点矩阵乘 |
| 3 | **MMVQ** | `mmvq.cu` | 权重是量化格式，batch ≤ 8（Blackwell 上 q4_k ≤ 5、q5_k ≤ 6、q6_k ≤ 7，注释写着"tuned on RTX 5090"） | 矩阵 × 向量，直接读量化块，用整数点积指令 |
| 4 | **MMQ** | `mmq.cu` | 量化格式，batch 更大；Turing 以后的卡（含 5080）只要格式支持就一律用 MMQ | 把激活也量化成 int8，在 Tensor Core 上做 int8 矩阵乘 |
| 5 | **cuBLAS** | `ggml_cuda_mul_mat_cublas` | 以上都不满足，比如 f16 权重 × 1500 帧 | NVIDIA 的库，f16 走 Tensor Core |

对应到 whisper：

| | batch（`ne11`） | f16 模型 | 量化模型（q4_0 等） |
|---|--:|---|---|
| Decoder 单 token | 1 | MMVF | MMVQ |
| Decoder 5 beam | 5 | MMF | MMVQ（q4_k 的上限正好是 5） |
| Prompt | 几十到几百 | cuBLAS（超过 16） | MMQ |
| Encoder | 1500 | cuBLAS | MMQ |

这张表是按源码里的条件推出来的，第 3 周用 Nsight Systems 看 kernel 名字逐条确认（练习 9.4）🔍。注意 decoder 里不是所有矩阵乘的 batch 都等于 token 数，比如 attention 里的 `KQ`、`KQV` 是和 KV cache 相乘，走的是 Flash Attention（5.4 节），不经过这张表。

**和 CPU 的对照**：CPU 上 q4_0 / q4_k 快，是因为它们有 REPACK 专用 kernel（chunk1 2.3 节）；GPU 上**所有**量化格式都走同一套 MMQ / MMVQ 模板，格式之间的差别只在"解包一个块要几条指令"。

### 5.4 Flash Attention

`whisper-cli` 和 `whisper-bench` 默认 `flash_attn = true` ✅（`examples/cli/cli.cpp:79`、`examples/bench/bench.cpp:17`），GPU 上 attention 走 `ggml_flash_attn_ext`（`fattn*.cu`）。原理：不把 `QKᵀ` 这个 `T × T` 的大矩阵写回显存，而是分块计算 softmax 和加权求和，全部在共享内存和寄存器里完成，**省掉了大量显存读写**。可以用 `-nfa` 关掉做对比（练习 9.6）。

### 5.5 CUDA Graphs 没开

`build-cuda/CMakeCache.txt` 里 `GGML_CUDA_GRAPHS:BOOL=OFF` ✅，CMake 选项的说明写着 `"ggml: use CUDA graphs (llama.cpp only)"`（`ggml/CMakeLists.txt:210`）。也就是说 whisper.cpp 每个 kernel 都是单独启动的，第 3.6 节说的启动开销一个都没省。

### 5.6 可以改的编译开关

| 开关 | 作用 | 用途 |
|---|---|---|
| `GGML_CUDA_FORCE_CUBLAS` | 量化矩阵乘也走 cuBLAS（先反量化成 f16） | 对比 MMQ 和 cuBLAS |
| `GGML_CUDA_FORCE_MMQ` | 能用 MMQ 就用 | |
| `GGML_CUDA_FA` | 编不编 Flash Attention kernel（默认 ON） | |
| `-ng`（运行时参数） | `whisper-bench` / `whisper-cli` 不用 GPU | **同一个 build-cuda 也能跑 CPU 基线** |

改编译开关时另开一个 build 目录（比如 `build-cuda-cublas`），不要覆盖 `build-cuda`。

---

## 6. 第一轮 GPU 数据和分析

### 6.1 数据（✅ `whisper-bench -t 4`，build-cuda，单次运行，2026-11-01）

| 模型 | 量化 | 加载 (ms) | encode (ms) | decode/token (ms) | batchd/token (ms) | prompt/token (ms) |
|---|---|--:|--:|--:|--:|--:|
| base | f16 | 285 | 3.71 | 1.57 | 0.39 | 0.012 |
| base | q8_0 | 180 | 3.83 | 1.74 | 0.41 | 0.012 |
| base | q5_0 | 129 | 4.00 | 1.72 | 0.38 | 0.012 |
| base | q4_k | 136 | 3.77 | 1.62 | 0.39 | 0.012 |
| base | q4_0 | 122 | 3.56 | 1.71 | 0.40 | 0.012 |
| small | f16 | 692 | 8.04 | 2.88 | 0.78 | 0.022 |
| small | q8_0 | 362 | 8.94 | 3.52 | 0.83 | 0.024 |
| small | q5_0 | 286 | 8.63 | 3.53 | 0.81 | 0.026 |
| small | q4_k | 262 | 9.78 | 3.59 | 0.81 | 0.024 |
| small | q4_0 | 257 | 8.61 | 3.28 | 0.84 | 0.025 |

（`whisper-bench` 在计时前会先把 encode 和三种 decode 各跑一遍预热 ✅，`examples/bench/bench.cpp:96–123`。所以这些是"热"的数字。）

### 6.2 Encoder：快了 160–440 倍 ✅

| | 4600H CPU（6 线程） | 5080 | 倍数 |
|---|--:|--:|--:|
| base f16 | 899 ms | 3.71 ms | 242× |
| small f16 | 3561 ms | 8.04 ms | 443× |
| small q4_k | 1584 ms | 9.78 ms | 162× |

按 chunk1 粗算的 encoder 计算量（base 约 87 GFLOP、small 约 344 GFLOP）：

- base f16：87 GFLOP ÷ 3.71 ms ≈ **23 TFLOPS**
- small f16：344 GFLOP ÷ 8.04 ms ≈ **43 TFLOPS**，已经接近 FP32 CUDA core 的峰值（57），说明矩阵乘确实走了 Tensor Core 🔍
- small 比 base 的利用率高，因为矩阵更大（768 vs 512），GPU 更容易被喂饱

**量化在 GPU 上对 encoder 没有帮助，甚至略慢**（small q4_k 9.78 ms vs f16 8.04 ms）。原因 🔍：encoder 是计算瓶颈，量化省的是带宽，没用；f16 走 cuBLAS 的 Tensor Core，量化走 MMQ，还要多一步把激活量化成 int8、把权重块解包。这和 CPU 上"q4 有 REPACK 专用 kernel 所以快 2 倍"正好相反，**同一个量化格式在不同硬件上的收益可以完全不同**，这是面试时值得讲的一点。

### 6.3 Decoder：只快 3–6 倍，量化反而变慢 ✅（数据）/ 🔍（解释）

| | 4600H CPU | 5080 | 倍数 |
|---|--:|--:|--:|
| base f16 decode/token | 5.78 ms | 1.57 ms | 3.7× |
| small f16 decode/token | 16.65 ms | 2.88 ms | 5.8× |
| small q4_k decode/token | 9.77 ms | 3.59 ms | 2.7× |

先算如果是纯带宽瓶颈应该多快。decoder 每生成一个 token 要把 decoder 的权重读一遍：

- 每层：自注意力 Q/K/V/O = 4d²，交叉注意力只有 Q/O = 2d²（交叉注意力的 K/V 在 encoder 之后算好存起来了），MLP = 2 × d × 4d = 8d²，**合计 14d²**
- 最后乘词表矩阵：n_vocab × d

| | 参数量 | f16 字节 | 703 GB/s 下的下限 | 实测 | 带宽利用率 |
|---|--:|--:|--:|--:|--:|
| base | 6 × 14 × 512² + 51865 × 512 ≈ 48.6 M | 97 MB | 0.14 ms | 1.57 ms | 约 9% |
| small | 12 × 14 × 768² + 51865 × 768 ≈ 139 M | 278 MB | 0.40 ms | 2.88 ms | 约 14% |

**实测比带宽下限慢 7–11 倍，所以 decoder 在 GPU 上不是带宽瓶颈，而是开销瓶颈** 🔍。可能的开销来源（第 3 周用 Nsight Systems 逐个确认）：

1. **kernel 启动开销**：每个 token 要跑完整个 decoder 的计算图，几百个 kernel，每个都有几微秒的固定开销，而且 CUDA Graphs 没开（5.5 节）
2. **每个 token 一次 CPU ↔ GPU 同步**（5.1 节）：GPU 算完 → logits 拷回 → CPU 采样 → CPU 构建下一张计算图 → 再提交
3. **CPU 端构建计算图、scheduler 分配后端**的时间
4. **小 kernel 喂不饱 GPU**：一个 512 × 512 的矩阵 × 向量，可能只用到几个 SM

这也解释了**为什么量化让 GPU decode 变慢**：量化省的是带宽，但带宽只占总时间的一成左右，省下的部分可以忽略；而 MMVQ 要解包量化块，每个 kernel 稍微重一点，加起来反而更慢。CPU 上 decoder 是带宽瓶颈，所以量化让它快了 1.4–1.7 倍（week1 表）；GPU 上不是，所以没用。

**为什么 small 的利用率比 base 高**：权重多了将近 3 倍，但 kernel 数量只多了 1 倍（12 层 vs 6 层），固定开销被摊薄了。

**batchd（5 个 token 一起）比单 token 便宜 4 倍**（base 0.39 vs 1.57 ms/token）：5 个 token 一起算，kernel 启动次数和同步次数都只有一份，开销被 5 个 token 分摊。这也从侧面说明单 token decode 的时间主要不在"算"上。

### 6.4 冷启动：第一次推理慢得多 ✅

`whisper-cli` 跑 jfk.wav（base-q5_0）报告的 `encode time = 68.94 ms`，而 `whisper-bench` 预热后同一个模型只要 4.00 ms，差了 17 倍。第一次调用要做的一次性工作 🔍：加载 kernel 模块到 GPU、cuBLAS 初始化和选算法、分配显存池。**这对第 2 周的评测脚本影响很大**，见第 7 节。

### 6.5 加载时间和 CPU 差不多 ✅

base f16 GPU 285 ms vs CPU 258 ms。把 141 MB 通过 PCIe 拷进显存只需要约 3 ms（141 MB ÷ 52 GB/s），所以加载时间主要是读盘、解析文件、初始化 CUDA 上下文 🔍，和用不用 GPU 关系不大。

### 6.6 这组数据的局限

- 每个模型只跑了一次，没有重复、没有取中位数
- 没有固定 GPU 时钟，GPU 会根据温度和功耗动态升降频
- CPU 对比用的是另一台机器（4600H 笔记本，电源计划"平衡"），不是这台的 9700X
- encoder 的 GFLOP 是粗算，TFLOPS 只能看量级

---

## 7. GPU 计时的坑（写第 2 周评测脚本前必读）

| 坑 | 后果 | 怎么办 |
|---|---|---|
| **异步执行** | kernel 启动后立刻用 CPU 计时，量到的只是排队时间 | 计时点前同步。用 `whisper_print_timings` 的数字没问题，因为 ggml 计算完会同步（5.2 节）；自己用 Python 量墙钟时间也行，因为 `whisper-cli` 退出前一定同步了 |
| **冷启动** | 第一次推理慢 10 倍以上（6.4 节） | 每个模型先跑一次不计时的预热；或者在同一个进程里跑多个音频，只统计第 2 个以后的 |
| **每条音频都启动一次 `whisper-cli`** | 每次都付一遍加载 + CUDA 初始化 + 冷启动 | 第 2 周跑 LibriSpeech 子集时，最好让一个进程处理所有文件（`whisper-cli` 支持一次传多个 `-f`），或者把 load 和推理分开统计 |
| **动态频率** | GPU 温度高了会降频，前后几轮数据不一样 | 多次重复取中位数；记录 `nvidia-smi` 报告的时钟和温度 |
| **RTF 的定义** | GPU 上 encode 只要几毫秒，总时间被 decode 和固定开销主导 | RTF 同时报"含加载"和"不含加载"两种；分开报 encode / decode |
| **显存占用** | Python 的 `psutil` 只能看主机内存，看不到显存 | 用 `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` 或 `pynvml`（NVML）在运行中采样峰值 |
| **CPU 也在干活** | mel 频谱、采样、beam search 在 CPU 上（5.1 节），`-t` 线程数对 GPU 版也有影响 | GPU 版也记录线程数；CPU 对比用 `-ng` 在同一个 build-cuda 上跑 |
| **两台机器的 CPU 不同** | 9700X（Zen 5，AVX-512）和 4600H（Zen 2，AVX2）不能混在一张 CPU 表里 | 结果按机器分目录存（4600h-cpu / 9700x-cpu / 9700x-cuda） |

---

## 8. 工具

| 工具 | 用来干什么 | 位置 / 命令 |
|---|---|---|
| `nvidia-smi` | 看 GPU 型号、驱动、显存占用、时钟、温度、功耗 | 系统 PATH 里有；`nvidia-smi -l 1` 每秒刷新 |
| **Nsight Systems**（`nsys`） | **时间线**：每个 kernel 什么时候开始、跑多久、中间空了多久，CPU 和 GPU 谁在等谁。**第 3 周验证"decode 是开销瓶颈"就用它** | `C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.3.2\`，有 GUI（`nsys-ui`）也有命令行 |
| **Nsight Compute**（`ncu`） | **单个 kernel 的细节**：占用率、显存吞吐、Tensor Core 利用率，自带 roofline 图 | `C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.3.0\` |
| `compute-sanitizer` | 查越界访问、竞争条件（自己写 kernel 时用） | `D:\NVIDIACUDA\NVIDIA_Toolkit\CUDA\v13.4\compute-sanitizer\` |
| Nsight Visual Studio Edition | 在 VS 里调试 kernel（单步、看每个线程的变量） | 已集成进 VS 2022 |

先用 `nsys` 看全局（时间花在哪），再用 `ncu` 看单个 kernel（为什么这个 kernel 慢）。顺序反了会陷进细节里。

---

## 9. 动手练习

### 9.1 跑设备查询（10 分钟）

在开始菜单里打开 "Developer PowerShell for VS 2022"（这样 `cl.exe` 在 PATH 里，`nvcc` 要用它编译主机端代码），进项目目录：

```bash
nvcc -O2 -arch=sm_120 -o $env:TEMP\devq.exe scripts/cuda/devq.cu
```

```bash
& $env:TEMP\devq.exe
```

对着 2.1 节的表看每个数是什么意思。思考：拷贝带宽 703 GB/s 为什么到不了 960？（提示：DRAM 刷新、读写切换的开销、内存控制器效率。）

### 9.2 带宽随数据大小的变化（1 小时）

改 `devq.cu`，把拷贝的大小从 1 MB 扫到 1 GB，画"带宽 vs 大小"的图。预期：小于约 32 MB（读 + 写都能放进 64 MB 的 L2）时带宽明显更高 🔍。这条曲线就是第 3 周 roofline 图的"屋顶"。

### 9.3 手写矩阵乘（半天到一天，最值得做）

依次写：① 朴素版（每个线程算一个输出元素）；② 共享内存分块版（tiling）；③ 调 cuBLAS。对 4096 × 4096 的 f32 矩阵测 GFLOPS，和 57 TFLOPS 峰值比较。参考 Simon Boehm 的文章（第 10 节），他从朴素版一步步优化到接近 cuBLAS。做完你会对"合并访存""数据复用""占用率"有体感，这是面试时最有说服力的经历。

### 9.4 用 Nsight Systems 看 whisper 的时间线（第 3 周的主线）

```bash
& "C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.3.2\target-windows-x64\nsys.exe" profile -o whisper_base --trace=cuda,nvtx third_party/whisper.cpp/build-cuda/bin/RelWithDebInfo/whisper-bench.exe -m models/ggml-base.bin -t 4
```

用 `nsys-ui` 打开 `whisper_base.nsys-rep`，回答：

1. encoder 用了哪些 kernel？矩阵乘是不是 cuBLAS（kernel 名字里有 `gemm` / `cutlass`）？
2. 一次单 token decode 启动了多少个 kernel？kernel 之间的空隙加起来占多少？
3. 量化模型的 decoder 里是不是 `mul_mat_vec_q`（MMVQ）？
4. 6.3 节的四个开销来源各占多少？

### 9.5 量化矩阵乘的三种路线对比

另开 `build-cuda-cublas`，加 `-DGGML_CUDA_FORCE_CUBLAS=ON`，跑 small q4_0 的 `whisper-bench`，和 build-cuda 的结果比 encode 时间。这就是在比 MMQ（int8 Tensor Core）和 cuBLAS（先反量化成 f16 再算）。

### 9.6 Flash Attention 开和关

```bash
third_party/whisper.cpp/build-cuda/bin/RelWithDebInfo/whisper-bench.exe -m models/ggml-small.bin -t 4 -nfa
```

和默认（开）比 encode 时间。encoder 的序列长度是 1500，attention 矩阵是 1500 × 1500，预期差别明显 🔍。

### 9.7 CPU 基线

同一个 build-cuda 加 `-ng` 跑一遍，得到 9700X 的 CPU 数据。和 4600H 比：AVX-512、Zen 5、DDR5 各带来多少提升？

---

## 10. 学习路线和资料

按项目进度排，每一步都和当周的任务挂钩：

| 时间 | 学什么 | 资料 |
|---|---|---|
| 第 2 周前半 | 编程模型（第 3 节）、跑通练习 9.1 | NVIDIA《CUDA C++ Programming Guide》第 1–5 章（Introduction、Programming Model、Programming Interface 的前半） |
| 第 2 周后半 | GPU 计时（第 7 节），写评测脚本的 GPU 部分 | 《CUDA C++ Best Practices Guide》的 Performance Metrics 一章 |
| 第 3 周 | 内存层级、合并访存、共享内存、占用率；练习 9.3、9.4 | 《Programming Massively Parallel Processors》（PMPP，Hwu / Kirk / El Hajj）第 2–6 章；Simon Boehm《How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance》 |
| 第 3 周 | Roofline、Nsight 的用法 | Nsight Systems / Nsight Compute 官方文档的 Getting Started；MIT 6.5940 讲 LLM 部署和量化的几节 |
| 有余力 | Tensor Core、Flash Attention 原理 | NVIDIA Blackwell 架构白皮书；Flash Attention 论文（Dao 等，2022）；GPU MODE 系列讲座 |

读 ggml 源码的顺序建议：`ggml-cuda.cu` 里的 `ggml_cuda_mul_mat()`（5.3 节那张表）→ `mmvq.cu`（最简单的量化 kernel，一个 warp 算一行点积）→ `mmq.cuh`（分块 + Tensor Core，最难）。

---

## 11. 踩过的坑

### 坑 1：C 盘空间不够，CUDA 装不上

`winget install Nvidia.CUDA` 默认装全部组件到 C 盘，安装日志（`DiagOutputDir\...\LOG.setup.exe.log`）里写着 `Extra space required on system for packages: "16629.75 MB"`。**解决**：手动下载安装包，选"自定义"，解压目录和安装目录都改到 D 盘，取消勾选驱动组件（系统里的驱动比安装包自带的新，装了反而是降级）。

### 坑 2：`-T cuda=<路径>` 报 "cannot detect Visual Studio integration files"

CMake 的 `-T cuda=` 会去 `<路径>\extras\visual_studio_integration\MSBuildExtensions` 找文件，这个版本的 Toolkit 没有这个目录。**解决**：不加 `-T`，VS 会用已经装进 `D:\Visual Studio\MSBuild\Microsoft\VC\v170\BuildCustomizations\` 的 `CUDA 13.4.props`。

### 坑 3：`The CUDA Toolkit directory '' does not exist`

VS 的 CUDA 集成靠环境变量 `CUDA_PATH_V13_4` 找 Toolkit。安装程序设好了系统环境变量，但**安装之前就开着的终端拿不到**。**解决**：关掉重开终端（或者重启 VS / Claude Code）。

### 坑 4：`nvcc` 编译时一堆 C4819 警告

`warning C4819: 该文件包含不能在当前代码页(936)中表示的字符`，来自 CUDA 自己的头文件。原因是系统语言是中文，MSVC 默认按 GBK 读源文件。**无害，可以忽略。**

### 坑 5：`nvcc` 找不到 `cl.exe`

`nvcc` 在 Windows 上要调用 MSVC 编译主机端代码。普通 PowerShell 里没有 `cl.exe`，用"Developer PowerShell for VS 2022"或者"x64 Native Tools Command Prompt"。CMake 构建不受影响，因为 VS 生成器自己会设好环境。

---

## 12. 术语表

| 术语 | 解释 |
|---|---|
| SM | Streaming Multiprocessor，GPU 的基本执行单元，5080 有 84 个 |
| CUDA core | SM 里的 FP32 运算单元，5080 共 10752 个 |
| Tensor Core | 专门做小块矩阵乘加的单元，支持 FP16 / BF16 / INT8 / FP8 / FP4 |
| Compute capability (cc) | GPU 架构版本号，5080 是 12.0；编译时的 `sm_120` 就是它 |
| Kernel | 在 GPU 上执行的函数，由 CPU 发起 |
| Thread / Warp / Block / Grid | 线程 / 32 个线程一组 / 最多 1024 个线程、在同一 SM 上 / 一次 kernel 的所有 block |
| SIMT | 单指令多线程，warp 内 32 个线程执行同一条指令 |
| Divergence | 分支发散，同一 warp 的线程走不同分支，要串行执行 |
| Coalescing | 合并访存，一个 warp 读连续地址时合成一次内存事务 |
| Shared memory | SM 上的高速内存，block 内共享，程序员手动管理 |
| Occupancy | 占用率，实际驻留 warp 数 ÷ 最大 warp 数 |
| Latency hiding | 用大量 warp 轮流执行来掩盖内存延迟 |
| Stream | GPU 的任务队列，同一 stream 内按顺序执行 |
| Pinned memory | 锁页内存，不会被操作系统换出，PCIe 传输更快，也是异步拷贝的前提 |
| CUDA Graphs | 把一串 kernel 录成一张图一次提交，省启动开销 |
| cuBLAS | NVIDIA 官方的线性代数库 |
| MMVF / MMVQ | ggml 的浮点 / 量化矩阵 × 向量 kernel（小 batch） |
| MMF / MMQ | ggml 的浮点 / 量化矩阵 × 矩阵 kernel（Tensor Core） |
| dp4a | 一条指令算 4 对 int8 的点积再累加，MMVQ 的核心 |
| Overhead-bound | 时间主要花在启动、同步、调度上，而不是计算或搬数据 |
| Flash Attention | 分块计算 attention，不把 `T × T` 的中间矩阵写回显存 |
| nsys / ncu | Nsight Systems（时间线）/ Nsight Compute（单 kernel 分析） |

---

## 13. 自测题

1. 5080 的 FP32 算力是 4600H 的约 100 倍，带宽只是约 19 倍。据此预测：encoder 和 decoder 哪个在 GPU 上提速更多？实测符合吗？
2. 为什么 GPU 需要"足够多的活"才能跑满？CPU 为什么不太需要？
3. 一个 warp 的 32 个线程分别读 `a[i * 32]`（`i` = 线程号）和读 `a[i]`，哪个快？为什么？
4. 写一个 kernel 后马上用 `time.perf_counter()` 计时，量到的是什么？
5. base 单 token decode 的带宽下限是 0.14 ms，实测 1.57 ms，剩下的时间可能花在哪？怎么验证？
6. 为什么 q4_0 在 CPU 上让 decode 快了 1.6 倍，在 GPU 上却没快？
7. 为什么 batchd（5 个 token）的每 token 成本只有单 token 的 1/4？
8. `whisper-cli` 报 encode 69 ms，`whisper-bench` 报 4 ms，同一个模型同一张卡，为什么？第 2 周的评测脚本该怎么处理？
9. ggml 对 f16 权重 × 1500 帧和 q4_0 权重 × 1 个 token，分别用哪个实现？
10. 如果把 whisper 的权重留在主机内存、每次推理通过 PCIe 传过去，base f16 每次要多花多少时间？

<details>
<summary>参考答案</summary>

1. encoder 是计算瓶颈，吃到 100 倍的算力差距；decoder 理论上是带宽瓶颈，最多吃到 19 倍。实测 encoder 快 240–440 倍（还有 Tensor Core 加成），decoder 只快 3.7–5.8 倍，比 19 倍还低得多，说明 GPU 上的 decoder 连带宽都没吃满（第 5 题）。
2. GPU 靠大量 warp 轮流执行来掩盖内存延迟，活少了就没有 warp 可切换，SM 也会闲着。CPU 靠大缓存、乱序执行、预取让单线程不停顿。
3. 读 `a[i]` 快：32 个线程读连续的 128 字节，合并成一次事务。`a[i * 32]` 每个线程的地址相隔 128 字节，要 32 次事务。
4. 只是 kernel 放进队列的时间（几微秒），因为启动是异步的。要先同步再计时，或者用 `cudaEvent` 计时。
5. kernel 启动开销、每 token 一次的 CPU ↔ GPU 同步、CPU 构建计算图、小 kernel 喂不饱 SM。用 Nsight Systems 看时间线：kernel 本身的总时长、kernel 之间的空隙、CPU 端的耗时分别是多少。
6. CPU 上 decode 是带宽瓶颈，量化少搬数据就变快；GPU 上 decode 是开销瓶颈，带宽只占一成左右，省带宽没意义，MMVQ 解包反而多了一点计算。
7. 5 个 token 共享一次计算图的 kernel 启动和同步，固定开销被分摊；而且权重读一次用 5 次。
8. `whisper-cli` 跑的是第一次推理（冷启动：加载 kernel、cuBLAS 初始化、显存分配），`whisper-bench` 计时前预热过。评测脚本要先预热，或者一个进程处理多个文件、丢掉第一个的时间。
9. f16 × 1500 帧：cuBLAS（MMVF 只接 batch = 1，MMF 只接 batch ≤ 16）。q4_0 × 1：MMVQ。
10. 141 MB ÷ 约 52 GB/s ≈ 2.7 ms，比 encode 本身（3.7 ms）都长。所以权重必须常驻显存。

</details>

---

## 14. 和下一周的关系

| 这份指南留下的问题 | 第几周解决 |
|---|---|
| GPU 上的 WER 和 CPU 一样吗？（浮点顺序不同，结果可能有细微差别） | 第 2 周：同一套评测分别跑 4600h-cpu / 9700x-cpu / 9700x-cuda |
| 冷启动对真实使用的 RTF 影响多大？ | 第 2 周：评测脚本分开统计首条和其余 |
| 峰值显存多少？ | 第 2 周：NVML 采样 |
| decode 的开销到底花在哪？ | 第 3 周：Nsight Systems（练习 9.4） |
| encoder 是不是真的走了 Tensor Core？利用率多少？ | 第 3 周：Nsight Compute 看 cuBLAS kernel |
| MMQ 和 cuBLAS 哪个更适合 encoder？ | 第 3 周：练习 9.5 |
| 打开 CUDA Graphs 能省多少？ | 第 4–5 周：可以作为一个真实改动或者开源贡献的方向 |
