# 岗位知识地图：按 JD 拆解要学什么

> 目标：把 JD 的每一条拆成**具体知识点**，标出它在这个项目的哪一周、哪段代码里练，以及面试时怎么用项目里的数字回答。写后面几周的代码时，遇到不懂的概念先查这里，再去 [chunk1.md](chunk1.md)（量化、roofline、CPU）和 [cuda_guide.md](cuda_guide.md)（GPU、CUDA）看细节。
>
> 标注约定：✅ = 已经从源码或实测数据里确认；🔍 = 合理的假设或待验证；📖 = 概念性知识，来自论文或公开资料，面试要会讲，但本项目不一定动手做。

---

## 目录

0. [JD 一张表](#0-jd-一张表)
1. [C/C++ 和 Python：读改框架代码](#1-cc-和-python读改框架代码)
2. [Transformer 和推理流程](#2-transformer-和推理流程)
3. [KV Cache](#3-kv-cache)
4. [量化](#4-量化)
5. [体系结构和异构计算](#5-体系结构和异构计算)
6. [性能分析方法论：定位瓶颈、验证优化](#6-性能分析方法论定位瓶颈验证优化)
7. [算子融合和内存优化](#7-算子融合和内存优化)
8. [推理框架：llama.cpp、vLLM、SGLang](#8-推理框架llamacppvllmsglang)
9. [开源贡献](#9-开源贡献)
10. [面试高频题](#10-面试高频题)
11. [简历和面试故事怎么讲](#11-简历和面试故事怎么讲)

---

## 0. JD 一张表

| JD 要求 | 拆成知识点 | 在项目哪里练 | 本文章节 |
|---|---|---|---|
| 熟悉 C/C++、Python，能读改框架代码，调试能力 | CMake、VS 调试器、ggml 数据结构和计算图、Python 评测脚本 | 第 1 周编译；第 3 周 debugger 跟 `whisper_full`；第 4–5 周改代码 | 1 |
| 理解 Transformer 与大模型推理流程 | attention、encoder/decoder、prefill vs decode、采样、beam search | 第 1 周的计时拆分；第 3 周读计算图 | 2 |
| 对模型量化有兴趣 | 量化格式、PTQ、GPTQ / AWQ / SmoothQuant、精度评估 | 第 1 周生成 5 种量化；第 2 周 WER | 4 |
| 对体系结构、异构计算有兴趣 | SIMD、缓存、带宽、roofline、CPU vs GPU vs NPU | 第 1 周 4600H；第 2 周 5080 | 5 |
| 结合代码、性能数据分析问题，定位瓶颈并验证优化效果 | 测量方法、profiler、Amdahl 定律、A/B 对比 | 第 2 周评测框架；第 3 周 profiling | 6 |
| 加分：llama.cpp / vLLM / SGLang，真实设备适配 | ggml 和 llama.cpp 共用；PagedAttention；RadixAttention | 整个项目就是 whisper.cpp 在两台真实设备上的适配 | 8 |
| 加分：推理优化、低比特量化、KV Cache 优化、算子融合、内存优化 | KV cache 大小公式、KV 量化、fusion、图内存分配 | 第 3 周分析；第 4–5 周改动 | 3、4、7 |
| 加分：CUDA / OpenCL / Vulkan / Metal / NPU，算子开发、性能分析、开源贡献 | CUDA 编程模型、Nsight、ggml 后端结构、提 PR | 第 2–3 周 CUDA；第 4–5 周 PR | 5、9，[cuda_guide.md](cuda_guide.md) |

**一句话定位**：这个 JD 要的是"**能在真实硬件上把模型跑快，并且能用数据说清楚为什么快**"的人。这个项目的每一周都在练这件事，面试时最有说服力的不是"我知道 X"，而是"我测了 X，数字是多少，原因是什么，我是怎么验证的"。

---

## 1. C/C++ 和 Python：读改框架代码

### 1.1 C/C++ 要会到什么程度

不需要会写模板元编程，但下面这些在 ggml 里天天见，读代码时要能一眼看懂：

| 知识点 | ggml 里的例子 | 为什么重要 |
|---|---|---|
| 指针、结构体、内存布局 | `struct ggml_tensor`（见 1.2） | 整个框架就是在传 tensor 指针 |
| 步长（stride）和视图（view） | `nb[]` 数组、`view_src` | 转置、reshape 不拷数据，只改步长 |
| 对齐和 padding | `GGML_PAD(n, 256)`，KV cache 长度 448 被补到 512 ✅ | SIMD 和 GPU 访存都要求对齐 |
| 函数指针 / 虚表式的 C 接口 | `ggml_backend_i`（[`ggml-backend-impl.h`](../third_party/whisper.cpp/ggml/src/ggml-backend-impl.h)） | 每个后端（CPU、CUDA、Vulkan……）实现同一组函数指针，这就是"异构"的软件结构 |
| SIMD intrinsics | `_mm256_fmadd_ps`（`ggml-cpu/` 下） | CPU kernel 的核心 |
| OpenMP / 线程池 | `ggml-cpu` 的多线程 | 线程扩展性分析（chunk1 9.3 节） |
| C++ 模板 | CUDA kernel 按量化类型、tile 大小实例化（`mmq.cuh`） | 一份代码生成几十个 kernel |
| 宏 | `GGML_ASSERT`、`GGML_TENSOR_LOCALS` | 读代码时要能展开 |

### 1.2 ggml 的核心数据结构（✅ 来自 [`ggml/include/ggml.h`](../third_party/whisper.cpp/ggml/include/ggml.h) 第 686 行）

```c
struct ggml_tensor {
    enum ggml_type type;                 // F32 / F16 / Q4_0 / ...
    struct ggml_backend_buffer * buffer; // 数据在哪个后端的内存里（主机内存 / 显存）
    int64_t ne[4];                       // 每一维的元素个数
    size_t  nb[4];                       // 每一维的步长（字节）
    enum ggml_op op;                     // 这个 tensor 是哪个算子算出来的（MUL_MAT、ADD……）
    int32_t op_params[...];              // 算子参数（比如 softmax 的 scale）
    struct ggml_tensor * src[...];       // 输入 tensor → 这就是计算图的边
    struct ggml_tensor * view_src;       // 如果是视图，指向原 tensor
    void * data;                         // 数据指针
    char name[...];
    void * extra;                        // 后端私有数据（比如 CUDA 用）
};
```

要记住的三点：

1. **`ne[0]` 是最内层（连续）的那一维**，和 NumPy 的习惯相反。`mul_mat(a, b)` 的维度规则和 NumPy 的 `@` 也不一样，第 3 周跟代码时画一下。
2. **建图和计算是分开的**：`ggml_mul_mat(ctx, a, b)` 只是创建一个 `op = MUL_MAT` 的节点，并不计算。整张图建好后交给 backend scheduler 执行（[cuda_guide.md 5.2 节](cuda_guide.md#52-计算图和-backend-scheduler)）。
3. **量化格式的 `nb[0]` 是一个"块"的字节数**，不是一个元素的（注释里写着 `nb[1] = nb[0] * (ne[0] / ggml_blck_size(type))`）。

### 1.3 调试能力

| 技巧 | 用在哪 |
|---|---|
| VS 里打开 `build/whisper.cpp.sln`，把 `whisper-cli` 设为启动项目，在"调试属性"里填命令行参数 | 第 3 周从 `whisper_full` 跟进 encoder |
| 条件断点（比如 `tensor->op == GGML_OP_MUL_MAT`） | 只停在矩阵乘节点 |
| 监视窗口看 `tensor->ne`、`tensor->name` | 搞清楚每个节点的形状 |
| `RelWithDebInfo` 而不是 `Debug` | 有符号又有优化，profiling 数字才有意义。`Debug` 版慢几十倍，测出来的瓶颈可能是假的 |
| 环境变量开关（比如 `GGML_CUDA_DISABLE_FUSION=1` ✅，`ggml-cuda.cu:171`） | 不重新编译就能做对比实验 |
| 二分法：`git bisect` | 定位"哪个提交让性能变差了"，提 issue 时很有用 |

### 1.4 Python 要会的

| 用途 | 库 | 项目里在哪 |
|---|---|---|
| 调用可执行文件、解析输出 | `subprocess`、`re` | `scripts/common.py` 的 `run()`，`week1_bench.py` 的正则 |
| WER / CER | `jiwer` | 第 2 周。上游已经有一套 LibriSpeech 评测可以参考：[`tests/librispeech/`](../third_party/whisper.cpp/tests/librispeech/)，里面带了 Whisper 论文用的 `EnglishTextNormalizer` ✅ |
| 峰值内存 | `psutil`（主机内存）、`pynvml`（显存） | 第 2 周 |
| 画图 | `matplotlib` | 第 2 周的"精度 vs 速度 vs 内存"图 |
| 统计 | `statistics.median`、`numpy` | 多次重复取中位数 |

**一个经验**：WER 一定要先做文本归一化（大小写、标点、数字写法）再算，不然"Mr." 和 "mister" 会被算成错误，量化之间的差异会被噪声淹没。这就是上游脚本要带 normalizer 的原因。

---

## 2. Transformer 和推理流程

### 2.1 Attention 一页纸

```
输入 X（T 个向量，每个 d 维）
Q = X·W_q,  K = X·W_k,  V = X·W_v           ← 三个矩阵乘
Attention(Q, K, V) = softmax(Q·Kᵀ / √d_head) · V
多头：把 d 拆成 h 个头，每个头 d_head = d / h，各自做 attention 再拼起来，最后乘 W_o
```

一个 Transformer 层 = attention + MLP（两个矩阵乘夹一个激活函数），每部分前面有 LayerNorm、外面有残差连接。**几乎所有的计算量都在矩阵乘里**，这是为什么性能分析总是先看 `mul_mat`。

### 2.2 Encoder-decoder（Whisper）和 decoder-only（LLM）

| | Whisper | LLM（Llama 等） |
|---|---|---|
| 结构 | encoder + decoder | 只有 decoder |
| 输入 | 30 秒音频 → 1500 帧（chunk1 1.1 节） | 文本 prompt |
| "prefill" 阶段 | encoder 一次处理 1500 帧 + decoder 处理初始 prompt | decoder 一次处理整个 prompt |
| "decode" 阶段 | decoder 一次一个 token | 一样 |
| 额外的 attention | **交叉注意力**：decoder 看 encoder 的输出 | 没有 |
| 归一化 / 激活 | LayerNorm、GELU | 多数是 RMSNorm、SwiGLU |

**面试时的说法**：Whisper 的 encoder 就相当于 LLM 的 prefill（大 batch、计算瓶颈），decoder 就是 LLM 的 decode（单 token、带宽或开销瓶颈）。所以在 whisper 上得到的结论可以直接迁移到 LLM 推理。

### 2.3 Prefill vs Decode（面试几乎必问）

| | Prefill | Decode |
|---|---|---|
| 一次处理 | 整段输入（几百到几千个 token） | 1 个 token（或 beam 数个） |
| 矩阵乘形状 | 矩阵 × 矩阵（GEMM） | 矩阵 × 向量（GEMV） |
| 算术强度 | 约等于 token 数，很高 | 约 1 FLOP/byte |
| 瓶颈 | 算力 | 带宽（CPU 上）、开销（GPU 上） |
| 本项目的证据 | 4600H encoder q4_k 快 2.3×（专用 kernel，算得快）；5080 encoder 达到约 43 TFLOPS | 4600H decode 量化快 1.4–1.7×（读得少）；5080 decode 只用了 9–14% 带宽 |
| 用户感受到的指标 | TTFT（Time To First Token，首 token 延迟） | TPOT / ITL（每个 token 的间隔），决定"打字速度" |

数据来源：[results/week1.md](../results/week1.md)、[cuda_guide.md 第 6 节](cuda_guide.md#6-第一轮-gpu-数据和分析)。

### 2.4 采样和 beam search

- **Greedy**：每步选概率最大的 token
- **Temperature / top-k / top-p**：在概率分布上采样，增加多样性
- **Beam search**：同时保留 k 条候选序列（whisper-cli 默认 5 beams ✅，日志 `5 beams + best of 5`），每步扩展后保留总分最高的 k 条。代价：decoder 要跑 k 份，KV cache 也要 k 份（第 3 节）
- whisper.cpp 的采样在 **CPU** 上做，每个 token 要把 logits 从 GPU 拷回来 ✅（[cuda_guide.md 5.1 节](cuda_guide.md#51-整体分工-来自源码)）

### 2.5 LLM 推理里的其他概念 📖

| 概念 | 一句话 |
|---|---|
| Continuous batching | 服务端不等一整批请求都完成，每生成一步就可以让新请求加入、让完成的请求退出。vLLM、SGLang 的核心调度方式 |
| Speculative decoding | 小模型先猜几个 token，大模型一次并行验证。把几次 decode（带宽瓶颈）换成一次"小 prefill"（计算瓶颈） |
| MoE（Mixture of Experts） | 每个 token 只激活部分专家的 MLP。ggml 里的 `mul_mat_id` 就是给 MoE 用的 |
| Tensor / pipeline parallel | 模型太大时拆到多张卡上 |

---

## 3. KV Cache

### 3.1 为什么要 KV cache

decode 第 t 步，新 token 要和**之前所有 token** 做 attention，需要它们的 K 和 V。如果每步都重算，就要把前 t 个 token 再跑一遍，总代价 O(T²)。**KV cache 把每个 token 的 K、V 存下来**，每步只算新 token 的 K、V 再追加进去，代价降到 O(T)。代价是显存：用内存换计算。

### 3.2 大小怎么算（面试必问）

```
KV cache 字节数 = 2（K 和 V）× 层数 × 序列长度 × d_kv × 每个元素的字节数 × batch（或 beam 数）
```

`d_kv` 在普通多头注意力（MHA）里就是 d；在 GQA 里是 `KV 头数 × d_head`（见 3.3）。

**用 whisper.cpp 的日志验证** ✅（`whisper-cli` 的 `whisper_init_state` 输出，f16，`src/whisper.cpp:3473–3498`）：

| | 公式 | 算出来 | 日志 |
|---|---|--:|--:|
| base 自注意力 | 2 × 6 层 × 512 × 512 × 2 B | 6.29 MB | `kv self size = 6.29 MB` |
| base 交叉注意力 | 2 × 6 层 × 1536 × 512 × 2 B | 18.87 MB | `kv cross size = 18.87 MB` |
| small 自注意力 | 2 × 12 层 × 512 × 768 × 2 B | 18.87 MB | `kv self size = 18.87 MB` |
| small 交叉注意力 | 2 × 12 层 × 1536 × 768 × 2 B | 56.62 MB | `kv cross size = 56.62 MB` |

两个细节：

- **长度被补齐了**：`n_text_ctx = 448` 补到 512，`n_audio_ctx = 1500` 补到 1536，都是 `GGML_PAD(..., 256)` ✅。补齐是为了让 kernel 的分块整齐
- **自注意力 cache 按 1 个 decoder 分配**，源码注释写着用到多个 decoder（beam search）时会重新分配 ✅。所以 5 beams 时实际是日志数字的 5 倍 🔍

**LLM 的例子**（面试时常问）📖：

| 模型 | 层数 | d_kv | 每 token（f16） | 4096 token |
|---|--:|--:|--:|--:|
| Llama-2-7B（MHA，32 个头） | 32 | 4096 | 2 × 32 × 4096 × 2 B = 512 KB | 2 GB |
| Llama-3-8B（GQA，8 个 KV 头 × 128） | 32 | 1024 | 128 KB | 512 MB |

**结论**：长上下文、大 batch 的时候，**KV cache 比模型权重还大**，而且 decode 每步都要把整个 KV cache 读一遍，所以它既占显存又占带宽。这就是为什么有那么多 KV cache 优化。

### 3.3 KV cache 优化（JD 加分项）📖

| 方法 | 思路 | 在哪见过 |
|---|---|---|
| **MQA / GQA** | 多个 Q 头共享一组 K、V 头，KV cache 缩小 4–8 倍 | Llama 2 70B、Llama 3、Mistral |
| **MLA** | 把 K、V 压缩到一个低维隐向量再存 | DeepSeek-V2 / V3 |
| **KV cache 量化** | K、V 存成 int8 / int4 | llama.cpp 的 `-ctk` / `-ctv` 参数；ggml 的 Flash Attention 支持 `q4_0-q4_0`、`q8_0-q8_0` 组合 ✅（`system_info` 里的 `FA_QUANTS`） |
| **PagedAttention** | KV cache 按固定大小的块分配，用块表映射，像操作系统的分页，消除碎片 | vLLM（第 8 节） |
| **Prefix caching** | 多个请求共享相同前缀（比如同一个 system prompt）的 KV | SGLang 的 RadixAttention、vLLM 的 automatic prefix caching |
| **滑动窗口 / 驱逐** | 只保留最近的 token，或者保留注意力分数高的 token | Mistral 的 sliding window、StreamingLLM（attention sink）、H2O |

**和本项目的关系**：Whisper 的 decoder 最多 448 个 token，KV cache 很小（small 自注意力 18.87 MB），不是瓶颈。但**交叉注意力 cache（56.62 MB）每生成一个 token 都要整个读一遍**，在带宽瓶颈的 CPU 上是 decode 开销的一部分 🔍（第 3 周 profiler 可以量这部分占比）。

---

## 4. 量化

（格式的字节布局见 [chunk1.md 第 5 节](chunk1.md#5-量化格式每个字节存的是什么)，这里讲概念和方法。）

### 4.1 基本概念

| 概念 | 解释 |
|---|---|
| 对称 / 非对称 | `x ≈ scale × q`（q4_0、q8_0）vs `x ≈ scale × q + min`（q4_1、q5_1，多存一个偏移） |
| 粒度 | per-tensor（整个矩阵一个 scale）→ per-channel（每行一个）→ **per-group / per-block**（每 32 个权重一个，ggml 的做法） |
| 二级量化 | k-quants 的 scale 本身也被量化（超级块里存 6 bit 的子 scale），所以 q4_k 每权重约 4.5 bit |
| Weight-only | 只量化权重，计算时还原成浮点（或量化激活再做整数点积）。端侧和 decode 主流做法，因为 decode 是带宽瓶颈 |
| W8A8 | 权重和激活都是 int8，可以直接用 int8 Tensor Core，对 prefill（计算瓶颈）更有利 |
| PTQ / QAT | 训练后量化（不重新训练，ggml 就是这种）/ 量化感知训练 |
| 校准数据 | 跑一些真实输入，统计激活分布，用来决定怎么量化 |

### 4.2 为什么量化难：异常值（outlier）📖

大模型的激活里，少数几个通道的数值比其他大几十倍。用同一个 scale 量化，小值就全被压成 0。各种方法都在解决这个问题：

| 方法 | 思路 |
|---|---|
| **LLM.int8()**（Dettmers 等，2022） | 异常值通道用 f16 算，其他用 int8，混合精度 |
| **SmoothQuant**（Xiao 等，2022） | 激活除以一个按通道的系数、权重乘上同一个系数，把量化难度从激活"转移"到权重，实现 W8A8 |
| **GPTQ**（Frantar 等，2022） | 逐列量化权重，用二阶信息（Hessian）把每列的量化误差补偿到还没量化的列上 |
| **AWQ**（Lin 等，2023，MIT 韩松组，就是 6.5940 的课题组） | 按激活大小找出约 1% 的重要通道，放大它们的权重再量化，降低重要通道的相对误差。weight-only |
| **llama.cpp 的 imatrix** | 用校准数据算每个权重的"重要性矩阵"，量化时最小化加权误差 |
| **FP8 / FP4（MXFP4、NVFP4）** | 用低比特浮点而不是整数，硬件直接支持（Hopper 支持 FP8，Blackwell 支持 FP4）。ggml 已经有 `MXFP4`、`NVFP4` 类型 ✅（`mmq.cu` 的支持列表），你的 5080 的 `system_info` 里有 `BLACKWELL_NATIVE_FP4 = 1` ✅ |

### 4.3 量化的收益取决于硬件和阶段（本项目的核心发现）

| | 4600H CPU | 5080 GPU |
|---|---|---|
| encoder（计算瓶颈） | q4_0 / q4_k 快 2×，因为有 REPACK 专用 AVX2 kernel；q5 几乎不快 | 量化不快，甚至略慢（f16 走 cuBLAS Tensor Core 更快）🔍 |
| decoder（CPU 带宽 / GPU 开销瓶颈） | 量化快 1.4–1.7× | 量化不快，甚至略慢 |
| 模型文件 / 显存 | 小 3 倍以上 | 小 3 倍以上 |

**面试金句**：量化不是"越低比特越快"，而是"**省下的是带宽，花掉的是解包计算**"。它在带宽瓶颈的地方有用，在计算瓶颈或开销瓶颈的地方可能没用甚至变慢，而且有没有为这个格式写过专用 kernel 影响巨大。这个结论我在两种硬件上都测过。

### 4.4 精度怎么评估（第 2 周）

- ASR：**WER**（词错误率）= (替换 + 删除 + 插入) ÷ 参考词数；中文、日文用 **CER**（字错误率），因为没有空格分词
- LLM：perplexity（llama.cpp 有 `llama-perplexity`）、下游任务准确率
- 要点：样本数要够（第 1 周的 jfk.wav 只有 11 秒，12 个模型全对，区分不出来）；报告置信区间或至少报样本数；所有模型用同一套归一化

---

## 5. 体系结构和异构计算

### 5.1 CPU 要懂的

| 知识点 | 本项目的例子 |
|---|---|
| SIMD 宽度 | 4600H 只有 AVX2（256 位），9700X 有 AVX-512（512 位）✅ |
| 缓存层级 | 4600H：L2 3 MB、L3 8 MB；模型放不进缓存，所以 decode 每步都要从内存读 |
| 内存带宽 | 4600H 双通道 DDR4-3200 理论 51.2 GB/s；带宽决定 decode 速度的上限 |
| 多线程扩展 | 4600H 上 small q4_0 的 decode 12 线程比 8 线程慢（chunk1 9.3 节）：超线程共享执行单元、带宽已饱和、同步开销 |
| 功耗和频率 | 电源计划"平衡"vs"高性能"会影响数字 |

### 5.2 GPU 要懂的

见 [cuda_guide.md](cuda_guide.md)：SM、warp、合并访存、共享内存、占用率、kernel 启动开销、PCIe。

### 5.3 NPU 和端侧芯片 📖

| | 特点 | 对软件的影响 |
|---|---|---|
| **NPU**（华为昇腾、高通 Hexagon、Intel NPU、苹果 ANE） | 大量 int8 / int4 乘加单元（常见是脉动阵列），能效比高，灵活性低 | 往往要求**静态图、固定形状**，算子支持有限，不支持的算子要回退到 CPU；量化几乎是必须的 |
| **手机 / 笔记本 SoC** | CPU、GPU、NPU **共享同一块内存**（统一内存） | 没有 PCIe 拷贝开销，但大家抢同一份带宽 |
| **独立 GPU** | 显存带宽高，但和 CPU 之间隔着 PCIe | 权重必须常驻显存（[cuda_guide.md 2.4 节](cuda_guide.md#24-内存层级从快到慢)） |

### 5.4 ggml 支持的后端（✅ `ggml/src/` 目录）

| 后端目录 | 硬件 / API |
|---|---|
| `ggml-cpu` | x86（AVX2 / AVX-512）、ARM（NEON / SVE）、RISC-V |
| `ggml-cuda` | NVIDIA GPU |
| `ggml-hip` | AMD GPU（复用 CUDA 代码，通过 HIP 编译） |
| `ggml-musa` | 摩尔线程 GPU（同样复用 CUDA 代码） |
| `ggml-metal` | 苹果 GPU |
| `ggml-vulkan` | 跨厂商 GPU（包括手机 GPU） |
| `ggml-opencl` | 主要面向高通 Adreno 手机 GPU 🔍 |
| `ggml-sycl` | Intel GPU |
| `ggml-cann` | 华为昇腾 NPU |
| `ggml-hexagon` | 高通 Hexagon NPU / DSP |
| `ggml-openvino` | Intel CPU / GPU / NPU（通过 OpenVINO） |
| `ggml-webgpu` | 浏览器 |
| `ggml-rpc` | 把计算分到网络上的其他机器 |
| `ggml-blas` | 调用 BLAS 库（OpenBLAS、Accelerate 等） |

**要理解的设计**：所有后端都实现同一组接口（`ggml_backend_i`），上层的 whisper.cpp / llama.cpp 不关心底层是什么硬件；backend scheduler 负责把计算图切开、分配到不同后端、在边界上插入拷贝。**这就是"异构计算"在软件层的样子。** 某个后端不支持某个算子时，scheduler 会把那个节点放回 CPU，代价是两次跨设备拷贝，这也是端侧适配时最常见的性能问题之一。

---

## 6. 性能分析方法论：定位瓶颈、验证优化

### 6.1 流程

```
① 测：端到端时间，拆成阶段（load / mel / encode / decode）
② 估：算理论上限（roofline：算力上限、带宽上限）
③ 比：实测离上限多远？离算力近 → 计算瓶颈；离带宽近 → 带宽瓶颈；两个都远 → 开销瓶颈
④ 看：profiler 看热点算子、kernel 时间线
⑤ 改：一次只改一个变量
⑥ 验：同样的测量方法再测一遍，重复多次，确认差异大于噪声
```

本项目第 1 周做了 ①②③，第 3 周做 ④，第 4–5 周做 ⑤⑥。[cuda_guide.md 6.3 节](cuda_guide.md#63-decoder只快-36-倍量化反而变慢-数据--解释)的"decode 只用了 9–14% 带宽"就是第 ③ 步的例子。

### 6.2 Amdahl 定律

```
总加速比 = 1 / ((1 − p) + p / s)        p = 被优化部分的占比，s = 这部分的加速比
```

例（✅ 用真实数据算）：4600H 上 base f16 跑 jfk.wav，总时间 1697 ms，其中 encode 944 ms，占比 p = 0.56（[week1_jfk.csv](../results/week1_jfk.csv)）。就算 encode 换成 5080 上快 242 倍的版本（[cuda_guide.md 6.2 节](cuda_guide.md#62-encoder快了-160440-倍-)），总加速比也只有 1 / (0.44 + 0.56 / 242) ≈ **2.2 倍**，因为剩下的加载、decode、beam search 一点没变。**优化前先看占比，别优化占 5% 的部分。**

### 6.3 测量的基本功

| 要点 | 做法 |
|---|---|
| 预热 | 第一次不计时（GPU 冷启动慢 17 倍 ✅） |
| 重复 | 至少 5 次，报中位数，最好也报最小值和波动范围 |
| 固定环境 | 电源计划、插电、关后台程序；GPU 记录时钟和温度 |
| 一次只改一个变量 | 量化等级、线程数、后端分别扫 |
| 记录版本 | whisper.cpp 的 commit（`482279b8`）、编译选项、驱动版本 |
| 区分延迟和吞吐 | 单条请求多快 vs 每秒能处理多少 |

### 6.4 Profiler

| 工具 | 平台 | 看什么 |
|---|---|---|
| Visual Studio Performance Profiler | CPU | 函数热点、调用树 |
| AMD uProf | AMD CPU | 热点 + 硬件计数器（缓存未命中、IPC），可以确认是不是带宽瓶颈 |
| Nsight Systems | NVIDIA | 时间线：kernel、拷贝、CPU 端 API 调用 |
| Nsight Compute | NVIDIA | 单个 kernel 的占用率、带宽、roofline |
| ggml 的 `test-backend-ops`（`perf` 模式） | ggml | 单个算子在各后端的性能和正确性。whisper.cpp 里的 ggml 没带 `tests/` 目录，要在 llama.cpp 或 ggml 仓库里编译 🔍 |

---

## 7. 算子融合和内存优化

### 7.1 算子融合（Operator fusion）

**是什么**：把几个连续的算子合成一个 kernel。比如 `y = gelu(x·W + b)` 不融合要 3 个 kernel，中间结果写回显存再读出来两次；融合后 1 个 kernel，中间结果留在寄存器里。

**省了什么**：
1. **显存读写**：中间 tensor 不落地（带宽瓶颈时有用）
2. **kernel 启动次数**：开销瓶颈时有用，正好对应 5080 上 decode 的情况 🔍

**常见例子** 📖：
- GEMM + bias + 激活（cuBLASLt 的 epilogue、CUTLASS）
- LayerNorm / RMSNorm + 逐元素乘
- **Flash Attention 本质上就是融合**：把 `QKᵀ → scale → mask → softmax → ·V` 融成一个 kernel，不写出 T × T 的矩阵

**ggml 里已有的融合**（✅ `ggml-cuda.cu`）：
- `ggml_cuda_should_fuse_rms_norm_mul_rope`：RMSNorm + 乘 + RoPE
- `ggml_cuda_should_fuse_mul_mat` / `..._mul_mat_q`：MLP 的 up、gate 两个矩阵乘 + GLU 激活
- `ggml_cuda_should_fuse_mul_mat_vec_f` / `..._vec_q`：MMVF / MMVQ 的融合版本
- 可以用环境变量 `GGML_CUDA_DISABLE_FUSION=1` 关掉做对比

**和 whisper 的关系** 🔍：上面这些融合大多是为 LLM 写的（RMSNorm、RoPE、SwiGLU），**Whisper 用的是 LayerNorm、GELU、绝对位置编码，可能一个都匹配不上**。第 3 周用 Nsight Systems 确认一下；如果真是这样，"给 whisper 的 decoder 加融合 / 打开 CUDA Graphs"就是第 4–5 周一个有数据支撑的优化方向。

### 7.2 内存优化

| 方法 | 解释 | ggml / whisper.cpp 里 |
|---|---|---|
| **图内存规划** | 分析计算图里每个中间 tensor 的生命周期，生命周期不重叠的 tensor 复用同一块内存 | `ggml-alloc.c` 的 graph allocator ✅；日志里的 `compute buffer (decode) = 97.29 MB` 就是规划后的峰值 ✅ |
| **in-place 算子** | 输出直接写在输入上 | ggml 的 `*_inplace` 系列函数 |
| **权重 mmap** | 模型文件直接映射到内存，不拷贝，多个进程共享 | llama.cpp 默认用 mmap 📖 |
| **量化** | 权重变小 3–4 倍 | 第 1 周 |
| **KV cache 管理** | 第 3 节 | |
| **内存池** | 避免反复 `cudaMalloc`（很慢） | ggml-cuda 的 `ggml_cuda_pool` ✅（`ggml-cuda/common.cuh`） |

whisper.cpp 的显存占用（base f16，✅ 日志）：权重 141 MB + KV（6.29 + 18.87 + 3.15 MB）+ 计算缓冲（17.24 + 23.09 + 4.66 + 97.29 MB）≈ 312 MB，再加上 CUDA 上下文本身的几百 MB。第 2 周用 NVML 实测对一下。

---

## 8. 推理框架：llama.cpp、vLLM、SGLang

### 8.1 llama.cpp：和这个项目最近的

**llama.cpp 和 whisper.cpp 用的是同一个 ggml**（同一个作者 Georgi Gerganov，ggml 在两个仓库之间同步）。你在 whisper.cpp 里学的计算图、后端、量化格式、CUDA kernel，**在 llama.cpp 里完全一样**。这是简历上"使用过 llama.cpp 系推理框架"最诚实也最有力的说法。

建议花半天做的事 📖：
1. 编译 llama.cpp（CMake 选项和这里一样，`-DGGML_CUDA=ON`）
2. 下载一个小模型的 GGUF（比如 1–3B 参数的 Q4_K_M）
3. 用 `llama-bench` 测 `pp512`（prefill）和 `tg128`（decode），在 4600H 和 5080 上各跑一遍
4. 用 `-ctk q8_0 -ctv q8_0` 试 KV cache 量化
5. 把结论和 whisper 的结论对照：prefill / decode 的瓶颈是不是一样？

| llama.cpp 概念 | 对应 whisper.cpp |
|---|---|
| GGUF 文件格式 | whisper 还用老的 ggml `.bin` 格式 |
| `llama-quantize` | `whisper-quantize` |
| `llama-bench` 的 `pp` / `tg` | `whisper-bench` 的 encode / decode |
| `-ngl`（放多少层到 GPU） | whisper 全放 GPU |
| `llama-server` | `whisper-server` |

### 8.2 vLLM 📖

- 定位：**服务端、高吞吐**，Python + PyTorch + 自定义 CUDA kernel
- 核心：**PagedAttention**（Kwon 等，SOSP 2023）。KV cache 按块分配，用块表记录每个请求的块在哪，就像操作系统的虚拟内存分页。解决了"按最大长度预分配导致大量浪费"和"碎片"的问题，让同一张卡能同时服务更多请求
- 配合 **continuous batching**，吞吐量比朴素实现高很多倍

### 8.3 SGLang 📖

- 定位：同样是服务端框架，强调**结构化生成**和**前缀复用**
- 核心：**RadixAttention**。用基数树（radix tree）管理 KV cache，不同请求的公共前缀（system prompt、few-shot 例子、多轮对话历史）只算一次、只存一份

### 8.4 三者对比

| | llama.cpp | vLLM | SGLang |
|---|---|---|---|
| 场景 | 端侧、单用户、各种硬件 | 服务端、高并发 | 服务端、高并发、复杂调用 |
| 语言 | C/C++ | Python + CUDA | Python + CUDA |
| 量化 | 自有格式（q4_0、k-quants、iq 系列） | GPTQ、AWQ、FP8 等 | 类似 vLLM |
| KV cache | 连续分配，支持量化 | PagedAttention | RadixAttention |
| 硬件 | CPU、各家 GPU、NPU | 以 NVIDIA / AMD GPU 为主 | 以 NVIDIA / AMD GPU 为主 |

**面试时可以说**：我的项目在端侧框架这一侧，重点是单请求延迟和多种硬件；服务端框架关注吞吐和显存利用率，KV cache 管理是核心。两边的共同点是 decode 都受带宽 / 开销限制，所以都在做量化、融合、更好的内存管理。

---

## 9. 开源贡献

### 9.1 怎么找题目

1. **你自己踩的坑**最好：第 1 周已经有 6 个（chunk1 第 11 节），本周又有 CUDA 安装的几个（[cuda_guide.md 第 11 节](cuda_guide.md#11-踩过的坑)）。比如"Windows + 中文系统下的编码问题""文档里没写某个参数"
2. GitHub 上 whisper.cpp / llama.cpp 的 `good first issue` 标签
3. 第 3 周 profiling 发现的性能问题，比如 7.1 节说的"whisper decode 没有用上融合 / CUDA Graphs"🔍。带着数据去开 issue 讨论，比直接扔一个 PR 更容易被接受

### 9.2 流程

1. 先读仓库根目录的 [`CONTRIBUTING.md`](../third_party/whisper.cpp/CONTRIBUTING.md) ✅（这个文件存在，提 PR 前一定要读）
2. fork → 新分支 → 小改动（一个 PR 只做一件事）
3. 改动涉及 ggml 算子时，用 ggml 的 `test-backend-ops` 验证各后端结果一致 🔍
4. PR 描述里写清楚：问题是什么、怎么复现、改了什么、**改动前后的数字**
5. 耐心回应 review

**注意**：ggml 的代码在 whisper.cpp、llama.cpp、ggml 三个仓库之间同步，改 ggml 核心部分的 PR 通常应该提到 llama.cpp 或 ggml 仓库，而不是 whisper.cpp 🔍（以 CONTRIBUTING.md 和维护者的说法为准）。

---

## 10. 面试高频题

<details>
<summary>1. Prefill 和 decode 有什么区别？为什么 decode 是 memory-bound？</summary>

Prefill 一次处理整段输入，矩阵乘是 GEMM，每个权重被 T 个 token 复用，算术强度约等于 T，是计算瓶颈。Decode 一次一个 token，矩阵乘是 GEMV，每个权重读进来只用一次（2 FLOP / 2 字节，约 1 FLOP/byte），远低于硬件的 ridge point，是带宽瓶颈。我在 4600H 上测到量化让 decode 快了 1.4–1.7 倍，正好印证了"读得少就快"。补充：在 GPU 上，小模型的 decode 甚至连带宽都吃不满，是 kernel 启动和同步的开销瓶颈。我在 5080 上测到 whisper base 的 decode 只用了约 9% 的带宽。
</details>

<details>
<summary>2. KV cache 的作用？内存占用怎么算？</summary>

存下历史 token 的 K、V，避免每步重算，把 decode 的总代价从 O(T²) 降到 O(T)。大小 = 2 × 层数 × 序列长度 × d_kv × 字节数 × batch。whisper small 的自注意力 cache 是 2 × 12 × 512 × 768 × 2 B = 18.87 MB，和 whisper.cpp 日志一致。Llama-2-7B 在 4096 长度下是 2 GB，所以长上下文时 KV cache 会比权重还大。
</details>

<details>
<summary>3. 什么是算术强度和 roofline？怎么用它判断瓶颈？</summary>

算术强度 = FLOP ÷ 访存字节数。Roofline：可达性能 = min(峰值算力, 算术强度 × 带宽)，两条线的交点叫 ridge point。强度低于 ridge point 是带宽瓶颈，高于是计算瓶颈。4600H 的 ridge point 约 19 FLOP/byte，5080（FP32）约 59。实测性能离两条线都很远时，说明是开销瓶颈，roofline 解释不了，要看 profiler 的时间线。
</details>

<details>
<summary>4. 量化为什么能加速？什么时候不能？</summary>

能加速是因为读的字节少了（带宽瓶颈时有用），或者能用整数 / 低精度硬件单元（比如 int8 Tensor Core）。不能加速的情况：计算瓶颈时解包反而增加了计算；开销瓶颈时带宽本来就不是问题；没有为这个格式写专用 kernel。我的数据：4600H 上 q4 的 encoder 快 2 倍，是因为有 REPACK 专用 kernel，q5 就没快；5080 上量化对 encoder 和 decoder 都没有帮助。
</details>

<details>
<summary>5. GPTQ、AWQ、SmoothQuant 分别解决什么问题？</summary>

都在处理异常值问题。GPTQ：逐列量化权重，用 Hessian 把误差补偿到后面的列，weight-only。AWQ：根据激活大小找出重要通道，缩放保护后再量化，weight-only，不需要反向传播。SmoothQuant：把激活的异常值通过按通道缩放转移到权重上，让激活也能 int8，实现 W8A8。
</details>

<details>
<summary>6. 为什么线程数加到一定程度就不提速了？</summary>

几个原因：带宽已经饱和（加线程只是更多核在等内存）；超线程共享同一个物理核的执行单元；线程同步和负载不均衡的开销；Amdahl 定律里的串行部分。我在 4600H 上测到 small q4_0 的 decode 在 12 线程时比 8 线程还慢（9.77 → 16.77 ms/token）。第 3 周用 profiler 看等待时间来区分是哪个原因。
</details>

<details>
<summary>7. 什么是算子融合？为什么有用？</summary>

把几个连续算子合成一个 kernel，中间结果留在寄存器 / 共享内存，不写回显存。省显存带宽，也省 kernel 启动次数。Flash Attention 就是 attention 的融合。ggml 的 CUDA 后端已经有 RMSNorm + mul + RoPE、MLP 的 up / gate + GLU 等融合，可以用 `GGML_CUDA_DISABLE_FUSION` 关掉做对比。
</details>

<details>
<summary>8. PagedAttention 解决了什么问题？</summary>

传统做法按最大长度给每个请求预分配连续的 KV cache，大部分空间用不到，还有碎片。PagedAttention 把 KV cache 切成固定大小的块，按需分配，用块表映射逻辑位置到物理块，像操作系统的分页。显存利用率高了，同一张卡能同时跑的请求就多了，吞吐量上去了。
</details>

<details>
<summary>9. CPU 和 GPU 做推理各自的优劣？什么时候用 NPU？</summary>

GPU 算力和带宽高一个数量级，适合大 batch 和 prefill；但有 PCIe 拷贝、kernel 启动开销，小 batch 时利用率低。CPU 灵活、延迟低、没有拷贝开销，适合小模型、单请求。NPU 能效比最高，适合端侧长时间运行，但要求静态图、算子支持有限、基本必须量化。我的数据：5080 的 encoder 比 4600H 快 240–440 倍，decoder 只快 3–6 倍。
</details>

<details>
<summary>10. 你怎么验证一个优化真的有效？</summary>

固定环境（电源、时钟、后台），预热，重复多次取中位数，一次只改一个变量；先估理论收益（Amdahl），再看实测是否吻合；用 profiler 确认变快的原因是不是你以为的那个（比如真的少了 kernel 启动，而不是碰巧降频少了）；精度也要一起测，不能只看速度。
</details>

---

## 11. 简历和面试故事怎么讲

### 11.1 简历条目的结构

**做了什么（框架 + 硬件）→ 测了什么（指标）→ 发现了什么（带数字）→ 做了什么改动（带数字）**

草稿（现在只有第 1 周和第一轮 GPU 数据，后面几周补数字）：

> **基于 whisper.cpp 的端侧语音识别量化与性能分析**
> - 在 AMD Ryzen 5 4600H（CPU）和 RTX 5080（CUDA）上部署 Whisper base / small，对比 f16 与 5 种 ggml 量化格式（q8_0 – q4_0）的速度、内存和 WER
> - 用 roofline 分析定位瓶颈：encoder 为计算瓶颈，q4_k 借助 AVX2 重排 kernel 提速 2.3×；decoder 在 CPU 上为带宽瓶颈（量化提速 1.7×），在 GPU 上为 kernel 启动开销瓶颈（仅用到约 10% 显存带宽）
> - （第 3 周）用 AMD uProf / Nsight Systems 验证……
> - （第 4–5 周）实现……，端到端延迟从 X 降到 Y / 向 ggml 提交 PR #……

### 11.2 面试时讲项目的顺序（STAR）

1. **背景**（10 秒）：端侧语音识别，想搞清楚量化在不同硬件上为什么有时快有时不快
2. **方法**（30 秒）：两台机器、12 个模型、扫线程数，拆分 encode / decode 计时，用 roofline 估上限
3. **发现**（1 分钟，重点）：挑一个反直觉的结论讲透。推荐"**同样是 q4，CPU 上 encoder 快 2 倍、GPU 上反而略慢**"，因为它同时涉及量化、体系结构、kernel 实现三件事，面试官一定会追问，而你都有数据
4. **验证**（30 秒）：profiler 看到了什么
5. **改动和结果**（30 秒）：第 4–5 周做的事

### 11.3 每周结束时问自己

- 这周的结论能用一句话 + 一个数字说出来吗？
- 如果面试官问"你怎么知道的"，我能指出是哪个数据、哪行代码、哪张 profiler 截图吗？
- 有没有哪个结论其实还是 🔍，需要下周验证？
