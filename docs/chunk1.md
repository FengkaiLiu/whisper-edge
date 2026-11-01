# Chunk 1：第 1 周完整学习笔记

> 目标：把第 1 周做过的**每一步**、**每个文件**、**每个数字**都讲清楚，读完你应该能自己从零重做一遍，并且能向面试官解释"为什么量化会变快"。
>
> 标注约定：✅ = 已经从源码或数据里确认；🔍 = 合理的假设，第 3 周用 profiler 验证。

---

## 目录

0. [一句话总结](#0-一句话总结)
1. [背景知识：Whisper 是怎么把声音变成文字的](#1-背景知识whisper-是怎么把声音变成文字的)
2. [背景知识：whisper.cpp 和 ggml](#2-背景知识whispercpp-和-ggml)
3. [背景知识：你的硬件](#3-背景知识你的硬件)
4. [实操步骤逐条讲解](#4-实操步骤逐条讲解)
5. [量化格式：每个字节存的是什么](#5-量化格式每个字节存的是什么)
6. [怎么读计时输出](#6-怎么读计时输出)
7. [四个脚本逐个讲](#7-四个脚本逐个讲)
8. [核心概念：计算瓶颈 vs 带宽瓶颈（Roofline）](#8-核心概念计算瓶颈-vs-带宽瓶颈roofline)
9. [用数字验证：结果逐条分析](#9-用数字验证结果逐条分析)
10. [测量方法的局限](#10-测量方法的局限)
11. [踩过的坑](#11-踩过的坑)
12. [术语表](#12-术语表)
13. [自测题](#13-自测题)
14. [动手练习](#14-动手练习)
15. [和下一周的关系](#15-和下一周的关系)

---

## 0. 一句话总结

我们把 OpenAI 的 Whisper 语音识别模型用 whisper.cpp 跑在 AMD 4600H 的 CPU 上，用 5 种量化格式压缩模型，在 1–12 个线程下计时，得出结论：**encoder 卡在算力，decoder 卡在内存带宽和固定开销；q4_k 在这台机器上速度最好，因为它既省带宽，又有专门优化过的 AVX2 kernel。**

---

## 1. 背景知识：Whisper 是怎么把声音变成文字的

### 1.1 整体流程

```
音频文件 (jfk.wav)
  │  16000 Hz 采样，单声道。11 秒 = 176000 个采样点
  ▼
① 预处理：log-mel 频谱图
  │  每 10 ms 取一帧（hop = 160 个采样点），每帧用 25 ms 窗口（n_fft = 400）做 FFT，
  │  再映射到 80 个 mel 频带 → 每秒 100 帧
  │  Whisper 永远按 30 秒处理：不够就补零 → 80 × 3000 的"图片"
  ▼
② Encoder（编码器）
  │  两层 1D 卷积（第二层 stride=2，3000 帧 → 1500 帧）
  │  + N 层 Transformer
  │  输出：1500 个向量，每个 d 维（base d=512，small d=768）
  ▼
③ Cross KV：把 encoder 输出投影成 K、V，只算一次，存起来
  ▼
④ Decoder（解码器）：自回归，一次一个 token
  │  输入：<|startoftranscript|><|en|><|transcribe|> + 已生成的 token
  │  每生成一个 token，都要把 decoder 整个跑一遍
  │  每一层：自注意力（看自己写过的） → 交叉注意力（看音频） → MLP
  │  最后乘上词表矩阵 → 51865 个词的概率
  ▼
⑤ 采样 / Beam search：挑下一个 token
  ▼
"And so my fellow Americans, ..."
```

### 1.2 两个模型的超参数（✅ 来自程序加载日志）

| 参数                              | 含义 | base | small |
|---                                |---|--:|--:|
| `n_mels`                          | mel 频带数 | 80 | 80 |
| `n_audio_ctx`                     | encoder 序列长度（帧数） | 1500 | 1500 |
| `n_audio_state` / `n_text_state`  | 隐藏维度 d | 512 | 768 |
| `n_audio_head` / `n_text_head`    | 注意力头数 | 8 | 12 |
| `n_audio_layer`                   | encoder 层数 | 6 | 12 |
| `n_text_layer`                    | decoder 层数 | 6 | 12 |
| `n_text_ctx`                      | decoder 最多 token 数 | 448 | 448 |
| `n_vocab`                         | 词表大小 | 51865 | 51865 |
| 参数量                            | | 约 74M | 约 244M |

### 1.3 要记住的两点（全周结论的基础）

- **Encoder 只跑 1 次，但一次处理 1500 帧。** 每个权重会被 1500 个向量重复使用 → 计算量大。
- **Decoder 跑很多次（每个 token 一次），每次只处理 1 个 token。** 每个权重只用 1 次 → 计算量小，但每次都要把权重从内存里读一遍。

### 1.4 Beam search 是什么

贪心解码每一步只挑概率最高的那个词。Beam search 同时保留 5 条最有希望的候选句子（`beam_size = 5`），每一步都扩展这 5 条，最后选总分最高的。更准，但 decoder 要同时算 5 条序列。whisper-cli 默认就是 `5 beams + best of 5`。

---

## 2. 背景知识：whisper.cpp 和 ggml

### 2.1 为什么有 whisper.cpp

原版 Whisper 用 Python + PyTorch，依赖重、启动慢，不适合手机、嵌入式这类端侧设备。whisper.cpp 用纯 C/C++ 重写了推理，没有任何依赖，可以编译到 x86、ARM、手机、浏览器（WASM）上。

### 2.2 ggml 是什么

ggml 是 whisper.cpp（以及 llama.cpp）底层的**张量计算库**。核心思想：

1. **先建图，再计算。** 代码不是直接算 `y = W·x`，而是先描述"y 等于 W 乘 x"，构成一张**计算图**（graph），然后交给 `ggml_graph_compute` 一次性执行。
2. **后端（backend）。** 同一张图可以交给不同的硬件执行：CPU、CUDA、Metal、Vulkan…… 这台笔记本只有 CPU 后端（日志里的 `no GPU found`）。
3. **CPU 后端用多线程 + SIMD。** 每个算子（比如矩阵乘）按行切给多个线程，每个线程内部用 AVX2 指令一次算多个数。
4. **量化是一等公民。** ggml 原生支持 q4_0、q8_0 这类格式，矩阵乘的时候直接在量化数据上算，不需要先还原成完整的 f16 矩阵。

### 2.3 REPACK 是什么（✅ 已从源码确认）

`system_info` 里的 `REPACK = 1` 表示 CPU 后端开了"权重重排"。加载模型时，ggml 会把某些量化格式的权重**重新排列成每 8 行交错存放**的布局，这样一次 AVX2 运算可以同时处理 8 行，矩阵乘效率更高。

源码 [`ggml/src/ggml-cpu/repack.cpp`](../third_party/whisper.cpp/ggml/src/ggml-cpu/repack.cpp) 里写明了在 x86 + AVX2 上**只有这几种格式**能重排：

| 格式 | 重排后的 kernel | 条件 |
|---|---|---|
| Q4_0 | `q4_0_8x8_q8_0` | AVX2，且行数 % 8 == 0 |
| Q4_K | `q4_K_8x8_q8_K` | AVX2，且行数 % 8 == 0 |
| IQ4_NL | （我们没用） | AVX2 |
| Q8_0 | ❌ 只在 ARM/RISC-V 上有 | — |
| Q5_0 / Q5_1 | ❌ 没有 | — |

**这就是本周最关键的发现之一的源头**：在你的机器上，q4_0/q4_k 有专门的快速 kernel，q8_0 和 q5 都没有。

---

## 3. 背景知识：你的硬件

### 3.1 CPU：AMD Ryzen 5 4600H

| 项目 | 值 | 意义 |
|---|---|---|
| 架构 | Zen 2（7nm，2020 年） | |
| 核心 / 线程 | 6 核 / 12 线程 | 每个物理核有 2 个硬件线程（SMT，超线程） |
| 基频 / 加速 | 3.0 / 4.0 GHz | 多核满载时通常在两者之间 |
| L2 缓存 | 每核 512 KB，共 3 MB | |
| L3 缓存 | 8 MB（两个 CCX 各 4 MB） | base 模型 141 MB，装不进缓存 → 必须从内存读 |
| SIMD | AVX2 + FMA + F16C，无 AVX-512 | |

### 3.2 SIMD、AVX2、FMA 是什么

- **SIMD**（Single Instruction, Multiple Data）：一条指令同时处理多个数。
- **AVX2**：256 位寄存器，一次可以处理 8 个 float32、16 个 int16，或 32 个 int8。
- **AVX-512**：512 位，一次处理 16 个 float32。4600H **没有**，这就是 CMake 日志里 `HAS_AVX512_1 - Failed` 的含义。
- **FMA**（Fused Multiply-Add）：一条指令算 `a*b + c`，算 2 次浮点运算（FLOP）。
- **F16C**：f16 和 f32 互相转换的指令。f16 模型的计算都靠它先把数据转成 f32。

**理论算力粗算**：Zen 2 每个核有 2 个 256 位 FMA 单元 → 每周期 2 × 8 × 2 = 32 FLOP（float32）。
6 核 × 32 × 3.0 GHz ≈ **576 GFLOPS**（基频下的理论峰值）。实际能跑到的远低于这个数，后面会看到。

### 3.3 内存：2 × 8 GB DDR4-3200，双通道

**理论带宽** = 3200 MT/s（每秒传输次数）× 8 字节（64 位总线）× 2 个通道 = **51.2 GB/s**。

实际能用到的一般只有理论值的 60–80%。

### 3.4 怎么查这些信息

```bash
# 在 PowerShell 里
Get-CimInstance Win32_Processor | Format-List Name,NumberOfCores,NumberOfLogicalProcessors,L2CacheSize,L3CacheSize
Get-CimInstance Win32_PhysicalMemory | Format-Table BankLabel,Capacity,Speed
powercfg /getactivescheme
```

- `BankLabel` 显示 `CHANNEL A` 和 `CHANNEL B` → 双通道 ✅
- 电源计划显示"平衡" → 测量时 CPU 可能会降频（见第 10 节）

---

## 4. 实操步骤逐条讲解

### 4.1 建项目、引入 whisper.cpp

```bash
mkdir whisper-edge && cd whisper-edge
git init
git submodule add https://github.com/ggml-org/whisper.cpp.git third_party/whisper.cpp
```

**git submodule 是什么**：在你的仓库里"嵌入"另一个仓库，但只记录**它的地址和某个具体 commit**，不复制它的代码。

**为什么用 submodule，而不是直接复制代码**：
1. **可复现。** 所有数字都是在 `482279b8` 这个版本上测的。上游以后改了代码，你的结果仍然能对上。
2. **干净。** 你的仓库里只有你自己写的东西，面试官一眼能看出哪些是你的工作。
3. **能升级。** 以后想用新版：`cd third_party/whisper.cpp && git pull`，然后在外层仓库提交新的 commit 指针。

生成的 `.gitmodules` 文件记录了这个对应关系。别人 clone 你的仓库后，要运行 `git submodule update --init` 才会真正下载 whisper.cpp。

### 4.2 CMake 配置（configure）

```bash
cmake -S third_party/whisper.cpp -B third_party/whisper.cpp/build -G "Visual Studio 17 2022" -A x64
```

| 参数 | 含义 |
|---|---|
| `-S` | 源码目录（里面有 `CMakeLists.txt`） |
| `-B` | 构建目录，所有生成的文件都放这里，不污染源码 |
| `-G "Visual Studio 17 2022"` | 生成器：生成 VS 2022 的 `.sln` 工程文件 |
| `-A x64` | 64 位 |

**CMake 不编译代码**，它只是"检测环境 + 生成工程文件"。日志逐行解释：

```
-- Found OpenMP: TRUE (found version "2.0")
```
找到了 OpenMP，ggml 用它来管理多线程。MSVC 只支持 OpenMP 2.0（很老的版本），但对 ggml 够用。

```
-- Performing Test HAS_AVX2_1 - Success
-- Performing Test HAS_AVX512_1 - Failed
```
CMake 编译了一小段使用 AVX2 / AVX-512 指令的测试程序并**在你的 CPU 上运行**，看会不会崩溃。AVX2 成功，AVX-512 失败 → 确认了 CPU 的能力。

```
-- Adding CPU backend variant ggml-cpu: /arch:AVX2 GGML_AVX2;GGML_FMA;GGML_F16C
```
最终决定：用 MSVC 的 `/arch:AVX2` 选项编译 CPU 后端，并打开 AVX2、FMA、F16C 三组代码路径。

### 4.3 编译（build）

```bash
cmake --build third_party/whisper.cpp/build --config RelWithDebInfo -j 12
```

| 参数 | 含义 |
|---|---|
| `--config RelWithDebInfo` | 编译配置（VS 的工程里四种配置都有，这里挑一种） |
| `-j 12` | 最多 12 个并行编译任务 |

**四种配置的区别**：

| 配置 | 优化 | 调试信息 | 用途 |
|---|---|---|---|
| Debug | ❌ 关 | ✅ | 调试，速度可能慢 10 倍以上，**测速没有意义** |
| Release | ✅ `/O2` | ❌ | 发布 |
| **RelWithDebInfo** | ✅ `/O2` | ✅ `.pdb` 文件 | **测速 + 第 3 周用 debugger 和 profiler** |
| MinSizeRel | 优化体积 | ❌ | 嵌入式 |

我们选 RelWithDebInfo：速度和 Release 基本一样，同时 profiler 能显示函数名，debugger 能打断点。

**编译产物**（在 `build/bin/RelWithDebInfo/`）：

| 程序 | 用途 |
|---|---|
| `whisper-cli.exe` | 转录音频（主程序） |
| `whisper-bench.exe` | 基准测试 |
| `whisper-quantize.exe` | 量化模型 |
| `whisper-server.exe` | HTTP 服务（本周没用） |
| `whisper-vad-speech-segments.exe` | VAD 语音段检测（第 4–5 周可能用） |
| `test-*.exe` | 单元测试 |
| `main.exe`、`bench.exe` | 旧名字，只打印"已改名"的提示 |

### 4.4 下载模型

```bash
py -3.11 scripts/setup_models.py
```

下载地址：`https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin`

**为什么选 `ggml-base.bin` 而不是 `ggml-base.en.bin`**：`.en` 是只支持英文的版本，在英文上略准。我们第 2 周要测中文和日文，必须用多语言版。

**ggml 模型文件里有什么**：文件头（超参数）+ mel 滤波器 + 词表 + 每个张量（名字、形状、类型、数据）。f16 模型里大部分张量是 f16，少数（bias、layernorm）是 f32。

### 4.5 量化

```bash
whisper-quantize.exe models/ggml-base.bin models/ggml-base-q4_0.bin q4_0
```

`setup_models.py` 对 base 和 small 各生成 5 种格式：q8_0、q5_1、q5_0、q4_k、q4_0。

**哪些张量会被量化**（✅ 来自 [`examples/quantize/quantize.cpp`](../third_party/whisper.cpp/examples/quantize/quantize.cpp) 和 [`examples/common-ggml.cpp`](../third_party/whisper.cpp/examples/common-ggml.cpp)）：

1. 只量化 **2 维张量**（矩阵）。1 维的 bias、layernorm 权重保持原样；**3 维的卷积核也不量化**（保持 f16）。
2. 明确跳过：`encoder.conv1.bias`、`encoder.conv2.bias`、`encoder.positional_embedding`、`decoder.positional_embedding`。
3. 其余所有矩阵：注意力的 Q/K/V/O 投影、MLP 的两层、词表矩阵（token embedding）都会被量化。

这就是为什么文件缩小的比例比"16 bit → 4.5 bit"略差：有一部分张量没被压缩。

### 4.6 第一次转录

```bash
whisper-cli.exe -m models/ggml-base.bin -f samples/jfk.wav -t 6
```

| 参数 | 含义 |
|---|---|
| `-m` | 模型文件 |
| `-f` | 音频文件 |
| `-t 6` | 6 个线程 |
| `-nt` | no timestamps，不打印时间戳（脚本里用） |

输出解释见第 6 节。

### 4.7 跑基准测试

```bash
py -3.11 scripts/week1_bench.py
```

做三件事：
1. **jfk**：12 个模型各转录一次 jfk.wav（6 线程），检查转录对不对、记录计时。
2. **bench**：12 个模型 × 6 种线程数（1, 2, 4, 6, 8, 12）= 72 次 `whisper-bench`。
3. **micro**：`whisper-bench -w 1`（内存复制速度）和 `-w 2`（矩阵乘速度）。

约 20 分钟。跑的时候**不要用电脑干别的**，否则会干扰计时。

### 4.8 生成报告表格

```bash
py -3.11 scripts/make_tables.py
```

---

## 5. 量化格式：每个字节存的是什么

### 5.1 量化的基本思想

一个 f16 权重占 16 bit。如果把 32 个权重分成一组（block），组内的数值范围通常差不多，可以：

1. 找一个公共缩放系数 `d`（存成 f16，占 2 字节）
2. 每个权重只存一个小整数 `q`
3. 用的时候还原：`w ≈ d × q`

### 5.2 各格式的内存布局（✅ 来自 [`ggml/src/ggml-common.h`](../third_party/whisper.cpp/ggml/src/ggml-common.h)）

**q8_0**：每组 32 个权重，34 字节
```c
typedef struct {
    ggml_half d;       // 2 字节：缩放系数
    int8_t  qs[32];    // 32 字节：每个权重一个 int8（-128..127）
} block_q8_0;          // 34 字节 / 32 个 = 8.5 bit/权重
```
还原：`w = d × q`

**q4_0**：每组 32 个权重，18 字节
```c
typedef struct {
    ggml_half d;          // 2 字节
    uint8_t qs[16];       // 16 字节：每字节装 2 个 4-bit 数（低 4 位一个，高 4 位一个）
} block_q4_0;             // 18 字节 / 32 个 = 4.5 bit/权重
```
还原：`w = d × (q − 8)`。4 bit 能表示 0..15，减 8 变成 −8..7，这样就能表示负数。

**q5_0**：每组 32 个权重，22 字节
```c
typedef struct {
    ggml_half d;          // 2 字节
    uint8_t qh[4];        // 4 字节 = 32 bit：每个权重的"第 5 位"单独放这里
    uint8_t qs[16];       // 16 字节：每个权重的低 4 位
} block_q5_0;             // 22 字节 / 32 个 = 5.5 bit/权重
```
还原：先把低 4 位和 `qh` 里对应的第 5 位**拼起来**，得到 0..31，再减 16：`w = d × (q − 16)`。

**这个"拼起来"就是 q5 慢的原因**：每个权重都要多做移位、与、或运算。源码里的还原代码长这样：
```c
const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;          // 取出第 5 位，挪到正确位置
const int32_t x0   = ((x[i].qs[j] & 0x0F) | xh_0) - 16;      // 和低 4 位拼起来
```

**q5_1**：在 q5_0 的基础上多存一个最小值 `m`（f16），24 字节，6 bit/权重。还原：`w = d × q + m`。多一个偏移量，能更好地表示不以 0 为中心的数据。

**q4_K（k-quant）**：每个"超级块"256 个权重，144 字节
```c
typedef struct {
    ggml_half d;           // 2 字节：缩放系数的缩放系数
    ggml_half dmin;        // 2 字节：最小值的缩放系数
    uint8_t scales[12];    // 12 字节：8 个子块各自的 scale 和 min，每个 6 bit
    uint8_t qs[128];       // 128 字节：256 个 4-bit 权重
} block_q4_K;              // 144 字节 / 256 个 = 4.5 bit/权重
```
**两级缩放**：256 个权重分成 8 个子块，每个子块 32 个，有自己的 scale 和 min（用 6 bit 存），这些 scale 和 min 本身再用 `d`、`dmin` 缩放。比 q4_0 更精细，所以通常更准，但体积一样。

### 5.3 汇总

| 格式 | 块大小 | 字节/块 | bit/权重 | 还原公式 | x86 上有 REPACK？ |
|---|--:|--:|--:|---|---|
| f16 | — | — | 16 | 直接用 | — |
| q8_0 | 32 | 34 | 8.5 | `d·q` | ❌ |
| q5_1 | 32 | 24 | 6.0 | `d·q + m` | ❌ |
| q5_0 | 32 | 22 | 5.5 | `d·(q−16)` | ❌ |
| q4_k | 256 | 144 | 4.5 | 两级缩放 | ✅ |
| q4_0 | 32 | 18 | 4.5 | `d·(q−8)` | ✅ |

**q4_0 和 q4_k 文件大小完全一样（base 都是 44.3 MB）**，因为都是 4.5 bit/权重。

### 5.4 量化后的矩阵乘怎么算

ggml 并**不是**先把整个权重矩阵还原成 f16 再乘。它是：
1. 把输入向量 x 也临时量化成 8 bit（q8_0 或 q8_K）
2. 按块做**整数**点积（int8 × int4/int8）
3. 最后乘上两边的缩放系数

整数运算比浮点快，而且数据小，能更多地留在缓存里。REPACK 的 kernel 名字 `q4_0_8x8_q8_0` 就是这个意思：q4_0 权重 × q8_0 输入，按 8×8 的块来算。

---

## 6. 怎么读计时输出

### 6.1 whisper-cli 的输出（base f16，6 线程，jfk.wav）

```
system_info: n_threads = 6 / 12 | CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | OPENMP = 1 | REPACK = 1 |
```
- `6 / 12`：用了 6 个线程，系统共 12 个逻辑处理器
- 后面是编译进去的 CPU 能力，**没有 AVX512** ✅

```
main: processing 'jfk.wav' (176000 samples, 11.0 sec), 6 threads, 1 processors, 5 beams + best of 5, lang = en, task = transcribe, timestamps = 1 ...
```
- 176000 个采样 ÷ 16000 Hz = 11.0 秒

```
whisper_print_timings:     load time =   246.51 ms
```
**加载**：读文件、分配内存、（如果是 q4）重排权重。只发生一次。

```
whisper_print_timings:     fallbacks =   0 p /   0 h
```
**温度回退**：当解码结果看起来不对（比如重复、压缩率异常）时，Whisper 会提高随机性重新解码。0 表示一次就成功了。

```
whisper_print_timings:      mel time =    10.40 ms
```
**mel 频谱图**：音频 → 80 × 3000 的图。很快，可以忽略。

```
whisper_print_timings:   sample time =   103.65 ms /   133 runs (     0.78 ms per run)
```
**采样**：从 51865 个词的概率里挑词、维护 beam。133 次 × 0.78 ms。

```
whisper_print_timings:   encode time =   828.18 ms /     1 runs (   828.18 ms per run)
```
**Encoder**：跑 1 次，占总时间一半以上。

```
whisper_print_timings:   decode time =    22.69 ms /     4 runs (     5.67 ms per run)
whisper_print_timings:   batchd time =   288.33 ms /   127 runs (     2.27 ms per run)
```
**Decoder** 分两类：
- `decode`：一次处理 1 条序列（beam 分叉之前的少数几步）
- `batchd`：一次处理多条序列（5 个 beam 打包成一批），127 次

```
whisper_print_timings:    total time =  1529.71 ms
```

**RTF**（Real-Time Factor，实时率）= 1529.71 ms ÷ 11000 ms = **0.139**。即处理 1 秒音频要 0.139 秒；小于 1 就比实时快。

### 6.2 whisper-bench 测的是什么（✅ 来自 [`examples/bench/bench.cpp`](../third_party/whisper.cpp/examples/bench/bench.cpp)）

whisper-bench 不读音频，直接喂一个全零的 mel，单独计时各个部分：

| 字段 | 实际做了什么 | 测的是 |
|---|---|---|
| `encode` | `whisper_encode` 跑 1 次（30 秒窗口） | encoder 一次的耗时 |
| `decode` | `whisper_decode(tokens, 1, i)` 循环 256 次，`n_past` 从 0 到 255 | **一次 1 个 token** 的耗时 |
| `batchd` | `whisper_decode(tokens, 5, 0)` 循环 64 次（共 320 个 token） | **一次 5 个 token** 时，平均每个 token 的耗时 |
| `prompt` | `whisper_decode(tokens, 256, 0)` 循环 16 次（共 4096 个 token） | **一次 256 个 token** 时，平均每个 token 的耗时 |

正式计时前，会先把每种操作各跑一遍预热。

**这三种 decode 的对比本身就是一个漂亮的实验**（base f16，6 线程）：

| 一批几个 token | 每个 token 耗时 |
|--:|--:|
| 1（decode） | 5.78 ms |
| 5（batchd） | 2.39 ms |
| 256（prompt） | 1.32 ms |

**同样的权重，一次处理的 token 越多，摊到每个 token 上就越便宜。** 因为权重只从内存读一次，就能给更多 token 用。这正是第 8 节"算术强度"的直观体现。

---

## 7. 四个脚本逐个讲

### 7.1 `scripts/common.py`：公共部分

```python
ROOT = Path(__file__).resolve().parent.parent   # 项目根目录 whisper-edge/
BIN = ROOT / "third_party/whisper.cpp/build/bin/RelWithDebInfo"
QUANTS = ["f16", "q8_0", "q5_1", "q5_0", "q4_k", "q4_0"]
```

- `Path(__file__).resolve().parent.parent`：脚本自己的路径 → 上一级 `scripts/` → 再上一级就是项目根目录。这样不管从哪个目录运行脚本，路径都对。
- `model_path("base", "q4_0")` → `models/ggml-base-q4_0.bin`；f16 对应原始文件名 `ggml-base.bin`。
- `run(cmd)`：运行一个外部程序，**分开**返回 stdout 和 stderr；如果程序出错就直接退出并打印错误。

**为什么 stdout 和 stderr 要分开**：whisper-cli 把**转录文字打到 stdout**，把**日志和计时打到 stderr**。第一版脚本把两者直接拼在一起，导致转录文字后面粘了一段日志（第 11 节的坑 4）。

### 7.2 `scripts/setup_models.py`：下载 + 量化

```python
def download(size):
    dst = model_path(size, "f16")
    if dst.exists():                       # 已经有了就跳过
        return
    tmp = dst.with_suffix(".part")         # 先写到 .part 临时文件
    with urllib.request.urlopen(URL) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)           # 流式复制，不会把 500 MB 一次读进内存
    tmp.rename(dst)                        # 下载完整后才改名
```

**为什么先写 `.part` 再改名**：如果下载到一半断网，留下的是 `.part` 文件，下次运行时 `dst.exists()` 仍然是 False，会重新下载。如果直接写 `dst`，半个文件也会被当成"已存在"而跳过，之后加载就会出错。

`quantize()` 的逻辑一样：存在就跳过，否则调用 `whisper-quantize.exe`。

### 7.3 `scripts/week1_bench.py`：测量

**解析计时的正则表达式**：

```python
TIMING_RE = re.compile(r"whisper_print_timings:\s+(\w+) time =\s+([\d.]+) ms(?: /\s+(\d+) runs \(\s*([\d.]+) ms per run\))?")
```

拆开看：

| 片段 | 匹配 | 捕获组 |
|---|---|---|
| `whisper_print_timings:\s+` | 固定前缀和空格 | |
| `(\w+) time =` | `encode time =` | ① 名字 `encode` |
| `\s+([\d.]+) ms` | `828.18 ms` | ② 总耗时 |
| `(?: ... )?` | 整个括号可有可无（`load time` 就没有 runs） | |
| `/\s+(\d+) runs` | `/ 1 runs` | ③ 次数 |
| `\(\s*([\d.]+) ms per run\)` | `( 828.18 ms per run)` | ④ 每次耗时 |

`parse_timings` 把这些整理成字典：`{"encode_ms": 828.18, "encode_per_run_ms": 828.18, ...}`。

**写 CSV**：

```python
keys = list(dict.fromkeys(k for r in rows for k in r))
```
收集所有行里出现过的列名，用 `dict.fromkeys` 去重并保持顺序（集合 `set` 不保证顺序）。

**`--steps` 参数**：`py -3.11 scripts/week1_bench.py --steps micro` 只跑微基准，调试时不用每次等 20 分钟。

### 7.4 `scripts/make_tables.py`：生成表格

读两个 CSV，按 `(模型, 量化, 线程数)` 建字典索引，然后格式化成 Markdown 表格。`sys.stdout.reconfigure(encoding="utf-8")` 是为了在 Windows 终端里正确输出中文（第 11 节的坑 5）。

---

## 8. 核心概念：计算瓶颈 vs 带宽瓶颈（Roofline）

**这是整个项目最重要的概念，面试必问。**

### 8.1 做一次矩阵乘，CPU 要干两件事

1. **搬数据**：把权重从内存搬进 CPU（受**内存带宽**限制）
2. **算**：做乘法和加法（受**算力**限制）

哪个更慢，哪个就是瓶颈。

### 8.2 算术强度（Arithmetic Intensity）

**算术强度 = 浮点运算次数 ÷ 搬运的字节数**（单位 FLOP/byte）

- 强度**高** → 每搬一个字节要算很多次 → 卡在**算力** → **计算瓶颈**（compute-bound）
- 强度**低** → 每搬一个字节只算一两次 → 卡在**带宽** → **带宽瓶颈**（memory-bound）

**分界点**（ridge point）= 峰值算力 ÷ 带宽 ≈ 576 GFLOPS ÷ 约 30 GB/s ≈ **19 FLOP/byte**。高于这个值是计算瓶颈，低于是带宽瓶颈。

### 8.3 算一算 Whisper 的两部分

一个 d × d 的权重矩阵（f16，每个权重 2 字节），乘以 T 个输入向量：
- 运算：每个权重对每个向量做 1 次乘加 = 2 FLOP → 共 `2 × d² × T` FLOP
- 搬运：权重读一次 = `2 × d²` 字节
- **算术强度 ≈ T**

| | T（一次处理几个向量） | 算术强度 | 结论 |
|---|--:|--:|---|
| Encoder | 1500 帧 | 约 1500 FLOP/byte | 远高于 19 → **计算瓶颈** |
| Decoder（decode） | 1 个 token | 约 1 FLOP/byte | 远低于 19 → **带宽瓶颈** |
| Decoder（batchd） | 5 个 token | 约 5 FLOP/byte | 仍是带宽瓶颈，但好一些 |
| Decoder（prompt） | 256 个 token | 约 256 FLOP/byte | 计算瓶颈 |

这就解释了 6.2 节的现象：batch 越大，每个 token 越便宜。

### 8.4 厨师的比喻

- 厨师 = CPU 的计算单元，仓库 = 内存，搬运工 = 内存带宽
- **Encoder** = 给 1500 桌客人做同一道菜：每样食材搬一次，切 1500 遍。**瓶颈是厨师的手速。**
- **Decoder** = 每次只给 1 桌做：食材搬一次，只切 1 遍。**厨师大部分时间在等搬运工。**

### 8.5 量化对两种瓶颈的作用

| | 量化让数据变小 | 量化让计算变复杂（要还原） |
|---|---|---|
| 带宽瓶颈（decoder） | ✅ 搬得少 → 变快 | 影响不大，反正在等数据 |
| 计算瓶颈（encoder） | 影响不大，本来就不在等数据 | ⚠️ 还原越复杂越慢；但整数运算和专用 kernel 可以更快 |

所以**同样是量化，对 encoder 和 decoder 的效果完全不同**。这就是本周数据的主线。

---

## 9. 用数字验证：结果逐条分析

### 9.1 Encoder：确实是计算瓶颈 ✅

**粗算 encoder 一次的计算量**（T = 1500，d = 隐藏维度）：

每层 Transformer：
- Q、K、V、O 四个投影：4 × (2·T·d²) = 8·T·d²
- MLP（d → 4d → d）：2 × (2·T·d·4d) = 16·T·d²
- 注意力分数 QKᵀ 和 加权求和 AV：2 × (2·T²·d) = 4·T²·d

两层卷积：conv1（80 → d，kernel 3，3000 帧），conv2（d → d，kernel 3，1500 帧）

| | base（d=512，6 层） | small（d=768，12 层） |
|---|--:|--:|
| 线性层（可量化） | 56.6 GFLOP | 254.8 GFLOP |
| 注意力（不可量化） | 27.6 GFLOP | 82.9 GFLOP |
| 卷积（f16，不量化） | 3.1 GFLOP | 6.4 GFLOP |
| **合计** | **约 87 GFLOP** | **约 344 GFLOP** |

**实际达到的算力 = 计算量 ÷ encode 时间**（6 线程）：

| | base | small | 对比：`-w 2` 微基准 4096² 矩阵乘 |
|---|--:|--:|--:|
| f16 | 87 / 0.899 s = **97 GFLOPS** | 344 / 3.561 s = **97 GFLOPS** | F16：100 GFLOPS |
| q8_0 | 146 | 137 | Q8_0：207 |
| q5_0 | 118 | 119 | Q5_0：156 |
| q4_0 | 191 | 180 | Q4_0：177 |
| q4_k | **220** | **217** | （没测） |

**怎么读这张表**：

1. **f16 的 encoder 几乎跑满了 f16 矩阵乘的极限**（97 vs 100）→ 确实是计算瓶颈 ✅
2. **q4_0 / q4_k 在真实 encoder 里比微基准还快**（180–220 vs 177）。原因（✅ 已从源码确认）：微基准里的矩阵乘用的是普通的 ggml context（`src/whisper.cpp` 里的 `ggml_graph_compute_helper`），**没有走 REPACK**；而真实推理加载模型时会把 q4_0/q4_k 权重重排，用 8×8 的专用 kernel。🔍 第 3 周用 profiler 确认 encoder 确实调用了 `q4_0_8x8_q8_0` / `q4_K_8x8_q8_K`。
3. **q8_0 在 encoder 里只有 137–146，远低于微基准的 207。** 部分原因是 Amdahl 定律（下一条）：encoder 里有约 1/3 的计算（注意力）不受量化影响。
4. **注意力部分不能被量化加速。** QKᵀ 和 AV 是"激活值 × 激活值"，不涉及权重，永远以 f16 计算。base 里它占约 32%，small 里约 24%。所以即使线性层的速度无限快，base encoder 最多也只能快约 3 倍。

**q5 为什么慢**：它既**没有 REPACK**，还原时又要**拼接第 5 位**，计算代价最高。在计算瓶颈的 encoder 里，"省下的带宽"毫无用处，"多出的计算"全部变成了时间。所以 base q5_1 的 encoder（949 ms）甚至比 f16（899 ms）还慢。

### 9.2 Decoder：带宽主导，但没吃满量化的收益

**粗算每生成 1 个 token 要从内存读多少字节**：

每层 decoder 用到的权重：
- 自注意力 Q、K、V、O：4·d²
- 交叉注意力 Q、O：2·d²（交叉注意力的 K、V 已经在第 ③ 步算好存起来了，见第 1 节流程图）
- MLP：8·d²
- 合计 14·d² 个参数/层

还有词表矩阵 51865 × d（最后算 51865 个词的概率时要用），以及**交叉注意力的 KV cache**（从加载日志读到：base 18.87 MB，small 56.62 MB）。**KV cache 是 f16，不受模型量化影响。**

| | base | small |
|---|--:|--:|
| decoder 各层权重 | 14 × 512² × 6 = 22.0M 参数 | 14 × 768² × 12 = 99.1M 参数 |
| 词表矩阵 | 51865 × 512 = 26.6M | 51865 × 768 = 39.8M |
| 权重合计 | 48.6M 参数 | 138.9M 参数 |
| f16 权重字节 | 97 MB | 278 MB |
| + cross KV cache（f16） | 19 MB | 57 MB |
| **f16 每 token 读取** | **约 116 MB** | **约 334 MB** |
| **q4 每 token 读取**（权重 × 4.5/16） | 27 + 19 = **约 46 MB** | 78 + 57 = **约 135 MB** |

**实际读取速度 = 字节数 ÷ decode/token 时间**：

| | base | small |
|---|--:|--:|
| f16 | 116 MB / 5.78 ms = **20 GB/s** | 334 MB / 16.65 ms = **20 GB/s** |
| q4_0 / q4_k | 46 MB / 3.43 ms = **13.5 GB/s** | 135 MB / 9.77 ms = **13.8 GB/s** |

**怎么读这张表**：

1. **两个模型在 f16 下都是约 20 GB/s**：数据量差了将近 3 倍，速度却完全一样 → 说明它们被同一个东西卡住，也就是内存带宽 ✅
2. **如果是纯带宽瓶颈**，q4 应该快 116/46 ≈ **2.5 倍**，实际只快了 **1.7 倍**。差距来自两方面：
   - **cross KV cache 不缩小**：量化只压缩权重，KV cache 永远是 f16，所以整体数据量只缩小了 2.5 倍，而不是 3.5 倍。（这一点 week1.md 里没考虑到。）
   - **固定开销**：q4 下实际读取速度从 20 GB/s 降到 13.5 GB/s，说明有一部分时间没在读内存，而是在构建计算图、同步线程、算 layernorm 和 softmax 等小算子。🔍 第 3 周用 profiler 量化。
3. **20 GB/s 离上限还有距离**：理论值 51.2 GB/s，实际可用一般是 30–40 GB/s。

**关于 memcpy 的 14 GB/s**（✅ 已读源码）：`whisper-bench -w 1` 计算的是"复制的字节数 ÷ 时间"，但复制 1 字节实际要**读 1 次 + 写 1 次**（CPU 写之前通常还要先读一次目标位置），所以真实内存流量是 28–42 GB/s。**不能直接拿 14 GB/s 当读带宽上限**。第 3 周需要一个只读的带宽测试。

### 9.3 线程扩展

| 线程 | base f16 encode | 加速比 | 效率（加速比 ÷ 线程数） |
|--:|--:|--:|--:|
| 1 | 3785 ms | 1.00× | 100% |
| 2 | 1954 ms | 1.94× | 97% |
| 4 | 1046 ms | 3.62× | 91% |
| 6 | 899 ms | 4.21× | 70% |
| 8 | 785 ms | 4.82× | 60% |
| 12 | 745 ms | 5.08× | 42% |

**1 → 4 线程接近线性**：活多，切开分给各个核就行。

**4 → 6 效率下降**：可能的原因有
- 多核同时满载 → 功耗和温度上升 → 降频（4600H 单核能到 4.0 GHz，全核满载一般更低）
- 有些算子（layernorm、softmax）并行度差
- 线程同步（每个算子结束都要等所有线程完成）

**6 → 12 只多出约 20%**：4600H 只有 6 个物理核，第 7–12 个线程是超线程（SMT）。**两个超线程共用同一个核的 FMA 单元**。encoder 已经把 FMA 单元占满了，再加线程只是排队。剩下那 20% 是因为一个线程在等内存或做别的事时，另一个可以顶上。

**Decoder 在 4–6 线程就饱和**：

| 线程 | small f16 decode/token | small q4_0 decode/token |
|--:|--:|--:|
| 1 | 41.62 ms | 31.61 ms |
| 4 | 19.37 ms | 11.77 ms |
| 6 | 16.65 ms | 10.64 ms |
| 8 | 16.70 ms | 9.77 ms |
| 12 | 16.14 ms | **16.77 ms** ⚠️ |

- **带宽瓶颈的任务，加线程不会让内存变快。** 4 个线程就差不多把能用的带宽占满了。
- **12 线程时 q4_0 反而大幅变慢**（9.77 → 16.77 ms）：每个 token 的计算本来就很少，切给 12 个线程后，每个线程只干几微秒，"分活 + 等所有人做完"的同步开销反而成了大头。🔍 这只是单次测量，需要多次重复确认（第 10 节）。

**Amdahl 定律**：如果程序里有比例为 p 的部分可以并行，用 N 个线程的最大加速比是 `1 / ((1 − p) + p/N)`。哪怕 95% 可并行，无限多线程也最多快 20 倍。实践中还要加上同步开销和带宽上限。

### 9.4 加载时间

| | f16 | q4_0 |
|---|--:|--:|
| base | 258 ms | 115 ms |
| small | 724 ms | 276 ms |

基本和文件大小成正比（读磁盘）。q4_k 的加载（small 383 ms）比 q4_0（276 ms）慢一些，🔍 可能是 REPACK 的重排过程更复杂。端侧应用冷启动时，这也是量化的收益之一。

### 9.5 精度

12 个模型对 jfk 的转录全部正确，只差逗号和句号。**这段音频太简单（清晰、标准发音、著名演讲，可能还在训练集里），区分不出精度差异。** 第 2 周用 LibriSpeech 和 Common Voice 测 WER/CER 才能看出来。

### 9.6 本周结论

1. **Encoder 是计算瓶颈**，量化能不能加速，取决于有没有快速 kernel，而不是数据小不小。在 x86 上，q4_0/q4_k 有 REPACK，最快；q5 最慢。
2. **Decoder 是带宽瓶颈**，量化能加速，但 cross KV cache 不缩小、固定开销不变，所以提速达不到权重缩小的比例。
3. **线程数**：encoder 用 6–8 线程，decoder 用 4–8 线程，12 线程不划算甚至变慢。
4. **当前最佳**：q4_k（encoder 快 2.25 倍，decoder 快 1.7 倍，体积 30%），精度待第 2 周确认。

---

## 10. 测量方法的局限

这些都是第 2 周要改进的：

| 问题 | 影响 | 改进办法 |
|---|---|---|
| 每个配置只跑 1 次 | 有离群值（base q8_0 8 线程 decode 5.98 ms，前后都是 4 ms 多） | 跑 3–5 次取中位数 |
| 电源计划"平衡" | CPU 可能动态降频，结果不稳定 | 插电，切到"高性能" |
| 没控制温度 | 笔记本长时间满载会降频，后面跑的配置吃亏 | 配置之间留冷却时间，或打乱顺序跑 |
| jfk 只有 11 秒 | encoder 永远按 30 秒算，短音频的 RTF 偏悲观 | 用不同长度的音频 |
| 精度只看了 1 段 | 无法比较量化损失 | WER/CER |
| memcpy 不是读带宽 | 不能当 roofline 的带宽上限 | 用只读的带宽测试 |

---

## 11. 踩过的坑

### 坑 1：目录没有写权限

```
New-Item : Access to the path 'whisper-edge' is denied.
```

**原因**：`D:\learnMLDSPNew` 的所有者是 Administrators，普通用户只有读权限。
**解决**：用管理员身份打开 PowerShell，运行
```bash
icacls "D:\learnMLDSPNew" /grant "chinaunmobile\harryliu:(OI)(CI)M"
```
`(OI)` 表示文件继承，`(CI)` 表示子目录继承，`M` 是修改权限。

### 坑 2：Git Bash 把 `/m` 变成了 `M:/`

```
开关:M:/
```

**原因**：在 Git Bash（MSYS）里，以 `/` 开头的参数会被当成 Unix 路径自动转换成 Windows 路径。MSBuild 的 `/m` 参数被转成了 `M:/`。
**解决**：去掉 `/m`（`cmake --build -j 12` 已经会让 MSBuild 并行），或者在命令前加 `MSYS_NO_PATHCONV=1`。

### 坑 3：`python` 打开了微软商店

```
Python was not found; run without arguments to install from the Microsoft Store
```

**原因**：Windows 自带一个叫 `python.exe` 的"应用执行别名"，它只会跳转到商店。
**解决**：用 Python 启动器 `py -3.11`，或者在"设置 → 应用 → 高级应用设置 → 应用执行别名"里关掉它。

### 坑 4：转录文字后面粘着日志

```
...for your country.whisper_init_from_file_with_params_no_state: loading model from ...
```

**原因**：第一版脚本用 `stdout + stderr` 直接拼接两个流，而 stdout 的最后一行没有换行符，于是和 stderr 的第一行连在了一起。
**解决**：分开处理两个流，转录只从 stdout 读，计时只从 stderr 读。**教训：不同用途的输出要分开处理，不要拼在一起再解析。**

### 坑 5：中文输出乱码

**原因**：Windows 中文版的默认控制台编码是 GBK，Python 打印 UTF-8 的中文时会乱码。
**解决**：`sys.stdout.reconfigure(encoding="utf-8")`。

### 坑 6：`LF will be replaced by CRLF` 警告

**原因**：Windows 用 CRLF（`\r\n`）换行，Linux 用 LF（`\n`）。Git 在 Windows 上默认会做转换。
**影响**：无害，可以忽略。

---

## 12. 术语表

| 术语 | 解释 |
|---|---|
| ASR | Automatic Speech Recognition，自动语音识别 |
| mel 频谱图 | 按人耳感知（mel 刻度）划分频带的频谱，是 Whisper 的输入 |
| Encoder / Decoder | 编码器（理解音频）/ 解码器（生成文字） |
| Token | 模型处理的最小文本单位，可能是一个词、半个词或一个标点 |
| 自回归 | 每一步的输出作为下一步的输入，只能一个一个生成 |
| Beam search | 同时保留多条候选序列的解码方法 |
| KV cache | 缓存注意力的 K、V，避免重复计算 |
| ggml | whisper.cpp 底层的张量计算库 |
| 计算图 | 描述"先算什么后算什么"的有向图，ggml 先建图再执行 |
| 量化 | 用更少的 bit 存权重 |
| q4_0 / q8_0 / ... | ggml 的量化格式，数字是每个权重的主要 bit 数 |
| k-quant（q4_k） | 用超级块和两级缩放的量化格式，更精细 |
| REPACK | 加载时把权重重排成适合 SIMD 的布局 |
| SIMD | 一条指令处理多个数据 |
| AVX2 / AVX-512 | x86 的 256 位 / 512 位 SIMD 指令集 |
| FMA | 乘加融合指令，`a×b+c` |
| SMT / 超线程 | 一个物理核运行两个硬件线程 |
| FLOP / GFLOPS | 浮点运算次数 / 每秒十亿次浮点运算 |
| 带宽 | 每秒能从内存搬多少字节（GB/s） |
| 算术强度 | FLOP ÷ 字节数 |
| Roofline 模型 | 用算术强度判断瓶颈是算力还是带宽的方法 |
| 计算瓶颈 | 速度受限于算力（compute-bound） |
| 带宽瓶颈 | 速度受限于内存带宽（memory-bound） |
| GEMM / GEMV | 矩阵 × 矩阵 / 矩阵 × 向量 |
| Amdahl 定律 | 串行部分决定了并行加速的上限 |
| RTF | Real-Time Factor，处理时间 ÷ 音频时长 |
| WER / CER | 词错误率 / 字错误率 |
| git submodule | 在一个仓库里引用另一个仓库的某个 commit |
| CMake | 跨平台的构建配置工具 |
| RelWithDebInfo | 带调试信息的优化编译配置 |

---

## 13. 自测题

<details>
<summary>1. 为什么 Whisper 的 encoder 不管音频多长，都按 30 秒处理？</summary>

Whisper 训练时输入固定是 30 秒（3000 帧 mel），位置编码也是固定 1500 个位置。短音频补零，长音频要切成多段。所以 11 秒的 jfk 和 30 秒的音频，encoder 的耗时是一样的。
</details>

<details>
<summary>2. q4_0 每个权重为什么是 4.5 bit，而不是 4 bit？</summary>

每 32 个权重共用一个 f16 的缩放系数 d（16 bit）。16 ÷ 32 = 0.5 bit 摊到每个权重上，所以是 4 + 0.5 = 4.5 bit。
</details>

<details>
<summary>3. q5_0 的文件比 q8_0 小，为什么 encoder 反而更慢？</summary>

encoder 是计算瓶颈，数据小不小不重要，重要的是算得快不快。q5_0 还原时要把第 5 位和低 4 位拼起来，额外的位运算让它比 q8_0（直接就是 int8）更慢。而且两者在 x86 上都没有 REPACK。
</details>

<details>
<summary>4. 为什么 batchd 每个 token 比 decode 便宜？</summary>

decode 一次处理 1 个 token，batchd 一次处理 5 个。权重从内存读一次，就能给 5 个 token 用，算术强度提高了 5 倍。decoder 是带宽瓶颈，读的次数少了，自然更快。
</details>

<details>
<summary>5. 为什么 decoder 的提速达不到权重缩小的比例？举两个原因。</summary>

① cross-attention 的 KV cache 是 f16，不随模型量化缩小；② 每个 token 都有构建计算图、线程同步、小算子等固定开销，这部分和权重大小无关。
</details>

<details>
<summary>6. 为什么线程数从 6 加到 12，encoder 只快了 20%？</summary>

4600H 只有 6 个物理核，第 7–12 个线程是超线程，和另一个线程共用同一个核的 FMA 单元。encoder 已经把 FMA 单元占满，加线程只是排队。
</details>

<details>
<summary>7. 为什么 q4_0 在真实 encoder 里比在 -w 2 微基准里还快？</summary>

真实推理加载模型时会对 q4_0 做 REPACK，用 8×8 的专用 kernel；微基准用的是普通 ggml context，没有重排。
</details>

<details>
<summary>8. memcpy 测出 14 GB/s，能说内存带宽只有 14 GB/s 吗？</summary>

不能。memcpy 按"复制的字节数"计算，但复制要读一次源、写一次目标（写之前常常还要先读一次目标），真实内存流量是 2–3 倍。而且 decoder 实测读取已经到了 20 GB/s，说明可用带宽高于 14 GB/s。
</details>

<details>
<summary>9. 为什么用 RelWithDebInfo 而不是 Debug 来测速？</summary>

Debug 关闭了编译优化，比优化后的代码慢很多倍，测出的数字不能代表真实性能。RelWithDebInfo 既有优化，又保留了调试符号。
</details>

<details>
<summary>10. 用自己的话解释"算术强度"，并说明 encoder 和 decoder 各自大约是多少。</summary>

算术强度 = 浮点运算次数 ÷ 搬运的字节数。对于 f16 权重的矩阵乘，强度约等于一次处理的向量个数 T。encoder T = 1500 → 约 1500 FLOP/byte（计算瓶颈）；decoder T = 1 → 约 1 FLOP/byte（带宽瓶颈）。
</details>

---

## 14. 动手练习

在 `D:\learnMLDSPNew\whisper-edge` 下运行。

**练习 1：看 encode 和 decode 分别怎么随量化变化**

```bash
third_party/whisper.cpp/build/bin/RelWithDebInfo/whisper-cli.exe -m models/ggml-small.bin -f third_party/whisper.cpp/samples/jfk.wav -t 6
```

```bash
third_party/whisper.cpp/build/bin/RelWithDebInfo/whisper-cli.exe -m models/ggml-small-q5_0.bin -f third_party/whisper.cpp/samples/jfk.wav -t 6
```

```bash
third_party/whisper.cpp/build/bin/RelWithDebInfo/whisper-cli.exe -m models/ggml-small-q4_k.bin -f third_party/whisper.cpp/samples/jfk.wav -t 6
```

对比三次的 `encode time` 和 `batchd time`。预期：q5_0 的 encode 只比 f16 快一点，q4_k 快 2 倍多。

**练习 2：体会超线程**

把练习 1 的 q4_k 命令改成 `-t 12`，看 encode 和 batchd 分别怎么变。

**练习 3：看哪些张量被量化了**

```bash
third_party/whisper.cpp/build/bin/RelWithDebInfo/whisper-quantize.exe models/ggml-base.bin C:/Users/harryliu/AppData/Local/Temp/test-q4_0.bin q4_0
```

输出里每一行是一个张量：名字、形状、原类型、新类型。找出哪些保持 f16/f32，对照第 4.5 节的规则。看完删掉这个临时文件。

**练习 4：看 REPACK 的效果**

```bash
third_party/whisper.cpp/build/bin/RelWithDebInfo/whisper-bench.exe -m models/ggml-base-q4_0.bin -t 6
```

```bash
third_party/whisper.cpp/build/bin/RelWithDebInfo/whisper-bench.exe -m models/ggml-base-q8_0.bin -t 6
```

q4_0 的 encode 应该比 q8_0 快约 1.3 倍，虽然 `-w 2` 微基准里 Q8_0 更快。想一想为什么（答案在第 9.1 节）。

**练习 5：观察 CPU 占用**

打开任务管理器 → 性能 → CPU → 右键 → 将图形更改为"逻辑处理器"。分别用 `-t 1`、`-t 6`、`-t 12` 跑 whisper-bench，看有几个逻辑处理器被占满。

**练习 6：自己算一遍**

用第 9.1 节的公式，算出 small 模型 encoder 的计算量，再用 `small q8_0 t=6` 的 encode 时间（2510 ms）算出实际 GFLOPS。答案约 137 GFLOPS。

---

## 15. 和下一周的关系

| 第 1 周留下的问题 | 第几周解决 |
|---|---|
| 量化损失了多少精度？ | 第 2 周：WER / CER |
| 数据噪声大 | 第 2 周：多次重复取中位数、固定电源计划 |
| 峰值内存多少？ | 第 2 周：psutil 测 peak working set |
| GPU 上是什么情况？ | 第 2 周：5080 + CUDA |
| q4 是不是真的走了 REPACK？ | 第 3 周：profiler 看函数调用 |
| decoder 的固定开销有多大？ | 第 3 周：profiler 看非 matmul 算子的占比 |
| 真实的读带宽上限是多少？ | 第 3 周：只读带宽测试，画 roofline |
| 12 线程为什么变慢？ | 第 3 周：profiler 看线程同步的等待时间 |
