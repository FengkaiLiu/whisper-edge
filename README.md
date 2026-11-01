# whisper-edge

基于 whisper.cpp 的端侧语音识别量化与性能分析（AMD Ryzen 5 4600H，CPU；对比 RTX 5080，CUDA）。

## 目录

- `third_party/whisper.cpp/`：上游仓库（git submodule，固定到某个 commit 保证可复现）
- `models/`：ggml 模型和量化版本（不入库，按下面步骤生成）
- `scripts/`：测量脚本
- `results/`：原始数据（CSV）和每周的分析，见 [第 1 周](results/week1.md)

## 复现

```bash
git submodule update --init
cmake -S third_party/whisper.cpp -B third_party/whisper.cpp/build -G "Visual Studio 17 2022" -A x64
cmake --build third_party/whisper.cpp/build --config RelWithDebInfo -j 12

# 模型：从 https://huggingface.co/ggerganov/whisper.cpp 下载 ggml-base.bin、ggml-small.bin 到 models/
# 量化：q8_0 q5_1 q5_0 q4_k q4_0
third_party/whisper.cpp/build/bin/RelWithDebInfo/whisper-quantize.exe models/ggml-base.bin models/ggml-base-q4_0.bin q4_0

py -3.11 scripts/week1_bench.py
```
