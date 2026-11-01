# whisper-edge

基于 whisper.cpp 的端侧语音识别量化与性能分析（AMD Ryzen 5 4600H，CPU；对比 RTX 5080，CUDA）。

## 目录

- `third_party/whisper.cpp/`：上游仓库（git submodule，固定到某个 commit 保证可复现）
- `models/`：ggml 模型和量化版本（不入库，按下面步骤生成）
- `scripts/`：脚本
  - `common.py`：共享的路径和模型列表
  - `setup_models.py`：下载 base / small 并生成 q8_0、q5_1、q5_0、q4_k、q4_0（已存在的跳过）
  - `week1_bench.py`：jfk 转录、bench 扫描、memcpy / mul_mat 微基准，输出到 `results/`
  - `make_tables.py`：从 CSV 生成 `results/week1.md` 里的表格
- `results/`：原始数据（CSV）和每周的分析，见 [第 1 周](results/week1.md)
- `docs/`：学习笔记，见 [chunk1.md](docs/chunk1.md)（第 1 周的完整推导）

## 复现

```bash
git submodule update --init
cmake -S third_party/whisper.cpp -B third_party/whisper.cpp/build -G "Visual Studio 17 2022" -A x64
cmake --build third_party/whisper.cpp/build --config RelWithDebInfo -j 12

py -3.11 scripts/setup_models.py                  # 约 600 MB 下载 + 量化
py -3.11 scripts/week1_bench.py                   # 全部测量，约 20 分钟；可用 --steps jfk bench micro 只跑一部分
py -3.11 scripts/make_tables.py > tables.md       # 生成报告表格
```
