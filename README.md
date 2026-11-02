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
  - `prepare_data.py`：下载 LibriSpeech test-clean / FLEURS（中、日）并抽样，写 `manifests/<lang>.jsonl`
  - `evaluate.py`：对每个（语言 × 模型 × 线程数）跑 whisper-cli，算 WER/CER、RTF、峰值内存，可断点续跑
  - `bootstrap_ci.py`：配对 bootstrap，判断量化模型相对 f16 的误差变化是否显著
  - `plot_week2.py`：精度-速度、时间分布、内存三张图
  - `cuda/devq.cu`：GPU 参数查询 + 显存 / PCIe 带宽测试
- `manifests/`：评测用的句子清单（音频本身在 `data/`，不入库）
- `results/`：原始数据（CSV）和每周的分析，见 [第 1 周](results/week1.md)、[第 2 周](results/week2.md)
- `docs/`：学习笔记，见 [chunk1.md](docs/chunk1.md)（第 1 周的完整推导）、[cuda_guide.md](docs/cuda_guide.md)（CUDA 入门和第一轮 GPU 数据，配合第 2–3 周）、[jd_knowledge.md](docs/jd_knowledge.md)（按岗位要求拆解的知识地图和面试题）

## 复现

```bash
git submodule update --init
cmake -S third_party/whisper.cpp -B third_party/whisper.cpp/build -G "Visual Studio 17 2022" -A x64
cmake --build third_party/whisper.cpp/build --config RelWithDebInfo -j 12

py -3.11 scripts/setup_models.py                  # 约 600 MB 下载 + 量化
py -3.11 scripts/week1_bench.py                   # 全部测量，约 20 分钟；可用 --steps jfk bench micro 只跑一部分
py -3.11 scripts/make_tables.py > tables.md       # 生成报告表格
```

第 2 周（评测框架，需要 Python 依赖）：

```bash
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe scripts/prepare_data.py                  # 下载约 1.3 GB 数据，生成 manifests/
.venv/Scripts/python.exe -u scripts/evaluate.py --tag week2_cpu   # 3 语言 × 12 模型，约 3 小时，中断后重跑会续上
.venv/Scripts/python.exe scripts/bootstrap_ci.py --tag week2_cpu
.venv/Scripts/python.exe scripts/plot_week2.py --tag week2_cpu
```

快速冒烟测试：`evaluate.py --tag smoke --langs en --models base-q4_k --limit 5`。在其他机器或 GPU 构建上用 `--bin <whisper-cli 所在目录>`。
