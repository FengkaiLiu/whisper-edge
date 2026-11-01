"""Print the markdown tables used in results/week1.md from the week 1 CSVs.

Usage: py -3.11 scripts/make_tables.py > tables.md
"""
import csv
import sys

from common import QUANTS, RESULTS, SIZES

JFK_SECONDS = 11.0
SWEEP_THREADS = 6


def load(name):
    with open(RESULTS / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def quant_table(bench, jfk):
    def b(size, quant, key):
        return float(bench[(size, quant, SWEEP_THREADS)][key])

    lines = [
        "| 模型 | 量化 | 文件 (MB) | 加载 (ms) | encode (ms) | encode 提速 | decode/token (ms) | decode 提速 | jfk 总耗时 (ms) | jfk RTF |",
        "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for size in SIZES:
        for quant in QUANTS:
            enc, dec = b(size, quant, "encode_ms"), b(size, quant, "decode_per_run_ms")
            total = float(jfk[(size, quant)]["total_ms"])
            lines.append(
                f"| {size} | {quant} | {b(size, quant, 'file_mb'):.1f} | {float(jfk[(size, quant)]['load_ms']):.0f} "
                f"| {enc:.0f} | {b(size, 'f16', 'encode_ms') / enc:.2f}× "
                f"| {dec:.2f} | {b(size, 'f16', 'decode_per_run_ms') / dec:.2f}× "
                f"| {total:.0f} | {total / 1000 / JFK_SECONDS:.3f} |"
            )
    return "\n".join(lines)


def thread_table(bench):
    threads = sorted({t for _, _, t in bench})

    def b(size, quant, t, key):
        return float(bench[(size, quant, t)][key])

    lines = [
        "| 线程数 | base f16 encode (ms) | 加速比 | small q4_k encode (ms) | 加速比 | small f16 decode/token (ms) | small q4_0 decode/token (ms) |",
        "|--:|--:|--:|--:|--:|--:|--:|",
    ]
    t0 = threads[0]
    for t in threads:
        lines.append(
            f"| {t} | {b('base', 'f16', t, 'encode_ms'):.0f} "
            f"| {b('base', 'f16', t0, 'encode_ms') / b('base', 'f16', t, 'encode_ms'):.2f}× "
            f"| {b('small', 'q4_k', t, 'encode_ms'):.0f} "
            f"| {b('small', 'q4_k', t0, 'encode_ms') / b('small', 'q4_k', t, 'encode_ms'):.2f}× "
            f"| {b('small', 'f16', t, 'decode_per_run_ms'):.2f} | {b('small', 'q4_0', t, 'decode_per_run_ms'):.2f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    bench = {(r["model"], r["quant"], int(r["threads"])): r for r in load("week1_bench.csv")}
    jfk = {(r["model"], r["quant"]): r for r in load("week1_jfk.csv")}
    print("## 量化等级对比\n")
    print(quant_table(bench, jfk))
    print("\n## 线程扩展\n")
    print(thread_table(bench))
