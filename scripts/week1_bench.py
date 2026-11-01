"""Week 1 first-pass measurements.

Runs whisper-bench (encoder / decoder timing) over every model x thread count,
and whisper-cli on samples/jfk.wav for a transcript sanity check per model.
Writes results/week1_bench.csv and results/week1_jfk.csv.
"""
import csv
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "third_party/whisper.cpp/build/bin/RelWithDebInfo"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
JFK = ROOT / "third_party/whisper.cpp/samples/jfk.wav"

SIZES = ["base", "small"]
QUANTS = ["f16", "q8_0", "q5_1", "q5_0", "q4_k", "q4_0"]
THREADS = [1, 2, 4, 6, 8, 12]

# e.g. "whisper_print_timings:   encode time =   828.18 ms /     1 runs (   828.18 ms per run)"
TIMING_RE = re.compile(r"whisper_print_timings:\s+(\w+) time =\s+([\d.]+) ms(?: /\s+(\d+) runs \(\s*([\d.]+) ms per run\))?")


def model_path(size, quant):
    return MODELS / (f"ggml-{size}.bin" if quant == "f16" else f"ggml-{size}-{quant}.bin")


def parse_timings(text):
    out = {}
    for name, total, runs, per_run in TIMING_RE.findall(text):
        out[f"{name}_ms"] = float(total)
        if per_run:
            out[f"{name}_per_run_ms"] = float(per_run)
    return out


def run(cmd):
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        sys.exit(f"command failed ({p.returncode}): {cmd}\n{p.stderr[-2000:]}")
    return p.stdout + "\n" + p.stderr


def bench_sweep():
    rows = []
    for size in SIZES:
        for quant in QUANTS:
            m = model_path(size, quant)
            for t in THREADS:
                tm = parse_timings(run([BIN / "whisper-bench.exe", "-m", m, "-t", t]))
                row = {"model": size, "quant": quant, "threads": t,
                       "file_mb": round(m.stat().st_size / 2**20, 1), **tm}
                rows.append(row)
                print(f"{size:5} {quant:5} t={t:2}  encode={tm.get('encode_ms', 0):8.1f} ms  "
                      f"decode/tok={tm.get('decode_per_run_ms', 0):6.2f} ms", flush=True)
    return rows


def jfk_sweep(threads=6):
    rows = []
    for size in SIZES:
        for quant in QUANTS:
            m = model_path(size, quant)
            text = run([BIN / "whisper-cli.exe", "-m", m, "-f", JFK, "-t", threads, "-nt"])
            transcript = " ".join(l.strip() for l in text.splitlines()
                                  if l.strip() and not re.match(r"^[\w ]+:|^\s*$|^read_audio|^main:", l.strip()))
            rows.append({"model": size, "quant": quant, "threads": threads,
                         "transcript": transcript, **parse_timings(text)})
            print(f"{size:5} {quant:5}  {transcript}", flush=True)
    return rows


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    write_csv(RESULTS / "week1_jfk.csv", jfk_sweep())
    write_csv(RESULTS / "week1_bench.csv", bench_sweep())
