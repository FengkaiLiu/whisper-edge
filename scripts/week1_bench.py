"""Week 1 first-pass measurements.

  jfk    whisper-cli on samples/jfk.wav per model (transcript sanity check)
         -> results/week1_jfk.csv
  bench  whisper-bench (encoder / decoder timing) over every model x thread count
         -> results/week1_bench.csv
  micro  whisper-bench memcpy and ggml_mul_mat microbenchmarks
         -> results/week1_microbench.txt

Usage: py -3.11 scripts/week1_bench.py [--steps jfk bench micro]
"""
import argparse
import csv

from common import QUANTS, RESULTS, ROOT, SIZES, exe, model_path, parse_timings, run

JFK = ROOT / "third_party/whisper.cpp/samples/jfk.wav"
THREADS = [1, 2, 4, 6, 8, 12]


def bench_sweep():
    rows = []
    for size in SIZES:
        for quant in QUANTS:
            m = model_path(size, quant)
            for t in THREADS:
                out, err = run([exe("whisper-bench"), "-m", m, "-t", t])
                tm = parse_timings(out + err)
                rows.append({"model": size, "quant": quant, "threads": t,
                             "file_mb": round(m.stat().st_size / 2**20, 1), **tm})
                print(f"{size:5} {quant:5} t={t:2}  encode={tm.get('encode_ms', 0):8.1f} ms  "
                      f"decode/tok={tm.get('decode_per_run_ms', 0):6.2f} ms", flush=True)
    return rows


def jfk_sweep(threads=6):
    rows = []
    for size in SIZES:
        for quant in QUANTS:
            # with -nt the transcript is the only thing on stdout; logs and timings go to stderr
            out, err = run([exe("whisper-cli"), "-m", model_path(size, quant), "-f", JFK, "-t", threads, "-nt"])
            transcript = " ".join(l.strip() for l in out.splitlines() if l.strip())
            rows.append({"model": size, "quant": quant, "threads": threads,
                         "transcript": transcript, **parse_timings(err)})
            print(f"{size:5} {quant:5}  {transcript}", flush=True)
    return rows


def microbench(threads=6):
    sections = []
    for what, title in [(1, "memcpy"), (2, "ggml_mul_mat")]:
        print(f"micro  {title} ...", flush=True)
        out, err = run([exe("whisper-bench"), "-w", what, "-t", threads])
        sections.append(f"## {title} (-w {what}, t={threads})\n{out}{err}")
    return "\n".join(sections)


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="+", choices=["jfk", "bench", "micro"], default=["jfk", "bench", "micro"])
    steps = ap.parse_args().steps

    RESULTS.mkdir(exist_ok=True)
    if "jfk" in steps:
        write_csv(RESULTS / "week1_jfk.csv", jfk_sweep())
    if "bench" in steps:
        write_csv(RESULTS / "week1_bench.csv", bench_sweep())
    if "micro" in steps:
        (RESULTS / "week1_microbench.txt").write_text(microbench(), encoding="utf-8")
