"""Paths, model lists and helpers shared by the scripts."""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "third_party/whisper.cpp/build/bin/RelWithDebInfo"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

SIZES = ["base", "small"]
QUANTS = ["f16", "q8_0", "q5_1", "q5_0", "q4_k", "q4_0"]

# e.g. "whisper_print_timings:   encode time =   828.18 ms /     1 runs (   828.18 ms per run)"
TIMING_RE = re.compile(r"whisper_print_timings:\s+(\w+) time =\s+([\d.]+) ms(?: /\s+(\d+) runs \(\s*([\d.]+) ms per run\))?")


def parse_timings(text):
    out = {}
    for name, total, runs, per_run in TIMING_RE.findall(text):
        out[f"{name}_ms"] = float(total)
        if per_run:
            out[f"{name}_per_run_ms"] = float(per_run)
    return out


def model_path(size, quant):
    return MODELS / (f"ggml-{size}.bin" if quant == "f16" else f"ggml-{size}-{quant}.bin")


def exe(name):
    p = BIN / f"{name}.exe"
    if not p.exists():
        sys.exit(f"{p} not found; build whisper.cpp first (see README)")
    return p


def run(cmd):
    """Run a command, exit on failure, return (stdout, stderr)."""
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        sys.exit(f"command failed ({p.returncode}): {cmd}\n{p.stderr[-2000:]}")
    return p.stdout, p.stderr
