"""Download the f16 models and generate every quantized variant.

Files that already exist are skipped, so this is safe to re-run.
Usage: py -3.11 scripts/setup_models.py
"""
import shutil
import urllib.request

from common import MODELS, QUANTS, SIZES, exe, model_path, run

URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{size}.bin"


def download(size):
    dst = model_path(size, "f16")
    if dst.exists():
        print(f"skip   {dst.name} (exists)")
        return
    print(f"fetch  {dst.name} ...", flush=True)
    tmp = dst.with_suffix(".part")
    with urllib.request.urlopen(URL.format(size=size)) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.rename(dst)
    print(f"done   {dst.name} ({dst.stat().st_size / 2**20:.1f} MB)")


def quantize(size, quant):
    dst = model_path(size, quant)
    if dst.exists():
        print(f"skip   {dst.name} (exists)")
        return
    run([exe("whisper-quantize"), model_path(size, "f16"), dst, quant])
    print(f"quant  {dst.name} ({dst.stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    MODELS.mkdir(exist_ok=True)
    for size in SIZES:
        download(size)
        for quant in QUANTS[1:]:
            quantize(size, quant)
