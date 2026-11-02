"""Accuracy / speed / memory evaluation of whisper.cpp models.

For every (language, model, threads, repeat) it runs whisper-cli over the
utterances in manifests/<lang>.jsonl and records:

  err        WER for en, CER for zh / ja (after text normalization)
  rtf        (total - load) time / audio duration, from whisper's own timings
  rtf_wall   process wall time / audio duration (includes model load)
  peak_mb    peak working set of the whisper-cli process
  load_ms, encode_ms, decode_ms, ...   per-utterance averages

Utterances are processed in batches inside one whisper-cli process, so the
model is loaded once per batch rather than once per file. Rows are appended to
results/<tag>.csv; configurations already present there are skipped, so an
interrupted run can be resumed by re-running the same command.

Usage:
  py scripts/evaluate.py --tag week2_cpu --langs en zh ja --threads 6
  py scripts/evaluate.py --tag smoke --langs en --models base-q4_k --limit 5
"""
import argparse
import csv
import json
import subprocess
import tempfile
import time
import unicodedata
from pathlib import Path

import jiwer
import opencc
import psutil
from whisper_normalizer.english import EnglishTextNormalizer

from common import BIN, QUANTS, RESULTS, ROOT, SIZES, model_path, parse_timings

MANIFESTS = ROOT / "manifests"
ALL_MODELS = [f"{s}-{q}" for s in SIZES for q in QUANTS]
BATCH = 40  # files per whisper-cli process; keeps the command line well under the Windows limit
KEY = ("lang", "model", "threads", "repeat")

_en_norm = EnglishTextNormalizer()
_t2s = opencc.OpenCC("t2s")


def normalize(text, lang):
    if lang == "en":
        return _en_norm(text)
    # zh / ja: compare characters only. NFKC folds full-width forms, zh output is
    # converted to simplified (Whisper often answers in traditional characters).
    text = unicodedata.normalize("NFKC", text).lower()
    if lang == "zh":
        text = _t2s.convert(text)
    return "".join(c for c in text if unicodedata.category(c)[0] not in "PSZC")


def error_rate(refs, hyps, lang):
    pairs = [(r, h) for r, h in zip(refs, hyps) if r]
    refs, hyps = [p[0] for p in pairs], [p[1] for p in pairs]
    return jiwer.wer(refs, hyps) if lang == "en" else jiwer.cer(refs, hyps)


def run_measured(cmd):
    """Run cmd, return (wall seconds, peak working set MB, stderr text)."""
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        t0 = time.perf_counter()
        p = subprocess.Popen([str(c) for c in cmd], stdout=out, stderr=err)
        ps, peak = psutil.Process(p.pid), 0
        while p.poll() is None:
            try:
                peak = max(peak, ps.memory_info().peak_wset)  # Windows: monotonic peak, so polling misses nothing
            except psutil.NoSuchProcess:
                break
            time.sleep(0.05)
        p.wait()
        wall = time.perf_counter() - t0
        err.seek(0)
        stderr = err.read().decode("utf-8", errors="replace")
    if p.returncode != 0:
        raise SystemExit(f"whisper-cli failed ({p.returncode}):\n{stderr[-2000:]}")
    return wall, peak / 2**20, stderr


def evaluate(cli, lang, model, threads, utts, hyp_dir, extra_args):
    size, quant = model.split("-", 1)
    mpath = model_path(size, quant)
    hyp_dir.mkdir(parents=True, exist_ok=True)

    wall, peak, timings = 0.0, 0.0, {}
    for i in range(0, len(utts), BATCH):
        cmd = [cli, "-m", mpath, "-t", threads, "-l", lang, "-nt", "-otxt", *extra_args]
        for u in utts[i:i + BATCH]:
            cmd += ["-f", ROOT / u["audio"], "-of", hyp_dir / u["id"]]
        w, pk, stderr = run_measured(cmd)
        wall, peak = wall + w, max(peak, pk)
        for k, v in parse_timings(stderr).items():
            if not k.endswith("per_run_ms"):
                timings[k] = timings.get(k, 0.0) + v

    hyps, rows = [], []
    for u in utts:
        f = hyp_dir / f"{u['id']}.txt"
        hyp = f.read_text(encoding="utf-8").strip() if f.exists() else ""
        hyps.append(hyp)
        rows.append({"id": u["id"], "ref": u["text"], "hyp": hyp,
                     "ref_norm": normalize(u["text"], lang), "hyp_norm": normalize(hyp, lang)})
    with open(hyp_dir.parent / f"{model}.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)

    audio_s = sum(u["duration"] for u in utts)
    n = len(utts)
    return {
        "n_utts": n,
        "audio_s": round(audio_s, 1),
        "file_mb": round(mpath.stat().st_size / 2**20, 1),
        "metric": "wer" if lang == "en" else "cer",
        "err": round(error_rate([r["ref_norm"] for r in rows], [r["hyp_norm"] for r in rows], lang), 4),
        "rtf": round((timings["total_ms"] - timings["load_ms"]) / 1000 / audio_s, 4),
        "rtf_wall": round(wall / audio_s, 4),
        "peak_mb": round(peak, 1),
        **{k: round(timings.get(f"{k}_ms", 0.0) / n, 1) for k in ["load", "mel", "sample", "encode", "decode", "batchd", "prompt"]},
    }


def load_rows(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_rows(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="results/<tag>.csv and results/hyp/<tag>/")
    ap.add_argument("--langs", nargs="+", default=["en", "zh", "ja"], choices=["en", "zh", "ja"])
    ap.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    ap.add_argument("--threads", nargs="+", type=int, default=[6])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--limit", type=int, help="only the first N utterances of each manifest")
    ap.add_argument("--bin", type=Path, default=BIN, help="directory containing whisper-cli.exe")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra whisper-cli arguments (put last)")
    args = ap.parse_args()

    cli = args.bin / "whisper-cli.exe"
    out_csv = RESULTS / f"{args.tag}.csv"
    rows = load_rows(out_csv)
    done = {tuple(str(r[k]) for k in KEY) for r in rows}

    for lang in args.langs:
        with open(MANIFESTS / f"{lang}.jsonl", encoding="utf-8") as f:
            utts = [json.loads(l) for l in f][:args.limit]
        for model in args.models:
            for threads in args.threads:
                for rep in range(args.repeats):
                    key = (lang, model, str(threads), str(rep))
                    if key in done:
                        continue
                    hyp_dir = RESULTS / "hyp" / args.tag / lang / model
                    r = evaluate(cli, lang, model, threads, utts, hyp_dir, args.extra)
                    rows.append(dict(zip(KEY, key)) | r)
                    save_rows(out_csv, rows)
                    print(f"{lang} {model:11} t={threads:2} rep={rep}  {r['metric']}={r['err']:.4f}  "
                          f"rtf={r['rtf']:.3f}  peak={r['peak_mb']:.0f} MB", flush=True)
