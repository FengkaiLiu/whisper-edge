"""Download the evaluation sets and write the subset manifests.

  en  LibriSpeech test-clean           (openslr.org)
  zh  FLEURS cmn_hans_cn test          (huggingface.co/datasets/google/fleurs)
  ja  FLEURS ja_jp test

Archives go to data/raw/, audio is extracted under data/, and the chosen subset
(fixed seed, so it is reproducible) is written to manifests/<lang>.jsonl.
Usage: py scripts/prepare_data.py [--n-en 200] [--n-zh 100] [--n-ja 100]
"""
import argparse
import json
import random
import shutil
import tarfile
import urllib.request

import soundfile as sf

from common import ROOT

DATA = ROOT / "data"
RAW = DATA / "raw"
MANIFESTS = ROOT / "manifests"
SEED = 0

FLEURS = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/{cfg}/{path}"
SOURCES = {
    "librispeech_test-clean.tar.gz": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "fleurs_cmn_hans_cn_test.tsv": FLEURS.format(cfg="cmn_hans_cn", path="test.tsv"),
    "fleurs_cmn_hans_cn_test.tar.gz": FLEURS.format(cfg="cmn_hans_cn", path="audio/test.tar.gz"),
    "fleurs_ja_jp_test.tsv": FLEURS.format(cfg="ja_jp", path="test.tsv"),
    "fleurs_ja_jp_test.tar.gz": FLEURS.format(cfg="ja_jp", path="audio/test.tar.gz"),
}


def fetch(name):
    dst = RAW / name
    if not dst.exists():
        print(f"fetch  {name} ...", flush=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        with urllib.request.urlopen(SOURCES[name]) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.rename(dst)
    return dst


def extract(archive, dest):
    """Extract once; a marker file records that extraction finished."""
    marker = dest / ".extracted"
    if marker.exists():
        return
    print(f"untar  {archive.name} -> {dest.relative_to(ROOT)}", flush=True)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")  # "data" filter rejects absolute paths, .., links outside dest
    marker.touch()


def librispeech(n):
    dest = DATA / "librispeech"
    extract(fetch("librispeech_test-clean.tar.gz"), dest)
    utts = []
    for trans in sorted((dest / "LibriSpeech/test-clean").rglob("*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            uid, text = line.split(" ", 1)
            utts.append({"id": uid, "audio": trans.parent / f"{uid}.flac", "text": text})
    chosen = sorted(random.Random(SEED).sample(utts, n), key=lambda u: u["id"])
    for u in chosen:
        u["duration"] = round(sf.info(u["audio"]).duration, 3)
    return chosen


def fleurs(cfg, n):
    dest = DATA / "fleurs" / cfg
    extract(fetch(f"fleurs_{cfg}_test.tar.gz"), dest)
    wavs = {p.name: p for p in dest.rglob("*.wav")}
    # columns: id, file, raw transcription, normalized transcription, chars, num_samples, gender.
    # The same sentence is read by several speakers; keep one recording per sentence.
    by_sentence = {}
    for line in fetch(f"fleurs_{cfg}_test.tsv").read_text(encoding="utf-8").splitlines():
        cols = line.split("\t")
        if cols[1] in wavs and cols[0] not in by_sentence:
            by_sentence[cols[0]] = {"id": f"{cfg}-{cols[0]}", "audio": wavs[cols[1]], "text": cols[2],
                                    "duration": round(int(cols[5]) / 16000, 3)}
    utts = sorted(by_sentence.values(), key=lambda u: u["id"])
    return sorted(random.Random(SEED).sample(utts, n), key=lambda u: u["id"])


def write_manifest(lang, utts):
    MANIFESTS.mkdir(exist_ok=True)
    path = MANIFESTS / f"{lang}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for u in utts:
            row = {"id": u["id"], "lang": lang, "audio": u["audio"].relative_to(ROOT).as_posix(),
                   "duration": u["duration"], "text": u["text"]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    hours = sum(u["duration"] for u in utts) / 3600
    print(f"wrote  {path.relative_to(ROOT)}: {len(utts)} utterances, {hours * 60:.1f} min of audio")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-en", type=int, default=200)
    ap.add_argument("--n-zh", type=int, default=100)
    ap.add_argument("--n-ja", type=int, default=100)
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    write_manifest("en", librispeech(args.n_en))
    write_manifest("zh", fleurs("cmn_hans_cn", args.n_zh))
    write_manifest("ja", fleurs("ja_jp", args.n_ja))
