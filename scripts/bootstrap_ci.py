"""Is a quantized model really worse than f16, or is it noise?

Paired bootstrap over utterances: resample the utterance set with replacement,
recompute corpus WER/CER for both the quantized model and the f16 model of the
same size on that same resample, and take the 2.5 / 97.5 percentiles of the
difference. If the interval excludes 0 the difference is significant at ~95%.

Reads the per-utterance results/hyp/<tag>/<lang>/<model>.jsonl written by
evaluate.py and writes results/<tag>_delta.csv.
Usage: py scripts/bootstrap_ci.py [--tag week2_cpu] [--n 2000]
"""
import argparse
import csv
import json

import jiwer
import numpy as np

from common import QUANTS, RESULTS, SIZES


def per_utt_counts(path, lang):
    """(errors, reference length) per utterance, in manifest order."""
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    out = []
    for r in rows:
        if not r["ref_norm"]:
            continue
        if lang == "en":
            o = jiwer.process_words(r["ref_norm"], r["hyp_norm"])
            n = len(r["ref_norm"].split())
        else:
            o = jiwer.process_characters(r["ref_norm"], r["hyp_norm"])
            n = len(r["ref_norm"])
        out.append((o.substitutions + o.deletions + o.insertions, n))
    return np.array(out, dtype=float)


def paired_ci(base, quant, n_boot, rng):
    idx = rng.integers(0, len(base), size=(n_boot, len(base)))
    rate = lambda c: c[idx, 0].sum(1) / c[idx, 1].sum(1)
    d = rate(quant) - rate(base)
    return np.percentile(d, [2.5, 97.5])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="week2_cpu")
    ap.add_argument("--n", type=int, default=2000, help="bootstrap resamples")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    hyp = RESULTS / "hyp" / args.tag
    rows = []
    for lang in ["en", "zh", "ja"]:
        if not (hyp / lang).exists():
            continue
        for size in SIZES:
            base = per_utt_counts(hyp / lang / f"{size}-f16.jsonl", lang)
            base_err = base[:, 0].sum() / base[:, 1].sum()
            for quant in QUANTS[1:]:
                q = per_utt_counts(hyp / lang / f"{size}-{quant}.jsonl", lang)
                err = q[:, 0].sum() / q[:, 1].sum()
                lo, hi = paired_ci(base, q, args.n, rng)
                rows.append({"lang": lang, "model": f"{size}-{quant}", "err": round(err, 4),
                             "f16_err": round(base_err, 4), "delta": round(err - base_err, 4),
                             "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "significant": lo > 0 or hi < 0})
                print(f"{lang} {size}-{quant:5}  delta={100 * (err - base_err):+5.2f} pt  "
                      f"95% CI [{100 * lo:+5.2f}, {100 * hi:+5.2f}]{'  *' if lo > 0 or hi < 0 else ''}")

    with open(RESULTS / f"{args.tag}_delta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
