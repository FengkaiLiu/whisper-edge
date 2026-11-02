"""Same model, two runs (e.g. CPU vs CUDA): do they transcribe the same text?

For every (language, model) present under both results/hyp/<a>/ and
results/hyp/<b>/ it reports the share of utterances whose normalized
transcripts are identical, both corpus error rates, and a paired-bootstrap
95% CI of the difference (b - a). Speed is compared from results/<tag>.csv.

Writes results/<a>_vs_<b>.csv.
Usage: py scripts/compare_tags.py week2_cpu week2_cuda [--n 2000]
"""
import argparse
import csv
import json

import numpy as np

from bootstrap_ci import paired_ci, per_utt_counts
from common import QUANTS, RESULTS, SIZES


def load_rtf(tag):
    with open(RESULTS / f"{tag}.csv", encoding="utf-8") as f:
        return {(r["lang"], r["model"]): float(r["rtf"]) for r in csv.DictReader(f) if r["repeat"] == "0"}


def hyps(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l)["hyp_norm"] for l in f]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--n", type=int, default=2000, help="bootstrap resamples")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    rtf_a, rtf_b = load_rtf(args.a), load_rtf(args.b)
    rows = []
    for lang in ["en", "zh", "ja"]:
        for model in [f"{s}-{q}" for s in SIZES for q in QUANTS]:
            pa = RESULTS / "hyp" / args.a / lang / f"{model}.jsonl"
            pb = RESULTS / "hyp" / args.b / lang / f"{model}.jsonl"
            if not (pa.exists() and pb.exists()):
                continue
            ha, hb = hyps(pa), hyps(pb)
            same = sum(x == y for x, y in zip(ha, hb)) / len(ha)
            ca, cb = per_utt_counts(pa, lang), per_utt_counts(pb, lang)
            ea, eb = ca[:, 0].sum() / ca[:, 1].sum(), cb[:, 0].sum() / cb[:, 1].sum()
            lo, hi = paired_ci(ca, cb, args.n, rng)
            sig = lo > 0 or hi < 0
            rows.append({"lang": lang, "model": model, "identical": round(same, 3),
                         "err_a": round(ea, 4), "err_b": round(eb, 4), "delta": round(eb - ea, 4),
                         "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "significant": sig,
                         "rtf_a": rtf_a[(lang, model)], "rtf_b": rtf_b[(lang, model)],
                         "speedup": round(rtf_a[(lang, model)] / rtf_b[(lang, model)], 1)})
            print(f"{lang} {model:11} identical={100 * same:5.1f}%  delta={100 * (eb - ea):+5.2f} pt  "
                  f"95% CI [{100 * lo:+5.2f}, {100 * hi:+5.2f}]{'  *' if sig else '   '}  "
                  f"speedup={rows[-1]['speedup']:.1f}x")

    out = RESULTS / f"{args.a}_vs_{args.b}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out.relative_to(RESULTS.parent)}")
