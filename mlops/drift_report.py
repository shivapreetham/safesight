"""Generate a score-distribution drift report from the live score log.

Compares recent production scores (logs/scores.jsonl) against a reference
window and writes an HTML report. Uses Evidently if installed; otherwise
falls back to a self-contained KS-test summary so the pipeline has no hard
dependency.

Usage:
    python mlops/drift_report.py --log logs/scores.jsonl --out docs/drift_report.html
"""

import argparse
import json
import math
import os
from datetime import datetime


def load_scores(path):
    scores = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                scores.append(float(json.loads(line)["score"]))
            except (ValueError, KeyError, json.JSONDecodeError):
                continue
    return scores


def ks_statistic(a, b):
    a, b = sorted(a), sorted(b)
    i = j = 0
    d = 0.0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            i += 1
        else:
            j += 1
        d = max(d, abs(i / len(a) - j / len(b)))
    return d


def render_fallback(reference, current, out_path):
    d = ks_statistic(reference, current)
    # Two-sample KS critical value at alpha=0.05.
    n, m = len(reference), len(current)
    critical = 1.358 * math.sqrt((n + m) / (n * m))
    drifted = d > critical

    def stats(xs):
        mean = sum(xs) / len(xs)
        var = sum((x - mean) ** 2 for x in xs) / len(xs)
        return mean, math.sqrt(var)

    ref_mean, ref_std = stats(reference)
    cur_mean, cur_std = stats(current)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>SafeSight drift report</title>
<style>body{{font-family:system-ui;max-width:720px;margin:40px auto}}
.badge{{padding:4px 12px;border-radius:6px;color:#fff;
background:{'#c0392b' if drifted else '#27ae60'}}}
table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:6px 14px}}</style>
</head><body>
<h1>SafeSight score drift report</h1>
<p>Generated {datetime.now().isoformat(timespec='seconds')}</p>
<p>Verdict: <span class="badge">{'DRIFT DETECTED' if drifted else 'NO DRIFT'}</span></p>
<table>
<tr><th></th><th>Reference (n={n})</th><th>Current (n={m})</th></tr>
<tr><td>Mean score</td><td>{ref_mean:.4f}</td><td>{cur_mean:.4f}</td></tr>
<tr><td>Std</td><td>{ref_std:.4f}</td><td>{cur_std:.4f}</td></tr>
</table>
<p>KS statistic: {d:.4f} (critical value at alpha=0.05: {critical:.4f})</p>
</body></html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return drifted


def render_evidently(reference, current, out_path):
    import pandas as pd
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=pd.DataFrame({"score": reference}),
        current_data=pd.DataFrame({"score": current}),
    )
    report.save_html(out_path)
    result = report.as_dict()
    return bool(result["metrics"][0]["result"]["dataset_drift"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default="logs/scores.jsonl")
    parser.add_argument("--out", default="docs/drift_report.html")
    parser.add_argument("--split", type=float, default=0.5,
                        help="fraction of the log treated as reference window")
    args = parser.parse_args()

    scores = load_scores(args.log)
    if len(scores) < 20:
        print(f"Only {len(scores)} scores logged; need at least 20 for a report.")
        return

    cut = int(len(scores) * args.split)
    reference, current = scores[:cut], scores[cut:]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    try:
        drifted = render_evidently(reference, current, args.out)
        engine = "evidently"
    except ImportError:
        drifted = render_fallback(reference, current, args.out)
        engine = "builtin-ks"

    print(f"Report written to {args.out} (engine={engine}, drift={drifted})")


if __name__ == "__main__":
    main()
