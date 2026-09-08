"""Review the user-feedback queue and export a retraining candidate list.

Closes the human-in-the-loop story: the extension and demo record user
corrections into logs/feedback.jsonl; this tool joins them with the score
log, summarizes disagreement, and exports the cases worth labeling for the
next training round.

Usage:
    python mlops/review_feedback.py [--scores logs/scores.jsonl]
        [--feedback logs/feedback.jsonl] [--out mlops/retraining_queue.csv]
"""

import argparse
import csv
import json
import os
from collections import Counter


def load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", default="logs/scores.jsonl")
    parser.add_argument("--feedback", default="logs/feedback.jsonl")
    parser.add_argument("--out", default="mlops/retraining_queue.csv")
    args = parser.parse_args()

    scores = load_jsonl(args.scores)
    feedback = load_jsonl(args.feedback)

    by_hash = {}
    for row in scores:
        if row.get("url_hash"):
            by_hash[row["url_hash"]] = row

    print(f"Score log entries   : {len(scores)}")
    print(f"Feedback entries    : {len(feedback)}")

    if not feedback:
        print("No feedback recorded yet; nothing to review.")
        return

    disagreements = []
    labels = Counter()
    for fb in feedback:
        labels[fb.get("user_label")] += 1
        scored = by_hash.get(fb.get("url_hash"), {})
        model_label = scored.get("label")
        if model_label is not None and model_label != fb.get("user_label"):
            disagreements.append({
                "url_hash": fb.get("url_hash"),
                "model_label": model_label,
                "model_score": fb.get("model_score"),
                "user_label": fb.get("user_label"),
                "preset": scored.get("preset", ""),
                "ts": fb.get("ts"),
            })

    print(f"User labels         : safe={labels.get(0, 0)} nsfw={labels.get(1, 0)}")
    print(f"Model disagreements : {len(disagreements)} "
          "(these are the retraining candidates)")

    if disagreements:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(disagreements[0].keys()))
            writer.writeheader()
            writer.writerows(disagreements)
        print(f"Exported retraining queue -> {args.out}")
        print("Next: label these cases and fold them into the training sets "
              "in the research repo (D:\\nude-MIL), then rerun training and "
              "the golden-set gate.")


if __name__ == "__main__":
    main()
