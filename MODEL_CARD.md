# Model card: RQSA-MIL v1

## Summary

RQSA-MIL (Region-Quality Supervised Attention for Multiple Instance Learning)
is a nudity detection model: a MobileNetV2 backbone with a spatial attention
gate conditioned on region quality. During training the gate receives the
IoU between each region and ground-truth annotations; at inference it
receives a pseudo-IoU of 0.5. Trained with weighted MIL on bag-level labels
plus an attention diversity regularizer.

- Architecture: MobileNetV2 (3.5M params) + IoU-conditioned attention
  (346K params, ~10% overhead) + linear head
- Input: 224x224 crops produced by dense sliding-window inference over
  images normalized to a 690px short side
- Output: per-region probability; image verdict via per-tier top-3 mean
  with calibrated dual thresholds (large tier 0.50, small tier 0.78)

## Intended use

Defensive content moderation: blurring or flagging NSFW imagery in browsers,
user-generated-content platforms, or safety pipelines. Not intended for
punitive automated decisions about individuals without human review.

## Performance (DS1 test set, dense evaluation)

| Metric | Value |
|---|---|
| Recall | 87.89% |
| Specificity | 85.52% |
| AUC | 92.98% |
| TPR at 1% FPR | 48.65% |

## Thresholding guidance

- `score` = max(large-tier, small-tier) top-3 mean. Use as a UI confidence.
- `score_large` is the recommended single-knob threshold (ROC/AUC numbers are
  computed on it). Lower threshold = stricter blocking, more false positives.
- The `fast` preset trades some sensitivity for ~8x fewer patches; production
  CPU deployments use it for interactive latency.

## Limitations

- Trained primarily on photographic content; drawings, anime, and heavily
  stylized imagery are out of distribution.
- Small embedded thumbnails (under ~140px on the normalized image) are scored
  by the noisier small-window tier only.
- Skin-tone-heavy but safe content (swimwear, dermatology, classical art) can
  produce elevated scores; keep a human review path for contested verdicts.
- The model detects nudity, not consent, age, or context. It must not be used
  as the sole basis for content takedowns or accusations.

## Training data

Trained on a bounding-box-annotated nudity detection dataset (positives) and
a nude/non-nude classification dataset (negatives). Datasets are not
redistributed with this repository.
