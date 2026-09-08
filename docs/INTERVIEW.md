# Interview playbook

Everything to have ready when presenting SafeSight in a data science /
ML engineering interview. Read `DECISIONS.md` once before any interview;
this file is the delivery script.

---

## 1. The 30-second pitch

"I took a research model I built - RQSA-MIL, a weakly-supervised nudity
detector with IoU-conditioned attention, 93% AUC - and shipped it as a real
product: a moderation API and a Chrome extension that blurs NSFW images as
you browse. The interesting parts are the inference system - a cascade that
decides per image how much compute it deserves - and the MLOps around it:
model registry, score-drift monitoring, a user feedback loop, a golden-set
regression gate in CI, and ONNX int8 quantization to make it fast on free
CPU hosting."

## 2. The 3-minute walkthrough (whiteboard order)

1. **Problem.** Detect nudity in arbitrary web images. Hard because
   deployment images are 100px-8000px while the model sees 224x224 regions,
   and because labels are weak (bag-level, not pixel-level) - that is the
   MIL setting.
2. **Model.** MobileNetV2 backbone; a spatial attention gate conditioned on
   region quality (IoU with annotations during training, pseudo-IoU 0.5 at
   test time); weighted MIL loss + attention diversity regularizer.
   87.9% recall, 85.5% specificity, 93.0% AUC on dense evaluation.
3. **Inference system.** Normalize short side to 690px, slide windows at the
   scale the model was trained on, aggregate per-tier top-3 means, dual
   calibrated thresholds. Then the cascade: full-frame look -> fast tier ->
   accurate tier, with asymmetric early exits (draw the diagram from
   `ml/strategies.py`).
4. **Product.** FastAPI service (score by upload or URL batch, explain
   endpoint returning evidence heatmaps); Chrome extension that pre-blurs
   images, batches URLs, applies a user-set sensitivity to `score_large`,
   click-to-reveal feeds the feedback queue.
5. **MLOps.** MLflow registry with the full baseline lineage; Prometheus
   score histogram + Grafana; Evidently/KS drift reports from the score log;
   golden-set gate; GitHub Actions -> Docker -> HF Spaces; ONNX fp32 for CPU
   (3.2x faster than eager torch - and the int8 story, section 4).

## 3. Numbers to memorize

| Number | Value | Where it comes from |
|---|---|---|
| Recall / Specificity / AUC | 87.9% / 85.5% / 93.0% | paper, dense eval on DS1 test |
| TPR at 1% FPR | 48.7% | paper |
| Attention overhead | 346K params, ~10% of 3.5M | model card |
| Decision thresholds | 0.50 large tier, 0.78 small tier | calibration in research repo |
| Cascade exits | NSFW >= 0.85 full-frame; safe <= 0.05 only if short side <= 400px; fast escalates in 0.35-0.65 | `ml/strategies.py` |
| fast vs accurate patches (800x600) | 24 vs 340 | measured |
| fast vs accurate CPU latency (800x600) | ~1.0s vs ~12s | measured, docs/benchmarks.md |
| cascade on small safe image | 1 patch, ~70ms (vs 2314ms accurate) | measured |
| soft-NMS vectorization win | accurate preset 39-44% faster (19.6s -> 12.0s medium) | measured |
| ONNX fp32 speedup over torch (CPU, 24 patches) | ~2.5x (315ms vs 780ms) | measured |
| int8 surprise | 3.7x smaller (2.8 vs 10.3 MB) but ~5x SLOWER than fp32 (1603ms) on this CPU | measured |
| int8 parity | max probability deviation ~0.04 | measured on structured inputs (see docs/benchmarks.md note: not reproduced by `ml.benchmark`, treat as illustrative) |
| Test suite | 34 tests, ruff clean, CI on every push | repo (`grep -c "^def test_" tests/*.py`; **Correction**: earlier draft said 31, actual count across `tests/*.py` is 34, matching this file's own section 6) |

## 4. "Why did you..." - the answers that matter

Each of these is a full decision record in `DECISIONS.md`; these are the
spoken versions.

**Why sliding windows instead of just classifying the image?**
"Because the model was trained on 200-330px crops of 690px images. On a
4000px photo, a nude region can be under 1% of the frame - resize the whole
thing to 224 and the evidence is gone. Our MobileNet-BCE baseline shows the
failure: it hits 100% recall but 27% specificity, because whole-frame
training turns it into a skin detector. Sliding windows restore the
region-to-input scale the model expects."

**Why not an object detector?**
"That inverts the research premise. Box annotations are expensive; the point
of MIL is learning from cheap bag-level labels. We had boxes on only one
training dataset - used to compute IoU weights for the attention gate - not
at deployment scale. And the published numbers are for this model."

**Why the cascade? Walk me through it.**
"Most images are obviously fine or obviously not - paying 340 patches on all
of them is waste. Stage one scores the full frame once: if it screams NSFW
at 0.85+, denser scanning only agrees, so we exit. The subtle part is the
asymmetry: a *low* full-frame score is only trustworthy on *small* images,
where the full frame is at training scale. On large images low-score
full-frame is precisely the known failure mode, so we never take that
shortcut and fall through to the sliding window. The fast tier then decides
unless it lands in its uncertainty band around the 0.50 threshold, in which
case the image has earned the paper-grade scan. Typical traffic costs 1-25
patches; only genuinely ambiguous images cost 365."

**Why did you rebuild the extension around per-image URLs?**
"The first version scraped the whole page server-side with Selenium and
blocked the entire page on one verdict. Fragile, slow, and heavy-handed.
Sending image URLs means tiny requests, per-image blurring, and global
caching - the same CDN image seen by any user is scored once. The trade-off
I accepted: some CDNs block server-side fetches and auth-protected images
are unreachable - those fail open, and a pixel-upload fallback is the next
iteration."

**Why fail open?**
"It is an assistive filter, not a compliance control. Breaking every image
on the web when a free-tier server cold-starts is worse than briefly losing
protection. For a parental-control product I would flip it - it is one
branch in the content script - and I documented that trade-off instead of
hiding it."

**How do you monitor a model with no production labels?**
"The score distribution is the leading indicator. Every verdict feeds a
Prometheus histogram and a JSONL log; a drift job compares recent scores to
a reference window - Evidently when installed, a two-sample KS test
otherwise. Distribution shift catches input drift before anyone complains.
For actual labels, the extension's click-to-reveal doubles as 'this was
wrong' feedback into a labeled queue - sparse, biased toward false
positives, but real."

**What stops a bad model or code change from shipping?**
"Three layers: deterministic inference tests in CI catch aggregation bugs;
a golden-set gate fails the release if recall or specificity regress below
thresholds - it runs locally because positive samples cannot live in a
public repo, which I will happily discuss as a real constraint; and the
registry keeps every baseline so rollback is a stage change, not an
archaeology project."

**Why ONNX? And why NOT int8?** (tell this story - it lands)
"Deployment is a 2-vCPU free tier, so I exported to ONNX and measured:
fp32 ONNX runs the fast-preset workload ~2.5x faster than eager PyTorch.
Then the textbook next step - int8 dynamic quantization - and here is the
part I like telling: the int8 model is 3.7x smaller but ran five times
SLOWER than fp32 on my hardware, because dynamic-quantized convolutions
fall back to ConvInteger kernels that have no fast path on CPUs without
VNNI. So I shipped fp32 ONNX, kept int8 as a size-optimized artifact, and
left the benchmark script in the repo so the decision re-derives itself on
any new host. The lesson I actually demonstrate with this: optimization
folklore gets re-measured on the target, always."

**How did you fit this on a free 512MB host?**
"By making the serving container torch-free. The inference pipeline -
transforms, sliding windows, soft-NMS, aggregation - never needed autograd,
so I rewrote it on numpy and put the frameworks behind a one-method backend
interface: score_regions(array) -> probabilities. The ONNX backend has no
torch dependency at all; the torch backend imports lazily and only exists
for local GPU use and the export tooling. Measured: ~317MB RSS under load
(a manual observation, not reproduced by a committed script - say so if
pressed on it), identical verdicts to the torch pipeline, same test suite
passes on both backends. Also a nice side effect: the requirements file for
serving is eleven small packages instead of a 2GB framework."

**Where else did measurement drive optimization?**
"The reference soft-NMS was a Python double loop over window pairs -
O(N^2) with tensor churn per pair. I replaced it with one vectorized
pairwise IoU matrix (torchvision box_iou) plus a per-rank vector decay -
same algorithm, identical outputs, and the accurate preset got 39-44%
faster end to end (19.6s to 12.0s on a medium image). Also chunked the
patch stacking so peak memory is one batch instead of a 200 MB tensor,
which matters on a 2-vCPU host. And in the serving path, URL downloads in
a batch now run concurrently under a bounded semaphore - I/O parallelizes
even when compute cannot - so a batch costs max(fetch) + sum(inference)
instead of the sum of both."

**Your API fetches arbitrary URLs. What could go wrong?**
"SSRF - the classic risk of any URL-fetching service. Someone submits
http://169.254.169.254 and reads cloud metadata through my server. The
fetcher is a dedicated module with layered defenses: scheme allowlist, DNS
resolution must yield only globally-routable addresses, and - the part most
implementations miss - redirects are followed manually with every hop
re-validated, because the easy bypass is a public URL that 302s to a
private address. Responses stream against a hard byte cap so a lying
Content-Length cannot force a giant allocation. Each defense has its own
unit test, including one asserting the private redirect hop is never
requested."

**How would you scale this to 1000 RPS?**
"The service is stateless except for an in-process cache, so first:
replicas behind a load balancer, cache moved to Redis, and the fetch layer
separated from the inference layer so I/O and compute scale independently.
At that point cross-request dynamic batching starts paying, which is when
I would move inference to Triton - the code already isolates inference
behind one function to make that migration cheap. The cascade matters more
at scale, not less: it is effectively per-request compute budgeting."

**How would you retrain?**
"The feedback queue plus drift reports tell me when and on what. Hard cases
(user-corrected false positives, drifted score bands) get labeled and folded
into the training sets in the research repo; the multiseed training script
reruns; the candidate goes through the golden gate and dense eval; MLflow
registers it; the Space redeploys on merge. The loop is designed, the
automation of the label-to-training handoff is the honest 'not built yet'."

## 5. Limitations to own before they ask

Raising these yourself is worth more than defending them later:

- Stylized content (anime, drawings) is out of distribution.
- The cascade exit thresholds are hand-calibrated, not learned; a small
  router model is the obvious upgrade.
- CI's golden gate runs on fixtures publicly; the real gate is
  pre-release-local because of data sensitivity.
- Single-replica cache; Redis is the known next step.
- Feedback is biased (users only report false positives - nobody clicks
  'this should have been blurred' on an image they wanted to see).
- The model detects nudity, not consent, age, or context - it must feed
  human review, not automatic punishment. Saying this unprompted signals
  responsible-ML maturity.

## 6. Five-minute live demo script

Open http://127.0.0.1:8000/demo (or the deployed URL) - the whole demo
lives on one page. Start the server beforehand: `powershell scripts/run.ps1`.

1. **The console header** (10s): "ONNX backend, CPU, cascade strategy" -
   the deployment story in one line.
2. **Score a URL** (1 min): paste any image URL, point at the verdict card:
   score, large-tier score, and the cascade path chips - "this image exited
   at full_frame: one patch, done in 80ms; ambiguous ones escalate."
   Score it again: cache hit, instant.
3. **The explain heatmap** (1 min): upload an image, show original vs
   evidence overlay side by side. "This is why the verdict fired - actual
   window scores, not a saliency approximation." The moment interviewers
   remember.
4. **The stat tiles** (30s): they updated live while you demoed - volume,
   NSFW rate, cache hit rate, latency p50/p95, cascade exit mix. "The same
   numbers feed Prometheus; the score distribution is my drift alarm."
5. **The extension** (1.5 min): browse an image-heavy page, images unblur
   as verdicts land; drag the sensitivity slider - "re-thresholds on
   score_large client-side, no re-scoring, thanks to the cache." Click
   reveal on a flagged image: "that just recorded feedback -
   mlops/review_feedback.py turns these into a retraining queue."
6. **The engineering** (1 min): flash /docs (OpenAPI), `git log`, CI run,
   and docs/benchmarks.md. Mention SAFESIGHT_API_KEY auth and the SSRF
   test that asserts a malicious redirect is never followed.

Fallbacks if wifi dies: upload files instead of URLs (everything but
score-urls works offline), `pytest -q` (34 tests), and the benchmarks
table tell the same story.

## 7. If they ask about the research itself

- MIL formulation: an image is a bag of region instances; only the bag label
  is known. Weighted MIL uses IoU-derived weights on positive regions so
  high-quality regions dominate the loss.
- The novelty: conditioning spatial attention *on the region's quality
  score*. Low-IoU (mostly background) regions get aggressive suppression,
  high-IoU regions gentle attention. At test time no IoU exists, so a
  neutral pseudo-IoU of 0.5 is fed - ablated in the paper.
- Attention diversity loss prevents all regions in a bag from attending to
  the same pattern (attention collapse).
- Versus baselines: +7.9 points recall over the same backbone with a plain
  spatial gate, +2.8 AUC - the attention conditioning, not capacity, is
  doing the work (parameter overhead is ~10%).
