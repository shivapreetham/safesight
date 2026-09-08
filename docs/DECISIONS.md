# Design decisions

Every significant choice in SafeSight, with the alternatives that were on the
table and why they lost. Format: context, options, decision, accepted costs.

---

## D1. Inference strategy: dense sliding windows with an early-exit cascade

**Context.** The model scores 224x224 regions and was trained on 200-330px
crops of 690px-normalized images. Production images range from 100px icons to
8000px photos. How do we turn region scores into an image verdict at
acceptable latency?

**Options considered.**
1. *Full-frame classification* (resize whole image to 224). Nearly free, but
   nudity occupying a small fraction of a high-res frame shrinks far below
   training scale and is missed. The MobileNet-BCE baseline demonstrates the
   failure: 100% recall / 27% specificity - it degenerates into a skin
   detector because whole frames were all it saw.
2. *Object detector (YOLO-style).* Would localize directly, but requires
   dense box annotations at deployment scale; the research premise was weak
   bag-level supervision (MIL) precisely because full box labeling is
   expensive. Also a different model, not the one with published numbers.
3. *Random crop sampling.* Cheaper than exhaustive sliding, but verdicts
   become non-deterministic. Moderation decisions must be reproducible for
   audits, and caching requires stable outputs.
4. *Multi-resolution pyramid of full frames.* Still scores whole frames, so
   the region-to-input ratio stays wrong; it downscales the evidence rather
   than isolating it.
5. *Dense sliding windows at training scale* (chosen, from the paper), plus
   a new *cascade* that runs cheap stages first and escalates only when
   uncertain.

**Decision.** Sliding windows for correctness; cascade for cost. The cascade
is asymmetric by design: full-frame may early-exit NSFW on extreme scores
(if nudity dominates a thumbnail-scale view, denser scanning only agrees),
but may early-exit SAFE only for small images where the full frame is
scale-appropriate. Large images always reach the sliding-window stage,
because a low full-frame score on a large image is exactly the known
failure mode. The fast tier escalates to the paper-grade accurate tier only
inside its uncertainty band (0.35-0.65 around the 0.50 threshold).

**Accepted costs.** Worst-case latency (ambiguous image) is the accurate
preset's full cost; thresholds for the exits are hand-calibrated rather than
learned (a learned router is a documented future step).

---

## D2. Two fixed presets instead of exposing raw knobs

**Context.** Window sizes, strides, TTA, NMS, and thresholds interact; the
paper numbers are valid only for one configuration.

**Options.** Expose all parameters per-request; expose nothing; or ship named
presets.

**Decision.** Two named presets - `accurate` (exact paper configuration, for
evaluation and the cascade's final stage) and `fast` (single 265px tier, no
TTA, ~8-15x fewer patches, for interactive serving) - plus `cascade`
combining them. Clients choose an intent, not a parameter soup, and the
served configuration is always one we have measured.

**Accepted costs.** Less flexibility for power users; changing a preset means
a code change and redeploy (which is the point - it forces re-benchmarking).

---

## D3. Extension sends image URLs; server fetches pixels

**Context.** The Chrome extension needs verdicts for every image on a page.

**Options.**
1. *Page-level verdict via server-side scraping* (the project's first
   iteration): Selenium loads the page server-side, scrapes text and images,
   returns one verdict; the extension blocks the whole page. Fragile
   (headless scraping breaks constantly), slow (full browser per request),
   heavy false positives (one bad thumbnail blocks the page), and it
   re-downloads content the browser already has.
2. *Extension uploads pixel data* of every image. Works for auth-protected
   images, but uploads user-viewed content (privacy), multiplies request
   size, and defeats cross-user caching.
3. *Extension sends URLs; server fetches and scores* (chosen).

**Decision.** Option 3: requests are tiny, verdicts cache globally by URL
hash + strategy (the same CDN image seen by any user is scored once), and
only URLs - already visible in page source - leave the browser. Per-image
granularity lets us blur one thumbnail instead of blocking a page.

Accepting URLs makes the server a fetch proxy, so the fetcher is a
dedicated hardened module (`service/fetcher.py`): scheme allowlist, DNS
resolution must yield only global addresses, every redirect hop re-validated
(a public URL cannot 302 into the metadata service), streamed reads with a
hard byte cap, content-type checks. Each defense has a unit test.

**Accepted costs.** Some CDNs (e.g. Wikimedia) block server-side fetches
regardless of user agent; those images fail open. Auth-protected images
cannot be fetched. A pixel-upload fallback for exactly these cases is the
documented next step.

---

## D4. Fail open in the extension

**Context.** What happens to blurred images when the API is unreachable?

**Options.** Keep everything blurred (fail closed) or unblur (fail open).

**Decision.** Fail open. This is an assistive consumer filter, not a
compliance control; breaking every image on the web when a hobby-tier server
sleeps is worse than temporarily losing protection, and the popup makes
protection status visible. A workplace/parental-control deployment should
flip this - it is one branch in `content.js` - and that trade-off is
deliberately documented rather than hidden.

---

## D5. Convert the checkpoint to a tensors-only format

**Context.** The research checkpoint pickled a training `Config` object, so
loading required `weights_only=False` plus a stub class injected into
`__main__` - arbitrary code execution risk and brittle deserialization.

**Options.** Keep the hack in serving code (what the first prototype did);
or convert once to a clean format.

**Decision.** A one-time conversion script produces a file containing only
tensors and JSON-safe metadata, loadable with `weights_only=True`. Serving
code never unpickles arbitrary objects, and the conversion script is the
single quarantined place that touches the legacy format.

**Accepted costs.** One extra artifact to manage; the original file is still
needed to re-convert.

---

## D6. CPU serving with ONNX int8, not GPU hosting

**Context.** Deployment target is Hugging Face Spaces free tier (2 vCPU).
GPU hosting starts at real money per month for a portfolio service.

**Options.** Pay for GPU; require self-hosting on the user's RTX 3060; or
optimize for CPU.

**Decision.** Optimize for CPU: MobileNetV2 was chosen by the research for
efficiency, and the graph exports cleanly to ONNX. Measured on the 24-patch
fast-preset workload: ONNX fp32 is ~2.5x faster than eager PyTorch (315ms vs
780ms). The plan was to serve int8 - but the measurement said no: despite
being 3.7x smaller (2.8 MB vs 10.3 MB), the dynamically quantized model ran
~5x SLOWER than fp32 (1603ms) because this CPU lacks fast ConvInteger kernel
paths. So the shipped decision is ONNX **fp32** for CPU serving, int8 kept
only as a size-optimized artifact, and the benchmark script kept in the repo
so the choice re-derives itself on any new host. The torch backend remains
the default locally (GPU-capable); the backend is a settings switch, not a
fork.

**Accepted costs.** int8 verdicts can deviate up to ~0.04 in probability
near the threshold if it is ever used; two model artifacts to version. The
benchmark and parity numbers are published in `docs/benchmarks.md` so the
trade is explicit - and the episode is a standing reminder that optimization
folklore must be re-measured per target.

---

## D7. FastAPI + threadpool + a model lock, not a batching inference server

**Context.** Serving needs async URL fetching (I/O bound) around synchronous
model inference (CPU bound).

**Options.** Triton/TorchServe (dynamic batching, model management);
Celery/queue workers; or FastAPI with inference in a threadpool guarded by a
lock.

**Decision.** FastAPI with `anyio.to_thread` and a single model lock. The
sliding-window pipeline already batches internally (up to 64 patches per
forward), so cross-request batching adds little; a lock serializes model
access, bounding memory on a 2-vCPU host. URL downloads run concurrently
under a bounded semaphore (I/O parallelizes; compute does not on this
host), so a batch costs roughly max(fetch) + sum(inference) instead of
sum(both). Triton is the right answer at fleet scale and is an easy
migration because inference is isolated behind one function.

**Accepted costs.** Single-image model throughput per replica; scaling is
horizontal (more replicas behind a balancer), which is the standard first
step anyway.

---

## D8. URL-hash LRU cache, in-process

**Context.** Pages repeat images (logos, CDNs, infinite scroll), and the
extension re-checks on every navigation.

**Options.** No cache; Redis; in-process LRU (chosen).

**Decision.** In-process LRU keyed by `sha256(url)[:32] + ":" + strategy`,
4096 entries, mirrored by a `chrome.storage` cache client-side (`service/cache.py`
`url_key()` + `service/app.py` `score_urls()`). Verdicts for a fixed model
version are immutable, making this the easiest correct cache. Redis becomes
necessary only with multiple replicas - noted, not built, because one
replica is the current reality.
**Correction**: earlier draft said the key was just `sha256(url)`; the actual
cache key also includes the scoring strategy/preset, so the same URL scored
under `fast` and `cascade` gets two separate entries.

**Accepted costs.** Cache lost on restart; no cross-replica sharing; entries
are not invalidated on model upgrade (mitigated: restart clears it).

---

## D9. Score-distribution histogram as the primary production metric

**Context.** Ground-truth labels do not exist in production, so accuracy
cannot be monitored directly. What signals drift?

**Options.** Log nothing until users complain; monitor only ops metrics
(latency/errors); or monitor the score distribution as a proxy.

**Decision.** Every verdict feeds a Prometheus histogram of scores plus a
JSONL log. A shift in the score distribution (mass moving toward the middle,
or the NSFW rate jumping) is the earliest observable symptom of input drift
or an upstream change. The drift job (Evidently, with a dependency-free KS
fallback) compares recent scores against a reference window and emits an
HTML report. User feedback from the extension supplies sparse true labels
for the cases the model got wrong.

**Accepted costs.** Score drift is a proxy - it can fire on benign traffic
shifts and miss compensating errors; the feedback queue exists precisely to
ground it.

---

## D10a. Torch-free serving build (numpy inference core)

**Context.** The chosen free host (Render, after Hugging Face put Docker
Spaces behind PRO) allots 512MB RAM. A container that imports PyTorch spends
most of that budget before the first request.

**Options.** Pay for a bigger host; keep torch and hope 512MB stretches; or
remove torch from the serving path entirely.

**Decision.** The inference pipeline (transforms, sliding windows, soft-NMS,
top-k aggregation) was rewritten on numpy + PIL - it never needed autograd
or GPU tensors. Backends now implement one method,
`score_regions(np.ndarray) -> np.ndarray`: `OnnxScorer` (no torch) and
`TorchScorer` (lazy torch import, GPU-capable, used locally and by the
export/benchmark tooling). Requirements split accordingly: `requirements.txt`
is the torch-free serving runtime; `requirements-torch.txt` layers torch on
top for dev. Measured result: the serving process runs at ~317MB RSS under
load - inside the free tier - and verdicts are numerically identical to the
torch pipeline (same test suite passes on both).
**Unverified**: no memory-profiling script exists in `scripts/` or `ml/` to
reproduce the ~317MB figure (the repo's setup/run scripts don't record RSS);
treat this as a manually-observed number rather than one traceable to a
committed measurement tool.

**Accepted costs.** Two backend adapters to maintain instead of one code
path; torchvision's transforms had to be reimplemented (PIL bilinear resize
+ normalize) and kept equivalent, which the deterministic and parity tests
now guard.

---

## D10. Golden-set gate in CI, with NSFW data kept out of the repo

**Context.** CI must stop a model or code change that silently degrades
detection, but positive samples cannot be committed to a public repository.

**Options.** No gate; commit an encrypted eval set; or split the gate.

**Decision.** The gate script (`ml/eval_golden.py`) computes recall and
specificity over a labeled directory and fails below thresholds. In public
CI it runs on checkpoint-independent tests only; the full gate with the
private golden set runs locally before a release (documented in the
runbook). Inference determinism tests in CI catch the class of bug -
aggregation changes, preset drift - that would invalidate the gate.

**Accepted costs.** The strongest check is manual-before-release rather than
automatic-on-push; acceptable for a single-maintainer project, and the gate
is one secret away from running on a private runner.
