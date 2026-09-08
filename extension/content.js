// SafeSight content script.
// Blurs candidate images until the API returns a verdict; keeps NSFW images
// blurred with a click-to-reveal badge.

(function () {
  if (window.__safesightInjected) return;
  window.__safesightInjected = true;

  const MIN_SIZE = 80;          // ignore icons and tiny thumbnails (css px)
  const BATCH_DELAY_MS = 300;   // debounce window for batching URL checks
  const seen = new Set();      // image URLs already handled on this page
  let pendingImgs = new Map();  // url -> [img elements]
  let batchTimer = null;
  let enabled = true;

  chrome.storage.local.get({ enabled: true }, (cfg) => {
    enabled = cfg.enabled;
    if (enabled) start();
  });

  function start() {
    scan(document);
    const observer = new MutationObserver((mutations) => {
      for (const m of mutations) {
        for (const node of m.addedNodes) {
          if (node.nodeType !== Node.ELEMENT_NODE) continue;
          if (node.tagName === "IMG") consider(node);
          else if (node.querySelectorAll) scan(node);
        }
        if (m.type === "attributes" && m.target.tagName === "IMG") {
          consider(m.target);
        }
      }
    });
    observer.observe(document.documentElement, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["src"],
    });
  }

  function scan(root) {
    root.querySelectorAll("img").forEach(consider);
  }

  function consider(img) {
    const url = absoluteUrl(img.currentSrc || img.src);
    if (!url || seen.has(url)) return;
    if (!url.startsWith("http")) return;

    if (img.complete && img.naturalWidth > 0) {
      enqueueIfBigEnough(img, url);
    } else {
      img.addEventListener("load", () => enqueueIfBigEnough(img, url), { once: true });
    }
  }

  function enqueueIfBigEnough(img, url) {
    if (seen.has(url)) return;
    const w = img.naturalWidth || img.width;
    const h = img.naturalHeight || img.height;
    if (Math.min(w, h) < MIN_SIZE) return;

    seen.add(url);
    img.classList.add("safesight-pending");
    if (!pendingImgs.has(url)) pendingImgs.set(url, []);
    pendingImgs.get(url).push(img);

    clearTimeout(batchTimer);
    batchTimer = setTimeout(flushBatch, BATCH_DELAY_MS);
  }

  function flushBatch() {
    if (pendingImgs.size === 0) return;
    const batch = pendingImgs;
    pendingImgs = new Map();
    const urls = Array.from(batch.keys());

    chrome.runtime.sendMessage({ type: "SCORE_URLS", urls: urls }, (resp) => {
      if (chrome.runtime.lastError || !resp || !resp.results) {
        // API unreachable: fail open, unblur everything from this batch.
        for (const imgs of batch.values()) {
          imgs.forEach((img) => img.classList.remove("safesight-pending"));
        }
        return;
      }
      for (const result of resp.results) {
        const imgs = batch.get(result.url) || [];
        imgs.forEach((img) => applyVerdict(img, result));
      }
    });
  }

  function applyVerdict(img, result) {
    img.classList.remove("safesight-pending");
    if (!result.ok || !result.nsfw) return;

    img.classList.add("safesight-blocked");
    attachBadge(img, result);
  }

  function attachBadge(img, result) {
    const badge = document.createElement("div");
    badge.className = "safesight-badge";
    badge.textContent = "NSFW " + Math.round((result.score || 0) * 100) + "% - click to reveal";
    positionBadge(badge, img);
    document.body.appendChild(badge);

    // All badge listeners share one controller so removal cleans up the
    // window-level scroll/resize handlers instead of leaking them.
    const controller = new AbortController();
    const removeBadge = () => {
      badge.remove();
      controller.abort();
    };

    badge.addEventListener("click", () => {
      img.classList.remove("safesight-blocked");
      removeBadge();
      chrome.runtime.sendMessage({
        type: "FEEDBACK",
        url: result.url,
        modelScore: result.score,
        userLabel: 0,
      });
    }, { signal: controller.signal });

    // Keep badge glued to the image on layout changes.
    const reposition = () => {
      if (!document.body.contains(img)) {
        removeBadge();
        return;
      }
      positionBadge(badge, img);
    };
    window.addEventListener("scroll", reposition, { passive: true, signal: controller.signal });
    window.addEventListener("resize", reposition, { passive: true, signal: controller.signal });
  }

  function positionBadge(badge, img) {
    const rect = img.getBoundingClientRect();
    badge.style.top = window.scrollY + rect.top + 8 + "px";
    badge.style.left = window.scrollX + rect.left + 8 + "px";
  }

  function absoluteUrl(src) {
    if (!src) return null;
    try {
      return new URL(src, document.baseURI).href;
    } catch {
      return null;
    }
  }
})();
