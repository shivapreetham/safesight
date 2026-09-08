// SafeSight background service worker.
// Batches image URL checks from content scripts, calls the moderation API,
// applies the user's sensitivity threshold, and caches verdicts.

const DEFAULTS = {
  enabled: true,
  apiUrl: "http://127.0.0.1:8000",
  apiKey: "",       // sent as X-API-Key when the server requires auth
  sensitivity: 0.5, // threshold on score_large; lower = stricter blocking
};

function authHeaders(cfg) {
  const headers = { "Content-Type": "application/json" };
  if (cfg.apiKey) headers["X-API-Key"] = cfg.apiKey;
  return headers;
}

const CACHE_MAX = 5000;

async function getConfig() {
  return await chrome.storage.local.get(DEFAULTS);
}

async function getCache() {
  const { verdictCache = {} } = await chrome.storage.local.get("verdictCache");
  return verdictCache;
}

async function putCache(cache) {
  const keys = Object.keys(cache);
  if (keys.length > CACHE_MAX) {
    // Naive eviction: drop the oldest half by insertion order.
    const drop = keys.slice(0, Math.floor(keys.length / 2));
    for (const k of drop) delete cache[k];
  }
  await chrome.storage.local.set({ verdictCache: cache });
}

function decideNsfw(entry, sensitivity) {
  if (!entry.ok) return false;
  return entry.label === 1 || entry.score_large >= sensitivity;
}

async function scoreUrls(urls) {
  const cfg = await getConfig();
  if (!cfg.enabled) {
    return { results: urls.map((u) => ({ url: u, ok: false, nsfw: false })) };
  }

  const cache = await getCache();
  const results = [];
  const toFetch = [];

  for (const url of urls) {
    if (cache[url]) {
      results.push(toResult(url, cache[url], cfg.sensitivity));
    } else {
      toFetch.push(url);
    }
  }

  if (toFetch.length > 0) {
    try {
      const resp = await fetch(cfg.apiUrl.replace(/\/$/, "") + "/v1/score-urls", {
        method: "POST",
        headers: authHeaders(cfg),
        body: JSON.stringify({ urls: toFetch }),
      });
      const data = await resp.json();
      for (const r of data.results || []) {
        const entry = {
          ok: r.ok,
          label: r.label,
          score: r.score,
          score_large: r.score_large,
        };
        cache[r.url] = entry;
        results.push(toResult(r.url, entry, cfg.sensitivity));
      }
      await putCache(cache);
    } catch (e) {
      console.warn("SafeSight API error:", e);
      for (const url of toFetch) {
        results.push({ url: url, ok: false, nsfw: false });
      }
    }
  }

  return { results: results };
}

function toResult(url, entry, sensitivity) {
  return {
    url: url,
    ok: entry.ok,
    nsfw: decideNsfw(entry, sensitivity),
    score: entry.score || 0,
  };
}

async function sha256Hex(text) {
  const data = new TextEncoder().encode(text);
  const digest = await crypto.subtle.digest("SHA-256", data);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

async function sendFeedback(msg) {
  const cfg = await getConfig();
  try {
    // Same key scheme as the server cache: sha256(url) truncated to 32 chars.
    const hash = (await sha256Hex(msg.url || "")).slice(0, 32);
    await fetch(cfg.apiUrl.replace(/\/$/, "") + "/v1/feedback", {
      method: "POST",
      headers: authHeaders(cfg),
      body: JSON.stringify({
        url_hash: hash,
        model_score: msg.modelScore || 0,
        user_label: msg.userLabel,
      }),
    });
  } catch (e) {
    console.warn("SafeSight feedback error:", e);
  }
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "SCORE_URLS") {
    scoreUrls(msg.urls).then(sendResponse);
    return true; // async response
  }
  if (msg.type === "FEEDBACK") {
    sendFeedback(msg);
    return false;
  }
  return false;
});
