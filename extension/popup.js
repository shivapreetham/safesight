const DEFAULTS = {
  enabled: true,
  apiUrl: "http://127.0.0.1:8000",
  apiKey: "",
  sensitivity: 0.5,
};

const enabledEl = document.getElementById("enabled");
const apiUrlEl = document.getElementById("apiUrl");
const apiKeyEl = document.getElementById("apiKeyInput");
const sensEl = document.getElementById("sensitivity");
const sensValueEl = document.getElementById("sensValue");
const statusEl = document.getElementById("status");

function renderSens() {
  sensValueEl.textContent = Number(sensEl.value).toFixed(2);
}

chrome.storage.local.get(DEFAULTS, (cfg) => {
  enabledEl.checked = cfg.enabled;
  apiUrlEl.value = cfg.apiUrl;
  apiKeyEl.value = cfg.apiKey;
  sensEl.value = cfg.sensitivity;
  renderSens();
});

sensEl.addEventListener("input", renderSens);

document.getElementById("save").addEventListener("click", async () => {
  const cfg = {
    enabled: enabledEl.checked,
    apiUrl: apiUrlEl.value.trim().replace(/\/$/, ""),
    apiKey: apiKeyEl.value.trim(),
    sensitivity: Number(sensEl.value),
  };
  await chrome.storage.local.set(cfg);
  // Clear cached verdicts since the threshold may have changed.
  await chrome.storage.local.remove("verdictCache");

  statusEl.className = "";
  statusEl.textContent = "Saved. Testing connection...";
  try {
    const resp = await fetch(cfg.apiUrl + "/healthz");
    const data = await resp.json();
    statusEl.textContent =
      "Connected: " + data.status + " on " + data.device;
  } catch (e) {
    statusEl.className = "err";
    statusEl.textContent = "Cannot reach API at " + cfg.apiUrl;
  }
});
