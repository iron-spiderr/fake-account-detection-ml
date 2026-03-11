/* ============================================================
   Fake Account Detection System — Frontend Logic
   ============================================================ */

"use strict";

// ---------------------------------------------------------------------------
// Tab navigation
// ---------------------------------------------------------------------------
document.querySelectorAll(".nav-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const tab = btn.dataset.tab;
    document
      .querySelectorAll(".nav-btn")
      .forEach((b) => b.classList.remove("active"));
    document
      .querySelectorAll(".tab-panel")
      .forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`tab-${tab}`).classList.add("active");
    if (tab === "history") loadHistory();
  });
});

// ---------------------------------------------------------------------------
// Status check
// ---------------------------------------------------------------------------
async function checkStatus() {
  try {
    const r = await fetch("/api/status");
    const d = await r.json();
    const dot = document.getElementById("status-dot");
    const text = document.getElementById("status-text");
    if (d.pipeline_ready) {
      dot.className = "status-dot ready";
      text.textContent = "Model ready";
    } else {
      dot.className = "status-dot not-ready";
      text.textContent = "Model not loaded";
    }
  } catch {
    document.getElementById("status-text").textContent = "Offline";
  }
}

checkStatus();
setInterval(checkStatus, 30000);

// ---------------------------------------------------------------------------
// Spinner helpers
// ---------------------------------------------------------------------------
const spinner = document.getElementById("spinner");
const showSpinner = () => spinner.classList.remove("hidden");
const hideSpinner = () => spinner.classList.add("hidden");

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------
async function callApi(endpoint, body = null) {
  showSpinner();
  try {
    const opts = {
      method: body ? "POST" : "GET",
      headers: { "Content-Type": "application/json" },
    };
    if (body) opts.body = JSON.stringify(body);
    const r = await fetch(endpoint, opts);
    const d = await r.json();
    return d;
  } finally {
    hideSpinner();
  }
}

// ---------------------------------------------------------------------------
// Result rendering
// ---------------------------------------------------------------------------
function renderResults(results, container) {
  if (!results || results.length === 0) {
    container.innerHTML = '<p style="color:var(--text-muted)">No results.</p>';
    return;
  }
  const grid = document.createElement("div");
  grid.className = "results-grid";
  results.forEach((r) => grid.appendChild(buildCard(r)));
  container.innerHTML = "";
  container.appendChild(grid);
}

function buildCard(r) {
  const isFake = r.label === "FAKE";
  const prob = (r.probability * 100).toFixed(1);
  const band = r.risk_band || "LOW";
  const uname = r.username || "unknown";

  const card = document.createElement("div");
  card.className = `result-card ${isFake ? "fake" : "genuine"}`;

  // Header
  const hdr = document.createElement("div");
  hdr.className = "card-header";
  hdr.innerHTML = `
    <span class="card-username">@${uname}</span>
    <span class="card-label ${isFake ? "label-fake" : "label-genuine"}">${r.label}</span>
  `;

  // Probability bar
  const barWrap = document.createElement("div");
  barWrap.className = "prob-bar-wrap";
  barWrap.innerHTML = `
    <div class="prob-bar-label">
      <span>P(fake): ${prob}%</span>
      <span>Conf: ${r.confidence || "—"}</span>
    </div>
    <div class="prob-bar-track">
      <div class="prob-bar-fill ${isFake ? "fill-fake" : "fill-genuine"}"
           style="width:${prob}%"></div>
    </div>
  `;

  // Meta chips
  const meta = document.createElement("div");
  meta.className = "card-meta";
  meta.innerHTML = `
    <span class="meta-chip chip-${band}">${band}</span>
    <span class="meta-chip chip-conf">Graph risk: ${(r.graph_risk * 100).toFixed(1)}%</span>
  `;

  // Explanation dropdown
  const details = document.createElement("details");
  details.className = "card-explanation-dropdown";
  const summary = document.createElement("summary");
  summary.className = "explanation-toggle";
  summary.textContent = "View Explanation";
  details.appendChild(summary);

  const explBody = document.createElement("div");
  explBody.className = "explanation-body";

  // Explanation text
  if (r.explanation) {
    const explText = document.createElement("p");
    explText.className = "card-explanation";
    explText.textContent = r.explanation;
    explBody.appendChild(explText);
  }

  // Top features — use annotated features if available for richer display
  const featWrap = document.createElement("div");
  featWrap.className = "card-features";
  if (r.annotated_features && r.annotated_features.length > 0) {
    r.annotated_features.forEach((af) => {
      const chip = document.createElement("span");
      // Positive impact = fake signal (red), negative = genuine signal (green)
      const isFakeSignal = af.impact > 0;
      chip.className = `feat-chip ${isFakeSignal ? "feat-fake" : "feat-genuine"}`;
      if (af.estimated) {
        chip.className += " feat-estimated";
        chip.title = "This feature was estimated (not directly observed)";
      } else {
        chip.title = "Directly observed from profile";
      }
      chip.textContent = af.name;
      featWrap.appendChild(chip);
    });
  } else {
    (r.top_features || []).forEach((f) => {
      const chip = document.createElement("span");
      chip.className = `feat-chip ${isFake ? "feat-fake" : "feat-genuine"}`;
      chip.textContent = f;
      featWrap.appendChild(chip);
    });
  }
  explBody.appendChild(featWrap);

  // SHAP impact bars (visual breakdown)
  const shapBars = document.createElement("div");
  shapBars.className = "shap-bars";
  const featsToShow =
    r.annotated_features ||
    (r.top_shap_values || []).map((t) => ({
      name: Array.isArray(t) ? (t.length === 3 ? t[1] : t[0]) : t,
      impact: Array.isArray(t) ? t[t.length - 1] : 0,
      estimated: false,
    }));
  if (featsToShow.length > 0) {
    const maxImpact = Math.max(
      ...featsToShow.map((f) => Math.abs(f.impact)),
      0.001,
    );
    featsToShow.forEach((f) => {
      const row = document.createElement("div");
      row.className = "shap-bar-row";
      const pct = Math.min((Math.abs(f.impact) / maxImpact) * 100, 100).toFixed(
        0,
      );
      const dir = f.impact > 0 ? "positive" : "negative";
      row.innerHTML = `
        <span class="shap-bar-label${f.estimated ? " shap-estimated" : ""}">${f.name}${f.estimated ? " *" : ""}</span>
        <div class="shap-bar-track">
          <div class="shap-bar-fill shap-${dir}" style="width:${pct}%"></div>
        </div>
        <span class="shap-bar-val">${f.impact > 0 ? "+" : "-"}${Math.abs(f.impact).toFixed(1)}%</span>
      `;
      shapBars.appendChild(row);
    });
    // Footnote for estimated features
    if (featsToShow.some((f) => f.estimated)) {
      const note = document.createElement("p");
      note.className = "shap-footnote";
      note.textContent =
        "* Estimated feature (not directly observed from profile)";
      shapBars.appendChild(note);
    }
  }
  explBody.appendChild(shapBars);

  // Data completeness indicator (shown for scrape/manual modes)
  if (r.data_completeness !== undefined) {
    const pct = Math.round(r.data_completeness * 100);
    const isLow = pct < 60;
    const completenessWrap = document.createElement("div");
    completenessWrap.className = "card-completeness";
    completenessWrap.innerHTML = `
      ${isLow ? '<div class="completeness-warning">\u26a0 Low data availability \u2014 prediction may be less reliable</div>' : ""}
      <span class="completeness-label">Data completeness: ${pct}%</span>
      <div class="completeness-bar-track">
        <div class="completeness-bar-fill${isLow ? " completeness-low" : ""}" style="width:${pct}%"></div>
      </div>
    `;
    explBody.appendChild(completenessWrap);
  }

  details.appendChild(explBody);

  // Ground truth badge (test suite only)
  if (r.ground_truth !== undefined) {
    const gt = document.createElement("div");
    gt.className = "card-ground-truth";
    const isCorrect = r.correct;
    gt.innerHTML = `
      <span class="gt-label">Expected: <strong>${r.ground_truth}</strong></span>
      <span class="gt-verdict ${isCorrect ? "gt-correct" : "gt-wrong"}">
        ${isCorrect ? "✔ Correct" : "✘ Wrong"}
      </span>
    `;
    card.append(hdr, barWrap, meta, details, gt);
  } else {
    card.append(hdr, barWrap, meta, details);
  }

  return card;
}

function showError(container, msg) {
  container.innerHTML = `<div class="error-banner">⚠️ ${msg}</div>`;
}

// ---------------------------------------------------------------------------
// Number formatter
// ---------------------------------------------------------------------------
function fmt(n) {
  if (n === undefined || n === null) return "—";
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "K";
  return n.toString();
}

// ---------------------------------------------------------------------------
// Profile summary card (scrape results)
// ---------------------------------------------------------------------------
function buildProfileSummary(d) {
  const p = d.profile || {};
  const priv = d.is_private;
  const el = document.createElement("div");
  el.className = "profile-summary";
  el.innerHTML = `
    <div class="profile-summary-header">
      <span class="ps-username">@${d.username}</span>
      <span class="ps-badge ${priv ? "badge-private" : "badge-public"}">
        ${priv ? "🔒 Private" : "✅ Public"}
      </span>
    </div>
    <div class="ps-stats">
      <div class="ps-stat">
        <span class="ps-stat-val">${fmt(p.followers)}</span>
        <span class="ps-stat-label">Followers</span>
      </div>
      <div class="ps-stat">
        <span class="ps-stat-val">${fmt(p.following)}</span>
        <span class="ps-stat-label">Following</span>
      </div>
      <div class="ps-stat">
        <span class="ps-stat-val">${fmt(p.posts)}</span>
        <span class="ps-stat-label">Posts</span>
      </div>
      <div class="ps-stat">
        <span class="ps-stat-val">${p.is_verified ? "✓" : "—"}</span>
        <span class="ps-stat-label">Verified</span>
      </div>
    </div>
    ${p.biography ? `<p class="ps-bio">&ldquo;${p.biography}&rdquo;</p>` : ""}
    ${p.external_url ? `<a class="ps-link" href="${p.external_url}" target="_blank" rel="noopener">${p.external_url}</a>` : ""}
  `;
  return el;
}

// ---------------------------------------------------------------------------
// Scrape & Analyse
// ---------------------------------------------------------------------------
document.getElementById("btn-scrape").addEventListener("click", async () => {
  const raw = document.getElementById("scrape-username").value.trim();
  const username = raw.replace(/^@/, "");
  const container = document.getElementById("results-scrape");

  if (!username) {
    showError(container, "Please enter an Instagram username.");
    return;
  }

  const d = await callApi("/api/scrape", { username });

  if (!d.ok) {
    if (d.rate_limited) {
      container.innerHTML = `
        <div class="warning-banner" style="max-width:520px">
          <strong>⏳ Instagram Rate Limited</strong><br>
          Instagram has temporarily blocked scraping from this IP (HTTP 429).<br>
          <br>
          <strong>What to do:</strong>
          <ul style="margin:.5rem 0 0 1.2rem;line-height:1.8">
            <li>Wait <strong>~30 minutes</strong> before trying again</li>
            <li>Avoid running multiple scrape requests back-to-back</li>
            <li>Use the <em>Manual Input</em> tab to analyse a profile by hand</li>
          </ul>
        </div>`;
    } else {
      showError(container, d.error || "Unknown error");
    }
    return;
  }

  container.innerHTML = "";

  // Profile summary
  container.appendChild(buildProfileSummary(d));

  // Warning for private / partial data
  if (d.warning) {
    const wb = document.createElement("div");
    wb.className = "warning-banner";
    wb.textContent = "⚠️ " + d.warning;
    container.appendChild(wb);
  }

  // Prediction results
  const resultSection = document.createElement("div");
  renderResults(d.results, resultSection);
  container.appendChild(resultSection);
});

// ---------------------------------------------------------------------------
// Demo Scan — helpers
// ---------------------------------------------------------------------------

function hideDemoSummary() {
  const el = document.getElementById("demo-summary");
  el.classList.add("hidden");
  el.innerHTML = "";
}

function showDemoSummary(s) {
  const el = document.getElementById("demo-summary");
  el.classList.remove("hidden");
  const f1Color = s.f1 >= 80 ? "#22c55e" : s.f1 >= 60 ? "#f59e0b" : "#ef4444";
  el.innerHTML = `
    <div class="summary-title">🧪 Test Suite Results</div>
    <div class="summary-stats">
      <div class="summary-stat">
        <span class="summary-val">${s.accuracy}%</span>
        <span class="summary-key">Accuracy</span>
      </div>
      <div class="summary-stat">
        <span class="summary-val">${s.precision}%</span>
        <span class="summary-key">Precision</span>
      </div>
      <div class="summary-stat">
        <span class="summary-val">${s.recall}%</span>
        <span class="summary-key">Recall</span>
      </div>
      <div class="summary-stat">
        <span class="summary-val" style="color:${f1Color}">${s.f1}%</span>
        <span class="summary-key">F1 Score</span>
      </div>
      <div class="summary-stat">
        <span class="summary-val">${s.correct} / ${s.total}</span>
        <span class="summary-key">Correct</span>
      </div>
    </div>
  `;
}

// Quick Demo (5 real + 5 fake curated profiles)
document.getElementById("btn-demo").addEventListener("click", async () => {
  const container = document.getElementById("results-demo");
  hideDemoSummary();
  const d = await callApi("/api/demo", {});
  if (!d.ok) {
    showError(container, d.error || "Unknown error");
    return;
  }
  if (d.summary) showDemoSummary(d.summary);
  renderResults(d.results, container);
});

// Full Test Suite (test_profiles.csv)
document.getElementById("btn-demo-test").addEventListener("click", async () => {
  const container = document.getElementById("results-demo");
  hideDemoSummary();
  const d = await callApi("/api/demo-test", {});
  if (!d.ok) {
    showError(container, d.error || "Unknown error");
    return;
  }
  if (d.summary) showDemoSummary(d.summary);
  renderResults(d.results, container);
});

// ---------------------------------------------------------------------------
// Self Scan
// ---------------------------------------------------------------------------
document.getElementById("btn-self").addEventListener("click", async () => {
  const token = document.getElementById("self-token").value.trim();
  const container = document.getElementById("results-self");
  if (!token) {
    showError(container, "Please enter your access token.");
    return;
  }
  const d = await callApi("/api/scan-self", { token });
  if (d.ok) renderResults(d.results, container);
  else showError(container, d.error || "Unknown error");
});

// ---------------------------------------------------------------------------
// Username Scan
// ---------------------------------------------------------------------------
document
  .getElementById("btn-scan-users")
  .addEventListener("click", async () => {
    const token = document.getElementById("user-token").value.trim();
    const usernames = document.getElementById("user-names").value.trim();
    const container = document.getElementById("results-username");
    if (!token || !usernames) {
      showError(container, "Token and at least one username are required.");
      return;
    }
    const d = await callApi("/api/scan-users", { token, usernames });
    if (d.ok) renderResults(d.results, container);
    else showError(container, d.error || "Unknown error");
  });

// ---------------------------------------------------------------------------
// Manual Input
// ---------------------------------------------------------------------------
document.getElementById("btn-manual").addEventListener("click", async () => {
  const container = document.getElementById("results-manual");
  const body = {
    username:
      document.getElementById("m-username").value.trim() || "manual_user",
    biography: document.getElementById("m-bio").value.trim(),
    followers: parseInt(document.getElementById("m-followers").value) || 0,
    following: parseInt(document.getElementById("m-following").value) || 0,
    posts: parseInt(document.getElementById("m-posts").value) || 0,
    website: document.getElementById("m-website").value.trim(),
    has_profile_pic: document.getElementById("m-pic").checked,
  };
  const d = await callApi("/api/manual", body);
  if (d.ok) renderResults(d.results, container);
  else showError(container, d.error || "Unknown error");
});

// ---------------------------------------------------------------------------
// History
// ---------------------------------------------------------------------------
async function loadHistory() {
  const container = document.getElementById("results-history");
  const d = await callApi("/api/history");
  if (!d.ok || d.history.length === 0) {
    container.innerHTML =
      '<p style="color:var(--text-muted)">No history yet.</p>';
    return;
  }
  container.innerHTML = "";
  d.history
    .slice()
    .reverse()
    .forEach((entry) => {
      const div = document.createElement("div");
      div.className = "history-entry";
      div.innerHTML = `<div class="history-meta">
      🕐 ${entry.timestamp} · Mode: <strong>${entry.mode}</strong> · ${entry.count} account(s)
    </div>`;
      const grid = document.createElement("div");
      grid.className = "results-grid";
      (entry.results || []).forEach((r) => grid.appendChild(buildCard(r)));
      div.appendChild(grid);
      container.appendChild(div);
    });
}

document
  .getElementById("btn-clear-history")
  .addEventListener("click", async () => {
    await callApi("/api/history/clear", {});
    loadHistory();
  });
