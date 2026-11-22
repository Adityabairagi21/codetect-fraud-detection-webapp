/* CoDetect front-end helpers: render Plotly charts declared with data-chart, network explorer. */
(function () {
  const BASE_LAYOUT = {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter, system-ui, Segoe UI, sans-serif", size: 12, color: "#52514e" },
    hoverlabel: { bgcolor: "#14213d", font: { color: "#fff" } },
  };
  const CONFIG = { displayModeBar: false, responsive: true };

  function merge(a, b) {
    const out = Object.assign({}, a);
    Object.keys(b || {}).forEach((k) => {
      out[k] = b[k] && typeof b[k] === "object" && !Array.isArray(b[k]) && a[k] ? merge(a[k], b[k]) : b[k];
    });
    return out;
  }

  function renderCharts() {
    if (!window.Plotly) return;
    document.querySelectorAll("[data-chart]").forEach((el) => {
      let spec;
      try { spec = JSON.parse(el.dataset.chart); } catch (e) { return; }
      Plotly.newPlot(el, spec.data, merge(BASE_LAYOUT, spec.layout), CONFIG);
    });
  }

  async function loadNetwork() {
    const box = document.getElementById("network");
    if (!box) return;
    const status = document.getElementById("network-status");
    const params = new URLSearchParams({ dataset: box.dataset.dataset });
    if (box.dataset.account) params.set("account", box.dataset.account);
    status.textContent = "Building the transaction network…";
    try {
      const res = await fetch(`${box.dataset.url}?${params}`, { headers: { Accept: "application/json" } });
      const fig = await res.json();
      if (!res.ok) { status.textContent = fig.error || "The network could not be loaded."; return; }
      if (fig.empty) {
        status.textContent = box.dataset.account
          ? "No transactions found for this account in the selected dataset."
          : "No suspicious transactions yet. Run fraud detection, or search for an account.";
        return;
      }
      Plotly.newPlot(box, fig.data, merge(BASE_LAYOUT, fig.layout), CONFIG);
      const s = fig.stats;
      status.textContent = `${s.accounts} accounts · ${s.transactions} transactions (${s.suspicious_transactions} suspicious)` +
        (s.truncated ? ` · limited to the ${s.max_edges} highest-scoring transactions` : "") +
        " · click an account to explore it";
      box.on("plotly_click", (ev) => {
        const p = ev.points && ev.points[0];
        if (p && p.customdata) {
          const u = new URL(window.location.href);
          u.searchParams.set("account", p.customdata);
          window.location.href = u.toString();
        }
      });
    } catch (e) {
      status.textContent = "The network could not be loaded.";
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    renderCharts();
    loadNetwork();
    document.querySelectorAll("form[data-busy]").forEach((f) => f.addEventListener("submit", () => {
      const b = f.querySelector("[type=submit]");
      if (b) { b.disabled = true; b.innerHTML = `<span class="spinner-border spinner-border-sm me-2"></span>${f.dataset.busy}`; }
    }));
  });
})();
