(() => {
  const $ = (id) => document.getElementById(id);
  const btn = $("demo-run");
  const dataNode = $("demo-data");
  if (!btn || !dataNode) return;

  let demo = null;
  try { demo = JSON.parse(dataNode.textContent); } catch { demo = null; }
  let series = (demo && demo.series) || [];
  // Returning visitors see the recorded line immediately; new points animate in.
  let shown = series.length;
  let mode = "idle"; // idle | driving | watching
  let lastProgressAt = Date.now();

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const fmt = (value) => (value == null ? "—" : String(Math.round(Number(value))));
  const run = () => (demo && demo.run) || null;
  const cooldownMs = () => (((demo && demo.cooldown_seconds) || 1) * 1000) + 250;
  const finished = () => { const r = run(); return !!r && r.status === "completed"; };
  const active = () => { const r = run(); return !!r && !["completed", "failed"].includes(r.status); };

  async function fetchState() {
    try {
      const response = await fetch("/v1/demo", {headers: {Accept: "application/json"}});
      if (!response.ok) return;
      demo = await response.json();
      const fresh = demo.series || [];
      if (fresh.length < shown) shown = fresh.length; // a reset shrank the series
      if (fresh.length > series.length) lastProgressAt = Date.now();
      series = fresh;
    } catch { /* transient network problem; the next tick retries */ }
  }

  async function act(payload) {
    const response = await fetch("/v1/replay/actions", {
      method: "POST",
      headers: {"Content-Type": "application/json", Accept: "application/json"},
      body: JSON.stringify(payload),
    });
    if (response.ok) {
      const body = await response.json();
      if (demo) demo.run = body.run;
      return body.run;
    }
    let detail = "The live system returned an error.";
    try {
      const body = await response.json();
      detail = (body.error && body.error.message) || detail;
    } catch { /* non-JSON body */ }
    const error = new Error(detail);
    error.transient = response.status === 409 || response.status === 429;
    throw error;
  }

  function chart() {
    if (!window.Plotly) return;
    const visible = series.slice(0, shown);
    const cycles = visible.map((p) => p.cycle);
    const layout = {
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      font: {color: "#8a93a0", size: 12},
      margin: {l: 46, r: 18, t: 8, b: 40},
      xaxis: {title: {text: "flight cycle"}, gridcolor: "#161b24", zeroline: false,
              range: [0, Math.max(60, (cycles[cycles.length - 1] || 0) + 12)]},
      yaxis: {title: {text: "flights remaining"}, gridcolor: "#161b24", rangemode: "tozero"},
      showlegend: false,
      hovermode: "x unified",
      shapes: [],
      annotations: [],
    };
    const r = run();
    if (finished() && shown === series.length && r.final_cycle != null) {
      layout.shapes.push({type: "line", x0: r.final_cycle, x1: r.final_cycle,
        yref: "paper", y0: 0, y1: 1, line: {color: "#ff6b74", width: 1.5, dash: "dot"}});
      layout.annotations.push({x: r.final_cycle, yref: "paper", y: 1, text: "failure",
        showarrow: false, font: {color: "#ff6b74", size: 11}, xanchor: "left", yanchor: "top"});
    }
    window.Plotly.react("demo-chart", [
      {x: cycles, y: visible.map((p) => p.upper_rul), mode: "lines",
       line: {width: 0}, hoverinfo: "skip"},
      {x: cycles, y: visible.map((p) => p.lower_rul), mode: "lines", fill: "tonexty",
       fillcolor: "rgba(99,167,255,.13)", line: {width: 0},
       hovertemplate: "90% range low: %{y:.0f}<extra></extra>"},
      {x: cycles, y: visible.map((p) => p.predicted_rul), mode: "lines",
       line: {color: "#63a7ff", width: 2.2},
       hovertemplate: "predicted: %{y:.0f} flights<extra></extra>"},
    ], layout, {responsive: true, displaylogo: false, displayModeBar: false});
  }

  function stats() {
    const point = series[shown - 1] || null;
    $("demo-cycle").textContent = point ? String(point.cycle) : "—";
    $("demo-rul").textContent = point ? fmt(point.predicted_rul) : "—";
    $("demo-interval").textContent =
      point && point.lower_rul != null ? `(${fmt(point.lower_rul)}–${fmt(point.upper_rul)})` : "";
    const risk = $("demo-risk");
    risk.textContent = point ? point.risk_level : "no data";
    risk.className = `risk ${point ? point.risk_level : "unknown"}`;
  }

  function button() {
    btn.disabled = false;
    if (!demo || demo.enabled === false) {
      btn.disabled = true;
      btn.textContent = "Demo unavailable";
    } else if (mode === "driving") {
      btn.disabled = true;
      btn.textContent = "Streaming…";
    } else if (mode === "watching") {
      btn.disabled = true;
      btn.textContent = "Watching live…";
    } else if (finished()) {
      btn.textContent = "↻ Run it again";
    } else {
      btn.textContent = "▶ Run simulation";
    }
  }

  function message(text) { $("demo-message").textContent = text; }

  function verdict() {
    const node = $("demo-punchline");
    const r = run();
    if (!finished() || shown < series.length || r.final_cycle == null) {
      node.hidden = true;
      return;
    }
    const flagged = series.find((p) => p.risk_level !== "healthy");
    if (flagged) {
      const lead = r.final_cycle - flagged.cycle;
      node.innerHTML =
        `The engine failed at cycle <strong>${r.final_cycle}</strong>. ` +
        `The model flagged trouble at cycle <strong>${flagged.cycle}</strong> — ` +
        `<span class="lead">${lead} flights of advance warning</span>.` +
        `<small>Every prediction was made in sequence, without seeing the future.</small>`;
    } else {
      node.innerHTML =
        `The engine failed at cycle <strong>${r.final_cycle}</strong>.` +
        `<small>Every prediction was made in sequence, without seeing the future.</small>`;
    }
    node.hidden = false;
  }

  function paint() { chart(); stats(); button(); verdict(); }

  // Animation: reveal one point per tick so the line draws itself continuously,
  // no matter how chunky the server responses are.
  setInterval(() => {
    if (shown < series.length) {
      shown += 1;
      paint();
      if (mode !== "idle") {
        message(`Streaming sensor data through the model — cycle ${series[shown - 1].cycle}`);
      }
    } else if (finished() && $("demo-punchline").hidden) {
      paint();
      if (mode === "idle") message("");
    }
  }, 110);

  // Background poll: the page stays live without any manual refresh, whether
  // this visitor, another visitor, or nobody is driving.
  setInterval(async () => {
    if (mode === "driving") return;
    if (!active()) return;
    await fetchState();
    if (mode === "watching") {
      if (finished()) { mode = "idle"; button(); message(""); return; }
      if (Date.now() - lastProgressAt > 15000) {
        mode = "idle";
        button();
        message("The run is paused — press the button to keep it going.");
      }
    }
  }, 3000);

  async function drive() {
    if (mode !== "idle") return;
    mode = "driving";
    button();
    try {
      let r = run();
      if (!r || r.status === "completed" || r.status === "failed") {
        message("Starting the engine…");
        const payload = r
          ? {action: "reset", source_asset_id: demo.demo_source_asset_id, confirm_reset: true}
          : {action: "start", source_asset_id: demo.demo_source_asset_id};
        await act(payload);
        if (payload.action === "reset") { series = []; shown = 0; $("demo-punchline").hidden = true; }
        await fetchState();
      }
      while (mode === "driving") {
        r = run();
        if (!r || r.status === "completed" || r.status === "failed") break;
        try {
          await act({action: "accelerate", run_id: r.run_id});
          lastProgressAt = Date.now();
        } catch (error) {
          if (error.transient) { mode = "watching"; button(); break; }
          throw error;
        }
        await fetchState();
        await sleep(cooldownMs());
      }
      await fetchState();
    } catch (error) {
      message(error.message || "The live system hit a snag — try again in a moment.");
    } finally {
      if (mode === "driving") mode = "idle";
      paint();
    }
  }

  btn.addEventListener("click", drive);

  if (!demo) {
    message("Waking the live system…");
    fetchState().then(() => { paint(); message(""); });
  }
  paint();
  if (demo && !series.length) {
    message("One click streams a real engine's sensor data through the live model.");
  } else if (demo && active()) {
    message("An engine run is under way — press run to keep it going.");
  }
})();
