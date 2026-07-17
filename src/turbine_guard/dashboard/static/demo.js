(() => {
  const $ = (id) => document.getElementById(id);
  const runButton = $("demo-run");
  const dataNode = $("demo-data");
  if (!runButton || !dataNode) return;

  let demo = null;
  try { demo = JSON.parse(dataNode.textContent); } catch { demo = null; }
  let driving = false;
  let spectating = false;

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const fmt = (value) => (value == null ? "—" : String(Math.round(Number(value))));
  const message = (text) => { $("demo-message").textContent = text; };

  async function refresh() {
    try {
      const response = await fetch("/v1/demo", {headers: {Accept: "application/json"}});
      if (response.ok) demo = await response.json();
    } catch { /* keep the last known state; the poller will retry */ }
    render();
  }

  async function action(payload) {
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
    let code = "unknown";
    let detail = "The live system returned an error.";
    try {
      const body = await response.json();
      code = (body.error && body.error.code) || code;
      detail = (body.error && body.error.message) || detail;
    } catch { /* non-JSON error body */ }
    const error = new Error(detail);
    error.code = code;
    error.transient = response.status === 409 || response.status === 429;
    throw error;
  }

  function firstAtLevel(series, levels) {
    return series.find((point) => levels.includes(point.risk_level)) || null;
  }

  function renderPunchline(series, run) {
    const node = $("demo-punchline");
    if (!run || run.status !== "completed" || run.final_cycle == null) {
      node.hidden = true;
      return;
    }
    const warning = firstAtLevel(series, ["warning", "critical"]);
    const critical = firstAtLevel(series, ["critical"]);
    if (warning) {
      const lead = run.final_cycle - warning.cycle;
      const criticalPart = critical
        ? ` and escalated to <strong>critical</strong> at cycle ${critical.cycle}`
        : "";
      node.innerHTML =
        `<strong>The engine failed at cycle ${run.final_cycle}.</strong> ` +
        `TurbineGuard flagged it at cycle ${warning.cycle}${criticalPart} — ` +
        `<strong>${lead} cycles of advance warning</strong> to schedule maintenance ` +
        `before the failure. Every prediction above was made without seeing the future.`;
    } else {
      node.innerHTML =
        `<strong>The engine failed at cycle ${run.final_cycle}.</strong> ` +
        `Every prediction above was made in sequence, without seeing the future.`;
    }
    node.hidden = false;
  }

  function renderButton(run) {
    runButton.disabled = false;
    if (!demo || demo.enabled === false) {
      runButton.disabled = true;
      runButton.textContent = "Demo unavailable";
    } else if (driving) {
      runButton.disabled = true;
      runButton.textContent = "Streaming…";
    } else if (spectating) {
      runButton.disabled = true;
      runButton.textContent = "Watching live…";
    } else if (run && run.status === "completed") {
      runButton.textContent = "↻ Run it again";
    } else if (run && run.last_confirmed_cycle > 0) {
      runButton.textContent = "▶ Continue the simulation";
    } else {
      runButton.textContent = "▶ Run live simulation";
    }
  }

  function drawChart(series, run) {
    if (!window.Plotly) return;
    const cycles = series.map((p) => p.cycle);
    const layout = {
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      font: {color: "#91a7ba"},
      margin: {l: 55, r: 20, t: 12, b: 45},
      xaxis: {title: "Flight cycle", gridcolor: "#203851", zeroline: false},
      yaxis: {title: "Predicted cycles remaining", gridcolor: "#203851", rangemode: "tozero"},
      legend: {orientation: "h", y: 1.08},
      hovermode: "x unified",
      shapes: [],
      annotations: [],
    };
    if (run && run.status === "completed" && run.final_cycle != null) {
      layout.shapes.push({
        type: "line", x0: run.final_cycle, x1: run.final_cycle, yref: "paper", y0: 0, y1: 1,
        line: {color: "#ff6b74", width: 2, dash: "dash"},
      });
      layout.annotations.push({
        x: run.final_cycle, yref: "paper", y: 1, text: "actual failure",
        showarrow: false, font: {color: "#ff6b74", size: 12}, xanchor: "left", yanchor: "top",
      });
    }
    const traces = [
      {x: cycles, y: series.map((p) => p.upper_rul), mode: "lines",
       line: {width: 0}, hoverinfo: "skip", showlegend: false},
      {x: cycles, y: series.map((p) => p.lower_rul), mode: "lines", fill: "tonexty",
       fillcolor: "rgba(99,167,255,.16)", line: {width: 0}, name: "90% confidence band",
       hovertemplate: "band: %{y:.0f}<extra></extra>"},
      {x: cycles, y: series.map((p) => p.predicted_rul), mode: "lines",
       line: {color: "#63a7ff", width: 2.5}, name: "Predicted cycles remaining",
       hovertemplate: "predicted: %{y:.0f} cycles<extra></extra>"},
    ];
    window.Plotly.react("demo-chart", traces, layout, {responsive: true, displaylogo: false});
  }

  function render() {
    const run = demo && demo.run;
    const series = (demo && demo.series) || [];
    const last = series[series.length - 1] || null;
    $("demo-engine").textContent = demo ? `NASA #${demo.demo_source_asset_id}` : "—";
    $("demo-cycle").textContent = run ? String(run.last_confirmed_cycle) : "—";
    $("demo-rul").textContent = last ? `${fmt(last.predicted_rul)} cycles` : "—";
    $("demo-interval").textContent =
      last && last.lower_rul != null
        ? `90% range: ${fmt(last.lower_rul)}–${fmt(last.upper_rul)}`
        : "90% range: —";
    const risk = $("demo-risk");
    const level = last ? last.risk_level : null;
    risk.textContent = level || "no data";
    risk.className = `badge large ${level || "unknown"}`;
    const pct = run ? Math.min(100, run.progress_percent || 0) : 0;
    $("demo-progress-pct").textContent = `${pct}%`;
    $("demo-progress-bar").style.width = `${pct}%`;
    $("demo-progress-note").textContent = run ? run.status.replace("_", " ") : "not started";
    drawChart(series, run);
    renderPunchline(series, run);
    renderButton(run);
  }

  async function spectate() {
    spectating = true;
    render();
    message("Another visitor is driving the engine — watching it live.");
    let stalled = 0;
    while (spectating) {
      const before = demo && demo.run ? demo.run.last_confirmed_cycle : 0;
      await sleep(2500);
      await refresh();
      const run = demo && demo.run;
      if (!run || run.status === "completed" || run.status === "failed") break;
      stalled = run.last_confirmed_cycle > before ? 0 : stalled + 1;
      if (stalled >= 4) break; // nobody is actually driving; offer the wheel back
    }
    spectating = false;
    render();
    message(demo && demo.run && demo.run.status === "completed"
      ? "Simulation complete — see the verdict below."
      : "Ready when you are.");
  }

  async function drive() {
    if (driving) return;
    driving = true;
    render();
    try {
      let run = demo && demo.run;
      if (!run || run.status === "completed" || run.status === "failed") {
        message(run ? "Starting a fresh engine run…" : "Starting the engine…");
        run = await action(
          run
            ? {action: "reset", source_asset_id: demo.demo_source_asset_id, confirm_reset: true}
            : {action: "start", source_asset_id: demo.demo_source_asset_id}
        );
        await refresh();
      }
      const pause = ((demo && demo.cooldown_seconds) || 1) * 1000 + 300;
      while (run && run.status !== "completed" && run.status !== "failed") {
        try {
          run = await action({action: "accelerate", run_id: run.run_id});
        } catch (error) {
          if (error.transient) {
            driving = false;
            await spectate();
            driving = true;
            run = demo && demo.run;
            if (!run || run.status === "completed" || run.status === "failed") break;
            continue;
          }
          throw error;
        }
        message(`Streaming real sensor data through the model… cycle ${run.last_confirmed_cycle}`);
        await refresh();
        await sleep(pause);
      }
      await refresh();
      message(
        demo && demo.run && demo.run.status === "completed"
          ? "Simulation complete — see the verdict below."
          : "The run paused; press the button to continue."
      );
    } catch (error) {
      message(error.message || "The demo hit a snag — try again in a moment.");
    } finally {
      driving = false;
      render();
    }
  }

  runButton.addEventListener("click", drive);

  if (demo) {
    render();
    message(
      demo.run && demo.run.status === "completed"
        ? "A finished run is shown — replay it yourself with one click."
        : "Press run to stream the engine's sensor data through the live model."
    );
  } else {
    message("Waking the live system…");
    refresh().then(() => message("Press run to stream the engine's sensor data through the live model."));
  }
})();
