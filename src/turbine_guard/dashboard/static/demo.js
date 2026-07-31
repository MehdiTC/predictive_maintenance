(() => {
  const $ = (id) => document.getElementById(id);
  const buttonNode = $("demo-run");
  const dataNode = $("demo-data");
  if (!buttonNode || !dataNode) return;

  let demo = null;
  try { demo = JSON.parse(dataNode.textContent); } catch { demo = null; }

  let series = (demo && demo.series) || [];
  let shown = series.length;
  let mode = "idle";
  let fetching = false;
  let cooldownUntil = 0;

  const formatCycles = (value) => value == null ? "—" : String(Math.round(Number(value)));
  const run = () => (demo && demo.run) || null;
  const finished = () => run() && run().status === "completed";
  const active = () => run() && !["completed", "failed"].includes(run().status);
  const warningHorizon = () => (demo && demo.warning_horizon) || 50;
  const criticalHorizon = () => (demo && demo.critical_horizon) || 30;

  async function fetchState() {
    if (fetching) return;
    fetching = true;
    try {
      const response = await fetch("/v1/demo", {headers: {Accept: "application/json"}});
      if (response.ok) {
        demo = await response.json();
        const fresh = demo.series || [];
        if (fresh.length < shown) shown = fresh.length;
        series = fresh;
      }
    } catch { /* A later poll retries transient network failures. */
    } finally {
      fetching = false;
    }
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
    let detail = "The live replay returned an error.";
    try {
      const body = await response.json();
      detail = (body.error && body.error.message) || detail;
    } catch { /* The error response was not JSON. */ }
    const error = new Error(detail);
    error.transient = response.status === 409 || response.status === 429;
    throw error;
  }

  function chart() {
    if (!window.Plotly) return;
    const visible = series.slice(0, shown);
    const cycles = visible.map((point) => point.cycle);
    const current = visible[visible.length - 1];
    const complete = finished() && shown === series.length;
    const finalCycle = complete ? run().final_cycle : null;
    const maxCycle = Math.max(60, finalCycle || 0, (cycles[cycles.length - 1] || 0) + 12);
    const upperValues = visible.map((point) => point.upper_rul);
    const predictedValues = visible.map((point) => point.predicted_rul);
    const yMax = Math.max(70, ...upperValues.filter((value) => value != null), ...predictedValues) + 8;

    const layout = {
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      font: {color: "#94a8bc", size: 11},
      margin: {l: 52, r: 14, t: 48, b: 48},
      xaxis: {
        title: {text: "operating cycle", standoff: 12},
        gridcolor: "#182a3d",
        zeroline: false,
        range: [0, maxCycle],
        fixedrange: true,
      },
      yaxis: {
        title: {text: "cycles remaining", standoff: 8},
        gridcolor: "#182a3d",
        zeroline: false,
        range: [0, yMax],
        fixedrange: true,
      },
      legend: {
        orientation: "h",
        x: 0,
        y: 1.14,
        xanchor: "left",
        yanchor: "top",
        font: {size: 10},
        bgcolor: "rgba(0,0,0,0)",
      },
      hovermode: "x unified",
      hoverlabel: {bgcolor: "#102035", bordercolor: "#294968", font: {color: "#eef5fc"}},
      shapes: [
        {type: "line", xref: "paper", x0: 0, x1: 1, y0: warningHorizon(), y1: warningHorizon(),
          line: {color: "rgba(244,191,88,.72)", width: 1, dash: "dot"}},
        {type: "line", xref: "paper", x0: 0, x1: 1, y0: criticalHorizon(), y1: criticalHorizon(),
          line: {color: "rgba(255,104,119,.72)", width: 1, dash: "dot"}},
      ],
      annotations: [
        {xref: "paper", x: 1, y: warningHorizon(), text: "warning · 50", showarrow: false,
          xanchor: "right", yanchor: "bottom", font: {color: "#f4bf58", size: 9}},
        {xref: "paper", x: 1, y: criticalHorizon(), text: "critical · 30", showarrow: false,
          xanchor: "right", yanchor: "bottom", font: {color: "#ff6877", size: 9}},
      ],
    };

    if (current) {
      layout.shapes.push({type: "line", x0: current.cycle, x1: current.cycle, yref: "paper", y0: 0, y1: 1,
        line: {color: "rgba(238,245,252,.3)", width: 1, dash: "dot"}});
      layout.annotations.push({x: current.cycle, yref: "paper", y: 1, text: "current", showarrow: false,
        xanchor: "right", yanchor: "bottom", font: {color: "#94a8bc", size: 9}});
    }
    if (finalCycle != null) {
      layout.shapes.push({type: "line", x0: finalCycle, x1: finalCycle, yref: "paper", y0: 0, y1: 1,
        line: {color: "#ff6877", width: 1.5, dash: "dot"}});
      layout.annotations.push({x: finalCycle, yref: "paper", y: .96, text: "failure", showarrow: false,
        xanchor: "right", yanchor: "top", font: {color: "#ff6877", size: 10}});
    }

    const traces = [
      {
        x: cycles,
        y: upperValues,
        mode: "lines",
        line: {width: 0},
        hoverinfo: "skip",
        showlegend: false,
      },
      {
        x: cycles,
        y: visible.map((point) => point.lower_rul),
        mode: "lines",
        name: "90% range",
        fill: "tonexty",
        fillcolor: "rgba(101,169,255,.14)",
        line: {width: 0},
        hovertemplate: "range low %{y:.0f} cycles<extra></extra>",
      },
      {
        x: cycles,
        y: predictedValues,
        mode: "lines",
        name: "Predicted RUL",
        line: {color: "#65a9ff", width: 2.4},
        hovertemplate: "%{y:.0f} cycles predicted<extra></extra>",
      },
    ];
    if (finalCycle != null) {
      traces.push({
        x: cycles,
        y: cycles.map((cycle) => Math.max(0, finalCycle - cycle)),
        mode: "lines",
        name: "Actual RUL",
        line: {color: "#eef5fc", width: 1.5, dash: "dash"},
        hovertemplate: "%{y:.0f} cycles actual<extra></extra>",
      });
    }

    window.Plotly.react("demo-chart", traces, layout, {
      responsive: true,
      displaylogo: false,
      displayModeBar: false,
    });
  }

  function updateDecision() {
    const point = series[shown - 1] || null;
    const risk = point ? point.risk_level.toLowerCase() : "unknown";

    $("demo-cycle").textContent = point ? String(point.cycle) : "—";
    $("demo-rul").textContent = point ? formatCycles(point.predicted_rul) : "—";
    $("demo-interval").textContent = point && point.lower_rul != null
      ? `${formatCycles(point.lower_rul)}–${formatCycles(point.upper_rul)} cycles`
      : "—";
    $("demo-model").textContent = demo && demo.model_version ? `v${demo.model_version}` : "—";

    const riskNode = $("demo-risk");
    riskNode.textContent = point ? risk : "No data";
    riskNode.className = `risk ${risk}`;

    const copy = {
      healthy: {
        title: "Forecast remains above the maintenance horizon.",
        recommendation: "Continue monitoring. No maintenance action is indicated by the current forecast.",
      },
      warning: {
        title: "The forecast has entered the warning horizon.",
        recommendation: "Plan maintenance. The predicted life has crossed the 50-cycle warning threshold.",
      },
      critical: {
        title: "The forecast has entered the critical horizon.",
        recommendation: "Prioritize maintenance. The predicted life is at or below the 30-cycle critical threshold.",
      },
      unknown: {
        title: "Start the replay to generate a forecast.",
        recommendation: "No maintenance decision is available yet.",
      },
    }[risk];
    $("demo-decision").textContent = copy.title;
    $("demo-recommendation").textContent = copy.recommendation;

    const replay = run();
    const progress = replay ? Math.max(0, Math.min(100, Number(replay.progress_percent) || 0)) : 0;
    $("demo-progress-bar").style.width = `${progress}%`;
    $("demo-progress").textContent = replay
      ? `${progress.toFixed(0)}% of the engine sequence observed`
      : "Replay not started";
  }

  function updateButton() {
    buttonNode.disabled = false;
    const backlog = series.length - shown;
    if (!demo || demo.enabled === false) {
      buttonNode.disabled = true;
      buttonNode.textContent = "Live replay unavailable";
    } else if (mode === "updating" || backlog > 0) {
      buttonNode.disabled = true;
      buttonNode.textContent = "Updating forecast…";
    } else if (Date.now() < cooldownUntil) {
      buttonNode.disabled = true;
      buttonNode.textContent = "Ready in a moment…";
    } else if (finished()) {
      buttonNode.textContent = "Replay from the start";
    } else if (active()) {
      const batch = demo.max_cycles_per_request || 10;
      buttonNode.textContent = `Advance up to ${batch} cycles`;
    } else {
      buttonNode.textContent = "Start live replay";
    }
  }

  function message(text) { $("demo-message").textContent = text; }

  function updateVerdict() {
    const node = $("demo-punchline");
    const replay = run();
    if (!finished() || shown < series.length || replay.final_cycle == null) {
      node.hidden = true;
      return;
    }
    const firstAlert = series.find((point) => point.risk_level !== "healthy");
    if (firstAlert) {
      const lead = replay.final_cycle - firstAlert.cycle;
      node.innerHTML =
        `Failure occurred at cycle <strong>${replay.final_cycle}</strong>. ` +
        `The first maintenance signal arrived at cycle <strong>${firstAlert.cycle}</strong>, ` +
        `providing <span class="lead">${lead} cycles of lead time</span>.` +
        `<small>Each forecast was generated in sequence; the failure cycle was hidden until completion.</small>`;
    } else {
      node.innerHTML =
        `Failure occurred at cycle <strong>${replay.final_cycle}</strong>, without an earlier maintenance signal.` +
        `<small>Each forecast was generated in sequence; the failure cycle was hidden until completion.</small>`;
    }
    node.hidden = false;
  }

  function paint() {
    chart();
    updateDecision();
    updateButton();
    updateVerdict();
  }

  function animationTick() {
    const backlog = series.length - shown;
    if (backlog > 0) {
      shown += 1;
      const point = series[shown - 1];
      message(`Forecast updated with operating cycle ${point.cycle}.`);
      paint();
    } else if (finished()) {
      updateVerdict();
      updateButton();
    }
    const delay = backlog >= 15 ? 70 : backlog >= 5 ? 120 : 220;
    setTimeout(animationTick, delay);
  }

  async function advanceReplay() {
    if (mode !== "idle" || Date.now() < cooldownUntil) return;
    mode = "updating";
    updateButton();
    message("Sending the next sensor cycles through the model…");
    try {
      let replay = run();
      if (!replay || replay.status === "completed" || replay.status === "failed") {
        const payload = replay
          ? {action: "reset", source_asset_id: demo.demo_source_asset_id, confirm_reset: true}
          : {action: "start", source_asset_id: demo.demo_source_asset_id};
        await act(payload);
        if (payload.action === "reset") {
          series = [];
          shown = 0;
          $("demo-punchline").hidden = true;
        }
        await fetchState();
        replay = run();
      }
      if (replay && !["completed", "failed"].includes(replay.status)) {
        await act({action: "accelerate", run_id: replay.run_id});
        await fetchState();
      }
      cooldownUntil = Date.now() + (((demo && demo.cooldown_seconds) || 1) * 1000);
      setTimeout(updateButton, Math.max(0, cooldownUntil - Date.now()) + 20);
    } catch (error) {
      message(error.transient
        ? "Another replay update is finishing. Try again in a moment."
        : (error.message || "The live replay hit a snag. Try again in a moment."));
    } finally {
      mode = "idle";
      paint();
    }
  }

  buttonNode.addEventListener("click", advanceReplay);
  animationTick();

  setInterval(async () => {
    if (!active() || mode !== "idle") return;
    await fetchState();
    paint();
  }, 3000);

  if (!demo) {
    message("Connecting to the live replay…");
    fetchState().then(() => {
      paint();
      message(demo ? "Start the replay when you're ready." : "The live replay is unavailable.");
    });
  } else if (!series.length) {
    message("Start the replay to reveal the engine sequence in bounded steps.");
  } else if (finished()) {
    message("Replay complete. The actual remaining life is now visible on the chart.");
  } else {
    message("Advance the replay to process the next sensor cycles.");
  }
  paint();
})();
