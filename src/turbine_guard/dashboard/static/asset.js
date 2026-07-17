(() => {
  const node = document.getElementById("asset-data");
  if (!node) return;
  const data = JSON.parse(node.textContent);
  const layout = {paper_bgcolor:"transparent",plot_bgcolor:"transparent",font:{color:"#91a7ba"},margin:{l:55,r:20,t:20,b:45},xaxis:{title:"Cycle",gridcolor:"#203851"},yaxis:{title:"RUL (cycles)",gridcolor:"#203851"},legend:{orientation:"h"}};
  if (window.Plotly && data.predictions.length) {
    const cycles = data.predictions.map(p => p.cycle);
    window.Plotly.newPlot("rul-chart", [
      {x:cycles,y:data.predictions.map(p=>p.upper_rul),name:"Upper",mode:"lines",line:{width:0},hoverinfo:"skip"},
      {x:cycles,y:data.predictions.map(p=>p.lower_rul),name:"Interval",mode:"lines",fill:"tonexty",fillcolor:"rgba(99,167,255,.18)",line:{width:0}},
      {x:cycles,y:data.predictions.map(p=>p.predicted_rul),name:"Predicted RUL",mode:"lines",line:{color:"#55d6be",width:3}},
    ], layout, {responsive:true,displaylogo:false});
    window.Plotly.newPlot("risk-chart", [{x:cycles,y:data.predictions.map(p=>p.risk_level),mode:"lines+markers",line:{color:"#63a7ff",shape:"hv",width:2},marker:{size:8,color:data.predictions.map(p=>p.risk_level==="critical"?"#ff6b74":p.risk_level==="warning"?"#f4c45c":"#55d6be")},name:"Risk"}], {...layout,yaxis:{title:"Risk",categoryorder:"array",categoryarray:["critical","warning","healthy"],gridcolor:"#203851"}}, {responsive:true,displaylogo:false});
  }
  if (window.Plotly && data.sensor_history.length) {
    window.Plotly.newPlot("sensor-chart", data.sensor_columns.map((column,index)=>({x:data.sensor_history.map(p=>p.cycle),y:data.sensor_history.map(p=>p.values[column]),name:column,mode:"lines",line:{width:2,color:["#55d6be","#63a7ff","#f4c45c","#ff6b74","#a78bfa","#f472b6"][index]}})), {...layout,yaxis:{title:"Recorded value",gridcolor:"#203851"}}, {responsive:true,displaylogo:false});
  }
  document.getElementById("apply-sensors")?.addEventListener("click", () => {
    const selected = [...document.querySelectorAll('input[name="sensor"]:checked')].map(el=>el.value).slice(0,6);
    const url = new URL(window.location.href); url.searchParams.set("sensors", selected.join(",")); window.location.assign(url);
  });
})();
