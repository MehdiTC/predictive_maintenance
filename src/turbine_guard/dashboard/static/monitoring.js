(() => {
  const node = document.getElementById("monitoring-data");
  if (!node || !window.Plotly) return;
  const {drift} = JSON.parse(node.textContent);
  if (!drift.top_features.length) return;
  window.Plotly.newPlot("drift-chart", [{type:"bar",orientation:"h",y:drift.top_features.map(x=>x.feature).reverse(),x:drift.top_features.map(x=>x.psi || 0).reverse(),marker:{color:drift.top_features.map(x=>x.drifted?"#ff6b74":x.warning?"#f4c45c":"#55d6be").reverse()},name:"PSI"}], {paper_bgcolor:"transparent",plot_bgcolor:"transparent",font:{color:"#91a7ba"},margin:{l:170,r:20,t:20,b:40},xaxis:{title:"Population Stability Index",gridcolor:"#203851"},yaxis:{automargin:true}}, {responsive:true,displaylogo:false});
})();
