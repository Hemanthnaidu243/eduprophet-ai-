const FIELD_IDS = [
  "cgpa", "attendance", "backlogs", "study_hours", "family_income",
  "financial_stress", "lms_engagement", "mentor_sessions", "extracurricular",
  "coding_skill", "communication_skill", "creativity_skill",
  "internships", "projects", "certifications",
];

let salaryChart = null;
let radarChart = null;
let lastResult = null;
let lastCareers = null;
let debounceTimer = null;

function readInputs() {
  const out = {};
  FIELD_IDS.forEach((id) => {
    const el = document.getElementById(id);
    out[id] = parseFloat(el.value);
  });
  return out;
}

function wireLiveLabels() {
  FIELD_IDS.forEach((id) => {
    const el = document.getElementById(id);
    const label = document.getElementById("v-" + id);
    el.addEventListener("input", () => {
      label.textContent = el.value;
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(runReading, 350);
    });
  });
}

function bandColor(band) {
  if (band === "High Risk") return "var(--red)";
  if (band === "Moderate Risk") return "var(--gold)";
  return "var(--teal)";
}

function renderBars(containerId, factors, colorVar) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  factors.forEach((f) => {
    const pct = Math.min(Math.abs(f.contribution) * 400, 100);
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <div class="bar-label"><span>${f.label}</span><span>${f.value}</span></div>
      <div class="bar-track"><div class="bar-fill" style="width:${pct}%; background:${colorVar}"></div></div>
    `;
    container.appendChild(row);
  });
}

function renderScoreRing(score) {
  const circumference = 2 * Math.PI * 52;
  const offset = circumference - (score / 100) * circumference;
  const ring = document.getElementById("ring-fg");
  ring.style.strokeDashoffset = offset;
  let color = "var(--red)";
  if (score >= 70) color = "var(--teal)";
  else if (score >= 45) color = "var(--gold)";
  ring.style.stroke = color;
  document.getElementById("prophet-score").textContent = score;
}

function renderSalaryChart(pred, low, high) {
  const ctx = document.getElementById("salary-chart");
  const data = {
    labels: ["Conservative", "Predicted", "Optimistic"],
    datasets: [{
      data: [low, pred, high],
      backgroundColor: ["#766c5c", "#a8792f", "#2f6d52"],
      borderRadius: 6,
    }],
  };
  if (salaryChart) { salaryChart.data = data; salaryChart.update(); return; }
  salaryChart = new Chart(ctx, {
    type: "bar",
    data,
    options: {
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#766c5c" }, grid: { display: false } },
        y: { ticks: { color: "#766c5c" }, grid: { color: "#ddd0ac" }, title: { display: true, text: "LPA", color: "#766c5c" } },
      },
    },
  });
}

function renderRadar(gap) {
  const ctx = document.getElementById("radar-chart");
  const data = {
    labels: gap.axes,
    datasets: [
      { label: "Student", data: gap.student, backgroundColor: "rgba(168,121,47,0.20)", borderColor: "#a8792f" },
      { label: "Ideal for role", data: gap.ideal, backgroundColor: "rgba(47,109,82,0.15)", borderColor: "#2f6d52" },
    ],
  };
  if (radarChart) { radarChart.data = data; radarChart.update(); return; }
  radarChart = new Chart(ctx, {
    type: "radar",
    data,
    options: {
      scales: {
        r: {
          min: 0, max: 10,
          angleLines: { color: "#ddd0ac" },
          grid: { color: "#ddd0ac" },
          pointLabels: { color: "#2b2420", font: { size: 11 } },
          ticks: { display: false },
        },
      },
      plugins: { legend: { labels: { color: "#766c5c" } } },
    },
  });
}

function renderCareers(list) {
  const container = document.getElementById("career-list");
  container.innerHTML = "";
  list.forEach((c) => {
    const div = document.createElement("div");
    div.className = "career-item";
    div.innerHTML = `
      <div class="name">${c.career}</div>
      <div class="fit">${c.fit_pct}% fit${c.avg_salary_lpa ? " · ~" + c.avg_salary_lpa + " LPA avg" : ""}</div>
      <div class="desc">${c.description}</div>
    `;
    container.appendChild(div);
  });
  document.getElementById("skillgap-target").textContent = "— best match: " + list[0].career;
  renderRadar(list[0].skill_gap);
}

async function runReading() {
  const inputs = readInputs();

  const [predRes, careerRes] = await Promise.all([
    fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(inputs) }),
    fetch("/api/careers", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(inputs) }),
  ]);
  const pred = await predRes.json();
  const careers = await careerRes.json();
  lastResult = pred;
  lastCareers = careers;

  document.getElementById("narrative").textContent = pred.narrative;
  document.getElementById("risk-msg").textContent = pred.dropout.message;
  document.getElementById("risk-msg").style.color = bandColor(pred.dropout.risk_band);
  renderScoreRing(pred.prophet_score);

  document.getElementById("dropout-prob").textContent = Math.round(pred.dropout.probability * 100) + "%";
  const band = document.getElementById("dropout-band");
  band.textContent = pred.dropout.risk_band;
  band.style.color = bandColor(pred.dropout.risk_band);
  renderBars("dropout-bars", pred.dropout.top_factors, "var(--red)");

  document.getElementById("placement-prob").textContent = Math.round(pred.placement.probability * 100) + "%";
  document.getElementById("placement-note").textContent =
    pred.placement.probability >= 0.6 ? "Strong placement outlook" : "Needs stronger placement signals";
  renderBars("placement-bars", pred.placement.top_factors, "var(--teal)");

  document.getElementById("salary-pred").textContent = pred.salary.predicted_lpa + " LPA";
  document.getElementById("salary-range").textContent =
    "Range " + pred.salary.range_lpa[0] + " – " + pred.salary.range_lpa[1] + " LPA";
  renderSalaryChart(pred.salary.predicted_lpa, pred.salary.range_lpa[0], pred.salary.range_lpa[1]);

  renderCareers(careers.recommendations);
}

function downloadReport() {
  if (!lastResult || !lastCareers) return;
  const p = lastResult;
  const c = lastCareers.recommendations;

  const { jsPDF } = window.jspdf;
  const doc = new jsPDF({ unit: "pt", format: "a4" });

  const margin = 48;
  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();
  const contentWidth = pageWidth - margin * 2;
  const gold = [168, 121, 47];
  const ink = [43, 36, 32];
  const muted = [118, 108, 92];
  const lineColor = [221, 208, 172];
  let y = margin;

  function ensureSpace(h) {
    if (y + h > pageHeight - margin) {
      doc.addPage();
      y = margin;
    }
  }

  function heading(text) {
    ensureSpace(30);
    doc.setFont("times", "bold");
    doc.setFontSize(13);
    doc.setTextColor(...gold);
    doc.text(text.toUpperCase(), margin, y);
    y += 6;
    doc.setDrawColor(...lineColor);
    doc.setLineWidth(0.75);
    doc.line(margin, y, pageWidth - margin, y);
    y += 18;
  }

  function bodyLine(label, value, opts = {}) {
    ensureSpace(18);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(10.5);
    doc.setTextColor(...ink);
    const labelText = label ? `${label}: ` : "";
    doc.text(labelText, margin, y);
    const labelWidth = label ? doc.getTextWidth(labelText) : 0;

    doc.setFont("helvetica", "normal");
    doc.setTextColor(...(opts.color || ink));
    const wrapped = doc.splitTextToSize(String(value), contentWidth - labelWidth);
    doc.text(wrapped[0] || "", margin + labelWidth, y);
    y += 15;
    for (let i = 1; i < wrapped.length; i++) {
      ensureSpace(15);
      doc.text(wrapped[i], margin, y);
      y += 15;
    }
  }

  function paragraph(text) {
    ensureSpace(15);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(10.5);
    doc.setTextColor(...ink);
    const wrapped = doc.splitTextToSize(text, contentWidth);
    wrapped.forEach((ln) => {
      ensureSpace(15);
      doc.text(ln, margin, y);
      y += 15;
    });
    y += 4;
  }

  // ---- Title block ----
  doc.setFont("times", "bold");
  doc.setFontSize(20);
  doc.setTextColor(...ink);
  doc.text("EduProphet AI", margin, y);
  y += 22;
  doc.setFont("times", "italic");
  doc.setFontSize(12);
  doc.setTextColor(...muted);
  doc.text("Student Reading Report", margin, y);
  y += 10;
  doc.setDrawColor(...gold);
  doc.setLineWidth(1.2);
  doc.line(margin, y, pageWidth - margin, y);
  y += 26;

  bodyLine("Prophet Score", `${p.prophet_score} / 100`);
  paragraph(p.narrative);
  y += 6;

  heading("Dropout Risk");
  bodyLine("Probability", `${(p.dropout.probability * 100).toFixed(1)}%  (${p.dropout.risk_band})`);
  bodyLine("Note", p.dropout.message);
  bodyLine("Top factors", p.dropout.top_factors.map(f => `${f.label} (${f.value})`).join(", "));
  y += 6;

  heading("Placement");
  bodyLine("Probability", `${(p.placement.probability * 100).toFixed(1)}%`);
  bodyLine("Top factors", p.placement.top_factors.map(f => `${f.label} (${f.value})`).join(", "));
  y += 6;

  heading("Salary Forecast");
  bodyLine("Predicted", `${p.salary.predicted_lpa} LPA`);
  bodyLine("Range", `${p.salary.range_lpa[0]} – ${p.salary.range_lpa[1]} LPA`);
  y += 6;

  heading("Career Recommendations");
  c.forEach((r) => {
    bodyLine(r.career, `${r.fit_pct}% fit`, { color: gold });
    paragraph(r.description);
  });

  // ---- Footer on every page ----
  const pageCount = doc.internal.getNumberOfPages();
  for (let i = 1; i <= pageCount; i++) {
    doc.setPage(i);
    doc.setFont("helvetica", "italic");
    doc.setFontSize(8.5);
    doc.setTextColor(...muted);
    doc.text(
      "Generated by EduProphet AI — synthetic-data demo. Not a substitute for institutional counseling.",
      margin,
      pageHeight - 24
    );
    doc.text(String(i), pageWidth - margin, pageHeight - 24, { align: "right" });
  }

  doc.save("eduprophet_reading.pdf");
}

document.getElementById("cast-btn").addEventListener("click", runReading);
document.getElementById("report-btn").addEventListener("click", downloadReport);
wireLiveLabels();
runReading();
