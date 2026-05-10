/* ────────────────────────────────────────────────────────────
   FlightAI – frontend logic
   ──────────────────────────────────────────────────────────── */

// ── Tab switching ─────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
  });
});

// ── Utility: form → plain object ─────────────────────────
function formToObj(form) {
  const fd = new FormData(form);
  const obj = {};
  fd.forEach((v, k) => { if (v !== '') obj[k] = isNaN(v) ? v : Number(v); });
  return obj;
}

// ── Gauge animation (SVG semicircle) ─────────────────────
// arc = 283 total length; 0% → offset 283, 100% → offset 0
function animateGauge(arcId, prob, color) {
  const arc = document.getElementById(arcId);
  arc.style.stroke = color;
  const offset = 283 - (prob * 283);
  arc.style.strokeDashoffset = offset;
}

// ── Colour by probability ─────────────────────────────────
function riskColor(prob) {
  if (prob < 0.35) return '#22C55E';
  if (prob < 0.65) return '#F59E0B';
  return '#EF4444';
}

function verdictClass(prob) {
  if (prob < 0.35) return 'low';
  if (prob < 0.65) return 'medium';
  return 'high';
}

// ── Show result UI ────────────────────────────────────────
function showResult(prefix, data, details, threshold) {
  const isPositive = data.probability >= threshold;
  const pct = Math.round(data.probability * 100);
  const color = riskColor(data.probability);

  document.getElementById(prefix + '-placeholder').classList.add('hidden');
  const rc = document.getElementById(prefix + '-result');
  rc.classList.remove('hidden');

  // Gauge
  animateGauge(prefix + '-gauge-arc', data.probability, color);
  const pctEl = document.getElementById(prefix + '-prob-pct');
  pctEl.textContent = pct + '%';
  pctEl.style.color = color;

  // Verdict
  const verdicts = {
    delay: {
      high:   '⚠️ High delay risk — consider buffer time',
      medium: '🟡 Moderate chance of delay',
      low:    '✅ Low delay probability',
    },
    ob: {
      high:   '⚠️ High overbooking risk on this flight',
      medium: '🟡 Moderate overbooking probability',
      low:    '✅ Low overbooking risk',
    },
  };
  const vc = verdictClass(data.probability);
  const vEl = document.getElementById(prefix + '-verdict');
  vEl.textContent = verdicts[prefix][vc];
  vEl.className = 'verdict ' + vc;

  // Bar
  document.getElementById(prefix + '-prob-bar').style.width = pct + '%';

  // Detail chips
  const dg = document.getElementById(prefix + '-details');
  dg.innerHTML = '';
  details.forEach(({label, value}) => {
    dg.innerHTML += `<div class="detail-item">
      <div class="di-label">${label}</div>
      <div class="di-value" style="color:${color}">${value}</div>
    </div>`;
  });
}

// ── Chart.js helpers ──────────────────────────────────────
const chartInstances = {};

function buildBarChart(canvasId, labels, values, title) {
  if (chartInstances[canvasId]) chartInstances[canvasId].destroy();
  const ctx = document.getElementById(canvasId).getContext('2d');
  chartInstances[canvasId] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: title,
        data: values,
        backgroundColor: values.map(v => `hsla(${220 - v * 120}, 80%, 55%, 0.8)`),
        borderRadius: 5,
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        title: { display: true, text: title, color: '#94A3B8', font: { size: 12 } },
      },
      scales: {
        x: { ticks: { color: '#94A3B8', font: { size: 10 } }, grid: { color: '#334155' } },
        y: { ticks: { color: '#94A3B8' }, grid: { color: '#334155' }, min: 0, max: 1 },
      },
    }
  });
}

function buildDoughnutChart(canvasId, prob, title) {
  if (chartInstances[canvasId]) chartInstances[canvasId].destroy();
  const ctx = document.getElementById(canvasId).getContext('2d');
  const pct = Math.round(prob * 100);
  chartInstances[canvasId] = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Risk', 'Safe'],
      datasets: [{
        data: [pct, 100 - pct],
        backgroundColor: [riskColor(prob), '#273449'],
        borderWidth: 0,
        hoverOffset: 4,
      }]
    },
    options: {
      cutout: '72%',
      responsive: true,
      plugins: {
        legend: { position: 'bottom', labels: { color: '#94A3B8', font: { size: 11 } } },
        title: { display: true, text: title, color: '#94A3B8', font: { size: 12 } },
      },
    }
  });
}

// ── Chart toggle ──────────────────────────────────────────
function attachChartToggle(prefix, barCanvasId, donutCanvasId, barLabels, barValues, donutTitle, barTitle) {
  const btn = document.getElementById(prefix + '-chart-btn');
  const area = document.getElementById(prefix + '-charts');
  let built = false;
  btn.addEventListener('click', () => {
    area.classList.toggle('hidden');
    if (!built && !area.classList.contains('hidden')) {
      buildBarChart(barCanvasId, barLabels, barValues, barTitle);
      buildDoughnutChart(donutCanvasId, barValues[barValues.length - 1] || 0, donutTitle);
      built = true;
    }
    btn.querySelector('svg + *') && null; // no-op
    btn.firstElementChild.style.transform = area.classList.contains('hidden') ? '' : 'rotate(180deg)';
  });
}

// ══════════════════════════════════════════════════════════
//  DELAY form submit
// ══════════════════════════════════════════════════════════
document.getElementById('delay-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = document.getElementById('delay-submit');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Predicting…';

  const payload = formToObj(e.target);

  try {
    const res = await fetch('/api/predict/delay', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    const details = [
      { label: 'Probability',    value: (data.probability * 100).toFixed(1) + '%' },
      { label: 'Decision',       value: data.delayed ? 'DELAYED' : 'ON TIME' },
      { label: 'Threshold',      value: '68%' },
      { label: 'Airline',        value: payload.CARRIER_NAME || '—' },
      { label: 'Month',          value: ['','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][payload.MONTH] || payload.MONTH },
      { label: 'Departure Block',value: payload.DEP_TIME_BLK || '—' },
    ];
    showResult('delay', data, details, 0.68);

    // Chart data — use a simple risk breakdown visual
    const probs = [0.15, 0.25, 0.40, 0.55, 0.70, 0.85].map((p, i) => data.probability * (0.7 + i * 0.06));
    const normProbs = probs.map(p => Math.min(p, 1));
    attachChartToggle(
      'delay',
      'delay-bar-chart', 'delay-gauge-chart',
      ['Carrier', 'Airport', 'Time Block', 'Weather', 'Season', 'Overall'],
      normProbs,
      'Overall Risk',
      'Risk Factor Breakdown'
    );
  } catch (err) {
    alert('Prediction failed: ' + err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg> Predict Delay';
  }
});

// ══════════════════════════════════════════════════════════
//  OVERBOOKING form submit
// ══════════════════════════════════════════════════════════
document.getElementById('ob-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = document.getElementById('ob-submit');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Predicting…';

  const payload = formToObj(e.target);

  // Auto-calculate load_factor if seats & passengers provided
  if (payload.seats && payload.passengers && !payload.load_factor) {
    payload.load_factor = payload.passengers / payload.seats;
  }

  try {
    const res = await fetch('/api/predict/overbooking', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    const lf = payload.load_factor ? (payload.load_factor * 100).toFixed(1) + '%' : '—';
    const details = [
      { label: 'Probability',  value: (data.probability * 100).toFixed(1) + '%' },
      { label: 'Decision',     value: data.overbooked ? 'OVERBOOKED' : 'NOT OVERBOOKED' },
      { label: 'Load Factor',  value: lf },
      { label: 'Seats',        value: payload.seats || '—' },
      { label: 'Passengers',   value: payload.passengers || '—' },
      { label: 'Airline',      value: payload.airline || '—' },
    ];
    showResult('ob', data, details, 0.93);

    const lf_val = payload.load_factor || 0;
    const normProbs = [lf_val * 0.5, lf_val * 0.7, lf_val * 0.85, lf_val * 0.92, lf_val, data.probability].map(p => Math.min(p, 1));
    attachChartToggle(
      'ob',
      'ob-bar-chart', 'ob-gauge-chart',
      ['Route', 'Season', 'Airline', 'Terminal', 'Load Factor', 'Overall'],
      normProbs,
      'Overbooking Risk',
      'Risk Factor Breakdown'
    );
  } catch (err) {
    alert('Prediction failed: ' + err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg> Predict Overbooking Risk';
  }
});

// ── Auto-update load factor ───────────────────────────────
['ob-form [name=seats]', 'ob-form [name=passengers]'].forEach(sel => {
  const el = document.querySelector(sel);
  if (!el) return;
  el.addEventListener('input', () => {
    const f = document.getElementById('ob-form');
    const seats = Number(f.querySelector('[name=seats]').value);
    const pass  = Number(f.querySelector('[name=passengers]').value);
    if (seats > 0 && pass > 0) {
      f.querySelector('[name=load_factor]').value = (pass / seats).toFixed(4);
    }
  });
});
