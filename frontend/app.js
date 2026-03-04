/* ── State ─────────────────────────────────────────────────── */
const DATASET_AVG = 206855;   // fallback; overwritten by /api/metadata

let avgPrice = DATASET_AVG;

/* ── Input bounds (match training data) ─────────────────────── */
const BOUNDS = {
  latitude: [32, 41], 
  housing_median_age: [1, 52],
  median_income: [1.5, 9],
  rooms_per_household: [1, 8],
  bedrooms_per_room: [0.05, 0.35],
};

function clamp(val, key) {
  const [lo, hi] = BOUNDS[key];
  const v = Math.max(lo, Math.min(hi, parseFloat(val)));
  return key === "latitude" ? Math.round(v) : v;
}

/* ── Slider ↔ Number sync ──────────────────────────────────── */
const pairs = [
  ["s-lat", "n-lat"],
  ["s-age", "n-age"],
  ["s-inc", "n-inc"],
  ["s-rph", "n-rph"],
  ["s-bpr", "n-bpr"],
];

pairs.forEach(([sliderId, numId]) => {
  const slider = document.getElementById(sliderId);
  const num    = document.getElementById(numId);
  slider.addEventListener("input", () => { num.value = slider.value; });
  num.addEventListener("input",    () => { slider.value = num.value; });
});

/* ── Load metadata (feature importance only) ── */
async function loadMetadata() {
  try {
    const res  = await fetch("/api/metadata");
    const data = await res.json();
    renderImportance(data.feature_importances || []);
  } catch (e) {
    console.error("Could not load metadata:", e);
  }
}

function renderImportance(items) {
  const container = document.getElementById("importance-bars");
  container.innerHTML = "";
  const max = items[0]?.importance ?? 1;

  items.forEach((item, i) => {
    const pct  = (item.importance / max * 100).toFixed(1);
    const pctAbs = (item.importance * 100).toFixed(1);
    const row  = document.createElement("div");
    row.className = "imp-row";

    row.innerHTML = `
      <span class="imp-name" title="${item.feature}">${item.feature}</span>
      <div class="imp-track">
        <div class="imp-fill ${i === 0 ? "top" : ""}" style="width:${pct}%"></div>
      </div>
      <span class="imp-pct">${pctAbs}%</span>
    `;
    container.appendChild(row);
  });
}

/* ── Load preset examples ──────────────────────────────────── */
async function loadPresets() {
  try {
    const res     = await fetch("/api/examples");
    const presets = await res.json();
    const grid    = document.getElementById("presets");
    grid.innerHTML = "";

    presets.forEach(preset => {
      const btn = document.createElement("button");
      btn.className  = "preset-btn";
      btn.type       = "button";
      btn.textContent = preset.label;
      btn.addEventListener("click", () => {
        applyPreset(preset.values);
        document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
      });
      grid.appendChild(btn);
    });
  } catch (e) {
    console.error("Could not load presets:", e);
  }
}

function applyPreset(values) {
    const set = (numId, sliderId, val, key) => {
    const clamped = clamp(val, key);
    document.getElementById(numId).value   = clamped;
    document.getElementById(sliderId).value = clamped;
  };
  set("n-lat", "s-lat", values.latitude, "latitude");
  set("n-age", "s-age", values.housing_median_age, "housing_median_age");
  set("n-inc", "s-inc", values.median_income, "median_income");
  set("n-rph", "s-rph", values.rooms_per_household, "rooms_per_household");
  set("n-bpr", "s-bpr", values.bedrooms_per_room, "bedrooms_per_room");
  document.getElementById("sel-ocean").value = values.ocean_proximity;
}

/* ── Predict ──────────────────────────────────────────────── */
document.getElementById("predict-form").addEventListener("submit", async (e) => {
  e.preventDefault();

  const btn    = document.getElementById("predict-btn");
  const btnTxt = document.getElementById("btn-text");
  const idle   = document.getElementById("result-idle");
  const live   = document.getElementById("result-live");
  const errEl  = document.getElementById("result-error");

  btnTxt.textContent = "Predicting…";
  btn.disabled = true;
  errEl.classList.add("hidden");
  live.classList.add("hidden");

  const form = document.getElementById("predict-form");
  const fd   = new FormData(form);

  const payload = {
    latitude:            clamp(fd.get("latitude"), "latitude"),
    housing_median_age:  clamp(fd.get("housing_median_age"), "housing_median_age"),
    median_income:       clamp(fd.get("median_income"), "median_income"),
    ocean_proximity:     fd.get("ocean_proximity"),
    rooms_per_household: clamp(fd.get("rooms_per_household"), "rooms_per_household"),
    bedrooms_per_room:   clamp(fd.get("bedrooms_per_room"), "bedrooms_per_room"),
  };

  try {
    const res  = await fetch("/api/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok) throw new Error(data.error || "Prediction failed");

    const pred = data.prediction.value;

    // Price
    document.getElementById("price-out").textContent = data.prediction.formatted;

    // Context chips
    const diffPct = ((pred - avgPrice) / avgPrice * 100).toFixed(1);
    const sign    = diffPct >= 0 ? "+" : "";
    document.getElementById("ctx-vs-avg").textContent =
      `${sign}${diffPct}% vs CA average`;

    document.getElementById("ctx-ocean").textContent =
      `Location: ${payload.ocean_proximity}`;

    const incDesc =
      payload.median_income >= 7   ? "High income" :
      payload.median_income >= 4   ? "Mid income"  : "Low income";
    document.getElementById("ctx-income").textContent =
      `${incDesc} ($${(payload.median_income * 10).toFixed(0)}k)`;

    // Show result
    idle.classList.add("hidden");
    live.classList.remove("hidden");

  } catch (err) {
    errEl.textContent = "Error: " + err.message;
    errEl.classList.remove("hidden");
  } finally {
    btnTxt.textContent = "Predict Price";
    btn.disabled = false;
  }
});

/* ── Boot ──────────────────────────────────────────────────── */
loadMetadata();
loadPresets();
