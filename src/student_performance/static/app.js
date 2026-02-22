async function fetchSchema() {
  const r = await fetch("/schema");
  if (!r.ok) throw new Error(`Schema fetch failed: ${r.status}`);
  return await r.json();
}

// Convert snake_case to "Title Case With Spaces"
function formatLabel(name) {
  return name
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

function makeField(feature) {
  const wrapper = document.createElement("div");
  wrapper.className = "field";

  const label = document.createElement("label");
  label.textContent = formatLabel(feature.name);
  label.htmlFor = feature.name;
  wrapper.appendChild(label);

  let input;

  // If schema provides categories → use a <select> dropdown
  if (feature.categories && feature.categories.length > 0) {
    const selectWrap = document.createElement("div");
    selectWrap.className = "select-wrap";

    input = document.createElement("select");
    input.id = feature.name;
    input.name = feature.name;

    // Placeholder option
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = `Select ${formatLabel(feature.name)}…`;
    placeholder.disabled = true;
    placeholder.selected = true;
    input.appendChild(placeholder);

    feature.categories.forEach((cat) => {
      const opt = document.createElement("option");
      opt.value = cat;
      // Capitalise first letter for display
      opt.textContent = cat.charAt(0).toUpperCase() + cat.slice(1);
      input.appendChild(opt);
    });

    selectWrap.appendChild(input);
    wrapper.appendChild(selectWrap);
  } else {
    // Fallback: plain text / number input
    input = document.createElement("input");
    input.id = feature.name;
    input.name = feature.name;
    input.type = feature.type === "number" ? "number" : "text";
    input.placeholder = `Enter ${formatLabel(feature.name)}`;
    wrapper.appendChild(input);
  }

  return { wrapper, input };
}

async function main() {
  const formDiv = document.getElementById("form");
  const resultDiv = document.getElementById("result");
  const rawPre = document.getElementById("raw");
  const submitBtn = document.getElementById("submit");

  let schema;
  try {
    schema = await fetchSchema();
  } catch (e) {
    resultDiv.innerHTML = `<div class="result-error">Failed to load schema: ${e.message}</div>`;
    return;
  }

  const inputs = {};
  schema.features.forEach((f) => {
    const { wrapper, input } = makeField(f);
    formDiv.appendChild(wrapper);
    inputs[f.name] = input;
  });

  submitBtn.onclick = async () => {
    // Loading state
    submitBtn.disabled = true;
    submitBtn.classList.add("loading");
    resultDiv.innerHTML = "";
    rawPre.textContent = "";

    const payload = {};
    schema.features.forEach((f) => {
      const v = inputs[f.name].value;
      payload[f.name] = f.type === "number" ? (v === "" ? null : Number(v)) : v;
    });

    try {
      const r = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const body = await r.json().catch(() => ({}));
      rawPre.textContent = JSON.stringify(body, null, 2);

      if (!r.ok) {
        const msg = body?.detail
          ? Array.isArray(body.detail)
            ? body.detail.map((e) => e.msg || JSON.stringify(e)).join(", ")
            : body.detail
          : JSON.stringify(body);
        resultDiv.innerHTML = `<div class="result-error">⚠️ ${msg}</div>`;
        return;
      }

      // Score is 0–100
      const score = Math.round(body.prediction * 10) / 10;
      const pct = Math.min(Math.max(score, 0), 100);

      resultDiv.innerHTML = `
        <div class="result-score">
          <div class="score-label">Predicted Math Score</div>
          <div class="score-value">${score}</div>
          <div class="score-unit">out of 100</div>
        </div>
        <div class="score-bar-wrap">
          <div class="score-bar" id="score-bar"></div>
        </div>
        <div class="score-range"><span>0</span><span>100</span></div>
      `;

      // Animate bar after render
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          document.getElementById("score-bar").style.width = `${pct}%`;
        });
      });

    } catch (e) {
      resultDiv.innerHTML = `<div class="result-error">Request failed: ${e.message}</div>`;
    } finally {
      submitBtn.disabled = false;
      submitBtn.classList.remove("loading");
    }
  };
}

main();