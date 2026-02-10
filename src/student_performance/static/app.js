async function fetchSchema() {
  const r = await fetch("/schema");
  if (!r.ok) throw new Error(`Schema fetch failed: ${r.status}`);
  return await r.json();
}

function makeInputRow(name, typeHint) {
  const row = document.createElement("div");
  row.className = "row";

  const label = document.createElement("label");
  label.textContent = name;

  const input = document.createElement("input");
  input.name = name;

  // Basic type hint: treat "number" as numeric, else text.
  if (typeHint === "number") input.type = "number";
  else input.type = "text";

  row.appendChild(label);
  row.appendChild(input);
  return { row, input };
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
    resultDiv.textContent = `Failed to load schema: ${e.message}`;
    return;
  }

  const inputs = {};
  schema.features.forEach((f) => {
    const { row, input } = makeInputRow(f.name, f.type);
    formDiv.appendChild(row);
    inputs[f.name] = input;
  });

  submitBtn.onclick = async () => {
    resultDiv.textContent = "Predicting...";
    rawPre.textContent = "";

    // Build payload
    const payload = {};
    schema.features.forEach((f) => {
      const v = inputs[f.name].value;

      // If numeric, cast to number; else keep string
      if (f.type === "number") payload[f.name] = (v === "" ? null : Number(v));
      else payload[f.name] = v;
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
        resultDiv.textContent = `Error: ${r.status} ${JSON.stringify(body)}`;
        return;
      }

      // Generic output
      resultDiv.textContent = `Prediction: ${body.prediction}`;
    } catch (e) {
      resultDiv.textContent = `Request failed: ${e.message}`;
    }
  };
}

main();
