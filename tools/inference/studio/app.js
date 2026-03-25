const els = {
  engineSummary: document.querySelector("#engine-summary"),
  engineMeta: document.querySelector("#engine-meta"),
  configSummary: document.querySelector("#config-summary"),
  sampleSelect: document.querySelector("#sample-select"),
  instruction: document.querySelector("#instruction"),
  text: document.querySelector("#text"),
  preserveTerms: document.querySelector("#preserve-terms"),
  reference: document.querySelector("#reference"),
  memory: document.querySelector("#memory"),
  checkInline: document.querySelector("#check-inline"),
  checkBullets: document.querySelector("#check-bullets"),
  checkBreaks: document.querySelector("#check-breaks"),
  modelPreset: document.querySelector("#model-preset"),
  controller: document.querySelector("#controller"),
  runButton: document.querySelector("#run-button"),
  runStatus: document.querySelector("#run-status"),
  metrics: document.querySelector("#metrics"),
  output: document.querySelector("#output"),
  copyButton: document.querySelector("#copy-button"),
  insights: document.querySelector("#insights"),
  scores: document.querySelector("#scores"),
  debug: document.querySelector("#debug"),
  modeQuick: document.querySelector("#mode-quick"),
  modeDeep: document.querySelector("#mode-deep"),
  shaderCanvas: document.querySelector("#shader-canvas"),
};

let productConfig = {};
let nightlyConfig = {};
let bestSummary = null;
let samples = [];
let currentMode = "quick";
let availablePresets = [];
let modelMatrix = {};

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

function formatScore(value) {
  if (typeof value !== "number") {
    return "n/a";
  }
  return value.toFixed(3);
}

function setStatus(message) {
  els.runStatus.textContent = message;
}

function parseTerms(value) {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function readChecks() {
  return {
    preserve_inline_code: els.checkInline.checked,
    preserve_bullets: els.checkBullets.checked,
    preserve_line_breaks: els.checkBreaks.checked,
  };
}

function applyChecks(checks = {}) {
  els.checkInline.checked = Boolean(checks.preserve_inline_code);
  els.checkBullets.checked = Boolean(checks.preserve_bullets);
  els.checkBreaks.checked = Boolean(checks.preserve_line_breaks);
}

function setMode(mode) {
  currentMode = mode === "deep" ? "deep" : "quick";
  for (const button of [els.modeQuick, els.modeDeep]) {
    const active = button.dataset.mode === currentMode;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  }
  renderEngineSummary();
}

function fillSample(sample) {
  if (!sample) {
    return;
  }
  els.instruction.value = sample.instruction ?? "";
  els.text.value = sample.input ?? "";
  els.reference.value = sample.reference ?? "";
  els.memory.value = (sample.memory ?? []).join("\n");
  els.preserveTerms.value = (sample.preserve_terms ?? []).join(", ");
  applyChecks(sample.checks ?? {});
  els.controller.value = "";
  setStatus(`Loaded sample ${sample.id}.`);
}

function renderEngineSummary() {
  const config = currentMode === "deep" ? nightlyConfig : productConfig;
  const model = config.model ?? "unknown model";
  if (currentMode === "deep") {
    els.engineSummary.textContent = `${model} with recursive memory routing`;
    els.engineMeta.textContent = `Deep mode uses the highest-quality promoted route · p95 ${bestSummary?.p95_latency_ms ? `${bestSummary.p95_latency_ms.toFixed(0)} ms` : "n/a"}`;
  } else {
    els.engineSummary.textContent = `${model} with a fast local pass`;
    els.engineMeta.textContent = "Quick mode uses the promoted fast route for lower-latency edits.";
  }

  els.configSummary.textContent = pretty({
    quick: productConfig,
    deep: nightlyConfig,
    best_quality: bestSummary?.avg_quality_score ?? null,
    best_p95_ms: bestSummary?.p95_latency_ms ?? null,
  });
}

function renderMetrics(metrics = {}, config = {}, mode = currentMode) {
  els.metrics.innerHTML = "";
  const entries = [
    ["mode", mode],
    ["route", config.controller ?? (mode === "deep" ? nightlyConfig.controller : productConfig.controller) ?? "default"],
    ["time", metrics.latency_ms ? `${metrics.latency_ms.toFixed(0)} ms` : "n/a"],
    ["memory steps", metrics.subcalls ?? "n/a"],
  ];

  for (const [label, value] of entries) {
    const pill = document.createElement("span");
    pill.className = "metric-pill";
    pill.textContent = `${label}: ${value}`;
    els.metrics.appendChild(pill);
  }
}

function renderInsights({ mode = currentMode, config = {}, scores = {}, debug = {}, metrics = {} } = {}) {
  const lines = [];
  const selectedMemory = Array.isArray(debug.selected_memory) ? debug.selected_memory : [];
  const preserveTerms = parseTerms(els.preserveTerms.value);
  const controller = config.controller ?? (mode === "deep" ? nightlyConfig.controller : productConfig.controller) ?? "default";

  if (mode === "deep") {
    lines.push(`Deep mode used ${controller} routing in ${metrics.subcalls ?? "n/a"} steps.`);
  } else {
    lines.push(`Quick mode used ${controller} routing in ${metrics.subcalls ?? "n/a"} steps.`);
  }

  if (selectedMemory.length > 0) {
    lines.push(`Pulled ${selectedMemory.length} memory note${selectedMemory.length === 1 ? "" : "s"} into the edit.`);
  } else {
    lines.push("Ran without extra memory notes.");
  }

  if (preserveTerms.length > 0) {
    lines.push(`Protected terms: ${preserveTerms.join(", ")}.`);
  }

  if (typeof scores.quality_score === "number") {
    lines.push(`Reference quality score: ${scores.quality_score.toFixed(3)}.`);
  } else if (metrics.latency_ms) {
    lines.push("Add reference text in Advanced if you want benchmark-style scoring.");
  }

  els.insights.innerHTML = "";
  for (const line of lines) {
    const item = document.createElement("li");
    item.textContent = line;
    els.insights.appendChild(item);
  }
}

function buildPayload() {
  const controller = els.controller.value.trim();
  const preset = els.modelPreset.value.trim();
  const config = {};
  if (preset) {
    config.preset = preset;
  }
  if (controller) {
    config.controller = controller;
  }

  return {
    mode: currentMode,
    instruction: els.instruction.value,
    text: els.text.value,
    reference: els.reference.value,
    preserve_terms: els.preserveTerms.value,
    memory: els.memory.value,
    checks: readChecks(),
    config,
  };
}

async function loadBootData() {
  const [configResp, examplesResp] = await Promise.all([
    fetch("/api/config"),
    fetch("/api/examples"),
  ]);
  const configJson = await configResp.json();
  const examplesJson = await examplesResp.json();
  productConfig = configJson.product_config ?? configJson.default_config ?? {};
  nightlyConfig = configJson.nightly_config ?? configJson.default_config ?? {};
  bestSummary = configJson.best_summary ?? null;
  availablePresets = configJson.presets ?? [];
  modelMatrix = configJson.model_matrix ?? {};
  samples = examplesJson.examples ?? [];

  renderEngineSummary();

  for (const preset of availablePresets) {
    const option = document.createElement("option");
    option.value = preset;
    const label = modelMatrix?.[preset]?.label ?? preset;
    option.textContent = label;
    els.modelPreset.appendChild(option);
  }

  for (const sample of samples) {
    const option = document.createElement("option");
    option.value = sample.id;
    option.textContent = sample.id;
    els.sampleSelect.appendChild(option);
  }

  if (samples.length > 0) {
    fillSample(samples[0]);
    els.sampleSelect.value = samples[0].id;
  }
}

async function runStudioPass() {
  const payload = buildPayload();
  if (!payload.text.trim()) {
    setStatus("Enter text first.");
    return;
  }

  els.runButton.disabled = true;
  setStatus(currentMode === "deep" ? "Running deep route..." : "Running quick route...");
  try {
    const response = await fetch("/api/fix", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const json = await response.json();
    if (!response.ok || !json.ok) {
      throw new Error(json.error ?? "Run failed.");
    }

    renderMetrics(json.metrics, json.config, json.mode);
    renderInsights(json);
    els.output.textContent = json.output ?? "";
    els.scores.textContent = json.scores ? pretty(json.scores) : "Reference-free run.";
    els.debug.textContent = pretty(json.debug ?? {});
    setStatus("Done.");
  } catch (error) {
    els.output.textContent = "";
    els.insights.innerHTML = "<li>Run failed before a usable result came back.</li>";
    els.scores.textContent = "No scores.";
    els.debug.textContent = "";
    setStatus(error.message || "Run failed.");
  } finally {
    els.runButton.disabled = false;
  }
}

function createShaderBackground() {
  const canvas = els.shaderCanvas;
  if (!canvas) {
    return;
  }

  const gl = canvas.getContext("webgl", {
    alpha: false,
    antialias: false,
    depth: false,
    stencil: false,
    powerPreference: "high-performance",
  });
  if (!gl) {
    document.body.classList.add("shader-fallback");
    return;
  }

  const vertexSource = `
    attribute vec2 a_position;
    void main() {
      gl_Position = vec4(a_position, 0.0, 1.0);
    }
  `;

  const fragmentSource = `
    precision mediump float;

    uniform vec2 u_resolution;
    uniform float u_time;

    float hash(vec2 p) {
      return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
    }

    float noise(vec2 p) {
      vec2 i = floor(p);
      vec2 f = fract(p);
      float a = hash(i);
      float b = hash(i + vec2(1.0, 0.0));
      float c = hash(i + vec2(0.0, 1.0));
      float d = hash(i + vec2(1.0, 1.0));
      vec2 u = f * f * (3.0 - 2.0 * f);
      return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
    }

    float fbm(vec2 p) {
      float value = 0.0;
      float amplitude = 0.5;
      for (int i = 0; i < 5; i++) {
        value += amplitude * noise(p);
        p = p * 2.0 + vec2(19.0, 7.0);
        amplitude *= 0.5;
      }
      return value;
    }

    void main() {
      vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution.xy) / min(u_resolution.x, u_resolution.y);
      float time = u_time * 0.08;

      vec3 c1 = vec3(0.212, 0.184, 0.310);
      vec3 c2 = vec3(0.357, 0.137, 1.000);
      vec3 c3 = vec3(0.000, 0.545, 1.000);
      vec3 c4 = vec3(0.894, 1.000, 0.188);

      float n1 = fbm(uv * 1.2 + vec2(time, -time * 0.35));
      float n2 = fbm(uv * 1.9 - vec2(time * 0.45, time * 0.2));
      float ribbon = 0.5 + 0.5 * sin((uv.x * 2.8 - uv.y * 1.5 + time) * 3.14159 + n2 * 3.5);
      vec3 color = mix(c1, c2, smoothstep(0.12, 0.9, n1));
      color = mix(color, c3, smoothstep(0.25, 0.95, ribbon));
      color = mix(color, c4, smoothstep(0.35, 0.98, n2));

      float vignette = smoothstep(1.6, 0.15, length(uv));
      vec3 base = vec3(0.022, 0.026, 0.042);
      color = mix(base, color, 0.72);
      color *= 0.35 + 0.65 * vignette;

      gl_FragColor = vec4(color, 1.0);
    }
  `;

  function compile(type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const error = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error(error || "Shader compile failed.");
    }
    return shader;
  }

  let rafId = 0;
  let visible = !document.hidden;
  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  try {
    const program = gl.createProgram();
    gl.attachShader(program, compile(gl.VERTEX_SHADER, vertexSource));
    gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fragmentSource));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(program) || "Shader link failed.");
    }

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([
        -1, -1,
        1, -1,
        -1, 1,
        -1, 1,
        1, -1,
        1, 1,
      ]),
      gl.STATIC_DRAW,
    );

    gl.useProgram(program);
    const position = gl.getAttribLocation(program, "a_position");
    const resolution = gl.getUniformLocation(program, "u_resolution");
    const time = gl.getUniformLocation(program, "u_time");
    gl.enableVertexAttribArray(position);
    gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);

    function resizeCanvas() {
      const ratio = Math.min(window.devicePixelRatio || 1, 1.5);
      const width = Math.floor(window.innerWidth * ratio);
      const height = Math.floor(window.innerHeight * ratio);
      if (canvas.width === width && canvas.height === height) {
        return;
      }
      canvas.width = width;
      canvas.height = height;
      gl.viewport(0, 0, width, height);
    }

    function draw(now) {
      resizeCanvas();
      gl.uniform2f(resolution, canvas.width, canvas.height);
      gl.uniform1f(time, prefersReducedMotion.matches ? 0 : now * 0.001);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      if (!prefersReducedMotion.matches && visible) {
        rafId = window.requestAnimationFrame(draw);
      }
    }

    function start() {
      if (rafId) {
        return;
      }
      rafId = window.requestAnimationFrame(draw);
    }

    function stop() {
      if (!rafId) {
        return;
      }
      window.cancelAnimationFrame(rafId);
      rafId = 0;
    }

    window.addEventListener("resize", () => {
      resizeCanvas();
      if (prefersReducedMotion.matches) {
        draw(0);
      }
    });
    document.addEventListener("visibilitychange", () => {
      visible = !document.hidden;
      if (!visible) {
        stop();
        return;
      }
      if (prefersReducedMotion.matches) {
        draw(0);
      } else {
        start();
      }
    });
    prefersReducedMotion.addEventListener("change", () => {
      stop();
      if (prefersReducedMotion.matches) {
        draw(0);
      } else if (visible) {
        start();
      }
    });

    resizeCanvas();
    if (prefersReducedMotion.matches) {
      draw(0);
    } else {
      start();
    }
  } catch {
    document.body.classList.add("shader-fallback");
  }
}

els.sampleSelect.addEventListener("change", () => {
  const sample = samples.find((item) => item.id === els.sampleSelect.value);
  fillSample(sample);
});

els.runButton.addEventListener("click", () => {
  void runStudioPass();
});

for (const button of [els.modeQuick, els.modeDeep]) {
  button.addEventListener("click", () => {
    setMode(button.dataset.mode);
  });
}

els.copyButton.addEventListener("click", async () => {
  const text = els.output.textContent?.trim();
  if (!text || text === "No run yet.") {
    setStatus("Run a pass before copying.");
    return;
  }

  try {
    await navigator.clipboard.writeText(text);
    setStatus("Copied edited text.");
  } catch {
    setStatus("Copy failed.");
  }
});

setMode("quick");
createShaderBackground();
void loadBootData();
