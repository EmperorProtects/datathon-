"use strict";

const POLL_MS = 1500;
const $ = (id) => document.getElementById(id);

const fileInput = $("file");
const drop = $("drop");
const submitBtn = $("submit");
const video = $("video");
let selectedFile = null;
let videoUrl = null;
let lastResult = null;

function setStatus(text, { busy = false, error = false } = {}) {
  const box = $("status");
  box.hidden = !text;
  box.classList.toggle("error", error);
  $("spinner").hidden = !busy;
  $("status-text").textContent = text || "";
}

function pickFile(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".mov")) {
    setStatus("Нужен файл в формате .mov", { error: true });
    return;
  }
  selectedFile = file;
  $("drop-text").textContent = `${file.name} (${(file.size / 1048576).toFixed(1)} МБ)`;
  submitBtn.disabled = false;
  setStatus("");
}

fileInput.addEventListener("change", () => pickFile(fileInput.files[0]));
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  pickFile(e.dataTransfer.files[0]);
});

$("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!selectedFile) return;
  submitBtn.disabled = true;
  $("result").hidden = true;
  setStatus("Загрузка видео…", { busy: true });

  const form = new FormData();
  form.append("file", selectedFile, selectedFile.name);
  try {
    const resp = await fetch("/ui/analyze", { method: "POST", body: form });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data.detail || `Ошибка загрузки (${resp.status})`);
    poll(data.job_id, Date.now());
  } catch (err) {
    setStatus(err.message, { error: true });
    submitBtn.disabled = false;
  }
});

async function poll(jobId, startedAt) {
  try {
    const resp = await fetch(`/ui/jobs/${encodeURIComponent(jobId)}`);
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || `Ошибка (${resp.status})`);
    if (data.status === "processing") {
      const sec = Math.round((Date.now() - startedAt) / 1000);
      setStatus(`Идёт анализ… ${sec} с`, { busy: true });
      setTimeout(() => poll(jobId, startedAt), POLL_MS);
      return;
    }
    submitBtn.disabled = false;
    if (data.status === "failed") {
      setStatus(`Ошибка обработки: ${data.error}`, { error: true });
      return;
    }
    setStatus("");
    renderResult(data);
  } catch (err) {
    setStatus(err.message, { error: true });
    submitBtn.disabled = false;
  }
}

function fmtTime(sec) {
  const m = Math.floor(sec / 60);
  const s = (sec - m * 60).toFixed(1).padStart(4, "0");
  return `${m}:${s}`;
}

function seek(sec, scroll = true) {
  if (!video.src) return;
  video.currentTime = sec;
  if (scroll) video.scrollIntoView({ behavior: "smooth", block: "center" });
}

function renderResult(data) {
  lastResult = data;
  const v = $("verdict");
  const n = data.violations.length;
  v.className = `verdict ${data.verdict}`;
  v.textContent = data.verdict === "violation"
    ? `Обнаружены признаки нарушений: ${n}`
    : "Признаков нарушений не обнаружено";

  if (videoUrl) URL.revokeObjectURL(videoUrl);
  videoUrl = URL.createObjectURL(selectedFile);
  video.src = videoUrl;

  const list = $("violations");
  list.replaceChildren();
  data.violations.forEach((viol, i) => list.appendChild(renderCard(viol, i)));
  renderTimeline();
  $("result").hidden = false;
}

function renderCard(viol, i) {
  const card = document.createElement("article");
  card.className = "card";
  card.id = `viol-${i}`;

  const h = document.createElement("h3");
  h.textContent = viol.type;
  card.appendChild(h);

  const time = document.createElement("button");
  time.className = "time";
  time.type = "button";
  time.textContent = `Таймкод: ${fmtTime(viol.timecode_sec)} (${viol.timecode_sec.toFixed(1)} с)`;
  time.addEventListener("click", () => seek(viol.timecode_sec));
  card.appendChild(time);

  const desc = document.createElement("p");
  desc.textContent = viol.description;
  card.appendChild(desc);

  if (viol.source) {
    const src = document.createElement("p");
    src.className = "meta";
    src.textContent = `Норматив: ${[viol.source.act, viol.source.clause && `п. ${viol.source.clause}`].filter(Boolean).join(", ")}`;
    card.appendChild(src);
  }
  if (typeof viol.confidence === "number") {
    const conf = document.createElement("p");
    conf.className = "meta";
    conf.textContent = `Уверенность: ${(viol.confidence * 100).toFixed(0)}%`;
    card.appendChild(conf);
  }

  const frames = document.createElement("div");
  frames.className = "frames";
  for (const f of viol.evidence_frames) {
    const fig = document.createElement("figure");
    const img = document.createElement("img");
    img.src = `data:image/jpeg;base64,${f.image_base64}`;
    img.alt = `Кадр ${f.timestamp_sec.toFixed(1)} с`;
    img.addEventListener("click", () => {
      $("lightbox-img").src = img.src;
      $("lightbox").showModal();
    });
    const cap = document.createElement("figcaption");
    cap.textContent = `${f.timestamp_sec.toFixed(1)} с`;
    fig.append(img, cap);
    frames.appendChild(fig);
  }
  card.appendChild(frames);
  return card;
}

function renderTimeline() {
  if (!lastResult) return;
  const tl = $("timeline");
  tl.replaceChildren();
  const maxT = Math.max(0, ...lastResult.violations.map((x) => x.timecode_sec));
  const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : Math.max(maxT * 1.1, 1);
  $("duration-label").textContent = `${duration.toFixed(1)} с`;
  lastResult.violations.forEach((viol, i) => {
    const m = document.createElement("button");
    m.className = "marker";
    m.type = "button";
    m.style.left = `${Math.min(100, (viol.timecode_sec / duration) * 100)}%`;
    m.title = `${fmtTime(viol.timecode_sec)} — ${viol.type}`;
    m.setAttribute("aria-label", m.title);
    m.addEventListener("click", () => {
      seek(viol.timecode_sec, false);
      $(`viol-${i}`).scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
    tl.appendChild(m);
  });
}

// Длительность известна только после загрузки метаданных — перерисовываем маркеры.
video.addEventListener("loadedmetadata", renderTimeline);
