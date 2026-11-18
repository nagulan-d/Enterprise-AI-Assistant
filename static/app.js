"use strict";

/* ---------- helpers ---------- */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
// allow file names to wrap after _ and . instead of mid-word
const wrapName = (s) => esc(s).replace(/([_.])/g, "$1<wbr>");
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

const state = {
  docs: [],
  scope: new Set(),        // names of documents ticked in the library
  history: [],             // chat memory sent with each question
  results: new Map(),      // result key -> {sources, terms, label}; lets a citation open its own evidence
  seq: 0,
  llm: { available: false, model: "", reason: "" },
};

let toastTimer;
function toast(msg, bad = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast" + (bad ? " bad" : "");
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), bad ? 6000 : 3000);
}

async function api(path, { method = "GET", json, form } = {}) {
  const opt = { method, headers: {} };
  if (json !== undefined) { opt.body = JSON.stringify(json); opt.headers["Content-Type"] = "application/json"; }
  if (form) opt.body = form;
  let res;
  try { res = await fetch(path, opt); }
  catch { throw new Error("Cannot reach the server. Is it still running?"); }
  let data = {};
  try { data = await res.json(); } catch { /* non-JSON error page */ }
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status}).`);
  return data;
}

async function busy(btn, label, fn) {
  const original = btn.textContent;
  btn.disabled = true; btn.textContent = label;
  try { return await fn(); }
  catch (e) { toast(e.message, true); }
  finally { btn.disabled = false; btn.textContent = original; }
}

// notes about a missing key are redundant with the header pill; only surface real model failures
const noteOf = (n) => (state.llm.available && n ? n : "");

const copyText = async (text) => {
  try { await navigator.clipboard.writeText(text); toast("Copied"); }
  catch { toast("Copy failed. Select the text and copy it manually.", true); }
};

function download(name, text) {
  const a = Object.assign(document.createElement("a"), {
    href: URL.createObjectURL(new Blob([text], { type: "text/markdown" })), download: name,
  });
  document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

/* ---------- markdown (small, safe: everything is escaped first) ---------- */
function inline(raw, resKey) {
  let s = esc(raw);
  s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/(^|[\s(])_([^_]+)_(?=[\s).,;:!?]|$)/g, "$1<em>$2</em>");
  s = s.replace(/(^|[\s(])\*([^*\s][^*]*)\*(?=[\s).,;:!?]|$)/g, "$1<em>$2</em>");
  if (resKey) {
    s = s.replace(/\[(S\d+(?:\s*[,;]\s*S\d+)*)\]/g, (_, grp) =>
      grp.split(/\s*[,;]\s*/).map((id) => {
        const n = id.slice(1);
        return `<button type="button" class="cite" data-res="${resKey}" data-id="${n}" aria-label="Show source ${n}">${n}</button>`;
      }).join(""));
  }
  return s;
}

function md(text, resKey) {
  const lines = String(text).replace(/\r/g, "").split("\n");
  const out = [];
  let i = 0;
  const isRow = (l) => /^\s*\|.*\|\s*$/.test(l);
  const cells = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i++; continue; }
    let m;
    if ((m = line.match(/^(#{1,4})\s+(.*)$/))) {
      const lvl = Math.min(m[1].length + 1, 4);
      out.push(`<h${lvl}>${inline(m[2], resKey)}</h${lvl}>`); i++;
    } else if (isRow(line) && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1])) {
      const head = cells(line); i += 2;
      const rows = [];
      while (i < lines.length && isRow(lines[i])) rows.push(cells(lines[i++]));
      out.push(`<div class="tablewrap"><table><thead><tr>${head.map((c) => `<th>${inline(c, resKey)}</th>`).join("")}</tr></thead><tbody>` +
        rows.map((r) => `<tr>${head.map((_, k) => `<td>${inline(r[k] ?? "", resKey)}</td>`).join("")}</tr>`).join("") +
        "</tbody></table></div>");
    } else if (/^\s*[-*•]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*[-*•]\s+/.test(lines[i])) items.push(lines[i++].replace(/^\s*[-*•]\s+/, ""));
      out.push("<ul>" + items.map((t) => `<li>${inline(t, resKey)}</li>`).join("") + "</ul>");
    } else if (/^\s*\d+[.)]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) items.push(lines[i++].replace(/^\s*\d+[.)]\s+/, ""));
      out.push("<ol>" + items.map((t) => `<li>${inline(t, resKey)}</li>`).join("") + "</ol>");
    } else {
      const para = [];
      while (i < lines.length && lines[i].trim() && !/^(#{1,4}\s|\s*[-*•]\s|\s*\d+[.)]\s)/.test(lines[i]) && !isRow(lines[i])) para.push(lines[i++]);
      if (!para.length) { para.push(lines[i++]); }
      out.push(`<p>${inline(para.join(" "), resKey)}</p>`);
    }
  }
  return out.join("");
}

/* ---------- evidence rail ---------- */
const NOISE = new Set("must shall need needs required what which when does have with about should would could there their many much from that this into your the and for are how who why can".split(" "));
const termsOf = (q) => [...new Set((q.toLowerCase().match(/[a-z0-9]{4,}/g) || []).filter((w) => !NOISE.has(w)))];

function highlight(text, terms) {
  const safe = esc(text);
  if (!terms.length) return safe;
  const stems = terms.map((t) => t.replace(/(ing|ed|es|s)$/, "")).filter((t) => t.length >= 3)
    .map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  if (!stems.length) return safe;
  const re = new RegExp(`(?<![&\\w])(?:${stems.join("|")})\\w*`, "gi");
  return safe.replace(re, (m) => `<mark>${m}</mark>`);
}

function storeResult(sources, query, label) {
  const key = "r" + ++state.seq;
  state.results.set(key, { sources, terms: termsOf(query || ""), label });
  return key;
}

function showEvidence(key, focusId) {
  const r = state.results.get(key);
  const box = $("#evidence");
  if (!r) return;
  $("#railFor").textContent = r.label || "";
  if (!r.sources.length) { box.innerHTML = '<p class="rail-empty">No passages were used for this result.</p>'; return; }
  box.innerHTML = r.sources.map((s) => `
    <article class="ev" id="ev-${s.id}">
      <div class="ev-head"><span class="ev-id">${s.id}</span>
        <div class="ev-doc">${wrapName(s.doc)}<span class="ev-loc">${esc(s.locator)}</span></div></div>
      <div class="ev-text">${highlight(s.text, r.terms)}</div>
      ${s.score ? `<div class="ev-foot">Relevance ${Math.round(s.score * 100)}%</div>` : ""}
    </article>`).join("");
  if (focusId) {
    const el = $(`#ev-${focusId}`, box);
    if (el) {
      el.scrollIntoView({ block: "nearest", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
      el.classList.add("flash"); setTimeout(() => el.classList.remove("flash"), 1800);
    }
  } else {
    box.scrollTop = 0; $(".rail").scrollTop = 0;
  }
}

document.addEventListener("click", (e) => {
  const c = e.target.closest(".cite");
  if (c) showEvidence(c.dataset.res, c.dataset.id);
});

/* ---------- library ---------- */
function renderLibrary() {
  const list = $("#docList");
  $("#libEmpty").hidden = state.docs.length > 0;
  $("#docCount").textContent = state.docs.length ? `(${state.docs.length})` : "";
  list.innerHTML = state.docs.map((d) => {
    const meta = [d.type, d.pages !== "–" ? `${d.pages} pages` : null, `${d.words.toLocaleString()} words`].filter(Boolean).join(", ");
    return `<li class="doc">
      <input type="checkbox" id="c-${esc(d.name)}" data-doc="${esc(d.name)}" ${state.scope.has(d.name) ? "checked" : ""}>
      <label for="c-${esc(d.name)}"><span class="doc-name">${wrapName(d.name)}</span>
        <span class="doc-meta">${esc(meta)}</span>
        ${d.key_terms ? `<span class="doc-terms">${esc(d.key_terms.split(", ").slice(0, 4).join(", "))}</span>` : ""}</label>
      <button class="x" type="button" data-remove="${esc(d.name)}" aria-label="Remove ${esc(d.name)}">
        <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><path d="M3 3l8 8M11 3l-8 8" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>
      </button></li>`;
  }).join("");
  const n = state.scope.size;
  $("#scopeNote").textContent = state.docs.length
    ? (n === 0 ? "No documents selected. Questions will search nothing." : n === state.docs.length ? "Searching all documents." : `Searching ${n} of ${state.docs.length} documents.`)
    : "";
  // summarize picker
  const sel = $("#sumDoc"), prev = sel.value;
  sel.innerHTML = state.docs.map((d) => `<option value="${esc(d.name)}">${esc(d.name)}</option>`).join("") || '<option value="">No documents yet</option>';
  if (state.docs.some((d) => d.name === prev)) sel.value = prev;
  // compare chips
  const picked = [...state.scope];
  $("#cmpPicked").innerHTML = picked.length
    ? `<span>Comparing</span>${picked.map((n) => `<span class="chip">${esc(n)}</span>`).join("")}${picked.length < 2 ? "<span>Select at least one more in the library.</span>" : ""}`
    : "<span>No documents selected. Tick two or more in the library.</span>";
  // suggestions
  renderSuggest();
}

function applyStatus(s, { selectNew = false } = {}) {
  const before = new Set(state.docs.map((d) => d.name));
  state.docs = s.docs;
  const names = new Set(s.docs.map((d) => d.name));
  state.scope = new Set([...state.scope].filter((n) => names.has(n)));
  s.docs.forEach((d) => { if (!before.has(d.name) || (selectNew && !state.scope.has(d.name) && !before.has(d.name))) state.scope.add(d.name); });
  state.llm = s.llm; state.suggestions = s.suggestions || [];
  renderMode(); renderLibrary();
}

function renderMode() {
  $("#modeDot").className = "dot" + (state.llm.available ? " on" : "");
  $("#modeText").textContent = state.llm.available ? `Claude · ${state.llm.model}` : "Offline mode";
}

function renderSuggest() {
  const box = $("#suggest");
  box.innerHTML = (state.suggestions || []).map((q) => `<button type="button">${esc(q)}</button>`).join("");
}

$("#docList").addEventListener("change", (e) => {
  const d = e.target.dataset.doc; if (!d) return;
  e.target.checked ? state.scope.add(d) : state.scope.delete(d);
  renderLibrary();
});
$("#docList").addEventListener("click", async (e) => {
  const name = e.target.closest("[data-remove]")?.dataset.remove; if (!name) return;
  if (!confirm(`Remove "${name}" from the library?`)) return;
  try { applyStatus(await api("/api/docs/" + encodeURIComponent(name), { method: "DELETE" })); toast("Removed " + name); }
  catch (err) { toast(err.message, true); }
});
$("#selAll").onclick = () => { state.scope = new Set(state.docs.map((d) => d.name)); renderLibrary(); };
$("#selNone").onclick = () => { state.scope.clear(); renderLibrary(); };

async function uploadFiles(files) {
  if (!files.length) return;
  const form = new FormData();
  [...files].forEach((f) => form.append("files", f));
  const drop = $("#drop"); drop.style.opacity = 0.6;
  try {
    const res = await api("/api/upload", { method: "POST", form });
    const bad = res.results.filter((r) => !r.ok), good = res.results.filter((r) => r.ok);
    applyStatus(res);
    good.forEach((r) => state.scope.add(r.name)); renderLibrary();
    if (bad.length) toast(bad.map((r) => `${r.name}: ${r.error}`).join("\n"), true);
    else toast(`Added ${good.length} document${good.length === 1 ? "" : "s"}`);
  } catch (err) { toast(err.message, true); }
  finally { drop.style.opacity = 1; $("#fileInput").value = ""; }
}
const drop = $("#drop");
drop.onclick = () => $("#fileInput").click();
drop.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("#fileInput").click(); } };
$("#fileInput").onchange = (e) => uploadFiles(e.target.files);
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => uploadFiles(e.dataTransfer.files));

$("#sampleBtn").onclick = (e) => busy(e.target, "Loading…", async () => {
  const res = await api("/api/samples", { method: "POST" });
  applyStatus(res); state.scope = new Set(res.docs.map((d) => d.name)); renderLibrary();
  toast(`Loaded ${res.loaded.length} sample documents`);
});

/* ---------- tabs ---------- */
function selectTab(id) {
  $$('[role="tab"]').forEach((t) => {
    const on = t.id === "t-" + id;
    t.setAttribute("aria-selected", on); t.tabIndex = on ? 0 : -1;
  });
  $$('[role="tabpanel"]').forEach((p) => (p.hidden = p.id !== "tab-" + id));
  const last = state.lastResult?.[id];
  if (last) showEvidence(last, null); else if (id !== "ask") { $("#railFor").textContent = ""; }
}
$$('[role="tab"]').forEach((t, i, all) => {
  t.onclick = () => selectTab(t.id.slice(2));
  t.onkeydown = (e) => {
    const d = { ArrowRight: 1, ArrowLeft: -1 }[e.key]; if (!d) return;
    const n = all[(i + d + all.length) % all.length]; n.focus(); selectTab(n.id.slice(2));
  };
});
state.lastResult = {};

/* ---------- ask ---------- */
const thread = $("#thread"), askInput = $("#askInput");
function autoGrow() { askInput.style.height = "auto"; askInput.style.height = Math.min(askInput.scrollHeight, 140) + "px"; }
askInput.addEventListener("input", autoGrow);
askInput.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#askForm").requestSubmit(); } });
$("#suggest").addEventListener("click", (e) => { if (e.target.tagName === "BUTTON") { askInput.value = e.target.textContent; $("#askForm").requestSubmit(); } });

$("#newChat").onclick = () => {
  state.history = [];
  $$(".turn", thread).forEach((t) => t.remove());
  $("#askEmpty").hidden = false; $("#evidence").innerHTML = '<p class="rail-empty">Passages behind an answer appear here. Select a citation number in an answer to jump to its passage.</p>';
  $("#railFor").textContent = ""; askInput.focus();
};

$("#askForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = askInput.value.trim(); if (!q) return;
  if (!state.docs.length) { toast("Add documents first, or load the sample set.", true); return; }
  askInput.value = ""; autoGrow();
  $("#askEmpty").hidden = true;
  const turn = document.createElement("article"); turn.className = "turn";
  turn.innerHTML = `<p class="q">${esc(q)}</p><div class="thinking">Searching your documents</div>`;
  thread.append(turn); thread.scrollTop = thread.scrollHeight;
  const btn = $("#askBtn"); btn.disabled = true;
  try {
    const a = await api("/api/ask", { method: "POST", json: { question: q, history: state.history, docs: [...state.scope] } });
    const key = storeResult(a.sources, a.standalone, `Sources for “${q.length > 60 ? q.slice(0, 57) + "…" : q}”`);
    const meta = [];
    if (a.grounded) {
      meta.push(`<span class="chip ${a.confidence}">${a.confidence} confidence</span>`);
      meta.push(`<span>${a.sources.length} source${a.sources.length === 1 ? "" : "s"}</span>`);
    }
    if (a.grounded) meta.push(`<button class="link" type="button" data-copy>Copy</button>`);
    turn.innerHTML = `<p class="q">${esc(q)}</p>
      <div class="a prose ${a.grounded ? "" : "muted"}">${md(a.text, key)}</div>
      <div class="meta">${meta.join("")}</div>
      ${a.standalone && a.standalone !== q ? `<p class="rewrite">Searched for: ${esc(a.standalone)}</p>` : ""}
      ${noteOf(a.note) ? `<p class="note">${esc(a.note)}</p>` : ""}`;
    const cp = $("[data-copy]", turn); if (cp) cp.onclick = () => copyText(a.text.replace(/\[S\d+\]/g, ""));
    state.history.push({ role: "user", content: q }, { role: "assistant", content: a.text });
    state.lastResult.ask = key; showEvidence(key, null);
  } catch (err) {
    turn.innerHTML = `<p class="q">${esc(q)}</p><p class="err">${esc(err.message)}</p>`;
  } finally { btn.disabled = false; askInput.focus(); thread.scrollTop = thread.scrollHeight; }
});

/* ---------- summarize ---------- */
function resultActions(markdown, filename) {
  const wrap = document.createElement("div"); wrap.className = "result-actions";
  wrap.innerHTML = '<button class="btn quiet" type="button" data-a="copy">Copy</button><button class="btn quiet" type="button" data-a="dl">Download .md</button>';
  $('[data-a="copy"]', wrap).onclick = () => copyText(markdown);
  $('[data-a="dl"]', wrap).onclick = () => download(filename, markdown);
  return wrap;
}
const shortSrc = (label) => {
  const [doc, rest = ""] = String(label).split(" · ");
  const sec = rest.split(" > ").pop();
  return sec && sec !== doc ? `${doc} · ${sec}` : doc;
};
const stripCites = (t) => t.replace(/\s*\[S\d+(?:\s*[,;]\s*S\d+)*\]/g, "");

$("#sumForm").addEventListener("submit", (e) => {
  e.preventDefault();
  const doc = $("#sumDoc").value; if (!doc) { toast("Add a document first.", true); return; }
  const style = $('input[name="style"]:checked').value;
  busy($("#sumBtn"), "Summarizing…", async () => {
    const s = await api("/api/summarize", { method: "POST", json: { doc, style } });
    const out = $("#sumOut");
    out.innerHTML = `<div class="result"><div class="result-head"><h2>${esc(s.doc)}</h2>
        <span class="chip">${s.words.toLocaleString()} words</span>
        <span class="chip">${s.mode === "llm" ? "Written by Claude" : "Key sentences extracted"}</span></div>
      ${s.key_terms.length ? `<div class="terms">${s.key_terms.map((t) => `<span class="chip">${esc(t)}</span>`).join("")}</div>` : ""}
      <div class="prose">${md(s.text)}</div>
      ${noteOf(s.note) ? `<p class="note" style="color:var(--warn);margin-top:10px">${esc(s.note)}</p>` : ""}</div>`;
    $(".result", out).append(resultActions(`# Summary of ${s.doc}\n\n${s.text}\n`, `summary-${s.doc.replace(/\.[^.]+$/, "")}.md`));
  });
});

/* ---------- compare ---------- */
$("#cmpForm").addEventListener("submit", (e) => {
  e.preventDefault();
  if (state.scope.size < 2) { toast("Tick at least two documents in the library to compare.", true); return; }
  const aspect = $("#cmpAspect").value.trim();
  busy($("#cmpBtn"), "Comparing…", async () => {
    const c = await api("/api/compare", { method: "POST", json: { docs: [...state.scope], aspect } });
    const key = storeResult(c.sources, aspect, `Sources for the comparison${aspect ? ` on “${aspect}”` : ""}`);
    const out = $("#cmpOut");
    out.innerHTML = `<div class="result wide"><div class="result-head"><h2>${c.docs.length} documents${aspect ? `, on “${esc(aspect)}”` : ""}</h2>
        <span class="chip">${c.mode === "llm" ? "Written by Claude" : "Matched sentences"}</span></div>
      <div class="prose">${md(c.text, key)}</div>
      ${noteOf(c.note) ? `<p class="note" style="color:var(--warn);margin-top:10px">${esc(c.note)}</p>` : ""}</div>`;
    $(".result", out).append(resultActions(`# Comparison\n\n${stripCites(c.text)}\n`, "comparison.md"));
    state.lastResult.compare = key; showEvidence(key, null);
  });
});

/* ---------- decide ---------- */
function planMarkdown(p, goal) {
  const rows = p.items.map((i) => `| ${i.action} | ${i.owner} | ${i.priority} | ${i.deadline} |`).join("\n");
  return `# Action plan${goal ? `: ${goal}` : ""}\n\n## Assessment\n${stripCites(p.summary)}\n\n## Recommendation\n${stripCites(p.recommendation)}\n\n` +
    `## Action items\n| Action | Owner | Priority | Deadline |\n|---|---|---|---|\n${rows}\n\n` +
    (p.risks.length ? `## Risks and open questions\n${p.risks.map((r) => `- ${stripCites(r)}`).join("\n")}\n` : "");
}

$("#actForm").addEventListener("submit", (e) => {
  e.preventDefault();
  if (!state.scope.size) { toast("Tick at least one document in the library.", true); return; }
  const goal = $("#actGoal").value.trim();
  busy($("#actBtn"), "Working…", async () => {
    const p = await api("/api/actions", { method: "POST", json: { goal, docs: [...state.scope] } });
    const key = storeResult(p.sources, goal, goal ? `Sources for “${goal.length > 50 ? goal.slice(0, 47) + "…" : goal}”` : "Sources for the action plan");
    const out = $("#actOut");
    const table = p.items.length ? `<div class="tablewrap"><table><thead><tr><th>Action</th><th>Owner</th><th>Priority</th><th>Deadline</th><th>Source</th></tr></thead><tbody>${
      p.items.map((i) => `<tr><td>${inline(i.action, key)}</td><td>${esc(i.owner)}</td><td><span class="pri ${esc(i.priority)}">${esc(i.priority)}</span></td><td>${esc(i.deadline)}</td><td class="src-link">${esc(shortSrc(i.source))}</td></tr>`).join("")
    }</tbody></table></div>` : '<p class="prose muted">No action items were found in the selected documents.</p>';
    out.innerHTML = `<div class="result wide">
      <div class="result-head"><h2>${goal ? esc(goal.length > 90 ? goal.slice(0, 87) + "…" : goal) : "Obligations and next steps"}</h2>
        <span class="chip">${p.mode === "llm" ? "Written by Claude" : "Extracted from the text"}</span></div>
      ${p.summary ? `<h3 class="sec-title">Assessment</h3><div class="prose callout">${md(p.summary, key)}</div>` : ""}
      ${p.recommendation ? `<h3 class="sec-title">Recommendation</h3><div class="prose callout">${md(p.recommendation, key)}</div>` : ""}
      <h3 class="sec-title">Action items (${p.items.length})</h3><div class="prose">${table}</div>
      ${p.risks.length ? `<h3 class="sec-title">Risks and open questions</h3><div class="prose"><ul class="risks">${p.risks.map((r) => `<li>${inline(r, key)}</li>`).join("")}</ul></div>` : ""}
      ${noteOf(p.note) ? `<p class="note" style="color:var(--warn);margin-top:10px">${esc(p.note)}</p>` : ""}</div>`;
    $(".result", out).append(resultActions(planMarkdown(p, goal), "action-plan.md"));
    state.lastResult.decide = key; showEvidence(key, null);
  });
});

/* ---------- settings ---------- */
const dlg = $("#settings");
function renderSettings() {
  $("#setState").textContent = state.llm.available ? `Connected to Claude (${state.llm.model})` : `Offline mode${state.llm.reason ? `: ${state.llm.reason}` : ""}`;
  $("#modelInput").value = state.llm.model || "";
  $("#keyInput").value = "";
  $("#disconnectBtn").hidden = !state.llm.available;
}
$("#modeBtn").onclick = () => { renderSettings(); dlg.showModal(); };
$("#closeSet").onclick = () => dlg.close();
$("#setForm").addEventListener("submit", (e) => {
  e.preventDefault();
  const key = $("#keyInput").value.trim();
  if (!key && !state.llm.available) { toast("Paste an API key to connect.", true); return; }
  busy($("#connectBtn"), "Connecting…", async () => {
    applyStatus(await api("/api/settings", { method: "POST", json: { api_key: key, model: $("#modelInput").value.trim() } }));
    dlg.close();
    toast(state.llm.available ? "Connected to Claude" : `Could not connect: ${state.llm.reason}`, !state.llm.available);
  });
});
$("#disconnectBtn").onclick = () => busy($("#disconnectBtn"), "Disconnecting…", async () => {
  applyStatus(await api("/api/settings", { method: "POST", json: { clear: true } })); dlg.close(); toast("Back to offline mode");
});

/* ---------- boot ---------- */
(async function init() {
  try { applyStatus(await api("/api/status")); } catch (e) { toast(e.message, true); }
  selectTab("ask");
})();
