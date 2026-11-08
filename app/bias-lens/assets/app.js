// Bias Lens web page: pairs with the board, sends text encrypted, and shows which words readers would flag.

import { startPairing, finishPairing, secureCall } from "./biaslens-client.js";

const $ = (id) => document.getElementById(id);
const STORE = "biaslens-pairing";
let pairing = null;
let pairingSid = null;
let lastText = "";  // the text of the analysis on show, for the perspective view
let perspectiveAvailable = false;  // whether the board has M3, the perspective model

// An invented example, written for this app.
const SAMPLE = `City council backs bus expansion
The city council voted 7 to 2 on Tuesday to approve a $40 million plan to extend bus service to the northern suburbs.
Critics slammed the scheme as a reckless giveaway to well-connected developers, while supporters hailed it as a long-overdue lifeline for working families.
The mayor, who has championed the plan for two years, said the first new routes could open next spring.
Opponents vowed to fight on, calling the vote a shameful betrayal of taxpayers.`;

function readers(share) {
  const n = Math.round(share * 10);
  return n < 1 ? "Fewer than 1 in 10" : `About ${n} in 10`;
}

function deviceName() {
  const ua = navigator.userAgent;
  const browser = /Edg\//.test(ua) ? "Edge" : /Firefox\//.test(ua) ? "Firefox" : /Chrome\//.test(ua) ? "Chrome" : "Browser";
  const system = /Android/.test(ua) ? "Android" : /iPhone|iPad/.test(ua) ? "iOS" : /Windows/.test(ua) ? "Windows"
    : /Mac OS/.test(ua) ? "macOS" : "Linux";
  return `${browser} on ${system}`;
}

// ---------------------------------------------------------------- pairing

function loadPairing() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORE));
    return saved && saved.baseUrl === window.location.origin ? saved : null;
  } catch {
    return null;
  }
}

function showPairingState() {
  const isPaired = Boolean(pairing);
  $("pairing").hidden = isPaired;
  $("analyze").disabled = !isPaired;
  $("sample").disabled = !isPaired;
  const line = $("paired-as");
  line.replaceChildren();
  if (isPaired) {
    line.append(`Paired as "${pairing.name}"; requests are encrypted. `);
    const forget = document.createElement("a");
    forget.href = "#";
    forget.textContent = "Forget this browser";
    forget.addEventListener("click", async (event) => {
      event.preventDefault();
      try { await secureCall(pairing, { op: "forget" }); } catch { /* forget locally anyway */ }
      localStorage.removeItem(STORE);
      pairing = null;
      showPairingState();
    });
    line.append(forget);
  }
}

async function beginPairing() {
  $("pair-status").textContent = "";
  try {
    const session = await startPairing(window.location.origin, deviceName());
    pairingSid = session.sid;
    $("pair-code").disabled = false;
    $("pair-finish").disabled = false;
    $("pair-code").value = "";
    $("pair-code").focus();
    $("pair-status").textContent = "Watch the board's LED matrix for six digits.";
  } catch (error) {
    $("pair-status").textContent = `Could not start pairing: ${error.message}`;
  }
}

async function completePairing() {
  const code = $("pair-code").value.trim();
  if (!/^\d{6}$/.test(code)) {
    $("pair-status").textContent = "Type the six digits from the board.";
    return;
  }
  $("pair-finish").disabled = true;
  $("pair-status").textContent = "Pairing…";
  try {
    pairing = await finishPairing(window.location.origin, pairingSid, code, deviceName());
    localStorage.setItem(STORE, JSON.stringify(pairing));
    $("pair-status").textContent = "";
    showPairingState();
  } catch (error) {
    $("pair-status").textContent = error.message;
    $("pair-code").disabled = true;
  }
}

// ---------------------------------------------------------------- analysis

function setStatus(message) {
  $("status").textContent = message;
}

async function analyze() {
  const text = $("text").value.trim();
  if (!text) {
    setStatus("Paste some text first.");
    return;
  }
  if (!pairing) {
    setStatus("Pair this browser first.");
    return;
  }
  setStatus("Analysing on the board…");
  $("analyze").disabled = true;
  try {
    render(await secureCall(pairing, { op: "analyze", text }));
    setStatus("");
  } catch (error) {
    if (/Unknown device|failed authentication/.test(error.message)) {
      localStorage.removeItem(STORE);
      pairing = null;
      showPairingState();
      setStatus("The board no longer knows this browser. Pair it again.");
    } else {
      setStatus(`Could not analyse the text: ${error.message}`);
    }
  } finally {
    $("analyze").disabled = !pairing;
  }
}

function listItem(text) {
  const li = document.createElement("li");
  li.textContent = text;
  return li;
}

function render(data) {
  const { text, sentences, summary } = data;

  // The article, with flagged words marked. Words arrive in reading order.
  const article = $("article");
  article.replaceChildren();
  let cursor = 0;
  for (const sentence of sentences) {
    for (const word of sentence.words) {
      if (word.band === "none") continue;
      article.append(text.slice(cursor, word.start));
      const mark = document.createElement("mark");
      mark.className = word.band;
      mark.textContent = text.slice(word.start, word.end);
      mark.style.setProperty("--share", word.share);
      mark.title = `${readers(word.share)} readers would flag "${word.text}"` +
        (word.lexicon ? " (also in the paper's bias lexicon)" : "");
      mark.dataset.start = word.start;
      mark.dataset.title = mark.title;
      mark.tabIndex = 0;
      article.append(mark);
      cursor = word.end;
    }
  }
  article.append(text.slice(cursor));

  // The heat strip, the same one the LED matrix shows.
  const strip = $("strip");
  strip.replaceChildren();
  for (const level of summary.strip) {
    const bar = document.createElement("span");
    bar.style.height = `${(level / 8) * 100}%`;
    strip.append(bar);
  }

  $("counts").replaceChildren(
    listItem(`${summary.flagged} flagged words, in ${summary.flagged_sentences} of ${summary.sentences} sentences`),
    listItem(`${summary.agreed} that most readers would flag; ${summary.contested} where readers split`),
    listItem(`${summary.words} words analysed on the board in ${summary.seconds.toFixed(1)} s`),
  );

  const top = $("top");
  top.replaceChildren();
  if (summary.top.length) {
    const heading = document.createElement("h2");
    heading.textContent = "Most flagged";
    const list = document.createElement("ol");
    for (const word of summary.top) {
      list.append(listItem(`${word.text}: ${readers(word.share).toLowerCase()} readers`));
    }
    top.append(heading, list);
  }
  lastText = text;
  $("perspective").hidden = !perspectiveAvailable;
  $("perspective-result").hidden = true;
  $("perspective-status").textContent = "";
  $("results").hidden = false;
}

// ---------------------------------------------------------------- liberal and conservative readers

function groups(word) {
  return `liberal readers ${readers(word.liberal).toLowerCase()}, conservative readers ` +
    `${readers(word.conservative).toLowerCase()}`;
}

function fillList(list, words) {
  if (!words.length) {
    const none = listItem("No flagged word differs by 1 reader in 10 or more.");
    none.className = "none";
    list.replaceChildren(none);
    return;
  }
  list.replaceChildren(...words.map((w) => listItem(`${w.text}: ${groups(w)}`)));
}

function renderPerspective(data) {
  const byStart = new Map(data.words.map((w) => [w.start, w]));
  for (const mark of $("article").querySelectorAll("mark")) {
    const word = byStart.get(Number(mark.dataset.start));
    mark.title = mark.dataset.title + (word ? `\nLiberal readers: ${readers(word.liberal).toLowerCase()}. ` +
      `Conservative readers: ${readers(word.conservative).toLowerCase()}.` : "");
  }
  fillList($("liberal-more"), data.liberal_more);
  fillList($("conservative-more"), data.conservative_more);
  $("perspective-result").hidden = false;
}

async function comparePerspectives() {
  if (!pairing || !lastText) return;
  $("perspective-run").disabled = true;
  $("perspective-status").textContent = "Comparing on the board…";
  try {
    renderPerspective(await secureCall(pairing, { op: "perspective", text: lastText }));
    $("perspective-status").textContent = "";
  } catch (error) {
    $("perspective-status").textContent = `Could not compare: ${error.message}`;
  } finally {
    $("perspective-run").disabled = false;
  }
}

async function showModel() {
  try {
    const info = await (await fetch("/api/info")).json();
    perspectiveAvailable = Boolean(info.perspective);
    const parts = ["Model: a small student network (MiniLM-L6), distilled from a RoBERTa-base teacher"];
    if (info.precision) parts.push(`${info.precision.toUpperCase()} on the board's CPU`);
    if (info.cv_f1) parts.push(`cross-validated F1 ${info.cv_f1.toFixed(2)} on MBIC`);
    parts.push(`words are flagged from ${readers(info.threshold).toLowerCase()} readers`);
    $("model").textContent = parts.join(" · ");
  } catch (error) {
    $("model").textContent = "Model information is not available.";
  }
}

// The phone bookmark: the shared overlay code plus a few lines that hand the article to this board's
// overlay.html in a new tab and mark the words when the result comes back.
async function setUpSharing() {
  const board = window.location.origin;
  $("board-address").textContent = board;
  $("api").textContent = "Programs pair once (POST /api/pair/start, /api/pair/exchange, /api/pair/confirm), then " +
    "send AES-256-GCM encrypted requests to POST /api/secure. See docs/BIAS_LENS_APP.md.";
  const core = await (await fetch("overlay-core.js")).text();
  const driver = `(function(){var B=${JSON.stringify(board)};var ex=BiasLensOverlay.extract(document);` +
    `if(ex.text.trim().length<200){alert("Bias Lens: no article text found on this page.");return}` +
    `var card=BiasLensOverlay.badge(document);` +
    `window.open(B+"/overlay.html#from="+encodeURIComponent(location.origin)+"&text="+encodeURIComponent(ex.text),"_blank");` +
    `window.addEventListener("message",function f(e){if(e.origin!==B||!e.data||e.data.type!=="biaslens-result")return;` +
    `window.removeEventListener("message",f);BiasLensOverlay.removeHighlights(document);` +
    `BiasLensOverlay.highlight(ex,e.data.result,document);card.show(e.data.result.summary)})})();`;
  const bookmarklet = `javascript:${encodeURIComponent(`${core}\n${driver}`)}`;
  $("bookmarklet").href = bookmarklet;
  $("bookmarklet-code").value = bookmarklet;
  $("copy-bookmarklet").addEventListener("click", () => {
    // navigator.clipboard needs HTTPS, so copy the old way from a text box.
    const box = $("bookmarklet-code");
    box.hidden = false;
    box.select();
    const copied = document.execCommand("copy");
    box.hidden = copied;
    $("copy-bookmarklet").textContent = copied ? "Copied" : "Select and copy the text below";
  });
}

document.addEventListener("DOMContentLoaded", () => {
  pairing = loadPairing();
  showPairingState();
  $("pair-start").addEventListener("click", beginPairing);
  $("pair-finish").addEventListener("click", completePairing);
  $("pair-code").addEventListener("keydown", (event) => { if (event.key === "Enter") completePairing(); });
  $("analyze").addEventListener("click", analyze);
  $("perspective-run").addEventListener("click", comparePerspectives);
  $("sample").addEventListener("click", () => {
    $("text").value = SAMPLE;
    analyze();
  });
  $("clear").addEventListener("click", () => {
    $("text").value = "";
    $("results").hidden = true;
    setStatus("");
  });
  setUpSharing();
  showModel();

  // Text sent by the bookmark arrives in the address's # part, which browsers never send to a server.
  if (window.location.hash.startsWith("#text=")) {
    $("text").value = decodeURIComponent(window.location.hash.slice(6));
    history.replaceState(null, "", window.location.pathname);
    if (pairing) analyze();
  }
});
