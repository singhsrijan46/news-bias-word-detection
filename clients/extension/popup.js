// Bias Lens extension popup: connect to the board, pair with the code on its LED matrix, and control highlighting.
const $ = (id) => document.getElementById(id);
const send = (message) => chrome.runtime.sendMessage(message).then((reply) => {
  if (reply && reply.error) throw new Error(reply.error);
  return reply;
});
let tab = null;

function say(text) {
  $("status").textContent = text;
}

async function refresh() {
  const status = await send({ type: "status" });
  $("board").value = status.boardUrl;
  $("connect").hidden = status.paired;
  $("pair").hidden = status.paired || !status.boardUrl;
  $("ready").hidden = !status.paired;
  $("auto").checked = status.auto;
  const host = tab && tab.url ? new URL(tab.url).hostname : "";
  $("host").textContent = host || "this site";
  $("pause").checked = status.pausedSites.includes(host);
}

$("save").onclick = async () => {
  const url = $("board").value.trim().replace(/\/+$/, "");
  if (!/^https?:\/\/[^/]+$/.test(url)) {
    say("Type the address shown on the board's page, like http://192.168.1.7:7000.");
    return;
  }
  // The extension may talk only to this board: ask for exactly this address.
  const granted = await chrome.permissions.request({ origins: [`${url}/*`] });
  if (!granted) {
    say("Bias Lens needs permission to reach your board.");
    return;
  }
  await send({ type: "set-board", boardUrl: url });
  say("");
  refresh();
};

$("pair-start").onclick = async () => {
  try {
    await send({ type: "pair-start" });
    $("code").disabled = false;
    $("pair-finish").disabled = false;
    $("code").focus();
    say("Watch the board's LED matrix for six digits (you have two minutes).");
  } catch (error) {
    say(`Could not reach the board: ${error.message}`);
  }
};

async function finish() {
  const code = $("code").value.trim();
  if (!/^\d{6}$/.test(code)) {
    say("Type the six digits from the board.");
    return;
  }
  try {
    say("Pairing…");
    await send({ type: "pair-finish", code });
    say("");
    refresh();
  } catch (error) {
    say(error.message);
    $("code").disabled = true;
    $("pair-finish").disabled = true;
  }
}
$("pair-finish").onclick = finish;
$("code").addEventListener("keydown", (event) => { if (event.key === "Enter") finish(); });

$("analyze").onclick = async () => {
  if (!tab) return;
  try {
    await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ["overlay-core.js", "content.js"] });
    await chrome.tabs.sendMessage(tab.id, { type: "biaslens-analyze-now" });
    window.close();
  } catch (error) {
    say(`This page cannot be analysed: ${error.message}`);
  }
};

$("auto").onchange = () => send({ type: "set-auto", auto: $("auto").checked });
$("pause").onchange = () => {
  const host = tab && tab.url ? new URL(tab.url).hostname : "";
  if (host) send({ type: "pause-site", host, paused: $("pause").checked });
};
$("forget").onclick = async (event) => {
  event.preventDefault();
  await send({ type: "forget" });
  refresh();
};

chrome.tabs.query({ active: true, currentWindow: true }).then(([active]) => {
  tab = active;
  refresh();
});
