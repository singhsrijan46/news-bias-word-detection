// The board tab that the phone bookmark opens: analyse the article's text over the encrypted
// channel, hand the result back to the article's tab, and close. The text arrives in the
// address's # part, which browsers never send over the network.

import { secureCall } from "./biaslens-client.js";

const STORE = "biaslens-pairing";
const message = document.getElementById("message");
const next = document.getElementById("next");
const params = new URLSearchParams(window.location.hash.slice(1));
const from = params.get("from");
const text = params.get("text") || "";
history.replaceState(null, "", window.location.pathname); // keep the text out of history

function savedPairing() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORE));
    return saved && saved.baseUrl === window.location.origin ? saved : null;
  } catch {
    return null;
  }
}

function link(label, href) {
  const a = document.createElement("a");
  a.href = href;
  a.textContent = label;
  return a;
}

async function main() {
  const pairing = savedPairing();
  if (!pairing) {
    message.textContent = "This browser is not paired with the board yet.";
    next.append(link("Pair it now", "/"), " — then use the bookmark again.");
    return;
  }
  if (!text.trim()) {
    message.textContent = "No article text arrived.";
    return;
  }
  try {
    const result = await secureCall(pairing, { op: "analyze", text });
    if (window.opener && from) {
      // Only the article's own page may receive the result.
      window.opener.postMessage({ type: "biaslens-result", result }, from);
      message.textContent = `Done: ${result.summary.flagged} flagged words. Returning to the article…`;
      setTimeout(() => window.close(), 400);
      next.append("If this tab stays open, switch back to the article: the highlights are there.");
    } else {
      // The article's tab cannot be reached (some sites block it): show the marked-up text here.
      window.location.replace(`/#text=${encodeURIComponent(text)}`);
    }
  } catch (error) {
    message.textContent = `Could not analyse the article: ${error.message}`;
  }
}

main();
