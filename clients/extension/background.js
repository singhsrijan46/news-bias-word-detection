// Bias Lens extension, background: keeps the pairing with the board and makes the encrypted calls.
// The article text goes only to the board the user paired with, encrypted with that pairing's key.

import { startPairing, finishPairing, secureCall } from "./biaslens-client.js";

const DEVICE_NAME = `Bias Lens extension (${navigator.userAgent.includes("Edg/") ? "Edge" : "Chrome"})`;

async function state() {
  return chrome.storage.local.get(["boardUrl", "pairing", "pendingSid", "auto", "pausedSites"]);
}

async function handle(message) {
  const s = await state();
  switch (message.type) {
    case "status":
      return { boardUrl: s.boardUrl || "", paired: Boolean(s.pairing), name: s.pairing && s.pairing.name,
               auto: s.auto !== false, pausedSites: s.pausedSites || [] };
    case "set-board":
      await chrome.storage.local.set({ boardUrl: message.boardUrl.replace(/\/+$/, "") });
      return { ok: true };
    case "pair-start": {
      const session = await startPairing(s.boardUrl, DEVICE_NAME);
      await chrome.storage.local.set({ pendingSid: session.sid });
      return { ok: true, expiresIn: session.expires_in };
    }
    case "pair-finish": {
      const pairing = await finishPairing(s.boardUrl, s.pendingSid, message.code, DEVICE_NAME);
      await chrome.storage.local.set({ pairing, pendingSid: null });
      return { ok: true };
    }
    case "forget":
      if (s.pairing) {
        try { await secureCall(s.pairing, { op: "forget" }); } catch { /* forget locally anyway */ }
      }
      await chrome.storage.local.remove("pairing");
      return { ok: true };
    case "set-auto":
      await chrome.storage.local.set({ auto: message.auto });
      return { ok: true };
    case "pause-site": {
      const paused = new Set(s.pausedSites || []);
      if (message.paused) paused.add(message.host); else paused.delete(message.host);
      await chrome.storage.local.set({ pausedSites: [...paused] });
      return { ok: true };
    }
    case "analyze":
      if (!s.pairing) throw new Error("Not paired with a Bias Lens board yet: open the extension's popup.");
      return { result: await secureCall(s.pairing, { op: "analyze", text: message.text }) };
    default:
      throw new Error(`Unknown request ${message.type}`);
  }
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  handle(message).then(sendResponse, (error) => sendResponse({ error: error.message }));
  return true; // answer asynchronously
});
