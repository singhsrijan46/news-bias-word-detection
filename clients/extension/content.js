// Bias Lens extension, content script: on news sites, finds the article and marks flagged words.
// Runs automatically on the sites listed in manifest.json, and on any page when the popup asks.
(function () {
  "use strict";
  if (window.__biasLensLoaded) return;
  window.__biasLensLoaded = true;

  let running = false;

  async function run(manual) {
    if (running) return;
    const extracted = BiasLensOverlay.extract(document);
    if (!manual && extracted.text.trim().length < 600) return; // not an article page
    running = true;
    BiasLensOverlay.removeHighlights(document);
    const card = BiasLensOverlay.badge(document);
    try {
      const reply = await chrome.runtime.sendMessage({ type: "analyze", text: extracted.text });
      if (reply.error) {
        card.error(reply.error);
        return;
      }
      BiasLensOverlay.highlight(extracted, reply.result, document);
      card.show(reply.result.summary);
    } catch (error) {
      card.error("Bias Lens could not reach the board: " + error.message);
    } finally {
      running = false;
    }
  }

  chrome.runtime.onMessage.addListener((message) => {
    if (message.type === "biaslens-analyze-now") run(true);
  });

  chrome.runtime.sendMessage({ type: "status" }).then((status) => {
    if (status && status.paired && status.auto && !status.pausedSites.includes(location.hostname)) {
      setTimeout(() => run(false), 1500); // give the page a moment to finish loading its text
    }
  });
})();
