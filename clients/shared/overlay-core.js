// Bias Lens overlay: find the article on a news page and mark flagged words where they stand.
// Shared by the browser extension (content script) and the phone bookmark. Needs no libraries,
// and defines one global, BiasLensOverlay.
(function (global) {
  "use strict";
  if (global.BiasLensOverlay) return;

  const MIN_PARAGRAPH = 40; // characters: shorter blocks are captions, bylines and buttons
  const SKIP = "nav, header, footer, aside, form, figure, figcaption, button, [role=navigation], " +
    "[aria-hidden=true], .biaslens-host, [class*=comment i], [id*=comment i], [class*=related i], " +
    "[class*=newsletter i], [class*=promo i], [class*=caption i]";
  const COLORS = { agreed: "205, 62, 52", contested: "214, 140, 20" };

  function readers(share) {
    const n = Math.round(share * 10);
    return n < 1 ? "Fewer than 1 in 10" : "About " + n + " in 10";
  }

  // The article body: the most specific element holding enough paragraph text. A page's <main>
  // often also holds comments and related stories, so it is only the fallback.
  function articleRoot(doc) {
    const score = function (el) {
      let total = 0;
      for (const p of el.querySelectorAll("p")) if (!p.closest(SKIP)) total += p.textContent.trim().length;
      return total;
    };
    for (const selector of ["[itemprop=articleBody]", "article", "main, [role=main]"]) {
      let best = null;
      let bestScore = 0;
      for (const el of doc.querySelectorAll(selector)) {
        const s = score(el);
        if (s > bestScore) {
          best = el;
          bestScore = s;
        }
      }
      if (best && bestScore >= 500) return best;
    }
    return doc.body;
  }

  // The article's text, and where each text node sits in it.
  function extract(doc, maxChars) {
    maxChars = maxChars || 20000;
    const blocks = [];
    const headline = doc.querySelector("h1");
    if (headline && !headline.closest(SKIP)) blocks.push(headline);
    for (const p of articleRoot(doc).querySelectorAll("p")) {
      if (!p.closest(SKIP) && p.textContent.trim().length >= MIN_PARAGRAPH) blocks.push(p);
    }
    const nodes = [];
    let text = "";
    for (const block of blocks) {
      if (text.length >= maxChars) break;
      const walker = doc.createTreeWalker(block, NodeFilter.SHOW_TEXT);
      let node;
      while ((node = walker.nextNode())) {
        if (node.parentElement && node.parentElement.closest("script, style, noscript")) continue;
        nodes.push({ node: node, start: text.length, end: text.length + node.nodeValue.length });
        text += node.nodeValue;
      }
      text += "\n";
    }
    return { text: text.slice(0, maxChars), nodes: nodes };
  }

  function addStyles(doc) {
    if (doc.getElementById("biaslens-styles")) return;
    const style = doc.createElement("style");
    style.id = "biaslens-styles";
    style.textContent =
      "mark.biaslens-mark{color:inherit!important;padding:0 1px;border-radius:2px;cursor:help}" +
      "mark.biaslens-agreed{background:rgba(" + COLORS.agreed + ",calc(.2 + var(--biaslens-share,.8)*.5))!important}" +
      "mark.biaslens-contested{background:rgba(" + COLORS.contested + ",calc(.12 + var(--biaslens-share,.4)*.45))!important;" +
      "border-bottom:2px dashed rgb(" + COLORS.contested + ")}" +
      "body.biaslens-hidden mark.biaslens-mark{background:none!important;border:0!important}";
    doc.head.appendChild(style);
  }

  // Wrap every flagged word in a <mark>. Words split across two text nodes are left alone.
  function highlight(extracted, result, doc) {
    addStyles(doc);
    const words = [];
    for (const sentence of result.sentences) {
      for (const word of sentence.words) if (word.band !== "none") words.push(word);
    }
    words.sort(function (a, b) { return a.start - b.start; });
    const byNode = new Map();
    let i = 0;
    for (const word of words) {
      while (i < extracted.nodes.length && extracted.nodes[i].end <= word.start) i++;
      const entry = extracted.nodes[i];
      if (!entry || word.start < entry.start || word.end > entry.end) continue;
      if (!byNode.has(entry)) byNode.set(entry, []);
      byNode.get(entry).push(word);
    }
    let count = 0;
    for (const [entry, list] of byNode) {
      for (const word of list.reverse()) { // from the end, so earlier offsets stay valid
        if (!entry.node.parentNode) continue;
        entry.node.splitText(word.end - entry.start);
        const piece = entry.node.splitText(word.start - entry.start);
        const mark = doc.createElement("mark");
        mark.className = "biaslens-mark biaslens-" + word.band;
        mark.style.setProperty("--biaslens-share", word.share);
        mark.dataset.biaslens = readers(word.share) + " readers would flag “" + word.text + "”" +
          (word.lexicon ? " (also in the paper's bias lexicon)" : "");
        mark.title = mark.dataset.biaslens;
        piece.parentNode.replaceChild(mark, piece);
        mark.appendChild(piece);
        count++;
      }
    }
    return count;
  }

  function removeHighlights(doc) {
    for (const mark of Array.from(doc.querySelectorAll("mark.biaslens-mark"))) {
      const parent = mark.parentNode;
      while (mark.firstChild) parent.insertBefore(mark.firstChild, mark);
      parent.removeChild(mark);
      parent.normalize();
    }
  }

  // A small card in the corner, in a shadow root so the page's styles cannot reach it.
  function badge(doc) {
    let host = doc.querySelector(".biaslens-host");
    if (!host) {
      host = doc.createElement("div");
      host.className = "biaslens-host";
      host.style.cssText = "position:fixed;right:12px;bottom:12px;z-index:2147483647";
      doc.body.appendChild(host);
      host.attachShadow({ mode: "open" });
    }
    const root = host.shadowRoot;
    root.innerHTML =
      "<style>" +
      ".card{font:13px/1.4 system-ui,sans-serif;color:#1d232a;background:#fff;border:1px solid #d9dde1;" +
      "border-radius:10px;box-shadow:0 4px 18px rgba(0,0,0,.18);padding:10px 12px;max-width:260px}" +
      ".top{display:flex;justify-content:space-between;align-items:center;gap:8px;font-weight:600}" +
      ".strip{display:flex;align-items:flex-end;gap:2px;height:28px;width:120px;padding:3px;background:#11161b;" +
      "border-radius:4px;margin:6px 0}.strip span{flex:1;background:#ff5a36;border-radius:1px}" +
      "button{font:inherit;border:1px solid #d9dde1;background:#fff;border-radius:6px;padding:2px 8px;cursor:pointer}" +
      ".x{border:0;font-size:16px;padding:0 4px}.tip{margin-top:6px;color:#5b6672}</style>" +
      "<div class=card><div class=top><span>Bias Lens</span><button class=x title=Close>×</button></div>" +
      "<div class=body>Reading the article…</div><div class=tip></div></div>";
    root.querySelector(".x").onclick = function () {
      doc.body.classList.remove("biaslens-hidden");
      removeHighlights(doc);
      host.remove();
    };
    const body = root.querySelector(".body");
    const tip = root.querySelector(".tip");
    // On phones there is no hover: tapping a marked word shows its line here.
    doc.addEventListener("click", function (event) {
      const mark = event.target.closest && event.target.closest("mark.biaslens-mark");
      if (mark) tip.textContent = mark.dataset.biaslens;
    });
    return {
      show: function (summary) {
        body.innerHTML = "";
        const strip = doc.createElement("div");
        strip.className = "strip";
        for (const level of summary.strip) {
          const bar = doc.createElement("span");
          bar.style.height = (level / 8) * 100 + "%";
          strip.appendChild(bar);
        }
        const line = doc.createElement("div");
        line.textContent = summary.flagged === 0 ? "No flagged words." :
          summary.flagged + " flagged: " + summary.agreed + " most readers would flag, " +
          summary.contested + " readers split";
        const toggle = doc.createElement("button");
        toggle.textContent = "Hide highlights";
        toggle.onclick = function () {
          const hidden = doc.body.classList.toggle("biaslens-hidden");
          toggle.textContent = hidden ? "Show highlights" : "Hide highlights";
        };
        body.append(strip, line, toggle);
        tip.textContent = "Tap or hover over a marked word to see how many readers would flag it.";
      },
      error: function (message) {
        body.textContent = message;
      },
    };
  }

  global.BiasLensOverlay = { extract: extract, highlight: highlight, removeHighlights: removeHighlights,
    badge: badge, readers: readers };
})(typeof window !== "undefined" ? window : this);
