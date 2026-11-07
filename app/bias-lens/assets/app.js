// Bias Lens web page: sends text to the board's API and shows which words readers would flag.

const $ = (id) => document.getElementById(id);

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

function setStatus(message) {
  $("status").textContent = message;
}

async function analyze() {
  const text = $("text").value.trim();
  if (!text) {
    setStatus("Paste some text first.");
    return;
  }
  setStatus("Analysing on the board…");
  $("analyze").disabled = true;
  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || response.statusText);
    }
    render(await response.json());
    setStatus("");
  } catch (error) {
    setStatus(`Could not analyse the text: ${error.message}`);
  } finally {
    $("analyze").disabled = false;
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
  $("results").hidden = false;
}

async function showModel() {
  try {
    const info = await (await fetch("/api/info")).json();
    const parts = ["Model: a small student network (MiniLM-L6), distilled from a RoBERTa-base teacher"];
    if (info.precision) parts.push(`${info.precision.toUpperCase()} on the board's CPU`);
    if (info.cv_f1) parts.push(`cross-validated F1 ${info.cv_f1.toFixed(2)} on MBIC`);
    parts.push(`words are flagged from ${readers(info.threshold).toLowerCase()} readers`);
    $("model").textContent = parts.join(" · ");
  } catch (error) {
    $("model").textContent = "Model information is not available.";
  }
}

function setUpSharing() {
  const origin = window.location.origin;
  const code = "(()=>{const t=(getSelection().toString()||document.body.innerText).slice(0,20000);" +
    `window.open('${origin}/#text='+encodeURIComponent(t),'_blank')})()`;
  $("bookmarklet").href = `javascript:${code}`;
  $("curl").textContent = `curl -X POST ${origin}/api/analyze \\\n  -H "Content-Type: application/json" \\\n` +
    `  -d '{"text": "Critics slammed the reckless plan."}'`;
}

document.addEventListener("DOMContentLoaded", () => {
  $("analyze").addEventListener("click", analyze);
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
    analyze();
  }
});
