// QuahogGPT chat page. The conversation lives here and is sent with every message;
// the server streams Server-Sent Events: meta, token, done, error (see web/server.py).

const messagesEl = document.getElementById("messages");
const emptyEl = document.getElementById("empty");
const chatEl = document.getElementById("chat");
const form = document.getElementById("composer");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");

let history = []; // [{role: "user" | "assistant", content}]
let busy = false;

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

// Minimal, safe formatting: escape everything, then **bold**, *italic*, paragraphs,
// and [n] citations (linked once the sources are known).
function render(text, sources = []) {
  const urls = new Map(sources.map((s) => [s.n, s.url]));
  let html = escapeHtml(text.trim())
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/\s?\[(\d+)\]/g, (_, n) => {
      const url = urls.get(Number(n));
      return url
        ? `<sup class="cite"><a href="${escapeHtml(url)}" target="_blank" rel="noopener" title="Source ${n}">${n}</a></sup>`
        : `<sup class="cite"><span>${n}</span></sup>`;
    });
  return html
    .split(/\n{2,}/)
    .map((p) => `<p>${p.replace(/\n/g, "<br>")}</p>`)
    .join("");
}

function addMessage(role) {
  emptyEl.hidden = true;
  const li = document.createElement("li");
  li.className = `msg ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  li.appendChild(bubble);
  messagesEl.appendChild(li);
  return { li, bubble };
}

function scrollDown() {
  chatEl.scrollTop = chatEl.scrollHeight;
}

function showSources(li, sources) {
  if (!sources.length) return;
  const list = document.createElement("ol");
  list.className = "sources";
  for (const s of sources) {
    const item = document.createElement("li");
    const snippet = s.text.length > 160 ? `${s.text.slice(0, 160)}...` : s.text;
    item.innerHTML = `<a href="${escapeHtml(s.url)}" target="_blank" rel="noopener">
      <span class="n">[${s.n}]</span><span class="title">${escapeHtml(s.title)}</span>
      <span class="snippet">${escapeHtml(snippet)}</span></a>`;
    list.appendChild(item);
  }
  li.appendChild(list);
}

// Parse a text/event-stream body incrementally; calls onEvent(name, data) per event.
async function readEvents(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      let name = "message";
      let data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) name = line.slice(7);
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      onEvent(name, JSON.parse(data));
    }
  }
}

async function send(message) {
  busy = true;
  sendBtn.disabled = true;
  addMessage("user").bubble.textContent = message;
  const { li, bubble } = addMessage("assistant");
  bubble.classList.add("typing");
  scrollDown();

  let answer = "";
  let final = null;
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, history }),
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.error || `HTTP ${response.status}`);
    }
    await readEvents(response, (name, data) => {
      if (name === "meta") {
        if (data.refused) li.classList.add("refused");
        else if (data.standalone && data.standalone !== message) {
          const note = document.createElement("p");
          note.className = "rewrite";
          note.textContent = `Searched as: ${data.standalone}`;
          li.appendChild(note);
        }
      } else if (name === "token") {
        answer += data;
        bubble.innerHTML = render(answer);
        scrollDown();
      } else if (name === "done") {
        final = data;
      } else if (name === "error") {
        throw new Error(data.error);
      }
    });
    if (!final) throw new Error("the answer stream ended early");
    bubble.innerHTML = render(final.answer, final.sources);
    showSources(li, final.sources);
    history.push({ role: "user", content: message }, { role: "assistant", content: final.answer });
  } catch (err) {
    li.classList.add("error");
    bubble.textContent = `Something went wrong: ${err.message}`;
  } finally {
    bubble.classList.remove("typing");
    busy = false;
    sendBtn.disabled = false;
    scrollDown();
    input.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message || busy) return;
  input.value = "";
  input.style.height = "";
  send(message);
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

input.addEventListener("input", () => {
  input.style.height = "";
  input.style.height = `${input.scrollHeight}px`;
});

for (const button of document.querySelectorAll(".example")) {
  button.addEventListener("click", () => {
    if (!busy) send(button.textContent);
  });
}

document.getElementById("new-chat").addEventListener("click", () => {
  if (busy) return;
  history = [];
  messagesEl.innerHTML = "";
  emptyEl.hidden = false;
  input.focus();
});

input.focus();
