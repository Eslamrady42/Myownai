(() => {
  const form = document.getElementById("chat-form");
  const prompt = document.getElementById("prompt");
  const messages = document.getElementById("messages");
  const welcome = document.getElementById("welcome");
  const send = document.getElementById("send");
  const status = document.getElementById("model-status");
  let busy = false;

  function addMessage(role, text, isError = false) {
    const row = document.createElement("div");
    row.className = "message " + role;
    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = role === "assistant" ? "M" : "You";
    const content = document.createElement("div");
    content.className = "message-content" + (isError ? " error" : "");
    content.textContent = text;
    row.append(avatar, content);
    messages.append(row);
    row.scrollIntoView({ behavior: "smooth", block: "nearest" });
    return content;
  }

  async function refreshStatus() {
    try {
      const response = await fetch("/model/status");
      const data = await response.json();
      status.textContent = data.loaded
        ? "Ready · " + Number(data.parameters).toLocaleString() + " parameters"
        : "No trained checkpoint loaded";
    } catch {
      status.textContent = "API not connected";
    }
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = prompt.value.trim();
    if (!text || busy) return;
    busy = true;
    send.disabled = true;
    welcome.hidden = true;
    addMessage("user", text);
    prompt.value = "";
    const responseNode = addMessage("assistant", "Generating locally…");
    try {
      const response = await fetch("/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: text, max_new_tokens: 160, temperature: 0.8, top_p: 0.9 })
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Inference request failed");
      responseNode.textContent = payload.text || "(The model returned an empty response.)";
      await refreshStatus();
    } catch (error) {
      responseNode.classList.add("error");
      responseNode.textContent = "Unable to generate: " + error.message +
        ". Make sure a trained local checkpoint is configured and the API is running.";
    } finally {
      busy = false;
      send.disabled = false;
      prompt.focus();
    }
  });

  prompt.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  document.getElementById("new-chat").addEventListener("click", () => {
    messages.replaceChildren();
    welcome.hidden = false;
    prompt.focus();
  });
  document.getElementById("clear-chat").addEventListener("click", () => {
    messages.replaceChildren();
    welcome.hidden = false;
  });
  refreshStatus();
})();
