/**
 * api.js — API Layer.
 *
 * All network communication with the backend lives here. No DOM access
 * happens in this file. Every function uses fetch + async/await, returns
 * parsed JSON, and throws a readable Error on failure so callers only
 * need one try/catch.
 */

const getBaseUrl = () => {
  if (
    typeof window !== "undefined" &&
    window.location &&
    window.location.hostname &&
    window.location.protocol !== "file:"
  ) {
    // If frontend is served on a different port (e.g., port 5500 or 3000 via python http.server),
    // point API requests to FastAPI backend on port 8000.
    if (window.location.port && window.location.port !== "8000") {
      return `${window.location.protocol}//${window.location.hostname}:8000`;
    }
    return window.location.origin;
  }
  return "http://127.0.0.1:8000";
};

const ROOT_URL = getBaseUrl();
const API_BASE = `${ROOT_URL}/api/v1`;
const MENU_ENDPOINT = `${API_BASE}/menu`;

// Whisper.cpp speech transcription endpoint (spec: POST /api/v1/speech/transcribe)
const SPEECH_TRANSCRIBE_ENDPOINT = `${API_BASE}/speech/transcribe`;

// Legacy voice endpoint (kept for reference)
const VOICE_TRANSCRIBE_ENDPOINT = `${API_BASE}/voice/transcribe`;

/** Shared response handler: parses JSON, throws a readable Error on
 * any non-2xx status, and treats 204 as "no content" success. */
async function handleResponse(response) {
  if (response.status === 204) return undefined;

  let body = null;
  try {
    body = await response.json();
  } catch {
    // No JSON body — fall through with body = null.
  }

  if (!response.ok) {
    const message = (body && body.detail) || `Request failed with status ${response.status}.`;
    throw new Error(message);
  }
  return body;
}

/** Backend health check, used to drive the header status indicator. */
export async function checkHealth() {
  const response = await fetch(`${ROOT_URL}/`);
  const body = await handleResponse(response);
  if (!body || body.message !== "Offline AI Voice Billing API") {
    throw new Error("Invalid backend health response");
  }
  return body;
}

/** GET /menu — fetch all menu items. */
export async function getMenus() {
  const response = await fetch(MENU_ENDPOINT);
  const data = await handleResponse(response);
  if (Array.isArray(data)) return data;
  if (data && Array.isArray(data.items)) return data.items;
  if (data && Array.isArray(data.data)) return data.data;
  return [];
}

/** POST /menu — create a menu item. data = { name, price } */
export async function createMenu(data) {
  const response = await fetch(MENU_ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return handleResponse(response);
}

/** PUT /menu/{id} — update a menu item. data = { name, price } */
export async function updateMenu(id, data) {
  const response = await fetch(`${MENU_ENDPOINT}/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return handleResponse(response);
}

/** DELETE /menu/{id} — delete a menu item. */
export async function deleteMenu(id) {
  const response = await fetch(`${MENU_ENDPOINT}/${id}`, {
    method: "DELETE",
  });
  return handleResponse(response);
}

const ORDER_PROCESS_ENDPOINT = `${API_BASE}/orders/process`;

/**
 * POST /api/v1/speech/transcribe — Whisper.cpp backend transcription.
 *
 * Uploads audio as multipart/form-data field named "audio".
 * Returns { success: true, transcript: "<text>" } on success.
 * Throws an Error with a user-readable message on any failure
 * (including { success: false, message: "Speech engine unavailable" }).
 *
 * @param {Blob} audioBlob - WAV (or webm/ogg) audio recorded by the browser.
 * @param {string} [filename="recording.wav"] - filename hint for the server.
 * @returns {Promise<{success: boolean, transcript: string}>}
 */
export async function transcribeVoiceWhisper(audioBlob, filename = "recording.wav") {
  const formData = new FormData();
  formData.append("audio", audioBlob, filename);

  const response = await fetch(SPEECH_TRANSCRIBE_ENDPOINT, {
    method: "POST",
    body: formData,
  });

  // Parse JSON regardless of HTTP status to capture { success, message }
  let body = null;
  try {
    body = await response.json();
  } catch {
    throw new Error(`Speech service returned non-JSON response (HTTP ${response.status}).`);
  }

  if (!response.ok || (body && body.success === false)) {
    const msg = (body && body.message) || `Transcription failed (HTTP ${response.status}).`;
    throw new Error(msg);
  }

  return body; // { success: true, transcript: "..." }
}

/** POST /api/v1/voice/transcribe — legacy WAV upload endpoint (kept for compat). */
export async function transcribeVoice(audioBlob) {
  const formData = new FormData();
  formData.append("audio_file", audioBlob, "recording.wav");

  const response = await fetch(VOICE_TRANSCRIBE_ENDPOINT, {
    method: "POST",
    body: formData,
  });
  return handleResponse(response);
}

/** POST /orders/process — process recognized text into structured order & bill. */
export async function processOrder(text) {
  const response = await fetch(ORDER_PROCESS_ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ speech: text }),
  });
  return handleResponse(response);
}

function fallbackProcessText(text) {
  if (!text || !text.trim()) {
    return { items: [], total: 0 };
  }

  const items = [];
  const defaultPrices = {
    dosa: 45,
    tea: 15,
    coffee: 20,
    idli: 30,
    vada: 25,
    poori: 40,
    samosa: 20,
  };

  // Match pattern like "2 dosa 1 tea" or "3 idli 2 coffee"
  const regex = /(\d+)\s+([a-zA-Z\s]+?)(?=\s+\d+|\s*$)/g;
  let match;

  while ((match = regex.exec(text.toLowerCase())) !== null) {
    const qty = parseInt(match[1], 10);
    const rawName = match[2].trim();
    if (qty > 0 && rawName) {
      const name = rawName
        .split(" ")
        .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
        .join(" ");
      const price = defaultPrices[rawName.toLowerCase()] || 50;
      items.push({
        name,
        quantity: qty,
        unit_price: price,
        total_price: qty * price,
      });
    }
  }

  if (items.length === 0) {
    items.push({
      name: text.trim().charAt(0).toUpperCase() + text.trim().slice(1),
      quantity: 1,
      unit_price: 50,
      total_price: 50,
    });
  }

  const total = items.reduce((sum, item) => sum + item.total_price, 0);
  return { items, total };
}


