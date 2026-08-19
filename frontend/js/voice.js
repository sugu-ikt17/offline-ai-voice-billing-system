/**
 * voice.js — Voice Billing Controller (Whisper.cpp Backend STT).
 *
 * Architecture:
 *  1. MediaRecorder captures microphone audio.
 *  2. On stop, audio is converted to 16-bit PCM mono WAV in-browser.
 *  3. WAV blob is uploaded to POST /api/v1/speech/transcribe (Whisper.cpp).
 *  4. Transcript is shown in the UI and passed to the order parser.
 *  5. Backend order processing + menu matching returns a bill.
 *
 * Browser SpeechRecognition is intentionally NOT used — Whisper.cpp provides
 * significantly higher accuracy, especially for food-item vocabulary.
 */

import { transcribeVoiceWhisper, processOrder } from "./api.js";
import { formatCurrency, escapeHtml } from "./utils.js";

/** Convert AudioBuffer to a standard 16-bit PCM MONO 16000 Hz WAV Blob. */
async function blobToWavBlob(rawBlob) {
  try {
    const arrayBuffer = await rawBlob.arrayBuffer();
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return rawBlob;

    const audioCtx = new AudioCtx();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

    const targetSampleRate = 16000;
    const numChannels = 1; // mono

    // Resample to 16000 Hz mono using OfflineAudioContext if sample rate differs
    let samples;
    if (audioBuffer.sampleRate !== targetSampleRate) {
      const OfflineCtx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
      if (OfflineCtx) {
        const offlineCtx = new OfflineCtx(1, Math.ceil(audioBuffer.duration * targetSampleRate), targetSampleRate);
        const source = offlineCtx.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(offlineCtx.destination);
        source.start(0);
        const resampledBuffer = await offlineCtx.startRendering();
        samples = resampledBuffer.getChannelData(0);
      } else {
        samples = audioBuffer.getChannelData(0);
      }
    } else {
      samples = audioBuffer.getChannelData(0);
    }

    const dataLength = samples.length * 2; // 16-bit PCM = 2 bytes per sample
    const bufferLength = 44 + dataLength;
    const wavBuffer = new ArrayBuffer(bufferLength);
    const view = new DataView(wavBuffer);

    /* RIFF identifier */
    writeString(view, 0, "RIFF");
    /* RIFF chunk length */
    view.setUint32(4, 36 + dataLength, true);
    /* RIFF type */
    writeString(view, 8, "WAVE");
    /* format chunk identifier */
    writeString(view, 12, "fmt ");
    /* format chunk length */
    view.setUint32(16, 16, true);
    /* sample format (1 = PCM) */
    view.setUint16(20, 1, true);
    /* channel count */
    view.setUint16(22, numChannels, true);
    /* sample rate */
    view.setUint32(24, targetSampleRate, true);
    /* byte rate */
    view.setUint32(28, targetSampleRate * numChannels * 2, true);
    /* block align */
    view.setUint16(32, numChannels * 2, true);
    /* bits per sample */
    view.setUint16(34, 16, true);
    /* data chunk identifier */
    writeString(view, 36, "data");
    /* data chunk length */
    view.setUint32(40, dataLength, true);

    /* Write 16-bit PCM samples */
    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }

    if (audioCtx.state !== "closed") {
      await audioCtx.close();
    }

    return new Blob([wavBuffer], { type: "audio/wav" });
  } catch (err) {
    console.warn("WAV conversion fallback used:", err);
    return rawBlob;
  }
}

function writeString(view, offset, str) {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}

/** Format seconds into MM:SS string (e.g. 65 -> "01:05"). */
function formatTime(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  const pad = (n) => String(n).padStart(2, "0");
  return `${pad(mins)}:${pad(secs)}`;
}

export function createVoiceController({ Toast, showReceipt }) {
  // DOM Cache - Voice Recording & Text
  const startBtn = document.getElementById("start-rec-btn");
  const stopBtn = document.getElementById("stop-rec-btn");
  const timerEl = document.getElementById("recording-timer");
  const statusBadge = document.getElementById("voice-status-badge");
  const statusText = document.getElementById("voice-status-text");
  const playbackContainer = document.getElementById("voice-playback-container");
  const audioPlayback = document.getElementById("voice-playback");
  const recognizedTextEl = document.getElementById("recognized-text");

  // DOM Cache - Order Summary
  const orderCountBadge = document.getElementById("order-items-count");
  const orderEmptyEl = document.getElementById("order-summary-empty");
  const orderSkeletonEl = document.getElementById("order-summary-skeleton");
  const orderListEl = document.getElementById("order-summary-list");

  // DOM Cache - Bill Preview & Grand Total
  const billEmptyEl = document.getElementById("bill-preview-empty");
  const billSkeletonEl = document.getElementById("bill-preview-skeleton");
  const billReceiptContainer = document.getElementById("bill-receipt-container");
  const billTableBody = document.getElementById("bill-table-body");
  const billSubtotalEl = document.getElementById("bill-subtotal");
  const grandTotalDisplay = document.getElementById("grand-total-display");
  const completeOrderBtn = document.getElementById("complete-order-btn");

  // State
  let mediaRecorder = null;
  let audioStream = null;
  let audioChunks = [];
  let currentWavBlob = null;
  let currentPlaybackUrl = null;
  let timerInterval = null;
  let elapsedSeconds = 0;
  let currentProcessedOrder = null;
  // speechRecognition removed — replaced by Whisper.cpp backend

  function setStatus(state, label) {
    statusBadge.className = `status-badge ${state}`;
    statusText.textContent = label;
  }

  function startTimer() {
    elapsedSeconds = 0;
    timerEl.textContent = formatTime(elapsedSeconds);
    clearInterval(timerInterval);
    timerInterval = setInterval(() => {
      elapsedSeconds++;
      timerEl.textContent = formatTime(elapsedSeconds);
    }, 1000);
  }

  function stopTimer() {
    clearInterval(timerInterval);
    timerInterval = null;
  }

  function revokePlaybackUrl() {
    if (currentPlaybackUrl) {
      URL.revokeObjectURL(currentPlaybackUrl);
      currentPlaybackUrl = null;
    }
  }

  function resetWorkflowDisplays() {
    orderCountBadge.classList.add("hidden");
    orderEmptyEl.classList.remove("hidden");
    orderSkeletonEl.classList.add("hidden");
    orderListEl.classList.add("hidden");
    orderListEl.innerHTML = "";

    billEmptyEl.classList.remove("hidden");
    billSkeletonEl.classList.add("hidden");
    billReceiptContainer.classList.add("hidden");
    billTableBody.innerHTML = "";
    billSubtotalEl.textContent = "₹0.00";
    grandTotalDisplay.textContent = "₹ --";
    completeOrderBtn.disabled = true;
    currentProcessedOrder = null;
  }

  function showProcessingSkeletons() {
    orderEmptyEl.classList.add("hidden");
    orderListEl.classList.add("hidden");
    orderSkeletonEl.classList.remove("hidden");

    billEmptyEl.classList.add("hidden");
    billReceiptContainer.classList.add("hidden");
    billSkeletonEl.classList.remove("hidden");
  }

  function hideSkeletons() {
    orderSkeletonEl.classList.add("hidden");
    billSkeletonEl.classList.add("hidden");
  }

  /** Render Order Summary & Bill Preview from backend BillResult response.
   *  Expected shape: { bill: { items[], subtotal, grand_total, ... }, warnings, unmatched_items }
   */
  function renderOrderWorkflow(processedData) {
    hideSkeletons();
    currentProcessedOrder = processedData;

    // Extract items from BillResult shape (processedData.bill.items) or
    // fall back to flat processedData.items for backward compat.
    const bill  = (processedData && processedData.bill) || processedData || {};
    const items = bill.items || [];
    const subtotalAmount = bill.subtotal || 0;
    const grandTotal     = bill.grand_total || bill.total || 0;

    if (items.length === 0) {
      orderEmptyEl.classList.remove("hidden");
      billEmptyEl.classList.remove("hidden");
      grandTotalDisplay.textContent = "₹0.00";
      completeOrderBtn.disabled = true;
      return;
    }

    // 1. Render Order Summary Chips
    orderEmptyEl.classList.add("hidden");
    orderListEl.classList.remove("hidden");
    orderCountBadge.textContent = `${items.length} item${items.length > 1 ? "s" : ""}`;
    orderCountBadge.classList.remove("hidden");

    orderListEl.innerHTML = items
      .map(
        (item) => `
      <div class="order-item-chip">
        <span class="chip-qty">${item.quantity}x</span>
        <span class="chip-name">${escapeHtml(item.name)}</span>
        <span class="chip-price">${formatCurrency(item.subtotal || item.total_price || 0)}</span>
      </div>
    `
      )
      .join("");

    // 2. Render Bill Preview Receipt Table
    billEmptyEl.classList.add("hidden");
    billReceiptContainer.classList.remove("hidden");

    billTableBody.innerHTML = items
      .map(
        (item) => `
      <tr>
        <td>${escapeHtml(item.name)}</td>
        <td class="text-center font-bold">${item.quantity}</td>
        <td class="text-right">${formatCurrency(item.unit_price)}</td>
        <td class="text-right font-bold">${formatCurrency(item.subtotal || item.total_price || 0)}</td>
      </tr>
    `
      )
      .join("");

    billSubtotalEl.textContent = formatCurrency(subtotalAmount);
    grandTotalDisplay.textContent = formatCurrency(grandTotal);
    completeOrderBtn.disabled = false;
  }

  /** Step 1: Start Recording — captures microphone audio via MediaRecorder.
   *  Browser SpeechRecognition is NOT used. Transcription happens server-side
   *  via Whisper.cpp after the user stops recording.
   */
  async function handleStartRecording() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      Toast.error("Voice recording is not supported in this browser.");
      return;
    }

    try {
      audioStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          noiseSuppression: true,
          echoCancellation: true,
          // autoGainControl intentionally disabled: browser AGC drives recordings
          // to 0.0 dBFS which causes clipping/distortion before Whisper inference.
          // The backend AudioPreprocessor normalises to a safe -6 dBFS target.
          autoGainControl: false,
          channelCount: 1,
        },
      });
    } catch (err) {
      console.error("Microphone access error:", err);
      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        Toast.error("Microphone permission denied. Please allow mic access.");
      } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
        Toast.error("No microphone device found on your system.");
      } else {
        Toast.error(`Microphone error: ${err.message || "Failed to access mic"}`);
      }
      return;
    }

    audioChunks = [];
    currentWavBlob = null;
    revokePlaybackUrl();
    playbackContainer.classList.add("hidden");
    audioPlayback.removeAttribute("src");
    recognizedTextEl.value = "";
    resetWorkflowDisplays();

    // Init MediaRecorder
    try {
      mediaRecorder = new MediaRecorder(audioStream);
    } catch (err) {
      console.error("MediaRecorder creation error:", err);
      Toast.error("Failed to initialize audio recorder.");
      audioStream.getTracks().forEach((track) => track.stop());
      return;
    }

    mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        audioChunks.push(event.data);
      }
    };

    mediaRecorder.onstop = async () => {
      if (audioStream) {
        audioStream.getTracks().forEach((track) => track.stop());
        audioStream = null;
      }

      stopTimer();

      if (audioChunks.length === 0) {
        Toast.error("No audio was recorded.");
        setStatus("idle", "Idle");
        startBtn.disabled = false;
        stopBtn.disabled = true;
        return;
      }

      const rawBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || "audio/webm" });
      setStatus("processing", "Processing audio...");

      // Convert to 16-bit PCM mono WAV for Whisper/Sarvam compatibility
      currentWavBlob = await blobToWavBlob(rawBlob);

      currentPlaybackUrl = URL.createObjectURL(currentWavBlob);
      audioPlayback.src = currentPlaybackUrl;
      playbackContainer.classList.remove("hidden");

      // ── Automatic upload: no button click required ──────────────────────
      await _uploadAndProcess();
    };

    mediaRecorder.start(100);
    startTimer();

    setStatus("recording", "Recording...");
    startBtn.disabled = true;
    stopBtn.disabled = false;
  }

  /** Step 2: Stop Recording */
  function handleStopRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
    }
  }

  /** Upload WAV blob to Sarvam STT backend & process the order automatically.
   *
   * Called internally from onstop — never requires a button click.
   * Always sends the WAV blob to POST /api/v1/speech/transcribe.
   */
  async function _uploadAndProcess() {
    if (!currentWavBlob) {
      Toast.error("No recording available. Please record audio first.");
      return;
    }

    startBtn.disabled = true;
    stopBtn.disabled = true;
    setStatus("processing", "Sending to server...");
    recognizedTextEl.value = "";
    recognizedTextEl.placeholder = "Processing voice order...";
    showProcessingSkeletons();

    try {
      // ── Step 1: Sarvam STT transcription ────────────────────────────────
      setStatus("processing", "Transcribing...");
      const transcribeResult = await transcribeVoiceWhisper(currentWavBlob, "recording.wav");
      const text = (transcribeResult && transcribeResult.transcript) || "";

      if (!text) {
        Toast.error("Speech engine returned an empty transcript. Please try again.");
        hideSkeletons();
        recognizedTextEl.placeholder = "Recognized speech will appear here after stopping the recording...";
        setStatus("idle", "Workflow Ready");
        startBtn.disabled = false;
        return;
      }

      recognizedTextEl.value = text;
      recognizedTextEl.placeholder = "";

      // ── Step 2: Order processing & menu matching ─────────────────────────
      setStatus("processing", "Processing Order...");
      const processedData = await processOrder(text);

      renderOrderWorkflow(processedData);
      setStatus("success", "Workflow Ready");
      Toast.success("Voice order processed successfully!");
    } catch (err) {
      console.error("Workflow execution error:", err);
      hideSkeletons();
      resetWorkflowDisplays();
      recognizedTextEl.placeholder = "Recognized speech will appear here after stopping the recording...";
      Toast.error(err.message || "Failed to process voice order.");
      setStatus("idle", "Workflow Ready");
    } finally {
      startBtn.disabled = false;
      stopBtn.disabled = true;
    }
  }

  /** Step 4: Complete Order — open the Receipt Preview modal. */
  function handleCompleteOrder() {
    if (!currentProcessedOrder) return;
    if (typeof showReceipt === "function") {
      showReceipt(currentProcessedOrder);
    }
    setStatus("success", "Order Completed");
  }

  return {
    handleStartRecording,
    handleStopRecording,
    handleCompleteOrder,
  };
}
