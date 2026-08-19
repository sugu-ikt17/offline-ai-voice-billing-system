import { checkHealth } from "./api.js";
import { createMenuController } from "./menu.js";
import { createVoiceController } from "./voice.js";
import { createReceiptController } from "./receipt.js";

// ---------------------------------------------------------------------
// Toast notifications
// ---------------------------------------------------------------------
const Toast = (() => {
  const container = document.getElementById("toast-container");
  const DISPLAY_MS = 3000;

  function show(message, type = "success") {
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    window.setTimeout(() => {
      toast.classList.add("fade-out");
      toast.addEventListener("animationend", () => toast.remove());
    }, DISPLAY_MS);
  }

  return {
    success: (message) => show(message, "success"),
    error: (message) => show(message, "error"),
  };
})();

// ---------------------------------------------------------------------
// Full-page loading overlay
// ---------------------------------------------------------------------
const LoadingOverlay = (() => {
  const overlay = document.getElementById("loading-overlay");
  const label = document.getElementById("loading-overlay-text");

  function show(message = "Loading...") {
    label.textContent = message;
    overlay.classList.remove("hidden");
  }

  function hide() {
    overlay.classList.add("hidden");
  }

  return { show, hide };
})();

// ---------------------------------------------------------------------
// Confirmation modal — returns a Promise<boolean> resolved by the
// user's choice, so callers can `await` it like a native confirm().
// ---------------------------------------------------------------------
const ConfirmModal = (() => {
  const backdrop = document.getElementById("confirm-modal-backdrop");
  const titleEl = document.getElementById("confirm-modal-title");
  const messageEl = document.getElementById("confirm-modal-message");
  const cancelBtn = document.getElementById("confirm-modal-cancel");
  const confirmBtn = document.getElementById("confirm-modal-confirm");

  function open({ title = "Are you sure?", message = "", confirmLabel = "Confirm" } = {}) {
    titleEl.textContent = title;
    messageEl.textContent = message;
    confirmBtn.textContent = confirmLabel;
    backdrop.classList.remove("hidden");

    return new Promise((resolve) => {
      function cleanup(result) {
        backdrop.classList.add("hidden");
        cancelBtn.removeEventListener("click", onCancel);
        confirmBtn.removeEventListener("click", onConfirm);
        backdrop.removeEventListener("click", onBackdropClick);
        resolve(result);
      }
      function onCancel() {
        cleanup(false);
      }
      function onConfirm() {
        cleanup(true);
      }
      function onBackdropClick(event) {
        if (event.target === backdrop) cleanup(false);
      }

      cancelBtn.addEventListener("click", onCancel);
      confirmBtn.addEventListener("click", onConfirm);
      backdrop.addEventListener("click", onBackdropClick);
    });
  }

  return { open };
})();

// ---------------------------------------------------------------------
// Backend status indicator (header)
// ---------------------------------------------------------------------
const StatusIndicator = (() => {
  const el = document.getElementById("status-indicator");
  const textEl = document.getElementById("status-text");

  function setState(state, label) {
    el.classList.remove("online", "offline", "checking");
    el.classList.add(state);
    textEl.textContent = label;
  }

  return {
    setChecking: () => setState("checking", "Checking backend..."),
    setOnline: () => setState("online", "Backend connected"),
    setOffline: () => setState("offline", "Backend unreachable"),
  };
})();

async function refreshBackendStatus() {
  StatusIndicator.setChecking();
  try {
    await checkHealth();
    StatusIndicator.setOnline();
  } catch {
    StatusIndicator.setOffline();
  }
}

// ---------------------------------------------------------------------
// Event registration — every button/form event for the page is wired
// here, pointing at handlers exposed by menu and voice controllers.
// ---------------------------------------------------------------------
function registerEvents(menu, voice) {
  const form = document.getElementById("menu-form");
  const cancelEditBtn = document.getElementById("cancel-edit-btn");
  const tableBody = document.getElementById("menu-table-body");
  const nameInput = document.getElementById("item-name");
  const priceInput = document.getElementById("item-price");

  form.addEventListener("submit", menu.handleFormSubmit);
  cancelEditBtn.addEventListener("click", menu.resetForm);
  tableBody.addEventListener("click", menu.handleTableClick);
  nameInput.addEventListener("input", menu.clearNameError);
  priceInput.addEventListener("input", menu.clearPriceError);

  const startRecBtn = document.getElementById("start-rec-btn");
  const stopRecBtn = document.getElementById("stop-rec-btn");
  const completeOrderBtn = document.getElementById("complete-order-btn");

  startRecBtn.addEventListener("click", voice.handleStartRecording);
  stopRecBtn.addEventListener("click", voice.handleStopRecording);
  if (completeOrderBtn) {
    completeOrderBtn.addEventListener("click", voice.handleCompleteOrder);
  }
}

// ---------------------------------------------------------------------
// Initialize application
// ---------------------------------------------------------------------
function initApp() {
  refreshBackendStatus();

  const menu = createMenuController({
    Toast,
    LoadingOverlay,
    ConfirmModal,
    refreshBackendStatus,
  });

  // Receipt controller — created first so we can inject showReceipt
  // into the voice controller.
  const receipt = createReceiptController({
    Toast,
    onNewOrder: () => {
      // Reset the voice workflow for a fresh order.
      // Re-enable start button, clear text, reset cards.
      document.getElementById("start-rec-btn").disabled = false;
      document.getElementById("stop-rec-btn").disabled = true;
      document.getElementById("recognized-text").value = "";
      document.getElementById("recording-timer").textContent = "00:00";

      // Reset voice status badge
      const badge = document.getElementById("voice-status-badge");
      badge.className = "status-badge idle";
      document.getElementById("voice-status-text").textContent = "Idle";

      // Hide playback
      document.getElementById("voice-playback-container").classList.add("hidden");

      // Reset order cards
      document.getElementById("order-items-count").classList.add("hidden");
      document.getElementById("order-summary-empty").classList.remove("hidden");
      document.getElementById("order-summary-list").classList.add("hidden");
      document.getElementById("order-summary-list").innerHTML = "";

      // Reset bill cards
      document.getElementById("bill-preview-empty").classList.remove("hidden");
      document.getElementById("bill-receipt-container").classList.add("hidden");
      document.getElementById("bill-table-body").innerHTML = "";
      document.getElementById("bill-subtotal").textContent = "₹0.00";
      document.getElementById("grand-total-display").textContent = "₹ --";
      document.getElementById("complete-order-btn").disabled = true;
    },
  });

  const voice = createVoiceController({
    Toast,
    showReceipt: receipt.showReceipt,
  });

  registerEvents(menu, voice);
  menu.loadMenu(); // load menu list automatically on page load
}

initApp();


