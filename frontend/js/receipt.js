/**
 * receipt.js — Receipt Preview Controller.
 *
 * Renders a thermal-receipt-style modal from BillResult data.
 * Handles Print Receipt (window.print()) and New Order (page reset).
 *
 * This module is entirely self-contained — it owns the receipt modal DOM,
 * formats all values, and manages show/hide. Nothing here touches the
 * voice recorder, menu, or order processing logic.
 */

import { formatCurrency, escapeHtml } from "./utils.js";

/**
 * Create and return a receipt controller.
 *
 * @param {Object} deps
 * @param {Object} deps.Toast — Toast.success() / Toast.error()
 * @param {Function} deps.onNewOrder — called when user clicks "New Order"
 * @returns {{ showReceipt: Function, hideReceipt: Function }}
 */
export function createReceiptController({ Toast, onNewOrder }) {
  // ── DOM Cache ──────────────────────────────────────────────────
  const backdrop        = document.getElementById("receipt-modal-backdrop");
  const shopNameEl      = document.getElementById("receipt-shop-name");
  const billNumberEl    = document.getElementById("receipt-bill-number");
  const dateTimeEl      = document.getElementById("receipt-date-time");
  const itemsBodyEl     = document.getElementById("receipt-items-body");
  const itemCountEl     = document.getElementById("receipt-item-count");
  const subtotalValEl   = document.getElementById("receipt-subtotal-val");
  const discountRowEl   = document.getElementById("receipt-discount-row");
  const discountValEl   = document.getElementById("receipt-discount-val");
  const taxRowEl        = document.getElementById("receipt-tax-row");
  const taxValEl        = document.getElementById("receipt-tax-val");
  const grandTotalValEl = document.getElementById("receipt-grand-total-val");
  const warningsEl      = document.getElementById("receipt-warnings-section");
  const warningsListEl  = document.getElementById("receipt-warnings-list");
  const printBtn        = document.getElementById("receipt-print-btn");
  const newOrderBtn     = document.getElementById("receipt-new-order-btn");

  // ── Helpers ────────────────────────────────────────────────────

  /** Format an ISO datetime string into a human-readable local format. */
  function formatDateTime(isoStr) {
    try {
      const d = new Date(isoStr);
      return d.toLocaleString("en-IN", {
        day:    "2-digit",
        month:  "short",
        year:   "numeric",
        hour:   "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: true,
      });
    } catch {
      return isoStr || "—";
    }
  }

  // ── Render ─────────────────────────────────────────────────────

  /**
   * Populate and show the receipt modal.
   *
   * @param {Object} billData — BillResult.to_dict() shape from backend:
   *   { bill: {bill_number, date_time, items[], item_count, total_quantity,
   *            subtotal, discount, tax, grand_total},
   *     warnings: [], unmatched_items: [] }
   * @param {string} [shopName="AI Voice Billing"] — displayed at top.
   */
  function showReceipt(billData, shopName = "AI Voice Billing") {
    if (!billData || !billData.bill) {
      Toast.error("No bill data to display.");
      return;
    }

    const bill = billData.bill;

    // Header
    shopNameEl.textContent  = shopName;
    billNumberEl.textContent = bill.bill_number || "—";
    dateTimeEl.textContent   = formatDateTime(bill.date_time);

    // Items table body
    const items = bill.items || [];
    if (items.length === 0) {
      itemsBodyEl.innerHTML = `
        <tr>
          <td colspan="4" style="text-align:center; color:#999; padding:12px 0;">
            No items in this bill.
          </td>
        </tr>`;
    } else {
      itemsBodyEl.innerHTML = items.map((item) => `
        <tr>
          <td class="col-item">${escapeHtml(item.name)}</td>
          <td class="col-qty">${item.quantity}</td>
          <td class="col-price">${formatCurrency(item.unit_price)}</td>
          <td class="col-total">${formatCurrency(item.subtotal)}</td>
        </tr>
      `).join("");
    }

    // Item count + total quantity
    itemCountEl.textContent =
      `${bill.item_count || 0} item${(bill.item_count || 0) !== 1 ? "s" : ""} · ` +
      `${bill.total_quantity || 0} unit${(bill.total_quantity || 0) !== 1 ? "s" : ""}`;

    // Subtotal
    subtotalValEl.textContent = formatCurrency(bill.subtotal || 0);

    // Discount row — hide when 0
    if (bill.discount && bill.discount > 0) {
      discountRowEl.classList.remove("hidden");
      discountValEl.textContent = `−${formatCurrency(bill.discount)}`;
    } else {
      discountRowEl.classList.add("hidden");
    }

    // Tax row — hide when 0
    if (bill.tax && bill.tax > 0) {
      taxRowEl.classList.remove("hidden");
      taxValEl.textContent = formatCurrency(bill.tax);
    } else {
      taxRowEl.classList.add("hidden");
    }

    // Grand total
    grandTotalValEl.textContent = formatCurrency(bill.grand_total || 0);

    // Warnings
    const warnings = billData.warnings || [];
    if (warnings.length > 0) {
      warningsEl.classList.remove("hidden");
      warningsListEl.innerHTML = warnings
        .map((w) => `<div class="receipt-warning-item">${escapeHtml(w)}</div>`)
        .join("");
    } else {
      warningsEl.classList.add("hidden");
    }

    // Show modal with animation
    backdrop.classList.add("visible");
    document.body.style.overflow = "hidden"; // prevent background scroll
  }

  /** Hide the receipt modal. */
  function hideReceipt() {
    backdrop.classList.remove("visible");
    document.body.style.overflow = "";
  }

  // ── Event Handlers ─────────────────────────────────────────────

  /** Print Receipt — uses window.print(); the CSS @media print rules
   *  ensure only the receipt paper is printed. */
  function handlePrint() {
    window.print();
  }

  /** New Order — close receipt, reset workflow for a fresh order. */
  function handleNewOrder() {
    hideReceipt();
    if (typeof onNewOrder === "function") {
      onNewOrder();
    }
    Toast.success("Ready for a new order!");
  }

  // ── Bind Events ────────────────────────────────────────────────
  if (printBtn)    printBtn.addEventListener("click",    handlePrint);
  if (newOrderBtn) newOrderBtn.addEventListener("click", handleNewOrder);

  // Close on backdrop click
  backdrop.addEventListener("click", (e) => {
    if (e.target === backdrop) hideReceipt();
  });

  // Close on Escape key
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && backdrop.classList.contains("visible")) {
      hideReceipt();
    }
  });

  return { showReceipt, hideReceipt };
}
