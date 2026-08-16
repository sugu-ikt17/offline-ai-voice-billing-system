/**
 * utils.js — small, pure, dependency-free helper functions.
 *
 * Nothing here touches the DOM or the network. Every function takes
 * input and returns output — easy to reason about and reuse anywhere.
 */

/** Format a number as an Indian Rupee amount, e.g. 45 -> "₹45.00". */
export function formatCurrency(amount) {
  const value = Number(amount);
  return `₹${Number.isFinite(value) ? value.toFixed(2) : "0.00"}`;
}

/** Escape text before inserting it as HTML, to avoid injection via
 * item names that happen to contain HTML-special characters. */
export function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text ?? "";
  return div.innerHTML;
}

/** True if `name` is non-empty after trimming whitespace. */
export function isValidName(name) {
  return typeof name === "string" && name.trim().length > 0;
}

/** True if `price` parses to a finite number greater than zero. */
export function isValidPrice(price) {
  const value = typeof price === "number" ? price : parseFloat(price);
  return Number.isFinite(value) && value > 0;
}

/** Parse a form price input into a number, or NaN if not parseable. */
export function parsePrice(rawValue) {
  return parseFloat(String(rawValue).trim());
}

/** Simple debounce — delays invoking `fn` until `delay` ms of silence. */
export function debounce(fn, delay = 250) {
  let timerId;
  return (...args) => {
    window.clearTimeout(timerId);
    timerId = window.setTimeout(() => fn(...args), delay);
  };
}
