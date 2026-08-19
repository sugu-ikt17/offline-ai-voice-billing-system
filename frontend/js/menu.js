/**
 * menu.js — Menu Logic.
 *
 * Implements the menu feature: loading, adding, editing, deleting,
 * table rendering, form validation/reset. Exposes a factory,
 * createMenuController(), rather than importing UI primitives (Toast,
 * LoadingOverlay, ConfirmModal) directly from app.js — that would
 * create a circular module dependency (app.js -> menu.js -> app.js).
 * Instead, app.js builds those primitives and injects them here, then
 * registers the returned handlers as the actual event listeners.
 */

import { getMenus, createMenu, updateMenu, deleteMenu } from "./api.js";
import { formatCurrency, isValidName, isValidPrice, parsePrice } from "./utils.js";

/**
 * @param {object} deps
 * @param {object} deps.Toast - { success(msg), error(msg) }
 * @param {object} deps.LoadingOverlay - { show(msg), hide() }
 * @param {object} deps.ConfirmModal - { open({title, message, confirmLabel}) -> Promise<boolean> }
 * @param {() => Promise<void>} deps.refreshBackendStatus
 */
export function createMenuController({ Toast, LoadingOverlay, ConfirmModal, refreshBackendStatus }) {
  // ----- DOM references -----
  const tableBody = document.getElementById("menu-table-body");
  const emptyState = document.getElementById("empty-state");
  const tableLoading = document.getElementById("table-loading");
  const itemCount = document.getElementById("item-count");

  const form = document.getElementById("menu-form");
  const nameInput = document.getElementById("item-name");
  const priceInput = document.getElementById("item-price");
  const nameError = document.getElementById("item-name-error");
  const priceError = document.getElementById("item-price-error");
  const formCardTitle = document.getElementById("form-card-title");
  const submitBtn = document.getElementById("submit-btn");
  const submitLabel = submitBtn.querySelector(".btn-label");
  const submitSpinner = submitBtn.querySelector(".btn-spinner");
  const cancelEditBtn = document.getElementById("cancel-edit-btn");

  // ----- State -----
  let currentItems = [];
  let editingId = null; // null = "add" mode, otherwise the id being edited
  let hasLoadedOnce = false;

  // ----- Field validation -----
  function setFieldError(inputEl, errorEl, message) {
    inputEl.classList.add("invalid");
    errorEl.textContent = message;
  }

  function clearFieldError(inputEl, errorEl) {
    inputEl.classList.remove("invalid");
    errorEl.textContent = "";
  }

  function clearAllFieldErrors() {
    clearFieldError(nameInput, nameError);
    clearFieldError(priceInput, priceError);
  }

  function validateForm() {
    let valid = true;
    clearAllFieldErrors();

    if (!isValidName(nameInput.value)) {
      setFieldError(nameInput, nameError, "Item name is required.");
      valid = false;
    }

    const priceRaw = priceInput.value.trim();
    if (priceRaw === "" || Number.isNaN(parsePrice(priceRaw))) {
      setFieldError(priceInput, priceError, "Price is required.");
      valid = false;
    } else if (!isValidPrice(priceRaw)) {
      setFieldError(priceInput, priceError, "Price must be greater than zero.");
      valid = false;
    }

    return valid;
  }

  // ----- Form mode (add vs edit) + reset -----
  function enterEditMode(item) {
    editingId = item.id;
    nameInput.value = item.name;
    priceInput.value = item.price;
    formCardTitle.textContent = "Edit Menu Item";
    submitLabel.textContent = "Update Menu Item";
    cancelEditBtn.classList.remove("hidden");
    clearAllFieldErrors();
    nameInput.focus();
  }

  function resetForm() {
    editingId = null;
    form.reset();
    formCardTitle.textContent = "Add Menu Item";
    submitLabel.textContent = "Add Menu Item";
    cancelEditBtn.classList.add("hidden");
    clearAllFieldErrors();
  }

  function setSubmitBusy(isBusy) {
    submitBtn.disabled = isBusy;
    submitSpinner.classList.toggle("hidden", !isBusy);
  }

  // ----- Table rendering -----
  function renderTable(items) {
    tableBody.innerHTML = "";
    const list = Array.isArray(items) ? items : (items && Array.isArray(items.items) ? items.items : []);
    itemCount.textContent = list.length
      ? `${list.length} item${list.length === 1 ? "" : "s"}`
      : "";

    if (!list || list.length === 0) {
      emptyState.classList.remove("hidden");
      return;
    }
    emptyState.classList.add("hidden");

    list.forEach((item) => {
      const row = document.createElement("tr");

      const nameCell = document.createElement("td");
      nameCell.textContent = item.name;

      const priceCell = document.createElement("td");
      priceCell.textContent = formatCurrency(item.price);

      const editCell = document.createElement("td");
      editCell.className = "col-action";
      const editBtn = document.createElement("button");
      editBtn.type = "button";
      editBtn.className = "btn-icon edit";
      editBtn.textContent = "Edit";
      editBtn.dataset.id = item.id;
      editBtn.dataset.action = "edit";
      editCell.appendChild(editBtn);

      const deleteCell = document.createElement("td");
      deleteCell.className = "col-action";
      const deleteBtn = document.createElement("button");
      deleteBtn.type = "button";
      deleteBtn.className = "btn-icon delete";
      deleteBtn.textContent = "Delete";
      deleteBtn.dataset.id = item.id;
      deleteBtn.dataset.action = "delete";
      deleteCell.appendChild(deleteBtn);

      row.append(nameCell, priceCell, editCell, deleteCell);
      tableBody.appendChild(row);
    });
  }

  function setTableLoading(isLoading) {
    tableLoading.classList.toggle("hidden", !isLoading);
    if (isLoading) {
      emptyState.classList.add("hidden");
      tableBody.innerHTML = "";
    }
  }

  // ----- Load menu (called automatically on page load by app.js) -----
  async function loadMenu() {
    if (!hasLoadedOnce) {
      LoadingOverlay.show("Loading menu...");
    } else {
      setTableLoading(true);
    }

    try {
      currentItems = await getMenus();
      renderTable(currentItems);
    } catch (error) {
      Toast.error(`Could not load menu: ${error.message}`);
      renderTable([]);
      refreshBackendStatus();
    } finally {
      LoadingOverlay.hide();
      setTableLoading(false);
      hasLoadedOnce = true;
    }
  }

  // ----- Add / Update -----
  async function handleFormSubmit(event) {
    event.preventDefault();
    if (!validateForm()) return;

    const name = nameInput.value.trim();
    const price = parsePrice(priceInput.value);
    const isEditing = editingId !== null;

    setSubmitBusy(true);
    try {
      if (isEditing) {
        await updateMenu(editingId, { name, price });
        Toast.success(`"${name}" updated successfully.`);
      } else {
        await createMenu({ name, price });
        Toast.success(`"${name}" added to the menu.`);
      }
      resetForm();
      await loadMenu(); // refresh table automatically after save
    } catch (error) {
      Toast.error(error.message);
    } finally {
      setSubmitBusy(false);
    }
  }

  // ----- Edit / Delete -----
  function handleTableClick(event) {
    const target = event.target.closest("button[data-action]");
    if (!target) return;

    const id = Number(target.dataset.id);
    const action = target.dataset.action;

    if (action === "edit") {
      const item = currentItems.find((i) => i.id === id);
      if (item) enterEditMode(item);
    } else if (action === "delete") {
      handleDelete(id);
    }
  }

  async function handleDelete(id) {
    const item = currentItems.find((i) => i.id === id);
    if (!item) return;

    const confirmed = await ConfirmModal.open({
      title: "Delete menu item?",
      message: `"${item.name}" will be permanently removed from the menu.`,
      confirmLabel: "Delete",
    });
    if (!confirmed) return;

    try {
      await deleteMenu(id);
      Toast.success(`"${item.name}" deleted.`);
      if (editingId === id) resetForm();
      await loadMenu(); // refresh table automatically after delete
    } catch (error) {
      Toast.error(error.message);
    }
  }

  // Handlers app.js registers as the actual event listeners.
  return {
    loadMenu,
    handleFormSubmit,
    handleTableClick,
    resetForm,
    clearNameError: () => clearFieldError(nameInput, nameError),
    clearPriceError: () => clearFieldError(priceInput, priceError),
  };
}
