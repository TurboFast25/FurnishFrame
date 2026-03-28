import { requestNanoBananaGeneration } from "./services/nanoBanana.js";

const GRID_COLUMNS = 24;
const GRID_ROWS = 24;

const furnitureCatalog = [
  createCatalogItem({
    id: "sofa",
    name: "Modular Sofa",
    color: "#b97a56",
    silhouette: "sofa",
  }),
  createCatalogItem({
    id: "chair",
    name: "Lounge Chair",
    color: "#8b6c8f",
    silhouette: "chair",
  }),
  createCatalogItem({
    id: "lamp",
    name: "Arc Lamp",
    color: "#d0a14b",
    silhouette: "lamp",
  }),
  createCatalogItem({
    id: "table",
    name: "Coffee Table",
    color: "#6f8e7c",
    silhouette: "table",
  }),
  createCatalogItem({
    id: "plant",
    name: "Floor Plant",
    color: "#5a8f62",
    silhouette: "plant",
  }),
  createCatalogItem({
    id: "shelf",
    name: "Open Shelf",
    color: "#857159",
    silhouette: "shelf",
  }),
];

const state = {
  items: [],
  selectedId: null,
  roomImageDataUrl: "",
  showGrid: false,
};

const els = {
  roomUpload: document.querySelector("#room-upload"),
  roomBoard: document.querySelector("#room-board"),
  roomImage: document.querySelector("#room-image"),
  dropHint: document.querySelector("#drop-hint"),
  furnitureLayer: document.querySelector("#furniture-layer"),
  furnitureLibrary: document.querySelector("#furniture-library"),
  libraryTemplate: document.querySelector("#furniture-card-template"),
  placedTemplate: document.querySelector("#placed-item-template"),
  scaleControl: document.querySelector("#scale-control"),
  rotationControl: document.querySelector("#rotation-control"),
  depthControl: document.querySelector("#depth-control"),
  selectionStatus: document.querySelector("#selection-status"),
  removeItem: document.querySelector("#remove-item"),
  promptInput: document.querySelector("#prompt-input"),
  generationLog: document.querySelector("#generation-log"),
  generateScene: document.querySelector("#generate-scene"),
  resetRoom: document.querySelector("#reset-room"),
  toggleGrid: document.querySelector("#toggle-grid"),
  generatedImage: document.querySelector("#generated-image"),
  resultPlaceholder: document.querySelector("#result-placeholder"),
  resultStatus: document.querySelector("#result-status"),
  customItemName: document.querySelector("#custom-item-name"),
  customItemImage: document.querySelector("#custom-item-image"),
  customItemLink: document.querySelector("#custom-item-link"),
  addCustomItem: document.querySelector("#add-custom-item"),
  selectionLink: document.querySelector("#selection-link"),
};

renderFurnitureLibrary();
attachEvents();
render();

function attachEvents() {
  els.roomUpload.addEventListener("change", handleRoomUpload);
  els.roomImage.addEventListener("load", render);
  window.addEventListener("resize", renderPlacedItems);
  els.furnitureLayer.addEventListener("pointerdown", beginExistingItemDrag);
  els.furnitureLayer.addEventListener("click", handlePlacedItemClick);
  els.roomBoard.addEventListener("dragover", handleBoardDragOver);
  els.roomBoard.addEventListener("drop", handleBoardDrop);
  els.scaleControl.addEventListener("input", updateSelectedItemFromControls);
  els.rotationControl.addEventListener("input", updateSelectedItemFromControls);
  els.depthControl.addEventListener("input", updateSelectedItemFromControls);
  els.removeItem.addEventListener("click", removeSelectedItem);
  els.generateScene.addEventListener("click", generateScene);
  els.resetRoom.addEventListener("click", resetRoom);
  els.toggleGrid.addEventListener("click", toggleGrid);
  els.addCustomItem.addEventListener("click", addCustomItemToLibrary);
  document.addEventListener("keydown", handleKeydown);
}

function renderFurnitureLibrary() {
  els.furnitureLibrary.innerHTML = "";

  furnitureCatalog.forEach((entry) => {
    const fragment = els.libraryTemplate.content.cloneNode(true);
    const button = fragment.querySelector(".furniture-card");

    button.dataset.furnitureId = entry.id;
    const image = fragment.querySelector(".furniture-image");
    image.src = entry.imageUrl;
    image.alt = entry.name;
    fragment.querySelector(".furniture-name").textContent = entry.name;
    fragment.querySelector(".furniture-link-copy").textContent = entry.productUrl
      ? "Linked product"
      : "No product link";

    button.addEventListener("dragstart", (event) => {
      event.dataTransfer.setData("text/plain", entry.id);
    });

    button.addEventListener("click", () => {
      if (!state.roomImageDataUrl) {
        setLog("Upload a room image before placing furniture.");
        return;
      }

      addFurniture(entry.id, Math.floor(GRID_COLUMNS / 2), Math.floor(GRID_ROWS / 2));
    });

    els.furnitureLibrary.appendChild(fragment);
  });
}

function addCustomItemToLibrary() {
  const name = els.customItemName.value.trim();
  const imageUrl = els.customItemImage.value.trim();
  const productUrl = els.customItemLink.value.trim();

  if (!name || !imageUrl) {
    setLog("Custom items require both a name and an image URL.");
    return;
  }

  furnitureCatalog.unshift({
    id: `custom-${crypto.randomUUID()}`,
    name,
    imageUrl,
    productUrl,
  });

  els.customItemName.value = "";
  els.customItemImage.value = "";
  els.customItemLink.value = "";
  renderFurnitureLibrary();
  setLog(`Added "${name}" to the furniture library.`);
}

function handleRoomUpload(event) {
  const [file] = event.target.files ?? [];
  if (!file) {
    return;
  }

  const reader = new FileReader();
  reader.onload = () => {
    state.roomImageDataUrl = String(reader.result);
    state.items = [];
    state.selectedId = null;
    render();
    setLog(`Loaded room image: ${file.name}`);
  };
  reader.readAsDataURL(file);
}

function handleBoardDragOver(event) {
  event.preventDefault();
}

function handleBoardDrop(event) {
  event.preventDefault();
  if (!state.roomImageDataUrl) {
    return;
  }

  const furnitureId = event.dataTransfer.getData("text/plain");
  if (!furnitureId) {
    return;
  }

  const { column, row } = getGridPositionFromPointer(event);
  addFurniture(furnitureId, column, row);
}

function addFurniture(furnitureId, column, row) {
  const definition = furnitureCatalog.find((item) => item.id === furnitureId);
  if (!definition) {
    return;
  }

  const safeColumn = clampGridIndex(column, GRID_COLUMNS);
  const safeRow = clampGridIndex(row, GRID_ROWS);

  const nextItem = {
    instanceId: crypto.randomUUID(),
    furnitureId,
    name: definition.name,
    imageUrl: definition.imageUrl,
    productUrl: definition.productUrl,
    gridColumn: safeColumn,
    gridRow: safeRow,
    scale: 100,
    rotation: 0,
    elevation: 0,
  };

  state.items = [...state.items, nextItem];
  state.selectedId = nextItem.instanceId;
  render();
}

function beginExistingItemDrag(event) {
  const button = event.target.closest(".placed-item");
  if (!button) {
    return;
  }

  event.preventDefault();
  state.selectedId = button.dataset.instanceId;
  renderSelectionControls();

  const move = (moveEvent) => {
    const { column, row } = getGridPositionFromPointer(moveEvent);

    state.items = state.items.map((item) =>
      item.instanceId === state.selectedId
        ? { ...item, gridColumn: column, gridRow: row }
        : item,
    );

    renderPlacedItems();
  };

  const stop = () => {
    window.removeEventListener("pointermove", move);
    window.removeEventListener("pointerup", stop);
  };

  window.addEventListener("pointermove", move);
  window.addEventListener("pointerup", stop);
}

function handlePlacedItemClick(event) {
  const button = event.target.closest(".placed-item");
  if (!button) {
    state.selectedId = null;
    renderSelectionControls();
    renderPlacedItems();
    return;
  }

  state.selectedId = button.dataset.instanceId;
  renderSelectionControls();
  renderPlacedItems();
}

function updateSelectedItemFromControls() {
  const selected = getSelectedItem();
  if (!selected) {
    return;
  }

  state.items = state.items.map((item) =>
    item.instanceId === selected.instanceId
      ? {
          ...item,
          scale: Number(els.scaleControl.value),
          rotation: Number(els.rotationControl.value),
          elevation: Number(els.depthControl.value),
        }
      : item,
  );

  renderPlacedItems();
}

function removeSelectedItem() {
  if (!state.selectedId) {
    return;
  }

  state.items = state.items.filter((item) => item.instanceId !== state.selectedId);
  state.selectedId = null;
  render();
}

function toggleGrid() {
  state.showGrid = !state.showGrid;
  els.roomBoard.classList.toggle("grid-hidden", !state.showGrid);
  els.toggleGrid.textContent = state.showGrid ? "Grid Off" : "Grid On";
}

function resetRoom() {
  state.items = [];
  state.selectedId = null;
  state.roomImageDataUrl = "";
  els.roomUpload.value = "";
  render();
  setLog("Room reset. Upload a new image to start again.");
}

function handleKeydown(event) {
  if (shouldIgnoreGlobalShortcut(event)) {
    return;
  }

  if (event.key === "Delete") {
    removeSelectedItem();
  }
}

function shouldIgnoreGlobalShortcut(event) {
  const target = event.target;

  if (!(target instanceof HTMLElement)) {
    return false;
  }

  return (
    target.isContentEditable ||
    target.tagName === "INPUT" ||
    target.tagName === "TEXTAREA" ||
    target.tagName === "SELECT"
  );
}

async function generateScene() {
  if (!state.roomImageDataUrl) {
    setLog("Generation skipped: upload a room image first.");
    return;
  }

  const payload = {
    roomImageDataUrl: state.roomImageDataUrl,
    prompt: els.promptInput.value.trim(),
    furniture: state.items.map(
      ({ name, imageUrl, productUrl, gridColumn, gridRow, scale, rotation }) => {
        const { x, y } = getPercentPositionFromGrid(gridColumn, gridRow);

        return {
          name,
          imageUrl,
          productUrl,
          x,
          y,
          scale,
          rotation,
        };
      },
    ),
  };

  setLog("Submitting staged room payload to Nano Banana service...");
  setResultState({ status: "Generating...", imageUrl: "" });

  try {
    const result = await requestNanoBananaGeneration(payload);
    setLog(JSON.stringify(result.meta, null, 2));
    setResultState({
      status: result.imageDataUrl ? "Image ready" : "No image returned",
      imageUrl: result.imageDataUrl,
    });
  } catch (error) {
    setLog(`Generation failed: ${error.message}`);
    setResultState({ status: "Generation failed", imageUrl: "" });
  }
}

function render() {
  const ready = Boolean(state.roomImageDataUrl);
  els.roomBoard.classList.toggle("is-ready", ready);
  els.dropHint.hidden = ready;
  els.roomImage.src = state.roomImageDataUrl;
  syncBoardToImage();
  renderPlacedItems();
  renderSelectionControls();
}

function renderPlacedItems() {
  syncBoardToImage();
  els.furnitureLayer.innerHTML = "";
  const imageRect = getDisplayedImageRect();

  state.items.forEach((item) => {
    const fragment = els.placedTemplate.content.cloneNode(true);
    const button = fragment.querySelector(".placed-item");
    const { x, y } = getPercentPositionFromGrid(item.gridColumn, item.gridRow);
    const placement = getArPlacementStyle(item, imageRect);

    button.dataset.instanceId = item.instanceId;
    button.style.left = `${x}%`;
    button.style.top = `${y}%`;
    button.style.width = `${placement.width}px`;
    button.style.height = `${placement.height}px`;
    button.style.setProperty("--surface-tilt", `${placement.tilt}deg`);
    button.style.setProperty("--surface-lift", `${placement.lift}px`);
    button.style.setProperty("--surface-sheen", placement.sheen);
    button.style.setProperty("--shadow-width", `${placement.shadowWidth}px`);
    button.style.setProperty("--shadow-height", `${placement.shadowHeight}px`);
    button.style.setProperty("--shadow-blur", `${placement.shadowBlur}px`);
    button.style.setProperty("--shadow-opacity", placement.shadowOpacity);
    button.style.setProperty("--shadow-offset", `${placement.shadowOffset}px`);
    button.style.transform =
      `translate(-50%, -50%) translateY(${item.elevation}px) rotate(${item.rotation}deg)`;
    button.classList.toggle("is-selected", item.instanceId === state.selectedId);
    const image = fragment.querySelector(".placed-item__image");
    image.src = item.imageUrl;
    image.alt = item.name;

    els.furnitureLayer.appendChild(fragment);
  });
}

function getArPlacementStyle(item, imageRect) {
  const depth = (item.gridRow + 1) / GRID_ROWS;
  const scaledWidth = imageRect.width * (0.06 + depth * 0.08) * (item.scale / 100);
  const width = clampValue(scaledWidth, 56, imageRect.width * 0.26);
  const height = width * (0.62 + depth * 0.12);

  return {
    width,
    height,
    tilt: 58 - depth * 28,
    lift: 10 + depth * 14,
    sheen: String(0.12 + depth * 0.16),
    shadowWidth: width * (0.72 + depth * 0.24),
    shadowHeight: 12 + depth * 14,
    shadowBlur: 16 + depth * 18,
    shadowOpacity: String(0.18 + depth * 0.24),
    shadowOffset: 8 + depth * 12,
  };
}

function renderSelectionControls() {
  const selected = getSelectedItem();
  if (!selected) {
    els.selectionStatus.textContent = "Nothing selected";
    els.scaleControl.value = "100";
    els.rotationControl.value = "0";
    els.depthControl.value = "0";
    els.selectionLink.hidden = true;
    els.selectionLink.removeAttribute("href");
    return;
  }

  els.selectionStatus.textContent = selected.name;
  els.scaleControl.value = String(selected.scale);
  els.rotationControl.value = String(selected.rotation);
  els.depthControl.value = String(selected.elevation);
  if (selected.productUrl) {
    els.selectionLink.href = selected.productUrl;
    els.selectionLink.hidden = false;
  } else {
    els.selectionLink.hidden = true;
    els.selectionLink.removeAttribute("href");
  }
}

function getSelectedItem() {
  return state.items.find((item) => item.instanceId === state.selectedId) ?? null;
}

function getGridPositionFromPointer(event) {
  const imageRect = getDisplayedImageRect();
  const relativeX = clamp01((event.clientX - imageRect.left) / imageRect.width);
  const relativeY = clamp01((event.clientY - imageRect.top) / imageRect.height);

  return {
    column: Math.min(GRID_COLUMNS - 1, Math.floor(relativeX * GRID_COLUMNS)),
    row: Math.min(GRID_ROWS - 1, Math.floor(relativeY * GRID_ROWS)),
  };
}

function getPercentPositionFromGrid(column, row) {
  return {
    x: ((column + 0.5) / GRID_COLUMNS) * 100,
    y: ((row + 0.5) / GRID_ROWS) * 100,
  };
}

function getDisplayedImageRect() {
  const boardRect = els.roomBoard.getBoundingClientRect();
  const naturalWidth = els.roomImage.naturalWidth || boardRect.width || 1;
  const naturalHeight = els.roomImage.naturalHeight || boardRect.height || 1;
  const scale = Math.min(boardRect.width / naturalWidth, boardRect.height / naturalHeight);
  const width = naturalWidth * scale;
  const height = naturalHeight * scale;
  const left = boardRect.left + (boardRect.width - width) / 2;
  const top = boardRect.top + (boardRect.height - height) / 2;

  return {
    left,
    top,
    width,
    height,
    leftPercent: (left - boardRect.left) / boardRect.width * 100,
    topPercent: (top - boardRect.top) / boardRect.height * 100,
    widthPercent: width / boardRect.width * 100,
    heightPercent: height / boardRect.height * 100,
  };
}

function syncBoardToImage() {
  const ready = Boolean(state.roomImageDataUrl);
  const imageRect = ready
    ? getDisplayedImageRect()
    : { leftPercent: 0, topPercent: 0, widthPercent: 100, heightPercent: 100 };

  els.roomBoard.style.setProperty("--image-left", `${imageRect.leftPercent}%`);
  els.roomBoard.style.setProperty("--image-top", `${imageRect.topPercent}%`);
  els.roomBoard.style.setProperty("--image-width", `${imageRect.widthPercent}%`);
  els.roomBoard.style.setProperty("--image-height", `${imageRect.heightPercent}%`);
  els.roomBoard.style.setProperty("--grid-columns", String(GRID_COLUMNS));
  els.roomBoard.style.setProperty("--grid-rows", String(GRID_ROWS));
}

function clamp01(value) {
  return Math.min(0.999999, Math.max(0, value));
}

function clampGridIndex(value, size) {
  return Math.min(size - 1, Math.max(0, Math.round(value)));
}

function clampValue(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function setLog(message) {
  els.generationLog.textContent = message;
}

function setResultState({ status, imageUrl }) {
  els.resultStatus.textContent = status;
  if (!imageUrl) {
    els.generatedImage.hidden = true;
    els.generatedImage.removeAttribute("src");
    els.resultPlaceholder.hidden = false;
    return;
  }

  els.generatedImage.src = imageUrl;
  els.generatedImage.hidden = false;
  els.resultPlaceholder.hidden = true;
}

function createCatalogItem({ id, name, color, silhouette }) {
  return {
    id,
    name,
    imageUrl: createFurnitureSwatch({ color, silhouette }),
    productUrl: "",
  };
}

function createFurnitureSwatch({ color, silhouette }) {
  const svg = {
    sofa: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120"><rect width="120" height="120" rx="28" fill="#f6efe4"/><rect x="22" y="52" width="76" height="28" rx="10" fill="${color}"/><rect x="18" y="44" width="24" height="36" rx="10" fill="${color}"/><rect x="78" y="44" width="24" height="36" rx="10" fill="${color}"/><rect x="28" y="36" width="64" height="18" rx="9" fill="#fff" fill-opacity=".38"/><rect x="26" y="80" width="6" height="12" rx="3" fill="#694f3d"/><rect x="88" y="80" width="6" height="12" rx="3" fill="#694f3d"/></svg>`,
    chair: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120"><rect width="120" height="120" rx="28" fill="#f6efe4"/><rect x="36" y="32" width="48" height="32" rx="14" fill="${color}"/><rect x="30" y="58" width="60" height="18" rx="9" fill="${color}"/><rect x="36" y="76" width="7" height="18" rx="3" fill="#694f3d"/><rect x="77" y="76" width="7" height="18" rx="3" fill="#694f3d"/></svg>`,
    lamp: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120"><rect width="120" height="120" rx="28" fill="#f6efe4"/><path d="M42 38h36L66 60H54z" fill="${color}"/><rect x="57" y="60" width="6" height="26" rx="3" fill="#694f3d"/><rect x="42" y="86" width="36" height="8" rx="4" fill="#694f3d"/></svg>`,
    table: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120"><rect width="120" height="120" rx="28" fill="#f6efe4"/><rect x="24" y="38" width="72" height="16" rx="8" fill="${color}"/><rect x="34" y="54" width="6" height="34" rx="3" fill="#694f3d"/><rect x="80" y="54" width="6" height="34" rx="3" fill="#694f3d"/></svg>`,
    plant: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120"><rect width="120" height="120" rx="28" fill="#f6efe4"/><path d="M60 26c14 8 16 24 7 36-11-4-17-18-7-36zM40 40c12 3 18 16 14 29-11 0-20-11-14-29zM80 40c6 18-3 29-14 29-4-13 2-26 14-29z" fill="${color}"/><path d="M44 72h32l-6 20H50z" fill="#9a6a44"/></svg>`,
    shelf: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120"><rect width="120" height="120" rx="28" fill="#f6efe4"/><rect x="30" y="24" width="8" height="72" rx="4" fill="${color}"/><rect x="82" y="24" width="8" height="72" rx="4" fill="${color}"/><rect x="30" y="30" width="60" height="8" rx="4" fill="${color}"/><rect x="30" y="54" width="60" height="8" rx="4" fill="${color}"/><rect x="30" y="78" width="60" height="8" rx="4" fill="${color}"/></svg>`,
  }[silhouette];

  return `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(svg)}`;
}
