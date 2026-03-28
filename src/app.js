import {
  requestNanoBananaGeneration,
  requestRoomAnalysis,
} from "./services/nanoBanana.js";

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
  roomAnalysis: null,
  isAnalyzingRoom: false,
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

      addFurniture(entry.id, 50, 50);
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

async function handleRoomUpload(event) {
  const [file] = event.target.files ?? [];
  if (!file) {
    return;
  }

  state.isAnalyzingRoom = true;
  state.roomImageDataUrl = "";
  state.roomAnalysis = null;
  state.items = [];
  state.selectedId = null;
  render();
  setLog(`Analyzing ${file.name} to build a room map...`);

  try {
    const roomImageDataUrl = await readFileAsDataUrl(file);
    const analysis = await requestRoomAnalysis({ roomImageDataUrl });

    state.roomImageDataUrl = roomImageDataUrl;
    state.roomAnalysis = analysis;
    setLog(formatRoomAnalysisLog(file.name, analysis));
  } catch (error) {
    state.roomImageDataUrl = "";
    state.roomAnalysis = null;
    event.target.value = "";
    setLog(`Image analysis failed: ${error.message}`);
  } finally {
    state.isAnalyzingRoom = false;
    render();
  }
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

  const boardRect = els.roomBoard.getBoundingClientRect();
  const x = ((event.clientX - boardRect.left) / boardRect.width) * 100;
  const y = ((event.clientY - boardRect.top) / boardRect.height) * 100;
  addFurniture(furnitureId, x, y);
}

function addFurniture(furnitureId, x, y) {
  const definition = furnitureCatalog.find((item) => item.id === furnitureId);
  if (!definition) {
    return;
  }

  const nextItem = {
    instanceId: crypto.randomUUID(),
    furnitureId,
    name: definition.name,
    imageUrl: definition.imageUrl,
    productUrl: definition.productUrl,
    x,
    y,
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

  const boardRect = els.roomBoard.getBoundingClientRect();

  const move = (moveEvent) => {
    const x = clampPercent(((moveEvent.clientX - boardRect.left) / boardRect.width) * 100);
    const y = clampPercent(((moveEvent.clientY - boardRect.top) / boardRect.height) * 100);

    state.items = state.items.map((item) =>
      item.instanceId === state.selectedId ? { ...item, x, y } : item,
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
  state.roomAnalysis = null;
  state.isAnalyzingRoom = false;
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

  if (!state.roomAnalysis) {
    setLog("Generation skipped: room mapping has not finished yet.");
    return;
  }

  const payload = {
    roomImageDataUrl: state.roomImageDataUrl,
    roomAnalysis: state.roomAnalysis,
    prompt: els.promptInput.value.trim(),
    furniture: state.items.map(({ name, imageUrl, productUrl, x, y, scale, rotation }) => ({
      name,
      imageUrl,
      productUrl,
      x,
      y,
      scale,
      rotation,
    })),
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
  els.dropHint.textContent = state.isAnalyzingRoom
    ? "Analyzing room image and building a placement map..."
    : "Upload a room image to start staging the scene.";
  els.generateScene.disabled = state.isAnalyzingRoom;
  els.toggleGrid.disabled = state.isAnalyzingRoom || !ready;
  els.roomImage.src = state.roomImageDataUrl;
  renderPlacedItems();
  renderSelectionControls();
}

function renderPlacedItems() {
  els.furnitureLayer.innerHTML = "";

  state.items.forEach((item) => {
    const fragment = els.placedTemplate.content.cloneNode(true);
    const button = fragment.querySelector(".placed-item");

    button.dataset.instanceId = item.instanceId;
    button.style.left = `${item.x}%`;
    button.style.top = `${item.y}%`;
    button.style.transform = `translate(-50%, -50%) scale(${item.scale / 100}) rotate(${item.rotation}deg) translateY(${item.elevation}px)`;
    button.classList.toggle("is-selected", item.instanceId === state.selectedId);
    const image = fragment.querySelector(".placed-item__image");
    image.src = item.imageUrl;
    image.alt = item.name;

    els.furnitureLayer.appendChild(fragment);
  });
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

function clampPercent(value) {
  return Math.min(96, Math.max(4, value));
}

function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error ?? new Error("Could not read the selected file."));
    reader.readAsDataURL(file);
  });
}

function formatRoomAnalysisLog(fileName, analysis) {
  const summary = analysis.summary || "No summary returned.";
  const floor = analysis.floorPolygon?.length
    ? `${analysis.floorPolygon.length} floor anchor points`
    : "no floor polygon";
  const guidance = analysis.placementGuidance?.join(", ") || "no placement guidance";

  return [
    `Mapped ${fileName} before upload staging.`,
    `Summary: ${summary}`,
    `Detected: ${floor}.`,
    `Placement guidance: ${guidance}.`,
  ].join("\n");
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
