import {
  requestRoomAnalysis,
  requestNanoBananaGeneration,
  requestSimilarProducts,
} from "./services/nanoBanana.js";

const GRID_COLUMNS = 24;
const GRID_ROWS = 24;
const THEME_STORAGE_KEY = "furnishframe-theme";

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
  theme: "light",
  items: [],
  selectedId: null,
  roomImageDataUrl: "",
  roomAnalysis: null,
  isAnalyzingRoom: false,
  showGrid: false,
  viewer: {
    offsetX: 0,
    offsetY: 0,
    zoom: 1,
    dragging: false,
    pointerId: null,
    lastX: 0,
    lastY: 0,
  },
  generatedTour: [],
  activeViewId: "",
  similarProductMatches: [],
  isSearchingProducts: false,
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
  searchSimilar: document.querySelector("#search-similar"),
  resetRoom: document.querySelector("#reset-room"),
  toggleGrid: document.querySelector("#toggle-grid"),
  toggleTheme: document.querySelector("#toggle-theme"),
  generatedImage: document.querySelector("#generated-image"),
  resultNav: document.querySelector("#result-nav"),
  resultFrame: document.querySelector("#result-frame"),
  resultViewer: document.querySelector("#result-viewer"),
  resultViewerHint: document.querySelector("#result-viewer-hint"),
  resultPlaceholder: document.querySelector("#result-placeholder"),
  similarProducts: document.querySelector("#similar-products"),
  resultStatus: document.querySelector("#result-status"),
  customItemName: document.querySelector("#custom-item-name"),
  customItemImage: document.querySelector("#custom-item-image"),
  customItemLink: document.querySelector("#custom-item-link"),
  wallColorInput: document.querySelector("#wall-color-input"),
  wallMaterialSelect: document.querySelector("#wall-material-select"),
  floorColorInput: document.querySelector("#floor-color-input"),
  floorMaterialSelect: document.querySelector("#floor-material-select"),
  addCustomItem: document.querySelector("#add-custom-item"),
  selectionLink: document.querySelector("#selection-link"),
};

renderFurnitureLibrary();
hydrateTheme();
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
  els.searchSimilar.addEventListener("click", searchSimilarProducts);
  els.resetRoom.addEventListener("click", resetRoom);
  els.toggleGrid.addEventListener("click", toggleGrid);
  els.toggleTheme.addEventListener("click", toggleTheme);
  els.addCustomItem.addEventListener("click", addCustomItemToLibrary);
  els.resultViewer.addEventListener("pointerdown", beginResultViewerDrag);
  els.resultViewer.addEventListener("wheel", handleResultViewerWheel, { passive: false });
  els.generatedImage.addEventListener("load", handleGeneratedImageLoad);
  document.addEventListener("keydown", handleKeydown);
}

function hydrateTheme() {
  const storedTheme = window.localStorage.getItem(THEME_STORAGE_KEY);
  const preferredTheme = storedTheme || "dark";
  applyTheme(preferredTheme);
}

function toggleTheme() {
  applyTheme(state.theme === "dark" ? "light" : "dark");
}

function applyTheme(theme) {
  state.theme = theme === "dark" ? "dark" : "light";
  document.body.dataset.theme = state.theme;
  els.toggleTheme.textContent = state.theme === "dark" ? "Light Mode" : "Dark Mode";
  window.localStorage.setItem(THEME_STORAGE_KEY, state.theme);
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
    color: "#8d9aa6",
    silhouette: "custom",
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

  const { x, y } = getPercentPositionFromPointer(event);
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
    color: definition.color,
    silhouette: definition.silhouette,
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

  const move = (moveEvent) => {
    const { x, y } = getPercentPositionFromPointer(moveEvent);

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
  state.generatedTour = [];
  state.activeViewId = "";
  state.similarProductMatches = [];
  state.isSearchingProducts = false;
  els.roomUpload.value = "";
  render();
  setResultState({ status: "No output yet", views: [], activeViewId: "" });
  renderSimilarProducts();
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
    roomFinishes: getRoomFinishes(),
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

  setLog("Submitting staged room payload to Nano Banana image service...");
  state.similarProductMatches = [];
  state.isSearchingProducts = false;
  renderSimilarProducts();
  setResultState({ status: "Generating...", views: [], activeViewId: "" });

  try {
    const result = await requestNanoBananaGeneration(payload);
    setLog(JSON.stringify(result.meta, null, 2));
    setResultState({
      status: result.imageDataUrl ? "Result ready" : "No image returned",
      views: result.imageDataUrl
        ? [
            {
              id: "result",
              label: "Result",
              hint: "Generated room image",
              imageDataUrl: result.imageDataUrl,
            },
          ]
        : [],
      activeViewId: "result",
    });
  } catch (error) {
    setLog(`Generation failed: ${error.message}`);
    setResultState({ status: "Generation failed", views: [], activeViewId: "" });
  }
}

async function searchSimilarProducts() {
  const activeView = getActiveGeneratedView();
  if (!activeView?.imageDataUrl) {
    setLog("Similar product search skipped: generate a room image first.");
    return;
  }

  if (!state.items.length) {
    setLog("Similar product search skipped: add at least one staged item first.");
    return;
  }

  state.isSearchingProducts = true;
  updateSimilarSearchButton();
  setLog("Searching the web for similar product matches...");

  try {
    const result = await requestSimilarProducts({
      generatedImageDataUrl: activeView.imageDataUrl,
      prompt: els.promptInput.value.trim(),
      roomAnalysis: state.roomAnalysis,
      items: state.items.map(({ name, productUrl }) => ({ name, productUrl })),
    });
    state.similarProductMatches = result.searches || [];
    renderSimilarProducts();
    setLog("Similar product search complete.");
  } catch (error) {
    setLog(`Similar product search failed: ${error.message}`);
  } finally {
    state.isSearchingProducts = false;
    updateSimilarSearchButton();
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
  updateSimilarSearchButton();
  syncBoardToImage();
  renderPlacedItems();
  renderSelectionControls();
}

function renderPlacedItems() {
  syncBoardToImage();
  els.furnitureLayer.innerHTML = "";

  state.items.forEach((item) => {
    const fragment = els.placedTemplate.content.cloneNode(true);
    const button = fragment.querySelector(".placed-item");
    const sprite = fragment.querySelector(".placed-item__sprite");
    const metrics = getSpriteMetrics(item);

    button.dataset.instanceId = item.instanceId;
    button.style.left = `${item.x}%`;
    button.style.top = `${item.y}%`;
    button.style.setProperty("--model-width", `${metrics.width}px`);
    button.style.setProperty("--model-height", `${metrics.height}px`);
    button.style.transform =
      `translate(-50%, calc(-100% + ${item.elevation}px)) rotate(${item.rotation}deg)`;
    button.classList.toggle("is-selected", item.instanceId === state.selectedId);
    button.dataset.silhouette = item.silhouette || "custom";
    sprite.src = item.imageUrl;
    sprite.alt = item.name;

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

function getRoomFinishes() {
  return {
    wallColor: els.wallColorInput.value.trim(),
    wallMaterial: els.wallMaterialSelect.value.trim(),
    floorColor: els.floorColorInput.value.trim(),
    floorMaterial: els.floorMaterialSelect.value.trim(),
  };
}

function clampPercent(value) {
  return Math.min(96, Math.max(4, value));
}

function getPercentPositionFromPointer(event) {
  const imageRect = getDisplayedImageRect();

  return {
    x: clampPercent(((event.clientX - imageRect.left) / imageRect.width) * 100),
    y: clampPercent(((event.clientY - imageRect.top) / imageRect.height) * 100),
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
    leftPercent: ((left - boardRect.left) / boardRect.width) * 100,
    topPercent: ((top - boardRect.top) / boardRect.height) * 100,
    widthPercent: (width / boardRect.width) * 100,
    heightPercent: (height / boardRect.height) * 100,
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

function getSpriteMetrics(item) {
  const base = {
    sofa: { width: 132, height: 70, depth: 48 },
    chair: { width: 84, height: 96, depth: 42 },
    lamp: { width: 56, height: 148, depth: 28 },
    table: { width: 110, height: 58, depth: 64 },
    plant: { width: 72, height: 118, depth: 40 },
    shelf: { width: 86, height: 136, depth: 36 },
    custom: { width: 104, height: 88, depth: 44 },
  }[item.silhouette || "custom"];

  const scale = item.scale / 100;

  return {
    width: Math.round(base.width * scale),
    height: Math.round(base.height * scale),
  };
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

function setResultState({ status, views, activeViewId }) {
  els.resultStatus.textContent = status;
  state.generatedTour = views || [];
  state.activeViewId = activeViewId || state.generatedTour[0]?.id || "";
  renderResultNav();
  updateSimilarSearchButton();

  const activeView = getActiveGeneratedView();
  if (!activeView?.imageDataUrl) {
    resetResultViewer();
    els.resultFrame.style.removeProperty("--result-aspect-ratio");
    els.resultNav.hidden = true;
    els.resultViewer.hidden = true;
    els.resultViewerHint.hidden = true;
    els.generatedImage.hidden = true;
    els.generatedImage.removeAttribute("src");
    els.resultPlaceholder.hidden = false;
    return;
  }

  resetResultViewer();
  els.resultNav.hidden = state.generatedTour.length <= 1;
  els.resultViewer.hidden = false;
  els.generatedImage.src = activeView.imageDataUrl;
  els.generatedImage.hidden = false;
  els.resultViewerHint.hidden = false;
  els.resultPlaceholder.hidden = true;
}

function handleGeneratedImageLoad() {
  syncResultFrameAspectRatio();
  resetResultViewer();
}

function renderResultNav() {
  els.resultNav.innerHTML = "";

  state.generatedTour.forEach((view) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "result-nav__button";
    button.textContent = view.label;
    button.title = view.hint || view.label;
    button.classList.toggle("is-active", view.id === state.activeViewId);
    button.addEventListener("click", () => {
      state.activeViewId = view.id;
      setResultState({
        status: "Result ready",
        views: state.generatedTour,
        activeViewId: view.id,
      });
    });
    els.resultNav.appendChild(button);
  });
}

function getActiveGeneratedView() {
  return state.generatedTour.find((view) => view.id === state.activeViewId) ?? null;
}

function updateSimilarSearchButton() {
  const canSearch = Boolean(getActiveGeneratedView()?.imageDataUrl) && state.items.length > 0;
  els.searchSimilar.disabled = !canSearch || state.isSearchingProducts;
  els.searchSimilar.textContent = state.isSearchingProducts
    ? "Searching Products..."
    : "Find Similar Products";
}

function renderSimilarProducts() {
  els.similarProducts.innerHTML = "";

  if (!state.similarProductMatches.length) {
    els.similarProducts.hidden = true;
    return;
  }

  state.similarProductMatches.forEach((match) => {
    const section = document.createElement("section");
    section.className = "similar-products__group";

    const heading = document.createElement("div");
    heading.className = "similar-products__heading";

    const title = document.createElement("h3");
    title.textContent = match.itemName;
    heading.appendChild(title);

    const query = document.createElement("p");
    query.className = "similar-products__query";
    query.textContent = `Search: ${match.query}`;
    heading.appendChild(query);

    if (match.traits?.length) {
      const traits = document.createElement("p");
      traits.className = "similar-products__traits";
      traits.textContent = `Detected: ${match.traits.join(", ")}`;
      heading.appendChild(traits);
    }

    section.appendChild(heading);

    if (match.sourceProductUrl) {
      const original = document.createElement("a");
      original.className = "similar-products__original";
      original.href = match.sourceProductUrl;
      original.target = "_blank";
      original.rel = "noreferrer";
      original.textContent = "Open linked product";
      section.appendChild(original);
    }

    const list = document.createElement("div");
    list.className = "similar-products__list";

    if (!match.results?.length) {
      const empty = document.createElement("p");
      empty.className = "similar-products__empty";
      empty.textContent = "No similar products found for this item.";
      list.appendChild(empty);
    } else {
      match.results.forEach((result) => {
        const link = document.createElement("a");
        link.className = "similar-products__card";
        link.href = result.url;
        link.target = "_blank";
        link.rel = "noreferrer";

        const resultTitle = document.createElement("strong");
        resultTitle.textContent = result.title;
        link.appendChild(resultTitle);

        const meta = document.createElement("span");
        meta.textContent = result.displayUrl || result.url;
        link.appendChild(meta);

        list.appendChild(link);
      });
    }

    section.appendChild(list);
    els.similarProducts.appendChild(section);
  });

  els.similarProducts.hidden = false;
}

function beginResultViewerDrag(event) {
  if (event.button !== 0 || els.generatedImage.hidden) {
    return;
  }

  event.preventDefault();
  state.viewer.dragging = true;
  state.viewer.pointerId = event.pointerId;
  state.viewer.lastX = event.clientX;
  state.viewer.lastY = event.clientY;
  els.resultViewer.setPointerCapture(event.pointerId);
  els.resultViewer.addEventListener("pointermove", handleResultViewerDrag);
  els.resultViewer.addEventListener("pointerup", endResultViewerDrag);
  els.resultViewer.addEventListener("pointercancel", endResultViewerDrag);
}

function handleResultViewerDrag(event) {
  if (!state.viewer.dragging || event.pointerId !== state.viewer.pointerId) {
    return;
  }

  const deltaX = event.clientX - state.viewer.lastX;
  const deltaY = event.clientY - state.viewer.lastY;
  state.viewer.lastX = event.clientX;
  state.viewer.lastY = event.clientY;
  state.viewer.offsetX += deltaX;
  state.viewer.offsetY += deltaY;
  clampViewerOffsets();
  renderResultViewer();
}

function endResultViewerDrag(event) {
  if (event.pointerId !== state.viewer.pointerId) {
    return;
  }

  state.viewer.dragging = false;
  els.resultViewer.releasePointerCapture(event.pointerId);
  els.resultViewer.removeEventListener("pointermove", handleResultViewerDrag);
  els.resultViewer.removeEventListener("pointerup", endResultViewerDrag);
  els.resultViewer.removeEventListener("pointercancel", endResultViewerDrag);
  state.viewer.pointerId = null;
}

function handleResultViewerWheel(event) {
  if (els.generatedImage.hidden) {
    return;
  }

  event.preventDefault();
  const zoomDelta = event.deltaY < 0 ? 0.12 : -0.12;
  state.viewer.zoom = Math.min(3.2, Math.max(1, state.viewer.zoom + zoomDelta));
  clampViewerOffsets();
  renderResultViewer();
}

function resetResultViewer() {
  state.viewer.offsetX = 0;
  state.viewer.offsetY = 0;
  state.viewer.zoom = 1;
  state.viewer.dragging = false;
  state.viewer.pointerId = null;
  renderResultViewer();
}

function clampViewerOffsets() {
  const {
    viewportWidth,
    viewportHeight,
    fittedWidth,
    fittedHeight,
  } = getResultImageViewportMetrics();
  const horizontalLimit = Math.max(0, (fittedWidth * state.viewer.zoom - viewportWidth) / 2);
  const verticalLimit = Math.max(0, (fittedHeight * state.viewer.zoom - viewportHeight) / 2);

  state.viewer.offsetX = Math.max(-horizontalLimit, Math.min(horizontalLimit, state.viewer.offsetX));
  state.viewer.offsetY = Math.max(-verticalLimit, Math.min(verticalLimit, state.viewer.offsetY));
}

function getResultImageViewportMetrics() {
  const viewportWidth = els.resultViewer.clientWidth || 1;
  const viewportHeight = els.resultViewer.clientHeight || 1;
  const naturalWidth = els.generatedImage.naturalWidth || viewportWidth;
  const naturalHeight = els.generatedImage.naturalHeight || viewportHeight;
  const fitScale = Math.min(viewportWidth / naturalWidth, viewportHeight / naturalHeight);

  return {
    viewportWidth,
    viewportHeight,
    fittedWidth: naturalWidth * fitScale,
    fittedHeight: naturalHeight * fitScale,
  };
}

function renderResultViewer() {
  els.generatedImage.style.transform =
    `translate(${state.viewer.offsetX}px, ${state.viewer.offsetY}px) scale(${state.viewer.zoom})`;
}

function syncResultFrameAspectRatio() {
  const naturalWidth = els.generatedImage.naturalWidth;
  const naturalHeight = els.generatedImage.naturalHeight;

  if (!naturalWidth || !naturalHeight) {
    els.resultFrame.style.removeProperty("--result-aspect-ratio");
    return;
  }

  els.resultFrame.style.setProperty("--result-aspect-ratio", `${naturalWidth} / ${naturalHeight}`);
}

function createCatalogItem({ id, name, color, silhouette }) {
  return {
    id,
    name,
    color,
    silhouette,
    imageUrl: createFurnitureSwatch({ color, silhouette }),
    productUrl: "",
  };
}

function createFurnitureSwatch({ color, silhouette }) {
  const svg = {
    sofa: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120"><rect x="22" y="52" width="76" height="28" rx="10" fill="${color}"/><rect x="18" y="44" width="24" height="36" rx="10" fill="${color}"/><rect x="78" y="44" width="24" height="36" rx="10" fill="${color}"/><rect x="28" y="36" width="64" height="18" rx="9" fill="#fff" fill-opacity=".38"/><rect x="26" y="80" width="6" height="12" rx="3" fill="#694f3d"/><rect x="88" y="80" width="6" height="12" rx="3" fill="#694f3d"/></svg>`,
    chair: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120"><rect x="36" y="32" width="48" height="32" rx="14" fill="${color}"/><rect x="30" y="58" width="60" height="18" rx="9" fill="${color}"/><rect x="36" y="76" width="7" height="18" rx="3" fill="#694f3d"/><rect x="77" y="76" width="7" height="18" rx="3" fill="#694f3d"/></svg>`,
    lamp: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120"><path d="M42 38h36L66 60H54z" fill="${color}"/><rect x="57" y="60" width="6" height="26" rx="3" fill="#694f3d"/><rect x="42" y="86" width="36" height="8" rx="4" fill="#694f3d"/></svg>`,
    table: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120"><rect x="24" y="38" width="72" height="16" rx="8" fill="${color}"/><rect x="34" y="54" width="6" height="34" rx="3" fill="#694f3d"/><rect x="80" y="54" width="6" height="34" rx="3" fill="#694f3d"/></svg>`,
    plant: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120"><path d="M60 26c14 8 16 24 7 36-11-4-17-18-7-36zM40 40c12 3 18 16 14 29-11 0-20-11-14-29zM80 40c6 18-3 29-14 29-4-13 2-26 14-29z" fill="${color}"/><path d="M44 72h32l-6 20H50z" fill="#9a6a44"/></svg>`,
    shelf: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120"><rect x="30" y="24" width="8" height="72" rx="4" fill="${color}"/><rect x="82" y="24" width="8" height="72" rx="4" fill="${color}"/><rect x="30" y="30" width="60" height="8" rx="4" fill="${color}"/><rect x="30" y="54" width="60" height="8" rx="4" fill="${color}"/><rect x="30" y="78" width="60" height="8" rx="4" fill="${color}"/></svg>`,
  }[silhouette];

  return `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(svg)}`;
}
