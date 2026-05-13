const PAGE_SIZE = 16;
let currentPage = 1;
let totalPages = 1;
let currentItems = [];
let currentRegion = '';
let availableRegions = [];
let uploadedFiles = [];
let lastPredictionResults = [];
let lastPredictionModel = null;
let availableCheckpoints = [];
let selectedCheckpoint = '';
const selected = new Set();

const gridEl = document.getElementById('grid');
const pageInfoEl = document.getElementById('pageInfo');
const statsInfoEl = document.getElementById('statsInfo');
const regionSelectEl = document.getElementById('regionSelect');
const prevBtn = document.getElementById('prevBtn');
const nextBtn = document.getElementById('nextBtn');
const savePageBtn = document.getElementById('savePageBtn');
const clearSelectionBtn = document.getElementById('clearSelectionBtn');

const tabButtons = [...document.querySelectorAll('.tab-button')];
const panels = [...document.querySelectorAll('.panel')];

const predictFormEl = document.getElementById('predictForm');
const predictInputEl = document.getElementById('predictInput');
const predictDropzoneEl = document.getElementById('predictDropzone');
const predictCheckpointEl = document.getElementById('predictCheckpoint');
const predictBtnEl = document.getElementById('predictBtn');
const clearUploadBtnEl = document.getElementById('clearUploadBtn');
const predictFeedbackEl = document.getElementById('predictFeedback');
const predictResultsEl = document.getElementById('predictResults');
const modelStatusEl = document.getElementById('modelStatus');
const predictSortEl = document.getElementById('predictSort');
const predictFilterEl = document.getElementById('predictFilter');
const predictLimitEl = document.getElementById('predictLimit');
const downloadPredictionsBtnEl = document.getElementById('downloadPredictionsBtn');
const imageLightboxEl = document.getElementById('imageLightbox');
const lightboxImageEl = document.getElementById('lightboxImage');
const lightboxCaptionEl = document.getElementById('lightboxCaption');
const lightboxCloseBtnEl = document.getElementById('lightboxCloseBtn');
const TAB_ROUTES = {
  labeling: '/',
  predict: '/predict',
};

function isSupportedImageFile(file) {
  const name = String(file?.name || '').toLowerCase();
  return file && (file.type === 'image/png' || file.type === 'image/jpeg' || name.endsWith('.png') || name.endsWith('.jpg') || name.endsWith('.jpeg'));
}

function syncPredictInputFiles(files) {
  if (!predictInputEl) {
    return;
  }
  try {
    const transfer = new DataTransfer();
    for (const file of files) {
      transfer.items.add(file);
    }
    predictInputEl.files = transfer.files;
  } catch (error) {
    // Some browsers do not allow setting input.files programmatically.
  }
}

function setActiveTab(tabName) {
  for (const button of tabButtons) {
    button.classList.toggle('active', button.dataset.tab === tabName);
  }
  for (const panel of panels) {
    panel.classList.toggle('active', panel.dataset.panel === tabName);
  }
}

function getTabFromLocation() {
  const path = window.location.pathname.replace(/\/+$/, '') || '/';
  if (path === '/predict') {
    return 'predict';
  }
  return 'labeling';
}

function navigateToTab(tabName) {
  const nextPath = TAB_ROUTES[tabName] || TAB_ROUTES.labeling;
  if (window.location.pathname !== nextPath) {
    window.history.pushState({ tab: tabName }, '', nextPath);
  }
  setActiveTab(tabName);
}

async function fetchRegions() {
  const res = await fetch('/api/regions');
  const data = await res.json();
  availableRegions = data.regions || [];

  regionSelectEl.innerHTML = '';
  if (availableRegions.length === 0) {
    currentRegion = '';
    return;
  }

  for (const region of availableRegions) {
    const option = document.createElement('option');
    option.value = region.region_id;
    option.textContent = `${region.region_id} (${region.total} offen)`;
    regionSelectEl.appendChild(option);
  }

  if (!currentRegion || !availableRegions.some((r) => r.region_id === currentRegion)) {
    currentRegion = availableRegions[0]?.region_id || '';
  }

  regionSelectEl.value = currentRegion;
}

async function loadPage(page) {
  if (!currentRegion) {
    currentItems = [];
    renderGrid();
    pageInfoEl.textContent = 'Keine offenen Bilder gefunden.';
    statsInfoEl.textContent = 'Lege neue Tiles unter data/unlabeled ab, um sie hier zu labeln.';
    return;
  }

  const res = await fetch(`/api/page?page=${page}&region_id=${encodeURIComponent(currentRegion)}&page_size=${PAGE_SIZE}`);
  const data = await res.json();

  currentRegion = data.region_id;
  currentPage = data.page;
  totalPages = data.total_pages;
  currentItems = data.items;
  selected.clear();

  regionSelectEl.value = currentRegion;
  renderGrid();
  renderMeta(data.total_items);
  await refreshStats();
}

async function refreshStats() {
  if (!currentRegion) {
    statsInfoEl.textContent = '';
    return;
  }

  const res = await fetch(`/api/stats?region_id=${encodeURIComponent(currentRegion)}`);
  const data = await res.json();
  statsInfoEl.textContent = `Offen: ${data.unlabeled} | Region: ${data.region_id}`;
}

function renderGrid() {
  gridEl.innerHTML = '';

  if (currentItems.length === 0) {
    gridEl.innerHTML = '<p class="empty-state">Keine offenen Bilder für diese Seite vorhanden.</p>';
    return;
  }

  if (currentItems.every((item) => item.file_exists === false)) {
    gridEl.innerHTML =
      '<p class="empty-state">Die Eintraege in <code>data/labels.csv</code> verweisen auf offene Bilder, die in <code>data/unlabeled</code> nicht mehr gefunden wurden.</p>';
    return;
  }

  for (const item of currentItems) {
    if (item.file_exists === false) {
      continue;
    }

    const card = document.createElement('article');
    card.className = 'card';
    card.dataset.index = String(item.index);

    const img = document.createElement('img');
    img.src = item.image_url || `/${item.image_path}`;
    img.alt = item.image_path;
    img.loading = 'lazy';

    const info = document.createElement('div');
    info.className = 'card-info';

    const filename = document.createElement('span');
    filename.className = 'filename';
    filename.textContent = item.image_path.split('/').pop() || item.image_path;

    info.appendChild(filename);
    card.appendChild(img);
    card.appendChild(info);

    if (String(item.label).trim() === '1') {
      selected.add(item.index);
      card.classList.add('selected');
    }

    card.addEventListener('click', () => {
      toggleSelect(item.index, card);
    });

    gridEl.appendChild(card);
  }

  if (!gridEl.children.length) {
    gridEl.innerHTML =
      '<p class="empty-state">Keine vorhandenen Bilddateien für diese Seite gefunden. Bitte <code>data/unlabeled</code> und <code>data/labels.csv</code> pruefen.</p>';
  }
}

function toggleSelect(index, cardEl) {
  if (selected.has(index)) {
    selected.delete(index);
    cardEl.classList.remove('selected');
  } else {
    selected.add(index);
    cardEl.classList.add('selected');
  }
}

function renderMeta(totalItems) {
  pageInfoEl.textContent = `Seite ${currentPage} / ${totalPages} | ${PAGE_SIZE} Bilder pro Seite | ${totalItems} offene Bilder`;
  prevBtn.disabled = currentPage <= 1;
  nextBtn.disabled = currentPage >= totalPages;
}

async function saveCurrentPageAsBinary(targetPage = currentPage) {
  if (currentItems.length === 0) {
    return false;
  }

  const updates = currentItems.map((item) => ({
    index: item.index,
    label: selected.has(item.index) ? '1' : '0',
  }));

  const res = await fetch('/api/update', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ updates }),
  });

  if (!res.ok) {
    alert('Speichern fehlgeschlagen.');
    return false;
  }

  await loadPage(targetPage);
  return true;
}

function renderModelStatus(payload, isError = false) {
  modelStatusEl.classList.toggle('error', isError);
  if (isError) {
    modelStatusEl.textContent = payload.error || 'Modell nicht verfügbar.';
    return;
  }
  modelStatusEl.innerHTML = `
    <strong>${payload.model_name}</strong>
    <span>Checkpoint: ${payload.checkpoint_relpath || payload.checkpoint_path || '-'}</span>
    <span>Device: ${payload.device}</span>
    <span>Threshold: ${Number(payload.threshold).toFixed(2)}</span>
  `;
}

function renderCheckpointOptions(items = [], activePath = '') {
  availableCheckpoints = items;
  predictCheckpointEl.innerHTML = '';

  for (const item of items) {
    const option = document.createElement('option');
    option.value = item.path;
    option.textContent = item.label;
    predictCheckpointEl.appendChild(option);
  }

  if (items.length === 0) {
    const option = document.createElement('option');
    option.value = '';
    option.textContent = 'Keine Modelle gefunden';
    predictCheckpointEl.appendChild(option);
    predictCheckpointEl.disabled = true;
    selectedCheckpoint = '';
    return;
  }

  predictCheckpointEl.disabled = false;
  selectedCheckpoint = activePath && items.some((item) => item.path === activePath) ? activePath : items[0].path;
  predictCheckpointEl.value = selectedCheckpoint;
}

function setPredictFeedback(message, isError = false) {
  predictFeedbackEl.textContent = message;
  predictFeedbackEl.classList.toggle('error', isError);
}

function openLightbox(imageSrc, caption = '') {
  if (!imageSrc || !imageLightboxEl) {
    return;
  }
  lightboxImageEl.src = imageSrc;
  lightboxImageEl.alt = caption;
  lightboxCaptionEl.textContent = caption;
  imageLightboxEl.hidden = false;
  document.body.style.overflow = 'hidden';
}

function closeLightbox() {
  if (!imageLightboxEl) {
    return;
  }
  imageLightboxEl.hidden = true;
  lightboxImageEl.src = '';
  lightboxImageEl.alt = '';
  lightboxCaptionEl.textContent = '';
  document.body.style.overflow = '';
}

function predictionUncertainty(result) {
  if (typeof result.score !== 'number' || typeof result.threshold !== 'number') {
    return Number.POSITIVE_INFINITY;
  }
  return Math.abs(result.score - result.threshold);
}

function getSortedPredictionResults(results) {
  const mode = predictSortEl?.value || 'original';
  const withIndex = results.map((result, index) => ({ result, index }));

  if (mode === 'uncertain') {
    withIndex.sort((a, b) => predictionUncertainty(a.result) - predictionUncertainty(b.result));
  } else if (mode === 'score_desc') {
    withIndex.sort((a, b) => (b.result.score ?? -1) - (a.result.score ?? -1));
  } else if (mode === 'score_asc') {
    withIndex.sort((a, b) => (a.result.score ?? 2) - (b.result.score ?? 2));
  }

  return withIndex;
}

function getVisiblePredictionResults(results) {
  const filterMode = predictFilterEl?.value || 'all';
  const limitMode = predictLimitEl?.value || 'all';

  let visible = getSortedPredictionResults(results);

  if (filterMode === 'positive') {
    visible = visible.filter(({ result }) => result.prediction === 1);
  } else if (filterMode === 'negative') {
    visible = visible.filter(({ result }) => result.prediction === 0);
  } else if (filterMode === 'uncertain') {
    visible = visible.filter(({ result }) => !result.error && predictionUncertainty(result) <= 0.08);
  }

  if (limitMode !== 'all') {
    visible = visible.slice(0, Number(limitMode));
  }

  return visible;
}

function renderPredictionResults(results = [], model = null) {
  predictResultsEl.innerHTML = '';

  if (results.length === 0) {
    predictResultsEl.innerHTML = '<p class="empty-state">Noch keine Vorhersagen. Lade Bilder hoch und starte die Inferenz.</p>';
    downloadPredictionsBtnEl.disabled = true;
    return;
  }

  downloadPredictionsBtnEl.disabled = false;
  const visibleResults = getVisiblePredictionResults(results);
  if (visibleResults.length === 0) {
    predictResultsEl.innerHTML = '<p class="empty-state">Mit der aktuellen Filterkombination gibt es keine Treffer.</p>';
    return;
  }

  for (const entry of visibleResults) {
    const { result, index } = entry;
    const card = document.createElement('article');
    card.className = 'predict-card';
    if (result.prediction === 1) {
      card.classList.add('positive');
    }
    if (!result.error && predictionUncertainty(result) <= 0.08) {
      card.classList.add('uncertain');
    }

    const media = document.createElement('div');
    media.className = 'predict-media';

    if (uploadedFiles[index]) {
      const img = document.createElement('img');
      const previewUrl = URL.createObjectURL(uploadedFiles[index]);
      img.src = previewUrl;
      img.alt = result.filename;
      img.addEventListener('click', () => {
        openLightbox(previewUrl, result.filename || '');
      });
      media.appendChild(img);
    }

    const body = document.createElement('div');
    body.className = 'predict-body';

    const title = document.createElement('h3');
    title.textContent = result.filename;
    body.appendChild(title);

    if (result.error) {
      const errorText = document.createElement('p');
      errorText.className = 'predict-error';
      errorText.textContent = result.error;
      body.appendChild(errorText);
    } else {
      const verdict = document.createElement('p');
      verdict.className = 'predict-verdict';
      verdict.textContent = result.label;
      body.appendChild(verdict);

      const metrics = document.createElement('div');
      metrics.className = 'predict-metrics';
      metrics.innerHTML = `
        <span>Prediction <strong>${result.prediction}</strong></span>
        <span>Score <strong>${Number(result.score).toFixed(3)}</strong></span>
        <span class="predict-threshold">Threshold ${Number(result.threshold).toFixed(2)}</span>
      `;
      body.appendChild(metrics);
    }

    card.appendChild(media);
    card.appendChild(body);
    predictResultsEl.appendChild(card);
  }
}

function downloadPredictionsCsv() {
  if (lastPredictionResults.length === 0) {
    return;
  }

  const visibleResults = getVisiblePredictionResults(lastPredictionResults);
  if (visibleResults.length === 0) {
    return;
  }

  const rows = [
    ['filename', 'score', 'prediction', 'label', 'threshold', 'uncertainty'],
    ...visibleResults.map(({ result }) => [
      result.filename || '',
      typeof result.score === 'number' ? result.score.toFixed(6) : '',
      result.prediction ?? '',
      result.label || '',
      typeof result.threshold === 'number' ? result.threshold.toFixed(4) : '',
      Number.isFinite(predictionUncertainty(result)) ? predictionUncertainty(result).toFixed(6) : '',
    ]),
  ];

  const csv = rows
    .map((row) => row.map((value) => `"${String(value).replaceAll('"', '""')}"`).join(','))
    .join('\n');

  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = 'predictions.csv';
  anchor.click();
  URL.revokeObjectURL(url);
}

async function fetchModelStatus() {
  try {
    const query = selectedCheckpoint ? `?checkpoint=${encodeURIComponent(selectedCheckpoint)}` : '';
    const res = await fetch(`/api/model-status${query}`);
    const data = await res.json();
    renderCheckpointOptions(data.available_checkpoints || [], data.checkpoint_relpath || selectedCheckpoint);
    if (!res.ok || !data.ready) {
      renderModelStatus(data, true);
      return;
    }
    selectedCheckpoint = data.checkpoint_relpath || selectedCheckpoint;
    renderModelStatus(data);
  } catch (error) {
    renderModelStatus({ error: 'Modellstatus konnte nicht geladen werden.' }, true);
  }
}

async function runPrediction(event) {
  event.preventDefault();

  if (uploadedFiles.length === 0) {
    setPredictFeedback('Bitte zuerst mindestens ein Bild auswählen.', true);
    return;
  }

  predictBtnEl.disabled = true;
  setPredictFeedback(`Inferenz läuft für ${uploadedFiles.length} Bild${uploadedFiles.length === 1 ? '' : 'er'} …`);

  const formData = new FormData();
  for (const file of uploadedFiles) {
    formData.append('images', file);
  }
  if (selectedCheckpoint) {
    formData.append('checkpoint', selectedCheckpoint);
  }

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      body: formData,
    });
    const data = await res.json();
    if (!res.ok) {
      setPredictFeedback(data.error || 'Vorhersage fehlgeschlagen.', true);
      return;
    }

    renderModelStatus(data.model);
    lastPredictionResults = data.results || [];
    lastPredictionModel = data.model || null;
    renderPredictionResults(lastPredictionResults, lastPredictionModel);
    setPredictFeedback(`Fertig. ${data.results?.length || 0} Bild${data.results?.length === 1 ? '' : 'er'} ausgewertet.`);
  } catch (error) {
    setPredictFeedback('Vorhersage fehlgeschlagen.', true);
  } finally {
    predictBtnEl.disabled = false;
  }
}

function handleUploadChange(event) {
  uploadedFiles = [...(event.target.files || [])].filter(isSupportedImageFile);
  syncPredictInputFiles(uploadedFiles);
  const message =
    uploadedFiles.length > 0
      ? `${uploadedFiles.length} Bild${uploadedFiles.length === 1 ? '' : 'er'} bereit zur Inferenz.`
      : 'Keine Bilder ausgewählt.';
  setPredictFeedback(message, false);
  lastPredictionResults = [];
  lastPredictionModel = null;
  renderPredictionResults();
}

function clearUploadSelection() {
  uploadedFiles = [];
  predictInputEl.value = '';
  predictDropzoneEl?.classList.remove('drag-active');
  setPredictFeedback('Auswahl geleert.');
  lastPredictionResults = [];
  lastPredictionModel = null;
  renderPredictionResults();
}

function setDroppedFiles(fileList) {
  const files = [...(fileList || [])].filter(isSupportedImageFile);
  uploadedFiles = files;
  syncPredictInputFiles(files);

  if (files.length === 0) {
    setPredictFeedback('Keine unterstützten Bilddateien erkannt. Bitte PNG oder JPG verwenden.', true);
  } else {
    setPredictFeedback(`${files.length} Bild${files.length === 1 ? '' : 'er'} per Drag-and-Drop bereit zur Inferenz.`);
  }

  lastPredictionResults = [];
  lastPredictionModel = null;
  renderPredictionResults();
}

function bindPredictDropzone() {
  if (!predictDropzoneEl) {
    return;
  }

  const preventDefaults = (event) => {
    event.preventDefault();
    event.stopPropagation();
  };

  for (const eventName of ['dragenter', 'dragover', 'dragleave', 'drop']) {
    predictDropzoneEl.addEventListener(eventName, preventDefaults);
  }

  for (const eventName of ['dragenter', 'dragover']) {
    predictDropzoneEl.addEventListener(eventName, () => {
      predictDropzoneEl.classList.add('drag-active');
    });
  }

  for (const eventName of ['dragleave', 'drop']) {
    predictDropzoneEl.addEventListener(eventName, () => {
      predictDropzoneEl.classList.remove('drag-active');
    });
  }

  predictDropzoneEl.addEventListener('drop', (event) => {
    setDroppedFiles(event.dataTransfer?.files);
  });
}

prevBtn.addEventListener('click', () => {
  if (currentPage > 1) {
    loadPage(currentPage - 1);
  }
});

nextBtn.addEventListener('click', async () => {
  if (currentPage < totalPages) {
    await saveCurrentPageAsBinary(currentPage + 1);
  }
});

regionSelectEl.addEventListener('change', async (event) => {
  currentRegion = event.target.value;
  await loadPage(1);
});

savePageBtn.addEventListener('click', saveCurrentPageAsBinary);

clearSelectionBtn.addEventListener('click', () => {
  selected.clear();
  document.querySelectorAll('.card.selected').forEach((el) => el.classList.remove('selected'));
});

for (const button of tabButtons) {
  button.addEventListener('click', () => {
    navigateToTab(button.dataset.tab);
  });
}

window.addEventListener('popstate', () => {
  setActiveTab(getTabFromLocation());
});

imageLightboxEl.addEventListener('click', (event) => {
  if (event.target.dataset.closeLightbox === 'true' || event.target === imageLightboxEl) {
    closeLightbox();
  }
});
lightboxCloseBtnEl.addEventListener('click', closeLightbox);
window.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && !imageLightboxEl.hidden) {
    closeLightbox();
  }
});

predictFormEl.addEventListener('submit', runPrediction);
predictInputEl.addEventListener('change', handleUploadChange);
bindPredictDropzone();
clearUploadBtnEl.addEventListener('click', clearUploadSelection);
predictCheckpointEl.addEventListener('change', async (event) => {
  selectedCheckpoint = event.target.value;
  await fetchModelStatus();
});
predictSortEl.addEventListener('change', () => {
  renderPredictionResults(lastPredictionResults, lastPredictionModel);
});
predictFilterEl.addEventListener('change', () => {
  renderPredictionResults(lastPredictionResults, lastPredictionModel);
});
predictLimitEl.addEventListener('change', () => {
  renderPredictionResults(lastPredictionResults, lastPredictionModel);
});
downloadPredictionsBtnEl.addEventListener('click', downloadPredictionsCsv);

async function init() {
  await Promise.all([fetchRegions(), fetchModelStatus()]);
  await loadPage(1);
  setActiveTab(getTabFromLocation());
  renderPredictionResults();
}

init();
