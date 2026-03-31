const PAGE_SIZE = 16;
let currentPage = 1;
let totalPages = 1;
let currentItems = [];
let currentRegion = '';
let availableRegions = [];
const selected = new Set();

const gridEl = document.getElementById('grid');
const pageInfoEl = document.getElementById('pageInfo');
const statsInfoEl = document.getElementById('statsInfo');
const regionSelectEl = document.getElementById('regionSelect');
const prevBtn = document.getElementById('prevBtn');
const nextBtn = document.getElementById('nextBtn');
const savePageBtn = document.getElementById('savePageBtn');
const clearSelectionBtn = document.getElementById('clearSelectionBtn');

async function fetchRegions() {
  const res = await fetch('/api/regions');
  const data = await res.json();
  availableRegions = data.regions || [];

  regionSelectEl.innerHTML = '';
  for (const region of availableRegions) {
    const option = document.createElement('option');
    option.value = region.region_id;
    option.textContent = `${region.region_id} (${region.total})`;
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
    pageInfoEl.textContent = 'Keine Regionen vorhanden.';
    statsInfoEl.textContent = '';
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
  statsInfoEl.textContent = `Gelabelt: ${data.labeled} | Offen: ${data.unlabeled} | Total: ${data.total}`;
}

function renderGrid() {
  gridEl.innerHTML = '';

  for (const item of currentItems) {
    const card = document.createElement('article');
    card.className = 'card';
    card.dataset.index = String(item.index);

    const img = document.createElement('img');
    img.src = `/${item.image_path}`;
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
  pageInfoEl.textContent = `Seite ${currentPage} / ${totalPages} | ${PAGE_SIZE} Bilder pro Seite | ${totalItems} Bilder`;
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

async function init() {
  await fetchRegions();
  await loadPage(1);
}

init();
