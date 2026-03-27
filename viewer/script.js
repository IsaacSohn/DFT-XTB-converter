const xyzInput = document.getElementById('xyzInput');
const parseBtn = document.getElementById('parseBtn');
const loadSampleBtn = document.getElementById('loadSampleBtn');
const fileInput = document.getElementById('fileInput');
const tableBody = document.getElementById('tableBody');
const meta = document.getElementById('meta');
const message = document.getElementById('message');
const searchInput = document.getElementById('searchInput');

let parsedRows = [];

const sample = `192
TotEnergy=-1103.24236000 cutoff=-1.00000000 nneightol=1.20000000 pbc="T T T" Lattice="23.46511000       0.00000000       0.00000000      -0.00000100      23.46511000       0.00000000      -0.00000100      -0.00000100      23.46511000" Properties=species:S:1:pos:R:3:force:R:3:Z:I:1
O              11.72590000     14.59020000     25.33440000     -0.04213780      0.03788820      0.00314949       8
H              12.69400000     16.13880000     24.72010000     -0.03709700     -0.03453660      0.01566490       1
H               9.70021000     15.03790000     25.76530000      0.07676920     -0.00101183     -0.02270490       1
O              10.68010000      3.41217000      4.43292000     -0.01918440      0.01516070      0.03966070       8
H              10.14500000      3.90822000      6.40047000      0.01092440     -0.00643783     -0.08715890       1
H               9.97507000      4.53606000      3.00742000      0.01153240     -0.01693960      0.04267200       1`;

function setMessage(text, isError = false) {
  message.textContent = text;
  message.className = isError ? 'error' : 'status';
}

function parseMetadataLine(line) {
  const result = {};
  const energyMatch = line.match(/TotEnergy=([^\s]+)/);
  const cutoffMatch = line.match(/cutoff=([^\s]+)/);
  const nneightolMatch = line.match(/nneightol=([^\s]+)/);
  const pbcMatch = line.match(/pbc="([^"]+)"/);
  const latticeMatch = line.match(/Lattice="([^"]+)"/);
  const propertiesMatch = line.match(/Properties=([^\s]+)/);

  if (energyMatch) result.TotEnergy = energyMatch[1];
  if (cutoffMatch) result.cutoff = cutoffMatch[1];
  if (nneightolMatch) result.nneightol = nneightolMatch[1];
  if (pbcMatch) result.pbc = pbcMatch[1];
  if (latticeMatch) result.lattice = latticeMatch[1];
  if (propertiesMatch) result.properties = propertiesMatch[1];

  return result;
}

function parseXYZ(text) {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  if (lines.length < 3) {
    throw new Error('Not enough lines for an XYZ/extXYZ file.');
  }

  const atomCount = Number(lines[0]);
  if (!Number.isFinite(atomCount)) {
    throw new Error('First line must be the atom count.');
  }

  const metadata = parseMetadataLine(lines[1]);
  const atomLines = lines.slice(2);
  const rows = [];

  for (let i = 0; i < atomLines.length; i++) {
    const parts = atomLines[i].split(/\s+/);
    if (parts.length < 8) continue;

    rows.push({
      index: i + 1,
      species: parts[0],
      x: Number(parts[1]),
      y: Number(parts[2]),
      z: Number(parts[3]),
      fx: Number(parts[4]),
      fy: Number(parts[5]),
      fz: Number(parts[6]),
      atomicNumber: Number(parts[7])
    });
  }

  return { atomCount, metadata, rows };
}

function renderMeta(atomCount, metadata, shownCount) {
  const items = [
    ['Atom count (header)', atomCount],
    ['Parsed atoms', shownCount],
    ['TotEnergy', metadata.TotEnergy ?? '—'],
    ['cutoff', metadata.cutoff ?? '—'],
    ['nneightol', metadata.nneightol ?? '—'],
    ['pbc', metadata.pbc ?? '—'],
    ['Properties', metadata.properties ?? '—']
  ];

  meta.innerHTML = items
    .map(
      ([label, value]) => `
        <div class="pill">
          <strong>${label}</strong>
          <span>${value}</span>
        </div>
      `
    )
    .join('');
}

function renderTable(rows) {
  tableBody.innerHTML = rows
    .map(
      (row) => `
        <tr>
          <td>${row.index}</td>
          <td>${row.species}</td>
          <td>${row.x}</td>
          <td>${row.y}</td>
          <td>${row.z}</td>
          <td>${row.fx}</td>
          <td>${row.fy}</td>
          <td>${row.fz}</td>
          <td>${row.atomicNumber}</td>
        </tr>
      `
    )
    .join('');
}

function applyFilter() {
  const q = searchInput.value.trim().toLowerCase();
  if (!q) {
    renderTable(parsedRows);
    return;
  }

  const filtered = parsedRows.filter(
    (row) =>
      row.species.toLowerCase().includes(q) ||
      String(row.index).includes(q) ||
      String(row.atomicNumber).includes(q) ||
      String(row.x).includes(q) ||
      String(row.y).includes(q) ||
      String(row.z).includes(q)
  );

  renderTable(filtered);
}

function handleParse() {
  try {
    const { atomCount, metadata, rows } = parseXYZ(xyzInput.value);
    parsedRows = rows;
    renderMeta(atomCount, metadata, rows.length);
    renderTable(rows);
    setMessage(`Parsed ${rows.length} atoms successfully.`);
  } catch (err) {
    meta.innerHTML = '';
    tableBody.innerHTML = '';
    parsedRows = [];
    setMessage(err.message, true);
  }
}

parseBtn.addEventListener('click', handleParse);

loadSampleBtn.addEventListener('click', () => {
  xyzInput.value = sample;
  handleParse();
});

searchInput.addEventListener('input', applyFilter);

fileInput.addEventListener('change', async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  const text = await file.text();
  xyzInput.value = text;
  handleParse();
});
