/**
 * export.js
 *
 * Client-side CSV and XLSX download for frame-by-frame AU data.
 *
 * Frame data schema (produced by ui.js):
 *   { frame: number, timestamp_ms: number, AU01: number, AU02: number, ... }
 */

// ─── CSV ──────────────────────────────────────────────────────────────────────

/**
 * Build a CSV string from an array of frame objects and trigger a download.
 * @param {Array<object>} frames
 * @param {string} [filename]
 */
export function downloadCSV(frames, filename = 'au_data.csv') {
  if (!frames.length) return;

  const headers = Object.keys(frames[0]);
  const rows    = frames.map(row =>
    headers.map(h => {
      const v = row[h];
      // Quote strings that contain commas
      return typeof v === 'string' && v.includes(',') ? `"${v}"` : v;
    }).join(',')
  );

  const csv  = [headers.join(','), ...rows].join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  triggerDownload(blob, filename);
}

// ─── XLSX ─────────────────────────────────────────────────────────────────────

/**
 * Build an XLSX workbook from frame data and trigger a download.
 * Uses SheetJS loaded dynamically to keep initial page load fast.
 *
 * @param {Array<object>} frames
 * @param {string} [filename]
 */
export async function downloadXLSX(frames, filename = 'au_data.xlsx') {
  if (!frames.length) return;

  // Lazy-load SheetJS (only fetched when the user first clicks XLSX)
  const XLSX = await loadSheetJS();

  const ws = XLSX.utils.json_to_sheet(frames);

  // Column widths: narrow for numbers, wider for names
  const headers = Object.keys(frames[0]);
  ws['!cols'] = headers.map(h =>
    h === 'frame' || h === 'timestamp_ms' ? { wch: 14 } : { wch: 8 }
  );

  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, 'AU Data');

  const buffer = XLSX.write(wb, { bookType: 'xlsx', type: 'array' });
  const blob   = new Blob([buffer], {
    type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  });
  triggerDownload(blob, filename);
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a   = document.createElement('a');
  a.href     = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

let _xlsxModule = null;

async function loadSheetJS() {
  if (_xlsxModule) return _xlsxModule;
  // Official SheetJS CDN — ES module build
  _xlsxModule = await import('https://cdn.sheetjs.com/xlsx-0.20.3/package/xlsx.mjs');
  return _xlsxModule;
}
