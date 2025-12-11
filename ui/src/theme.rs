pub const GLOBAL_CSS: &str = r#"
:root {
  --bg: #05090f;
  --bg-elev-1: #0b111a;
  --bg-elev-2: #111a26;
  --panel: #0d1520;
  --border: rgba(255, 255, 255, 0.08);
  --border-strong: rgba(255, 255, 255, 0.16);
  --text: #e6edf7;
  --text-dim: #b7c6d9;
  --text-muted: #7f8ba0;
  --accent: #5cb0ff;
  --accent-strong: #7ac6ff;
  --positive: #3fb68b;
  --negative: #f0635c;
  --warning: #f7c843;
  --surface-hover: rgba(255, 255, 255, 0.05);
  --surface-active: rgba(255, 255, 255, 0.1);
  --shadow-soft: 0 14px 42px rgba(0, 0, 0, 0.38);
  --radius: 10px;
  --radius-pill: 999px;
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-5: 20px;
  --space-6: 24px;
  --font-body: "Inter", "SF Pro Text", system-ui, -apple-system, sans-serif;
  --font-mono: "JetBrains Mono", "SFMono-Regular", ui-monospace, monospace;
  --font-size-xs: 11px;
  --font-size-sm: 13px;
  --font-size-md: 15px;
  --font-size-lg: 17px;
  --transition: 140ms ease-out;
}

.light-theme {
  --bg: #f8fbff;
  --bg-elev-1: #ffffff;
  --bg-elev-2: #edf1f7;
  --panel: #ffffff;
  --border: rgba(0, 0, 0, 0.06);
  --border-strong: rgba(0, 0, 0, 0.12);
  --text: #0c1625;
  --text-dim: #2c3a4f;
  --text-muted: #5b6678;
  --accent: #2563eb;
  --accent-strong: #1d4ed8;
  --positive: #0ea66c;
  --negative: #e11d48;
  --warning: #d97706;
  --surface-hover: rgba(0, 0, 0, 0.04);
  --surface-active: rgba(0, 0, 0, 0.08);
  --shadow-soft: 0 10px 36px rgba(0, 0, 0, 0.14);
}

* { box-sizing: border-box; }
html, body {
  padding: 0;
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-body);
  font-size: var(--font-size-sm);
  line-height: 1.4;
  letter-spacing: 0.01em;
  min-height: 100%;
}

a { color: var(--accent); text-decoration: none; }
a:hover { color: var(--accent-strong); }

button {
  font-family: var(--font-body);
}

input, select, textarea {
  background: var(--bg-elev-1);
  border: 1px solid var(--border);
  color: var(--text);
  padding: var(--space-2) var(--space-3);
  border-radius: var(--radius);
  font-size: var(--font-size-sm);
  outline: none;
  transition: border-color var(--transition), box-shadow var(--transition), background var(--transition);
}

input:focus, select:focus, textarea:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 1px rgba(77, 163, 255, 0.35);
}

.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

.app-shell {
  display: grid;
  grid-template-rows: 56px 1fr 220px;
  grid-template-columns: 280px 1fr 320px;
  grid-template-areas:
    "topbar topbar topbar"
    "sidebar main rightbar"
    "bottombar bottombar bottombar";
  min-height: 100vh;
  background: var(--bg);
  gap: var(--space-3);
  padding: var(--space-3);
}

.app-shell.light-theme { background: var(--bg); }

.panel { background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius); box-shadow: var(--shadow-soft); }
.topbar { grid-area: topbar; display: flex; align-items: center; justify-content: space-between; gap: var(--space-3); padding: 0 var(--space-3); }
.sidebar { grid-area: sidebar; display: flex; flex-direction: column; gap: var(--space-3); padding: var(--space-3); }
.main { grid-area: main; position: relative; display: flex; flex-direction: column; gap: var(--space-2); min-height: 0; }
.rightbar { grid-area: rightbar; display: flex; flex-direction: column; gap: var(--space-3); padding: var(--space-3); }
.bottombar { grid-area: bottombar; display: flex; flex-direction: column; gap: var(--space-2); padding: var(--space-3); }

.section-label { font-size: var(--font-size-xs); color: var(--text-muted); letter-spacing: 0.04em; text-transform: uppercase; }
.flex-row { display: flex; gap: var(--space-2); align-items: center; }
.flex-col { display: flex; flex-direction: column; gap: var(--space-2); }
.flex-between { display: flex; justify-content: space-between; align-items: center; }
.chip { padding: var(--space-1) var(--space-2); border-radius: var(--radius-pill); background: var(--surface-hover); border: 1px solid var(--border); font-size: var(--font-size-xs); color: var(--text-dim); }
.watchlist { display: flex; flex-direction: column; gap: var(--space-2); }
.watchlist-item { display: flex; justify-content: space-between; align-items: center; padding: var(--space-2) var(--space-3); border: 1px solid var(--border); border-radius: var(--radius); background: var(--bg-elev-1); }
.watchlist-item:hover { background: var(--surface-hover); }
.chart-grid { display: grid; width: 100%; height: 100%; gap: var(--space-2); align-items: stretch; justify-items: stretch; min-height: 0; }
.chart-cell { position: relative; display: flex; flex-direction: column; height: 100%; min-height: 420px; background: var(--bg-elev-1); border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; }
.indicator-panel, .alerts-panel { display: flex; flex-direction: column; gap: var(--space-2); }
.indicator-row { display: flex; align-items: center; justify-content: space-between; gap: var(--space-2); padding: var(--space-2); border: 1px solid var(--border); border-radius: var(--radius); background: var(--bg-elev-1); }
.flyout { position: absolute; top: 46px; left: 0; z-index: 10; min-width: 320px; padding: var(--space-2); border-radius: var(--radius); background: var(--bg-elev-2); border: 1px solid var(--border-strong); box-shadow: var(--shadow-soft); }
.flyout-row { width: 100%; display: flex; align-items: center; justify-content: space-between; gap: var(--space-2); padding: var(--space-2) var(--space-3); border-radius: var(--radius); border: 1px solid transparent; background: transparent; color: var(--text); cursor: pointer; }
.flyout-row:hover, .flyout-row.active { background: var(--surface-hover); border-color: var(--border); }
.flyout-meta { color: var(--text-muted); font-size: var(--font-size-xs); }
.status-pill { display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: var(--radius-pill); font-size: var(--font-size-xs); border: 1px solid var(--border); background: var(--bg-elev-1); }
.status-good { border-color: rgba(63, 182, 139, 0.4); color: var(--positive); }
.status-warn { border-color: rgba(247, 200, 67, 0.4); color: var(--warning); }
.status-bad { border-color: rgba(240, 99, 92, 0.4); color: var(--negative); }
.status-muted { color: var(--text-muted); }
.status-spinner { width: 10px; height: 10px; border: 2px solid rgba(92,176,255,0.4); border-top-color: var(--accent); border-radius: 50%; display: inline-block; animation: spin 0.9s linear infinite; }
.status-elapsed { color: var(--text-muted); }

.lab-app { background: var(--bg); color: var(--text); min-height: 100vh; }
.lab-shell { display: flex; flex-direction: column; gap: var(--space-4); padding: var(--space-4) var(--space-4) var(--space-5); }
.panel { background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius); box-shadow: var(--shadow-soft); }

.lab-topbar { display: grid; grid-template-columns: 240px 1fr auto; align-items: center; height: 64px; padding: 0 var(--space-4); gap: var(--space-4); position: sticky; top: 0; z-index: 8; border: 1px solid var(--border-strong); }
.topbar-left { display: flex; align-items: center; gap: var(--space-2); }
.brand-mark { display: flex; align-items: center; gap: var(--space-2); font-weight: 600; letter-spacing: 0.03em; }
.brand-title { font-size: var(--font-size-lg); }
.topbar-center { display: flex; align-items: center; gap: var(--space-3); justify-content: space-between; }
.topbar-center .input-stack.wide { flex: 1 1 auto; }
.topbar-controls { display: flex; gap: var(--space-3); align-items: flex-end; }
.topbar-actions { display: flex; align-items: center; gap: var(--space-2); }

.lab-body { display: grid; grid-template-columns: minmax(0, 1fr) 340px; gap: var(--space-4); min-height: calc(100vh - 140px); }
.left-column { display: flex; flex-direction: column; gap: var(--space-3); min-height: 0; }
.chart-card { display: flex; flex-direction: column; gap: var(--space-3); padding: var(--space-3); min-height: 640px; }
.chart-meta { display: flex; align-items: center; justify-content: space-between; gap: var(--space-3); }
.chart-meta-actions { display: flex; align-items: center; gap: var(--space-2); }
.chart-stage { position: relative; background: var(--bg-elev-1); border: 1px solid var(--border); border-radius: var(--radius); padding: var(--space-2); min-height: 520px; display: flex; flex-direction: column; }
.status-floating { position: absolute; right: var(--space-3); bottom: var(--space-3); background: rgba(0,0,0,0.45); border: 1px solid var(--border); padding: 6px 10px; border-radius: var(--radius-pill); font-size: var(--font-size-xs); color: var(--text-dim); }

.chart-grid { display: grid; width: 100%; height: 100%; gap: var(--space-2); align-items: stretch; justify-items: stretch; min-height: 0; }
.chart-cell { position: relative; display: flex; flex-direction: column; height: 100%; background: var(--bg-elev-1); border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; }
.chart-stage > .chart-grid { flex: 1 1 auto; min-height: 0; }
.chart-banner { display: flex; align-items: center; justify-content: space-between; padding: var(--space-1) var(--space-2) var(--space-2); gap: var(--space-2); }
.chart-tools { display: flex; gap: var(--space-2); flex-wrap: wrap; justify-content: flex-end; }
.chart-surface { position: relative; flex: 1 1 auto; min-height: 420px; }
.chart-canvas { position: relative; flex: 1 1 auto; width: 100%; height: 100%; }

.pane-grid { display: grid; gap: var(--space-3); grid-template-columns: 1fr; }
.pane { background: var(--bg-elev-1); border: 1px solid var(--border); border-radius: var(--radius); padding: var(--space-3); display: flex; flex-direction: column; gap: var(--space-2); }
.resizable-pane { resize: vertical; overflow: auto; min-height: 180px; }
.pane-header { display: flex; align-items: center; justify-content: space-between; gap: var(--space-2); }
.pane-title { font-size: var(--font-size-md); font-weight: 600; letter-spacing: 0.01em; }
.pane-subtitle { font-size: var(--font-size-xs); color: var(--text-muted); }
.pane-kpis { display: flex; gap: var(--space-2); flex-wrap: wrap; }
.tab-bar { display: flex; gap: var(--space-2); flex-wrap: wrap; margin: var(--space-1) 0 var(--space-2); }
.tab-bar .pill { border-style: dashed; }
.tab-bar .pill.active { border-color: var(--accent); background: rgba(92,176,255,0.14); color: var(--text); }

.pill-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: var(--space-2); }
.pill-card { text-align: left; border: 1px solid var(--border); background: var(--bg-elev-1); border-radius: var(--radius); padding: var(--space-2) var(--space-3); display: flex; flex-direction: column; gap: 4px; cursor: pointer; transition: border-color var(--transition), background var(--transition); }
.pill-card:hover { border-color: var(--border-strong); background: var(--surface-hover); }
.pill-card-title { font-weight: 600; }
.pill-card-sub { font-size: var(--font-size-xs); color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.04em; }

.heatmap-grid { display: flex; flex-direction: column; gap: var(--space-2); }
.heatmap-legend { display: flex; gap: var(--space-2); flex-wrap: wrap; }
.heatmap-body { display: grid; gap: 6px; }
.heatmap-cell { border: 1px solid var(--border); border-radius: 6px; padding: 10px 0; color: var(--text); font-weight: 600; text-align: center; cursor: pointer; transition: transform var(--transition), border-color var(--transition); }
.heatmap-cell:hover { transform: translateY(-1px); border-color: var(--accent); }
.heatmap-cell.empty { background: var(--bg-elev-2); border-style: dashed; color: var(--text-muted); }

.runs-table { display: flex; flex-direction: column; gap: var(--space-1); }
.runs-head, .runs-row { display: grid; grid-template-columns: 2fr repeat(4, 0.8fr) 0.6fr; align-items: center; gap: var(--space-2); }
.runs-head { font-size: var(--font-size-xs); color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.04em; }
.runs-row { text-align: left; padding: var(--space-2); border: 1px solid var(--border); border-radius: var(--radius); background: var(--bg-elev-1); cursor: pointer; transition: background var(--transition), border-color var(--transition); }
.runs-row:hover { background: var(--surface-hover); border-color: var(--border-strong); }

.sweep-summary { display: flex; gap: var(--space-2); flex-wrap: wrap; align-items: center; margin-bottom: var(--space-2); }
.sweep-group { display: flex; flex-direction: column; gap: var(--space-2); }
.sweep-row { display: grid; grid-template-columns: 1fr 100px 72px; align-items: center; gap: var(--space-2); }
.sweep-label { font-weight: 600; color: var(--text-dim); }
.sweep-range { display: grid; grid-template-columns: repeat(3, 1fr); gap: var(--space-2); align-items: center; }
.range-sep { text-align: center; color: var(--text-muted); }
.toggle { padding: 8px 10px; border-radius: var(--radius); border: 1px solid var(--border); background: var(--bg-elev-1); color: var(--text-muted); cursor: pointer; transition: background var(--transition), border-color var(--transition), color var(--transition); }
.toggle.on { border-color: var(--accent); background: rgba(92,176,255,0.14); color: var(--text); }
.input-compact.tight { padding: 6px 8px; }

.list-row.compact { padding: var(--space-2); gap: var(--space-2); }

@media (max-width: 1100px) {
  .runs-head, .runs-row { grid-template-columns: 1.6fr repeat(3, 1fr) 0.9fr; }
  .sweep-row { grid-template-columns: 1fr 120px 80px; }
}

@media (max-width: 780px) {
  .lab-topbar { grid-template-columns: 1fr; height: auto; }
  .topbar-center { flex-direction: column; align-items: stretch; gap: var(--space-2); }
  .topbar-controls { width: 100%; justify-content: flex-start; flex-wrap: wrap; gap: var(--space-2); }
  .runs-head, .runs-row { grid-template-columns: 1fr 0.8fr 0.8fr; }
  .runs-head span:nth-child(n+4), .runs-row span:nth-child(n+4) { display: none; }
  .sweep-row { grid-template-columns: 1fr 1fr; }
  .sweep-range { grid-template-columns: repeat(3, minmax(0, 1fr)); }
}

@media (max-width: 640px) {
  .lab-shell { padding: var(--space-3); }
  .tab-bar { gap: var(--space-1); }
  .runs-head, .runs-row { grid-template-columns: 1fr 0.8fr; }
  .runs-head span:nth-child(n+3), .runs-row span:nth-child(n+3) { display: none; }
  .heatmap-body { gap: 4px; }
  .sweep-row { grid-template-columns: 1fr; }
  .sweep-range { grid-template-columns: 1fr; }
}

.sidebar { position: sticky; top: 82px; align-self: start; display: flex; flex-direction: column; gap: var(--space-3); max-height: calc(100vh - 120px); overflow: auto; padding: var(--space-3); }
.sidebar-section { display: flex; flex-direction: column; gap: var(--space-2); border-bottom: 1px solid var(--border); padding-bottom: var(--space-3); }
.sidebar-section:last-child { border-bottom: none; padding-bottom: 0; }
.section-head { display: flex; align-items: center; justify-content: space-between; }
.section-title { font-size: var(--font-size-sm); font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; color: var(--text-dim); }
.section-subtitle { font-size: var(--font-size-xs); color: var(--text-muted); }
.section-subtitle.muted { color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.04em; }

.watchlist { display: flex; flex-direction: column; gap: var(--space-2); }
.list-row { width: 100%; text-align: left; padding: var(--space-2) var(--space-3); border: 1px solid var(--border); border-radius: var(--radius); background: var(--bg-elev-1); color: var(--text); display: flex; align-items: center; justify-content: space-between; gap: var(--space-3); cursor: pointer; transition: background var(--transition), border-color var(--transition); }
.list-row:hover { background: var(--surface-hover); border-color: var(--border-strong); }
.list-row.active-row { background: var(--surface-active); border-color: var(--accent); }
.list-left { display: flex; flex-direction: column; gap: 4px; align-items: flex-start; }
.list-right { display: flex; gap: var(--space-2); align-items: center; }
.list-title { font-weight: 600; }
.list-sub { font-size: var(--font-size-xs); color: var(--text-muted); }
.list-stack { display: flex; flex-direction: column; gap: var(--space-2); }

.watchlist-add { display: flex; gap: var(--space-2); align-items: center; }

.input-stack { display: flex; flex-direction: column; gap: 6px; min-width: 240px; }
.control-stack { display: flex; flex-direction: column; gap: 6px; }
.input-label { font-size: var(--font-size-xs); color: var(--text-muted); letter-spacing: 0.04em; text-transform: uppercase; }
.input-wrap { position: relative; display: flex; flex-direction: column; gap: 6px; }
.input-lg { padding: 12px 14px; font-size: var(--font-size-md); border-radius: 12px; }
.input-compact { padding: 8px 10px; font-size: var(--font-size-sm); border-radius: var(--radius); }
.compact-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: var(--space-2); }
.form-grid { display: grid; gap: var(--space-2); }

.btn { border: 1px solid var(--border); background: var(--bg-elev-1); color: var(--text); padding: 8px 12px; border-radius: var(--radius); font-size: var(--font-size-sm); cursor: pointer; transition: background var(--transition), border-color var(--transition), color var(--transition), transform var(--transition); }
.btn:hover { background: var(--surface-hover); border-color: var(--border-strong); }
.btn:active { background: var(--surface-active); transform: translateY(1px); }
.btn.primary { background: linear-gradient(135deg, var(--accent), var(--accent-strong)); border-color: transparent; color: #02111f; font-weight: 600; }
.btn.primary:hover { filter: brightness(1.05); }
.btn.secondary { background: var(--bg-elev-2); border-color: var(--border); color: var(--text); font-weight: 600; }
.btn.ghost { background: transparent; border-style: dashed; color: var(--text-dim); }
.btn.micro { padding: 4px 8px; font-size: var(--font-size-xs); border-radius: 8px; }
.chart-tools { position: relative; display: flex; align-items: center; gap: var(--space-2); }
.chart-tools .tool-trigger { padding: 6px 8px; border-radius: var(--radius); border: 1px solid var(--border); background: var(--bg-elev-1); color: var(--text-dim); cursor: pointer; }
.chart-tools .tool-tray { display: flex; gap: var(--space-2); align-items: center; }
.chart-tools.compact .tool-tray { display: none; }
.chart-tools.compact:hover .tool-tray,
.chart-tools.compact:focus-within .tool-tray { display: flex; }
.chart-tools.compact .tool-trigger { display: inline-flex; align-items: center; gap: 6px; }
.chart-tools.compact .tool-trigger .caret { font-size: 10px; color: var(--text-muted); }

.pill { padding: 4px 10px; border-radius: var(--radius-pill); background: var(--surface-hover); border: 1px solid var(--border); font-size: var(--font-size-xs); color: var(--text-dim); display: inline-flex; gap: 6px; align-items: center; }
.pill-soft { background: rgba(92, 176, 255, 0.08); border-color: rgba(92, 176, 255, 0.2); color: var(--text); }
.pill-accent { background: rgba(92, 176, 255, 0.12); border-color: rgba(92, 176, 255, 0.3); color: var(--text); }
.pill-strong { background: rgba(255,255,255,0.08); border-color: var(--border-strong); color: var(--text); font-weight: 600; }
.pill.selectable { cursor: pointer; border-style: dashed; }
.pill.selectable.active { border-color: var(--accent); color: var(--text); background: rgba(92, 176, 255, 0.14); }
.pill-row { display: flex; gap: var(--space-2); flex-wrap: wrap; }
.pill-up { color: var(--positive); border-color: rgba(63, 182, 139, 0.4); background: rgba(63, 182, 139, 0.08); }
.pill-down { color: var(--negative); border-color: rgba(240, 99, 92, 0.4); background: rgba(240, 99, 92, 0.08); }

.status-pill { display: inline-flex; align-items: center; gap: 8px; padding: 6px 10px; border-radius: var(--radius-pill); font-size: var(--font-size-xs); border: 1px solid var(--border); background: var(--bg-elev-1); }
.status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--text-muted); display: inline-block; }
.status-good { border-color: rgba(63, 182, 139, 0.4); color: var(--positive); }
.status-good .status-dot { background: var(--positive); }
.status-warn { border-color: rgba(247, 200, 67, 0.4); color: var(--warning); }
.status-warn .status-dot { background: var(--warning); }
.status-bad { border-color: rgba(240, 99, 92, 0.4); color: var(--negative); }
.status-bad .status-dot { background: var(--negative); }
.status-muted { color: var(--text-muted); }

.flyout { position: absolute; top: 46px; left: 0; z-index: 10; min-width: 320px; padding: var(--space-2); border-radius: var(--radius); background: var(--bg-elev-2); border: 1px solid var(--border-strong); box-shadow: var(--shadow-soft); }
.flyout-row { width: 100%; display: flex; align-items: center; justify-content: space-between; gap: var(--space-2); padding: var(--space-2) var(--space-3); border-radius: var(--radius); border: 1px solid transparent; background: transparent; color: var(--text); cursor: pointer; }
.flyout-row:hover { background: var(--surface-hover); border-color: var(--border); }
.flyout-symbol { font-weight: 600; }
.flyout-meta { color: var(--text-muted); font-size: var(--font-size-xs); }

.indicator-controls { display: flex; flex-direction: column; gap: var(--space-2); }
.indicator-actions { display: flex; gap: var(--space-2); align-items: center; }

.equity-placeholder { position: relative; height: 160px; background: linear-gradient(180deg, rgba(92,176,255,0.08), rgba(11,17,26,0.7)); border: 1px dashed var(--border); border-radius: var(--radius); overflow: hidden; }
.equity-line { position: absolute; left: 0; right: 0; bottom: 36%; height: 2px; background: var(--accent); box-shadow: 0 0 0 1px rgba(92, 176, 255, 0.3); }
.drawdown-line { position: absolute; left: 0; right: 0; bottom: 18%; height: 2px; background: var(--negative); opacity: 0.7; }

.drawer { position: fixed; left: 0; right: 0; bottom: 0; background: var(--panel); border-top: 1px solid var(--border-strong); box-shadow: 0 -12px 40px rgba(0,0,0,0.5); transform: translateY(100%); transition: transform var(--transition); padding: var(--space-4); z-index: 20; }
.drawer.open { transform: translateY(0); }
.drawer-narrow { right: 24px; left: auto; width: 420px; border-left: 1px solid var(--border-strong); }
.drawer-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: var(--space-3); }
.drawer-body { display: grid; grid-template-columns: minmax(0, 1fr) 220px; gap: var(--space-3); }
.drawer-body-column { grid-template-columns: 1fr; }
.drawer-side { display: flex; flex-direction: column; gap: var(--space-2); padding: var(--space-3); background: var(--bg-elev-1); border: 1px solid var(--border); border-radius: var(--radius); }
.drawer-actions { display: flex; justify-content: flex-end; gap: var(--space-2); margin-top: var(--space-2); }

.script-area { width: 100%; min-height: 260px; background: var(--bg-elev-1); color: var(--text); border: 1px solid var(--border); border-radius: var(--radius); padding: var(--space-3); font-family: var(--font-mono); font-size: var(--font-size-sm); }

@media (max-width: 1350px) {
  .lab-topbar { grid-template-columns: 1fr; height: auto; padding: var(--space-3); gap: var(--space-3); }
  .topbar-center { flex-wrap: wrap; }
  .lab-body { grid-template-columns: 1fr; }
  .sidebar { position: relative; top: 0; max-height: none; }
}

@media (max-width: 900px) {
  .lab-shell { padding: var(--space-3); }
  .chart-card { min-height: 480px; }
  .chart-stage { min-height: 360px; }
}

@media (max-width: 1100px) {
  .app-shell {
    grid-template-columns: 1fr;
    grid-template-rows: 56px auto auto auto 220px;
    grid-template-areas:
      "topbar"
      "sidebar"
      "main"
      "rightbar"
      "bottombar";
  }
  .sidebar, .rightbar { display: flex; position: relative; top: 0; max-height: none; }
  .chart-cell { min-height: 360px; }
}

@media (max-width: 640px) {
  .app-shell { padding: var(--space-2); gap: var(--space-2); }
  .topbar { padding: 0 var(--space-2); flex-wrap: wrap; gap: var(--space-2); }
  .sidebar, .rightbar, .bottombar { padding: var(--space-2); }
  .chart-cell { min-height: 240px; }
  .flyout { width: calc(100vw - 32px); }
}
"#;
