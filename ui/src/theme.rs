pub const GLOBAL_CSS: &str = r#"
:root {
  --bg: #05070b;
  --bg-elev-1: #0a0f16;
  --bg-elev-2: #0f1623;
  --panel: #0c1018;
  --border: rgba(255, 255, 255, 0.06);
  --border-strong: rgba(255, 255, 255, 0.12);
  --text: #e3edf7;
  --text-dim: #9fb1c7;
  --text-muted: #6f7c8f;
  --accent: #4da3ff;
  --accent-strong: #69c0ff;
  --positive: #3fb68b;
  --negative: #f0635c;
  --warning: #f7c843;
  --surface-hover: rgba(255, 255, 255, 0.04);
  --surface-active: rgba(255, 255, 255, 0.08);
  --shadow-soft: 0 8px 24px rgba(0, 0, 0, 0.35);
  --radius: 4px;
  --radius-pill: 999px;
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-5: 20px;
  --space-6: 24px;
  --font-body: "Inter", "SF Pro Text", system-ui, -apple-system, sans-serif;
  --font-size-xs: 11px;
  --font-size-sm: 13px;
  --font-size-md: 15px;
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
  --shadow-soft: 0 8px 24px rgba(0, 0, 0, 0.12);
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

body {
  display: flex;
  flex-direction: column;
}

a { color: var(--accent); text-decoration: none; }

a:hover { color: var(--accent-strong); }

button {
  font-family: var(--font-body);
  font-size: var(--font-size-sm);
  border: 1px solid var(--border);
  background: var(--panel);
  color: var(--text);
  padding: var(--space-2) var(--space-3);
  border-radius: var(--radius);
  cursor: pointer;
  transition: background var(--transition), border-color var(--transition), color var(--transition), transform var(--transition);
}

button:hover { background: var(--surface-hover); border-color: var(--border-strong); }
button:active { background: var(--surface-active); transform: translateY(1px); }

input, select {
  background: var(--panel);
  border: 1px solid var(--border);
  color: var(--text);
  padding: var(--space-2);
  border-radius: var(--radius);
  font-size: var(--font-size-sm);
  outline: none;
  transition: border-color var(--transition), box-shadow var(--transition);
}

input:focus, select:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 1px rgba(77, 163, 255, 0.35);
}

.app-shell {
  display: grid;
  grid-template-rows: 48px 1fr 200px;
  grid-template-columns: 280px 1fr 320px;
  grid-template-areas:
    "topbar topbar topbar"
    "sidebar main right"
    "bottombar bottombar bottombar";
  min-height: 100vh;
  background: var(--bg);
  gap: var(--space-2);
  padding: var(--space-2);
}

.app-shell.mobile {
  grid-template-columns: 1fr;
  grid-template-rows: 48px 1fr 220px;
  grid-template-areas:
    "topbar"
    "main"
    "bottombar";
}

.panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow-soft);
}

.topbar { grid-area: topbar; display: flex; align-items: center; gap: var(--space-2); padding: 0 var(--space-3); }
.sidebar { grid-area: sidebar; display: flex; flex-direction: column; gap: var(--space-2); padding: var(--space-3); }
.main { grid-area: main; position: relative; display: flex; flex-direction: column; gap: var(--space-2); }
.rightbar { grid-area: right; display: flex; flex-direction: column; gap: var(--space-2); padding: var(--space-3); }
.bottombar { grid-area: bottombar; display: flex; flex-direction: column; gap: var(--space-2); padding: var(--space-3); }

.section-label { font-size: var(--font-size-xs); color: var(--text-muted); letter-spacing: 0.04em; text-transform: uppercase; }

.flex-row { display: flex; gap: var(--space-2); align-items: center; }
.flex-col { display: flex; flex-direction: column; gap: var(--space-2); }
.flex-between { display: flex; justify-content: space-between; align-items: center; }

.chip {
  padding: var(--space-1) var(--space-2);
  border-radius: var(--radius-pill);
  background: var(--surface-hover);
  border: 1px solid var(--border);
  font-size: var(--font-size-xs);
  color: var(--text-dim);
}

.chart-grid {
  display: grid;
  width: 100%;
  height: 100%;
  gap: var(--space-2);
}

.chart-cell {
  position: relative;
  min-height: 240px;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
}

.chart-header {
  padding: var(--space-2) var(--space-3);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: var(--space-2);
  justify-content: space-between;
}

.chart-canvas {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
}

.watchlist { gap: var(--space-2); display: flex; flex-direction: column; }
.watchlist-item { display: flex; justify-content: space-between; align-items: center; padding: var(--space-2) var(--space-3); border: 1px solid var(--border); border-radius: var(--radius); background: var(--bg-elev-1); }
.watchlist-item:hover { background: var(--surface-hover); }

.order-ticket, .indicator-panel, .alerts-panel { display: flex; flex-direction: column; gap: var(--space-2); }

.lab-panel { padding: var(--space-3); border: 1px dashed var(--border); }
.lab-panel progress { width: 100%; height: 8px; }
.regime-bar { height: 10px; border-radius: var(--radius); opacity: 0.9; }

@media (max-width: 1100px) {
  .app-shell { grid-template-columns: 1fr; grid-template-rows: 48px 1fr 200px; grid-template-areas: "topbar" "main" "bottombar"; }
  .sidebar, .rightbar { display: none; }
}

@media (max-width: 640px) {
  .app-shell { padding: var(--space-1); gap: var(--space-1); }
  .chart-cell { min-height: 200px; }
}
"#;
