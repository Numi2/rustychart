use serde::{Deserialize, Serialize};
use ta_engine::IndicatorConfig;

#[cfg(target_arch = "wasm32")]
use std::cell::RefCell;
#[cfg(target_arch = "wasm32")]
use std::collections::HashMap;
#[cfg(target_arch = "wasm32")]
use std::rc::Rc;
#[cfg(target_arch = "wasm32")]
use std::str::FromStr;

#[cfg(target_arch = "wasm32")]
use chart_frontend::ChartHandle;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::closure::Closure;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
use web_sys::{window, Document, HtmlCanvasElement, HtmlDivElement, HtmlElement, Storage};

/// Simple theme model, extensible if needed.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub enum Theme {
    #[default]
    Dark,
    Light,
    Custom {
        background: String,
        grid: String,
        text: String,
    },
}

/// Drawing kinds kept in the central state. This mirrors frontend drawing types.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DrawingKind {
    HorizontalLine,
    VerticalLine,
    TrendLine,
    Rectangle,
}

/// Drawing state persisted in layouts.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DrawingState {
    pub id: u64,
    pub kind: DrawingKind,
    // Normalized coordinates for portability.
    pub ts1: i64,
    pub price1: f64,
    pub ts2: Option<i64>,
    pub price2: Option<f64>,
    pub color: String,
    pub width: f64,
}

/// Trading overlays persisted in chart state.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct OrderState {
    pub id: String,
    pub side: String,
    pub price: f64,
    pub qty: f64,
    pub label: String,
    pub stop_price: Option<f64>,
    pub take_profit_price: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct AlertState {
    pub id: String,
    pub ts: i64,
    pub price: Option<f64>,
    pub label: String,
    pub fired: bool,
}

/// Per-chart configuration: symbol, timeframe, indicators, drawings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartState {
    pub id: u32,
    pub symbol: String,
    pub timeframe: String,
    pub indicators: Vec<IndicatorConfig>,
    pub drawings: Vec<DrawingState>,
    /// Charts in the same link_group mirror crosshair/zoom/symbol.
    pub link_group: Option<String>,
    pub orders: Vec<OrderState>,
    pub positions: Vec<OrderState>,
    pub alerts: Vec<AlertState>,
}

/// Layout type: single or grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayoutKind {
    Single,
    Grid { rows: u8, cols: u8 },
}

/// Layout = topology + charts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutState {
    pub id: u32,
    pub kind: LayoutKind,
    pub charts: Vec<ChartState>,
}

/// Global app state.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AppState {
    pub theme: Theme,
    pub layouts: Vec<LayoutState>,
    pub active_layout_id: Option<u32>,
}

/// Snapshot-based state store with undo/redo.
/// State is small vs time-series; cloning is cheap and predictable.
pub struct StateStore {
    state: AppState,
    undo_stack: Vec<AppState>,
    redo_stack: Vec<AppState>,
}

impl StateStore {
    pub fn new(initial: AppState) -> Self {
        Self {
            state: initial,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
        }
    }

    pub fn with_default() -> Self {
        Self::new(AppState::default())
    }

    pub fn state(&self) -> &AppState {
        &self.state
    }

    pub fn state_mut(&mut self) -> &mut AppState {
        &mut self.state
    }

    /// Mutate with automatic undo/redo snapshot.
    pub fn mutate<F: FnOnce(&mut AppState)>(&mut self, f: F) {
        self.undo_stack.push(self.state.clone());
        self.redo_stack.clear();
        f(&mut self.state);
    }

    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    pub fn undo(&mut self) {
        if let Some(prev) = self.undo_stack.pop() {
            let cur = std::mem::replace(&mut self.state, prev);
            self.redo_stack.push(cur);
        }
    }

    pub fn redo(&mut self) {
        if let Some(next) = self.redo_stack.pop() {
            let cur = std::mem::replace(&mut self.state, next);
            self.undo_stack.push(cur);
        }
    }
}

// ---------- Persistence: localStorage ---------------------------------------
#[cfg(target_arch = "wasm32")]
fn local_storage() -> Result<Storage, JsValue> {
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("no window"))?;
    let storage = window
        .local_storage()?
        .ok_or_else(|| JsValue::from_str("localStorage unavailable"))?;
    Ok(storage)
}

/// Save app state to localStorage as JSON.
#[cfg(target_arch = "wasm32")]
pub fn save_state_to_local_storage(key: &str, state: &AppState) -> Result<(), JsValue> {
    let storage = local_storage()?;
    let json = serde_json::to_string(state).map_err(|e| JsValue::from_str(&e.to_string()))?;
    storage.set_item(key, &json)?;
    Ok(())
}

/// Load app state from localStorage; returns Ok(None) if not found.
#[cfg(target_arch = "wasm32")]
pub fn load_state_from_local_storage(key: &str) -> Result<Option<AppState>, JsValue> {
    let storage = local_storage()?;
    let value = storage.get_item(key)?;
    if let Some(json) = value {
        let state: AppState =
            serde_json::from_str(&json).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Some(state))
    } else {
        Ok(None)
    }
}

// ---------- Persistence: backend (simple HTTP API) --------------------------

#[cfg(target_arch = "wasm32")]
pub async fn save_state_to_backend(
    api_base: &str,
    auth_token: Option<&str>,
    state: &AppState,
) -> Result<(), JsValue> {
    use gloo_net::http::Request;

    let url = format!("{}/layout/save", api_base.trim_end_matches('/'));
    let json = serde_json::to_string(state).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let mut req = Request::post(&url).header("Content-Type", "application/json");
    if let Some(token) = auth_token {
        req = req.header("Authorization", &format!("Bearer {}", token));
    }
    let resp = req
        .body(json)
        .map_err(|e| JsValue::from_str(&e.to_string()))?
        .send()
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    if !resp.ok() {
        return Err(JsValue::from_str("backend save failed"));
    }
    Ok(())
}

#[cfg(target_arch = "wasm32")]
pub async fn load_state_from_backend(
    api_base: &str,
    auth_token: Option<&str>,
) -> Result<Option<AppState>, JsValue> {
    use gloo_net::http::Request;

    let url = format!("{}/layout/load", api_base.trim_end_matches('/'));
    let mut req = Request::get(&url);
    if let Some(token) = auth_token {
        req = req.header("Authorization", &format!("Bearer {}", token));
    }
    let resp = req
        .send()
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    if resp.status() == 404 {
        return Ok(None);
    }
    if !resp.ok() {
        return Err(JsValue::from_str("backend load failed"));
    }
    let state: AppState =
        resp.json().await.map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(Some(state))
}

// ---------- UI shell (wasm) -------------------------------------------------

#[cfg(target_arch = "wasm32")]
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
enum ChartEvent {
    Click { x: f64, y: f64, ts: i64, price: f64, button: i16 },
    CrosshairMove { x: f64, y: f64, ts: i64, price: f64 },
    ViewChanged { start: i64, end: i64 },
}

#[cfg(target_arch = "wasm32")]
#[derive(Debug, Serialize)]
pub struct PerfScenario {
    pub name: &'static str,
    pub candles: usize,
    pub overlays: usize,
    pub panes: usize,
    pub charts: usize,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct UiShell {
    state_store: StateStore,
    charts: Rc<RefCell<HashMap<u32, ChartHandle>>>,
    root: HtmlElement,
    persist_key: String,
    http_base: String,
    ws_base: String,
    _last_persist_ts: f64,
}

#[cfg(target_arch = "wasm32")]
fn document() -> Result<Document, JsValue> {
    window()
        .and_then(|w| w.document())
        .ok_or_else(|| JsValue::from_str("no document"))
}

#[cfg(target_arch = "wasm32")]
fn create_canvas(container: &HtmlDivElement, id: &str) -> Result<HtmlCanvasElement, JsValue> {
    let doc = document()?;
    let canvas: HtmlCanvasElement = doc
        .create_element("canvas")?
        .dyn_into::<HtmlCanvasElement>()
        .map_err(|_| JsValue::from_str("not a canvas"))?;
    canvas.set_id(id);
    canvas
        .style()
        .set_property("width", "100%")
        .expect("style");
    canvas
        .style()
        .set_property("height", "100%")
        .expect("style");
    container
        .append_child(&canvas)
        .map_err(|_| JsValue::from_str("append canvas failed"))?;
    Ok(canvas)
}

#[cfg(target_arch = "wasm32")]
fn build_grid(root: &HtmlElement, rows: u8, cols: u8) -> Result<(), JsValue> {
    let style = root.style();
    style.set_property("display", "grid")?;
    style.set_property("grid-template-columns", &format!("repeat({}, 1fr)", cols))?;
    style.set_property("grid-template-rows", &format!("repeat({}, 1fr)", rows))?;
    style.set_property("gap", "6px")?;
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn clear_root(root: &HtmlElement) {
    let _ = root.set_inner_html("");
}

#[cfg(target_arch = "wasm32")]
fn resize_canvas_to_parent(canvas: &HtmlCanvasElement) -> Result<(), JsValue> {
    let rect = canvas
        .get_bounding_client_rect()
        .unchecked_into::<web_sys::DomRect>();
    let w = rect.width().max(1.0);
    let h = rect.height().max(1.0);
    canvas.set_width(w as u32);
    canvas.set_height(h as u32);
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn apply_indicator_configs(handle: &ChartHandle, configs: &[IndicatorConfig]) {
    handle.clear_indicators();
    for cfg in configs {
        if let Ok(json) = serde_json::to_string(cfg) {
            let _ = handle.add_indicator_from_config(&json);
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn apply_drawings(handle: &ChartHandle, drawings: &[DrawingState]) {
    handle.clear_drawings();
    for d in drawings {
        match d.kind {
            DrawingKind::HorizontalLine => {
                handle.add_horizontal_line(d.price1, &d.color, d.width);
            }
            DrawingKind::VerticalLine => {
                handle.add_vertical_line(d.ts1, &d.color, d.width);
            }
            DrawingKind::TrendLine => {
                if let (Some(ts2), Some(price2)) = (d.ts2, d.price2) {
                    handle.add_trend_line(d.ts1, d.price1, ts2, price2, &d.color, d.width);
                }
            }
            DrawingKind::Rectangle => {
                if let (Some(ts2), Some(price2)) = (d.ts2, d.price2) {
                    handle.add_rectangle(d.ts1, d.price1, ts2, price2, &d.color, d.width);
                }
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn apply_orders(handle: &ChartHandle, orders: &[OrderState]) {
    if let Ok(json) = serde_json::to_string(orders) {
        let _ = handle.set_orders(&json);
    }
}

#[cfg(target_arch = "wasm32")]
fn apply_positions(handle: &ChartHandle, positions: &[OrderState]) {
    if let Ok(json) = serde_json::to_string(positions) {
        let _ = handle.set_positions(&json);
    }
}

#[cfg(target_arch = "wasm32")]
fn apply_alerts(handle: &ChartHandle, alerts: &[AlertState]) {
    if let Ok(json) = serde_json::to_string(alerts) {
        let _ = handle.set_alerts(&json);
    }
}

#[cfg(target_arch = "wasm32")]
impl UiShell {
    fn persist_state(&mut self) {
        if let Err(err) = save_state_to_local_storage(&self.persist_key, self.state_store.state()) {
            web_sys::console::error_1(&err);
        }
    }

    fn subscribe_chart_events(&mut self, chart_id: u32, handle: &ChartHandle) {
        let groups: HashMap<u32, Option<String>> = self
            .state_store
            .state()
            .layouts
            .iter()
            .flat_map(|l| l.charts.iter().map(|c| (c.id, c.link_group.clone())))
            .collect();
        let link_group = groups.get(&chart_id).cloned().unwrap_or(None);
        let charts = self.charts.clone();
        let lg = link_group.clone();
        let cb = Closure::<dyn FnMut(JsValue)>::wrap(Box::new(move |val: JsValue| {
            if let Some(txt) = val.as_string() {
                if let Ok(ev) = serde_json::from_str::<ChartEvent>(&txt) {
                    match ev {
                        ChartEvent::CrosshairMove { ts, price, .. } => {
                            if let Some(group) = lg.clone() {
                                let map = charts.borrow();
                                for (other_id, other_handle) in map.iter() {
                                    if *other_id == chart_id {
                                        continue;
                                    }
                                    if groups
                                        .get(other_id)
                                        .and_then(|g| g.clone())
                                        == Some(group.clone())
                                    {
                                        other_handle.show_crosshair(ts, price);
                                    }
                                }
                            }
                        }
                        ChartEvent::ViewChanged { start, end } => {
                            if let Some(group) = lg.clone() {
                                let map = charts.borrow();
                                for (other_id, other_handle) in map.iter() {
                                    if *other_id == chart_id {
                                        continue;
                                    }
                                    if groups
                                        .get(other_id)
                                        .and_then(|g| g.clone())
                                        == Some(group.clone())
                                    {
                                        let _ = other_handle.sync_view(start, end);
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }));
        let _ = handle.subscribe_events(cb.as_ref().unchecked_ref());
        cb.forget();
    }

    fn rebuild_layout(&mut self) -> Result<(), JsValue> {
        clear_root(&self.root);
        let state = self.state_store.state().clone();
        let layout = state
            .active_layout_id
            .and_then(|id| state.layouts.iter().find(|l| l.id == id).cloned())
            .or_else(|| state.layouts.first().cloned())
            .unwrap_or(LayoutState {
                id: 1,
                kind: LayoutKind::Single,
                charts: vec![ChartState {
                    id: 1,
                    // Use Coinbase/Yahoo friendly default to ensure data loads.
                    symbol: "BTC-USD".to_string(),
                    timeframe: "1m".to_string(),
                    indicators: Vec::new(),
                    drawings: Vec::new(),
                    link_group: None,
                    orders: Vec::new(),
                    positions: Vec::new(),
                    alerts: Vec::new(),
                }],
            });

        let (rows, cols) = match layout.kind {
            LayoutKind::Single => (1, 1),
            LayoutKind::Grid { rows, cols } => (rows.max(1), cols.max(1)),
        };

        build_grid(&self.root, rows, cols)?;
        self.charts.borrow_mut().clear();

        for (idx, chart_state) in layout.charts.iter().enumerate() {
            let container: HtmlDivElement = document()?
                .create_element("div")?
                .dyn_into::<HtmlDivElement>()
                .map_err(|_| JsValue::from_str("div cast failed"))?;
            container
                .style()
                .set_property("position", "relative")
                .ok();
            container
                .style()
                .set_property("min-height", "200px")
                .ok();
            let canvas_id = format!("chart-canvas-{}", chart_state.id);
            let canvas = create_canvas(&container, &canvas_id)?;
            self.root.append_child(&container)?;
            resize_canvas_to_parent(&canvas)?;

            let handle = ChartHandle::new(
                &canvas_id,
                &chart_state.symbol,
                &chart_state.timeframe,
                &self.http_base,
                &self.ws_base,
            )?;

            handle.set_symbol(&chart_state.symbol);
            handle.set_timeframe(&chart_state.timeframe)?;
            apply_indicator_configs(&handle, &chart_state.indicators);
            apply_drawings(&handle, &chart_state.drawings);
            self.subscribe_chart_events(chart_state.id, &handle);
            {
                let mut charts = self.charts.borrow_mut();
                charts.insert(chart_state.id, handle);
            }

            let rect = canvas.get_bounding_client_rect();
            if let Some(h) = self.charts.borrow().get(&chart_state.id) {
                h.resize(rect.width(), rect.height());
            }
            let _ = idx; // suppress unused warning for idx; kept for future grid mapping
        }

        Ok(())
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl UiShell {
    #[wasm_bindgen(constructor)]
    pub fn new(
        root_id: &str,
        persist_key: &str,
        http_base: &str,
        ws_base: &str,
    ) -> Result<UiShell, JsValue> {
        let doc = document()?;
        let root_elem = doc
            .get_element_by_id(root_id)
            .ok_or_else(|| JsValue::from_str("root element not found"))?
            .dyn_into::<HtmlElement>()
            .map_err(|_| JsValue::from_str("root is not HTMLElement"))?;

        let initial = load_state_from_local_storage(persist_key)?
            .unwrap_or_else(AppState::default);

        let mut shell = UiShell {
            state_store: StateStore::new(initial),
            charts: Rc::new(RefCell::new(HashMap::new())),
            root: root_elem,
            persist_key: persist_key.to_string(),
            http_base: http_base.to_string(),
            ws_base: ws_base.to_string(),
            _last_persist_ts: 0.0,
        };

        shell.rebuild_layout()?;
        Ok(shell)
    }

    /// Replace the current AppState (e.g. after loading from backend) and rebuild charts.
    #[wasm_bindgen]
    pub fn set_state_json(&mut self, json: &str) -> Result<(), JsValue> {
        let state: AppState =
            serde_json::from_str(json).map_err(|e| JsValue::from_str(&e.to_string()))?;
        self.state_store = StateStore::new(state);
        self.persist_state();
        self.rebuild_layout()
    }

    /// Serialize AppState for host-side persistence.
    #[wasm_bindgen]
    pub fn state_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(self.state_store.state())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Predefined performance scenarios for benchmarking in hosts.
    #[wasm_bindgen]
    pub fn default_perf_scenarios_json() -> Result<String, JsValue> {
        let scenarios = vec![
            PerfScenario {
                name: "single-100k-5overlays",
                candles: 100_000,
                overlays: 5,
                panes: 1,
                charts: 1,
            },
            PerfScenario {
                name: "grid-2x2-50k",
                candles: 50_000,
                overlays: 3,
                panes: 3,
                charts: 4,
            },
            PerfScenario {
                name: "mobile-2pane-25k",
                candles: 25_000,
                overlays: 4,
                panes: 2,
                charts: 1,
            },
        ];
        serde_json::to_string(&scenarios).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Mutate a chart's symbol/timeframe and mirror to ChartHandle + state.
    #[wasm_bindgen]
    pub fn set_chart_symbol(&mut self, chart_id: u32, symbol: &str) -> Result<(), JsValue> {
        if let Some(chart) = self.state_store.state_mut().layouts.iter_mut().flat_map(|l| l.charts.iter_mut()).find(|c| c.id == chart_id) {
            chart.symbol = symbol.to_string();
        }
        if let Some(handle) = self.charts.borrow().get(&chart_id) {
            handle.set_symbol(symbol);
        }
        self.persist_state();
        Ok(())
    }

    #[wasm_bindgen]
    pub fn set_chart_timeframe(&mut self, chart_id: u32, timeframe: &str) -> Result<(), JsValue> {
        if let Some(chart) = self.state_store.state_mut().layouts.iter_mut().flat_map(|l| l.charts.iter_mut()).find(|c| c.id == chart_id) {
            chart.timeframe = timeframe.to_string();
        }
        if let Some(handle) = self.charts.borrow().get(&chart_id) {
            handle.set_timeframe(timeframe)?;
        }
        self.persist_state();
        Ok(())
    }

    /// Replace indicator configs for a chart and mirror to handle.
    #[wasm_bindgen]
    pub fn set_chart_indicators(&mut self, chart_id: u32, configs_json: &str) -> Result<(), JsValue> {
        let configs: Vec<IndicatorConfig> =
            serde_json::from_str(configs_json).map_err(|e| JsValue::from_str(&e.to_string()))?;
        if let Some(chart) = self.state_store.state_mut().layouts.iter_mut().flat_map(|l| l.charts.iter_mut()).find(|c| c.id == chart_id) {
            chart.indicators = configs.clone();
        }
        if let Some(handle) = self.charts.borrow().get(&chart_id) {
            apply_indicator_configs(handle, &configs);
        }
        self.persist_state();
        Ok(())
    }

    /// Replace drawings for a chart (used after undo/redo or external edit).
    #[wasm_bindgen]
    pub fn set_chart_drawings(&mut self, chart_id: u32, drawings_json: &str) -> Result<(), JsValue> {
        let drawings: Vec<DrawingState> =
            serde_json::from_str(drawings_json).map_err(|e| JsValue::from_str(&e.to_string()))?;
        if let Some(chart) = self.state_store.state_mut().layouts.iter_mut().flat_map(|l| l.charts.iter_mut()).find(|c| c.id == chart_id) {
            chart.drawings = drawings.clone();
        }
        if let Some(handle) = self.charts.borrow().get(&chart_id) {
            apply_drawings(handle, &drawings);
        }
        self.persist_state();
        Ok(())
    }

    /// Refresh DOM grid and ChartHandle instances to match current state.
    #[wasm_bindgen]
    pub fn rebuild(&mut self) -> Result<(), JsValue> {
        self.rebuild_layout()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn undo_redo_restores_state() {
        let mut store = StateStore::with_default();
        store.mutate(|s| s.layouts.push(LayoutState {
            id: 1,
            kind: LayoutKind::Single,
            charts: vec![ChartState {
                id: 1,
                symbol: "TEST".into(),
                timeframe: "1m".into(),
                indicators: Vec::new(),
                drawings: Vec::new(),
                link_group: None,
                orders: Vec::new(),
                positions: Vec::new(),
                alerts: Vec::new(),
            }],
        }));
        assert_eq!(store.state().layouts.len(), 1);
        store.undo();
        assert!(store.state().layouts.is_empty());
        store.redo();
        assert_eq!(store.state().layouts.len(), 1);
    }

    #[test]
    fn state_roundtrip() {
        let state = AppState {
            theme: Theme::Dark,
            layouts: vec![LayoutState {
                id: 7,
                kind: LayoutKind::Grid { rows: 2, cols: 2 },
                charts: Vec::new(),
            }],
            active_layout_id: Some(7),
        };
        let json = serde_json::to_string(&state).unwrap();
        let decoded: AppState = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.active_layout_id, Some(7));
        assert_eq!(decoded.layouts.len(), 1);
    }
}
