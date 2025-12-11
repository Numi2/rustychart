use app_shell::{AppState, ChartState, LayoutKind, LayoutState, StateStore, Theme};
use leptos::*;

#[derive(Clone)]
pub struct AppCtx {
    pub store: RwSignal<StateStore>,
    pub api_base: RwSignal<String>,
    pub ws_base: RwSignal<String>,
}

#[derive(Clone, Debug)]
pub struct LinkCrosshair {
    pub chart_id: u32,
    pub link_group: Option<String>,
    pub ts: i64,
    pub price: f64,
}

#[derive(Clone, Debug)]
pub struct LinkView {
    pub chart_id: u32,
    pub link_group: Option<String>,
    pub start: i64,
    pub end: i64,
}

#[derive(Clone)]
pub struct LinkBus {
    pub crosshair: RwSignal<Option<LinkCrosshair>>,
    pub view: RwSignal<Option<LinkView>>,
}

pub fn provide_app_ctx(api_base: String, ws_base: String) -> AppCtx {
    let default_state = AppState {
        theme: Theme::Dark,
        layouts: vec![LayoutState {
            id: 1,
            kind: LayoutKind::Single,
            charts: vec![ChartState {
                id: 1,
                // Default to a symbol supported by both Yahoo and Coinbase.
                symbol: "BTC-USD".into(),
                timeframe: "1m".into(),
                indicators: Vec::new(),
                drawings: Vec::new(),
                link_group: Some("A".into()),
                orders: Vec::new(),
                positions: Vec::new(),
                alerts: Vec::new(),
                price_scale_log: false,
                price_pane_weight: 1.0,
                pane_layout: Vec::new(),
                pane: 0,
                height_ratio: 1.0,
                inputs: Vec::new(),
            }],
        }],
        active_layout_id: Some(1),
    };

    #[cfg(target_arch = "wasm32")]
    let initial_state = app_shell::load_state_from_local_storage("rustychart-state")
        .ok()
        .flatten()
        .unwrap_or(default_state);
    #[cfg(not(target_arch = "wasm32"))]
    let initial_state = default_state;

    let store = create_rw_signal(StateStore::new(initial_state));
    let api = create_rw_signal(api_base);
    let ws = create_rw_signal(ws_base);
    let ctx = AppCtx {
        store,
        api_base: api,
        ws_base: ws,
    };
    provide_context(ctx.clone());
    ctx
}

pub fn provide_link_bus() -> LinkBus {
    let bus = LinkBus {
        crosshair: create_rw_signal(None),
        view: create_rw_signal(None),
    };
    provide_context(bus.clone());
    bus
}

pub fn use_app_ctx() -> AppCtx {
    use_context::<AppCtx>().expect("AppCtx not provided")
}

pub fn use_link_bus() -> LinkBus {
    use_context::<LinkBus>().expect("LinkBus not provided")
}
