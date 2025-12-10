use crate::state::use_app_ctx;
#[cfg(target_arch = "wasm32")]
use app_shell::{AlertState, DrawingKind, DrawingState, OrderState};
#[cfg(not(target_arch = "wasm32"))]
use app_shell::{DrawingKind, DrawingState};
#[cfg(target_arch = "wasm32")]
use crate::state::{use_link_bus, LinkCrosshair, LinkView};
use leptos::*;
use ta_engine::{IndicatorConfig, IndicatorKind, IndicatorParams, OutputKind, SourceField};

#[cfg(target_arch = "wasm32")]
use chart_frontend::ChartHandle;
#[cfg(target_arch = "wasm32")]
use std::rc::Rc;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use serde_json;
#[cfg(target_arch = "wasm32")]
use serde::Deserialize;
#[cfg(target_arch = "wasm32")]
use js_sys::Date;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
use gloo_timers::future::TimeoutFuture;
#[cfg(not(target_arch = "wasm32"))]
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(target_arch = "wasm32")]
type HandleSignal = RwSignal<Option<Rc<ChartHandle>>>;
#[cfg(not(target_arch = "wasm32"))]
type HandleSignal = ();

#[cfg(target_arch = "wasm32")]
fn now_ms() -> i64 {
    Date::now() as i64
}

#[cfg(not(target_arch = "wasm32"))]
fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(target_arch = "wasm32")]
fn apply_indicators(handle: &ChartHandle, indicators: &[ta_engine::IndicatorConfig]) {
    handle.clear_indicators();
    for cfg in indicators {
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
fn apply_orders(handle: &ChartHandle, orders: &[OrderState], positions: &[OrderState], alerts: &[AlertState]) {
    if let Ok(json) = serde_json::to_string(orders) {
        let _ = handle.set_orders(&json);
    }
    if let Ok(json) = serde_json::to_string(positions) {
        let _ = handle.set_positions(&json);
    }
    if let Ok(json) = serde_json::to_string(alerts) {
        let _ = handle.set_alerts(&json);
    }
}

#[cfg(target_arch = "wasm32")]
#[derive(Deserialize)]
#[serde(tag = "type")]
enum ChartEventPayload {
    CrosshairMove { ts: i64, price: f64, x: f64, y: f64 },
    ViewChanged { start: i64, end: i64 },
    Click { x: f64, y: f64, ts: i64, price: f64, button: i16 },
}

#[component]
pub fn ChartView(
    chart_id: u32,
    http_base: String,
    ws_base: String,
) -> impl IntoView {
    #[cfg(not(target_arch = "wasm32"))]
    let _ = (&http_base, &ws_base);
    let canvas_id = format!("chart-canvas-{chart_id}");
    let _node_ref = create_node_ref::<leptos::html::Canvas>();
    #[cfg(target_arch = "wasm32")]
    let bus = use_link_bus();
    #[cfg(not(target_arch = "wasm32"))]
    let _bus = ();
    let app_ctx = use_app_ctx();
    let (symbol, set_symbol) = create_signal(String::from("…"));
    let (timeframe, set_timeframe) = create_signal(String::from("…"));
    #[cfg(not(target_arch = "wasm32"))]
    let _ = (&set_symbol, &set_timeframe);

    #[cfg(target_arch = "wasm32")]
    let handle = create_rw_signal::<Option<Rc<ChartHandle>>>(None);
    #[cfg(not(target_arch = "wasm32"))]
    let _handle: HandleSignal = ();
    #[cfg(target_arch = "wasm32")]
    let last_cross = create_rw_signal::<Option<i64>>(None);
    #[cfg(target_arch = "wasm32")]
    let last_view = create_rw_signal::<Option<(i64, i64)>>(None);
    #[cfg(not(target_arch = "wasm32"))]
    let handle: HandleSignal = ();

    #[cfg(target_arch = "wasm32")]
    {
        let canvas_id_clone = canvas_id.clone();
        let http = http_base.clone();
        let ws = ws_base.clone();
        let bus_cross = bus.crosshair;
        let bus_view = bus.view;
        let store_links = app_ctx.store;

        spawn_local(async move {
            TimeoutFuture::new(0).await;
            console_error_panic_hook::set_once();
            let id = canvas_id_clone.clone();
            let store = app_ctx.store;
            let http = http.clone();
            let ws = ws.clone();
            let bus_cross = bus_cross.clone();
            let bus_view = bus_view.clone();
            let link_group = store.with(|s| {
                s.state()
                    .layouts
                    .iter()
                    .flat_map(|l| l.charts.iter())
                    .find(|c| c.id == chart_id)
                    .and_then(|c| c.link_group.clone())
            });
            let initial = store.with(|s| {
                s.state()
                    .layouts
                    .iter()
                    .flat_map(|l| l.charts.iter())
                    .find(|c| c.id == chart_id)
                    .cloned()
            });
            let Some(chart_state) = initial else { return; };
            set_symbol.set(chart_state.symbol.clone());
            set_timeframe.set(chart_state.timeframe.clone());
            let symbol = chart_state.symbol.clone();
            let tf = chart_state.timeframe.clone();
            let chart = ChartHandle::new(&id, &symbol, &tf, &http, &ws);
            if let Ok(h) = chart {
                h.set_symbol(&symbol);
                let _ = h.set_timeframe(&tf);
                apply_indicators(&h, &chart_state.indicators);
                apply_drawings(&h, &chart_state.drawings);
                apply_orders(&h, &chart_state.orders, &chart_state.positions, &chart_state.alerts);
                let cb_bus_cross = bus_cross.clone();
                let cb_bus_view = bus_view.clone();
                let cb_link_group = link_group.clone();
                let cb_chart_id = chart_id;
                let cb_last_cross = last_cross;
                let cb_last_view = last_view;
                let callback = Closure::<dyn FnMut(JsValue)>::wrap(Box::new(move |val: JsValue| {
                    if let Some(txt) = val.as_string() {
                        if let Ok(ev) = serde_json::from_str::<ChartEventPayload>(&txt) {
                            match ev {
                                ChartEventPayload::CrosshairMove { ts, price, .. } => {
                                    let last = cb_last_cross.get_untracked();
                                    if last == Some(ts) {
                                        return;
                                    }
                                    cb_last_cross.set(Some(ts));
                                    cb_bus_cross.set(Some(LinkCrosshair {
                                        chart_id: cb_chart_id,
                                        link_group: cb_link_group.clone(),
                                        ts,
                                        price,
                                    }));
                                }
                                ChartEventPayload::ViewChanged { start, end } => {
                                    let last = cb_last_view.get_untracked();
                                    if last == Some((start, end)) {
                                        return;
                                    }
                                    cb_last_view.set(Some((start, end)));
                                    cb_bus_view.set(Some(LinkView {
                                        chart_id: cb_chart_id,
                                        link_group: cb_link_group.clone(),
                                        start,
                                        end,
                                    }));
                                }
                                _ => {}
                            }
                        }
                    }
                }));
                let _ = h.subscribe_events(callback.as_ref().unchecked_ref());
                callback.forget();
                handle.set(Some(Rc::new(h)));
            }
        });

        // React to link bus crosshair updates.
        {
            let handle = handle.clone();
            let bus_cross = bus.crosshair;
            let this_chart = chart_id;
            create_effect(move |_| {
                let link_group = store_links.with(|s| {
                    s.state()
                        .layouts
                        .iter()
                        .flat_map(|l| l.charts.iter())
                        .find(|c| c.id == this_chart)
                        .and_then(|c| c.link_group.clone())
                });
                if let Some(ev) = bus_cross.get() {
                    if ev.chart_id == this_chart {
                        return;
                    }
                    if ev.link_group.is_some() && ev.link_group == link_group {
                        if let Some(h) = handle.get() {
                            h.show_crosshair(ev.ts, ev.price);
                        }
                    }
                }
            });
        }

        // React to linked view changes.
        {
            let handle = handle.clone();
            let bus_view = bus.view;
            let this_chart = chart_id;
            create_effect(move |_| {
                let link_group = store_links.with(|s| {
                    s.state()
                        .layouts
                        .iter()
                        .flat_map(|l| l.charts.iter())
                        .find(|c| c.id == this_chart)
                        .and_then(|c| c.link_group.clone())
                });
                if let Some(ev) = bus_view.get() {
                    if ev.chart_id == this_chart {
                        return;
                    }
                    if ev.link_group.is_some() && ev.link_group == link_group {
                        if let Some(h) = handle.get() {
                            let _ = h.sync_view(ev.start, ev.end);
                        }
                    }
                }
            });
        }

        // Sync handle when chart state changes.
        {
            let handle = handle.clone();
            let store = app_ctx.store;
            create_effect(move |_| {
                let maybe_chart = store.with(|s| {
                    s.state()
                        .layouts
                        .iter()
                        .flat_map(|l| l.charts.iter())
                        .find(|c| c.id == chart_id)
                        .cloned()
                });
                if let (Some(h), Some(chart)) = (handle.get(), maybe_chart) {
                    set_symbol.set(chart.symbol.clone());
                    set_timeframe.set(chart.timeframe.clone());
                    let _ = h.set_timeframe(&chart.timeframe);
                    h.set_symbol(&chart.symbol);
                    apply_indicators(&h, &chart.indicators);
                    apply_drawings(&h, &chart.drawings);
                    apply_orders(&h, &chart.orders, &chart.positions, &chart.alerts);
                }
            });
        }

        on_cleanup(move || {
            if let Some(h) = handle.get_untracked() {
                h.destroy();
            }
        });
    }

    let add_indicator = {
        let store = app_ctx.store;
        #[cfg(target_arch = "wasm32")]
        let handle = handle.clone();
        move |_| {
            let cfg = IndicatorConfig::with_default_styles(
                IndicatorKind::Sma,
                IndicatorParams::Sma {
                    period: 20,
                    source: SourceField::Close,
                },
                OutputKind::Overlay,
                None,
            );
            store.update(|s| {
                if let Some(chart) = s
                    .state_mut()
                    .layouts
                    .iter_mut()
                    .flat_map(|l| l.charts.iter_mut())
                    .find(|c| c.id == chart_id)
                {
                    chart.indicators.push(cfg.clone());
                }
            });
            #[cfg(target_arch = "wasm32")]
            if let Some(h) = handle.get() {
                if let Ok(json) = serde_json::to_string(&cfg) {
                    let _ = h.add_indicator_from_config(&json);
                }
            }
        }
    };

    let add_hline = {
        let store = app_ctx.store;
        #[cfg(target_arch = "wasm32")]
        let handle = handle.clone();
        move |_| {
            let price = 100.0;
            #[cfg(target_arch = "wasm32")]
            let new_id = handle
                .get()
                .map(|h| h.add_horizontal_line(price, "#4da3ff", 1.5) as u64);
            #[cfg(not(target_arch = "wasm32"))]
            let new_id: Option<u64> = None;

            store.update(|s| {
                if let Some(chart) = s
                    .state_mut()
                    .layouts
                    .iter_mut()
                    .flat_map(|l| l.charts.iter_mut())
                    .find(|c| c.id == chart_id)
                {
                    let id = if let Some(id) = new_id {
                        id
                    } else {
                        chart.drawings.len() as u64 + 1
                    };
                    chart.drawings.push(DrawingState {
                        id,
                        kind: DrawingKind::HorizontalLine,
                        ts1: 0,
                        price1: price,
                        ts2: None,
                        price2: None,
                        color: "#4da3ff".into(),
                        width: 1.5,
                    });
                }
            });
        }
    };

    let add_vline = {
        let store = app_ctx.store;
        #[cfg(target_arch = "wasm32")]
        let handle = handle.clone();
        move |_| {
            let ts = 0_i64;
            #[cfg(target_arch = "wasm32")]
            let new_id = handle
                .get()
                .map(|h| h.add_vertical_line(ts, "#69c0ff", 1.0) as u64);
            #[cfg(not(target_arch = "wasm32"))]
            let new_id: Option<u64> = None;

            store.update(|s| {
                if let Some(chart) = s
                    .state_mut()
                    .layouts
                    .iter_mut()
                    .flat_map(|l| l.charts.iter_mut())
                    .find(|c| c.id == chart_id)
                {
                    let id = if let Some(id) = new_id {
                        id
                    } else {
                        chart.drawings.len() as u64 + 1
                    };
                    chart.drawings.push(DrawingState {
                        id,
                        kind: DrawingKind::VerticalLine,
                        ts1: ts,
                        price1: 0.0,
                        ts2: None,
                        price2: None,
                        color: "#69c0ff".into(),
                        width: 1.0,
                    });
                }
            });
        }
    };

    view! {
        <div class="chart-cell">
            <div class="chart-header">
                <div class="flex-row">
                    <span class="chip">{move || symbol.get()}</span>
                    <span class="chip">{move || timeframe.get()}</span>
                </div>
                <div class="flex-row" style="gap: 6px;">
                    <button aria-label="Add indicator" on:click=add_indicator>+Ind</button>
                    <button aria-label="Add H-line" on:click=add_hline>H</button>
                    <button aria-label="Add V-line" on:click=add_vline>V</button>
                    <button aria-label="Undo" on:click=move |_| { #[cfg(target_arch = "wasm32")] if let Some(h)=handle.get(){ h.undo(); } }>Undo</button>
                    <button aria-label="Redo" on:click=move |_| { #[cfg(target_arch = "wasm32")] if let Some(h)=handle.get(){ h.redo(); } }>Redo</button>
                </div>
            </div>
            <canvas _ref=_node_ref id=canvas_id class="chart-canvas"></canvas>
            <OrderForm chart_id=chart_id handle=handle_signal(handle) />
        </div>
    }
}

#[cfg(target_arch = "wasm32")]
fn handle_signal(sig: HandleSignal) -> HandleSignal {
    sig
}

#[cfg(not(target_arch = "wasm32"))]
fn handle_signal(_: HandleSignal) -> HandleSignal {
}

#[component]
fn OrderForm(chart_id: u32, handle: HandleSignal) -> impl IntoView {
    let app_ctx = use_app_ctx();
    let (price, set_price) = create_signal(100.0_f64);
    let (qty, set_qty) = create_signal(1.0_f64);
    let (label, set_label) = create_signal(String::from("Order"));
    let (alert_price, set_alert_price) = create_signal(100.0_f64);
    #[cfg(not(target_arch = "wasm32"))]
    let _ = &handle;

    #[cfg(target_arch = "wasm32")]
    let flush_handle = {
        let store = app_ctx.store;
        move || {
            if let Some(h) = handle.get() {
                if let Some(chart) = store.with(|s| {
                    s.state()
                        .layouts
                        .iter()
                        .flat_map(|l| l.charts.iter())
                        .find(|c| c.id == chart_id)
                        .cloned()
                }) {
                    apply_orders(&h, &chart.orders, &chart.positions, &chart.alerts);
                }
            }
        }
    };
    #[cfg(not(target_arch = "wasm32"))]
    let flush_handle = || {};

    let add_order = move |_| {
        let now = now_ms();
        let price = price.get();
        let qty = qty.get();
        let label_val = label.get();
        app_ctx.store.update(|s| {
            if let Some(chart) = s
                .state_mut()
                .layouts
                .iter_mut()
                .flat_map(|l| l.charts.iter_mut())
                .find(|c| c.id == chart_id)
            {
                chart.orders.push(app_shell::OrderState {
                    id: format!("ord-{now}"),
                    side: "buy".into(),
                    price,
                    qty,
                    label: label_val.clone(),
                    stop_price: None,
                    take_profit_price: None,
                });
            }
        });
        flush_handle();
    };

    let add_position = move |_| {
        let now = now_ms();
        let price = price.get();
        let qty = qty.get();
        let label_val = label.get();
        app_ctx.store.update(|s| {
            if let Some(chart) = s
                .state_mut()
                .layouts
                .iter_mut()
                .flat_map(|l| l.charts.iter_mut())
                .find(|c| c.id == chart_id)
            {
                chart.positions.push(app_shell::OrderState {
                    id: format!("pos-{now}"),
                    side: "long".into(),
                    price,
                    qty,
                    label: label_val.clone(),
                    stop_price: None,
                    take_profit_price: None,
                });
            }
        });
        flush_handle();
    };

    let add_alert = move |_| {
        let now = now_ms();
        let price_val = alert_price.get();
        let label_val = label.get();
        app_ctx.store.update(|s| {
            if let Some(chart) = s
                .state_mut()
                .layouts
                .iter_mut()
                .flat_map(|l| l.charts.iter_mut())
                .find(|c| c.id == chart_id)
            {
                chart.alerts.push(app_shell::AlertState {
                    id: format!("al-{now}"),
                    ts: now,
                    price: Some(price_val),
                    label: label_val.clone(),
                    fired: false,
                });
            }
        });
        flush_handle();
    };

    view! {
        <div class="panel" style="padding: var(--space-2); border-top: 1px solid var(--border);">
            <div class="section-label">Trading</div>
            <div class="flex-row" style="gap:8px;">
                <input type="number" step="0.1" value=move || price.get() on:input=move |ev| {
                    if let Ok(v) = event_target_value(&ev).parse::<f64>() { set_price.set(v); }
                }/>
                <input type="number" step="0.1" value=move || qty.get() on:input=move |ev| {
                    if let Ok(v) = event_target_value(&ev).parse::<f64>() { set_qty.set(v); }
                }/>
                <input type="text" value=move || label.get() on:input=move |ev| set_label.set(event_target_value(&ev)) />
            </div>
            <div class="flex-row" style="gap:6px; margin-top:6px;">
                <button on:click=add_order>+ Order</button>
                <button on:click=add_position>+ Position</button>
            </div>
            <div class="flex-row" style="gap:8px; margin-top:6px;">
                <input type="number" step="0.1" value=move || alert_price.get() on:input=move |ev| {
                    if let Ok(v) = event_target_value(&ev).parse::<f64>() { set_alert_price.set(v); }
                }/>
                <button on:click=add_alert>+ Alert</button>
            </div>
        </div>
    }
}
