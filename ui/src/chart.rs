use crate::state::use_app_ctx;
#[cfg(target_arch = "wasm32")]
use crate::state::{use_link_bus, LinkCrosshair, LinkView};
use app_shell::{ChartInput, InputKind, PaneConfig};
#[cfg(target_arch = "wasm32")]
use app_shell::{DrawingKind, DrawingState};
#[cfg(not(target_arch = "wasm32"))]
use app_shell::{DrawingKind, DrawingState};
use leptos::{event_target_checked, event_target_value, *};
use ta_engine::{IndicatorConfig, IndicatorKind, IndicatorParams, OutputKind, SourceField};

#[cfg(target_arch = "wasm32")]
use chart_frontend::ChartHandle;
#[cfg(target_arch = "wasm32")]
use gloo_timers::future::TimeoutFuture;
#[cfg(target_arch = "wasm32")]
use serde::Deserialize;
#[cfg(target_arch = "wasm32")]
use serde_json;
#[cfg(target_arch = "wasm32")]
use std::rc::Rc;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

#[cfg(target_arch = "wasm32")]
type HandleSignal = RwSignal<Option<Rc<ChartHandle>>>;
#[cfg(not(target_arch = "wasm32"))]
type HandleSignal = ();

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
fn apply_panes(handle: &ChartHandle, chart: &app_shell::ChartState) {
    let pane_ids: Vec<u32> = chart.pane_layout.iter().map(|p| p.id).collect();
    let pane_weights: Vec<f64> = chart.pane_layout.iter().map(|p| p.weight as f64).collect();
    handle.set_pane_layout(chart.price_pane_weight as f64, pane_ids, pane_weights);
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
#[derive(Deserialize)]
#[serde(tag = "type")]
#[allow(dead_code)]
enum ChartEventPayload {
    CrosshairMove {
        ts: i64,
        price: f64,
        x: f64,
        y: f64,
    },
    ViewChanged {
        start: i64,
        end: i64,
    },
    Click {
        x: f64,
        y: f64,
        ts: i64,
        price: f64,
        button: i16,
    },
}

#[component]
pub fn ChartView(chart_id: u32, http_base: String, ws_base: String) -> impl IntoView {
    #[cfg(not(target_arch = "wasm32"))]
    let _ = (&http_base, &ws_base);
    let canvas_id = format!("chart-canvas-{chart_id}");
    let _node_ref = create_node_ref::<leptos::html::Canvas>();
    #[cfg(target_arch = "wasm32")]
    let bus = use_link_bus();
    #[cfg(not(target_arch = "wasm32"))]
    let _bus = ();
    let app_ctx = use_app_ctx();
    let (symbol, set_symbol) = create_signal(String::from("..."));
    let (timeframe, set_timeframe) = create_signal(String::from("..."));
    #[cfg(not(target_arch = "wasm32"))]
    let _ = (&set_symbol, &set_timeframe);

    #[cfg(target_arch = "wasm32")]
    let handle = create_rw_signal::<Option<Rc<ChartHandle>>>(None);
    #[cfg(not(target_arch = "wasm32"))]
    let _handle: HandleSignal = ();
    #[cfg(target_arch = "wasm32")]
    let last_cross = create_rw_signal::<Option<(i64, f64)>>(None);
    #[cfg(target_arch = "wasm32")]
    let last_view = create_rw_signal::<Option<(i64, i64)>>(None);

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
            let Some(chart_state) = initial else {
                return;
            };
            set_symbol.set(chart_state.symbol.clone());
            set_timeframe.set(chart_state.timeframe.clone());
            let symbol = chart_state.symbol.clone();
            let tf = chart_state.timeframe.clone();
            let chart = ChartHandle::new(&id, &symbol, &tf, &http, &ws);
            if let Ok(h) = chart {
                h.set_symbol(&symbol);
                let _ = h.set_timeframe(&tf);
                apply_panes(&h, &chart_state);
                apply_indicators(&h, &chart_state.indicators);
                apply_drawings(&h, &chart_state.drawings);
                let cb_bus_cross = bus_cross.clone();
                let cb_bus_view = bus_view.clone();
                let cb_link_group = link_group.clone();
                let cb_chart_id = chart_id;
                let cb_last_cross = last_cross;
                let cb_last_view = last_view;
                let callback =
                    Closure::<dyn FnMut(JsValue)>::wrap(Box::new(move |val: JsValue| {
                        if let Some(txt) = val.as_string() {
                            if let Ok(ev) = serde_json::from_str::<ChartEventPayload>(&txt) {
                                match ev {
                                    ChartEventPayload::CrosshairMove { ts, price, .. } => {
                                        let last = cb_last_cross.get_untracked();
                                        if last.map(|(t, _)| t) == Some(ts) {
                                            return;
                                        }
                                        cb_last_cross.set(Some((ts, price)));
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
                    apply_panes(&h, &chart);
                    apply_indicators(&h, &chart.indicators);
                    apply_drawings(&h, &chart.drawings);
                }
            });
        }

        on_cleanup(move || {
            if let Some(h) = handle.get_untracked() {
                h.destroy();
            }
        });
    }

    // Utilities to read/update this chart state.
    let update_chart = {
        let store = app_ctx.store;
        move |f: &mut dyn FnMut(&mut app_shell::ChartState)| {
            store.update(|s| {
                if let Some(chart) = s
                    .state_mut()
                    .layouts
                    .iter_mut()
                    .flat_map(|l| l.charts.iter_mut())
                    .find(|c| c.id == chart_id)
                {
                    f(chart);
                }
            });
        }
    };

    let pane_layout = {
        let store = app_ctx.store;
        create_memo(move |_| {
            store
                .with(|s| {
                    s.state()
                        .layouts
                        .iter()
                        .flat_map(|l| l.charts.iter())
                        .find(|c| c.id == chart_id)
                        .map(|c| c.pane_layout.clone())
                })
                .unwrap_or_default()
        })
    };
    let price_weight = {
        let store = app_ctx.store;
        create_memo(move |_| {
            store
                .with(|s| {
                    s.state()
                        .layouts
                        .iter()
                        .flat_map(|l| l.charts.iter())
                        .find(|c| c.id == chart_id)
                        .map(|c| c.price_pane_weight)
                })
                .unwrap_or(1.0)
        })
    };
    let input_specs = {
        let store = app_ctx.store;
        create_memo(move |_| {
            store
                .with(|s| {
                    s.state()
                        .layouts
                        .iter()
                        .flat_map(|l| l.charts.iter())
                        .find(|c| c.id == chart_id)
                        .map(|c| c.inputs.clone())
                })
                .unwrap_or_default()
        })
    };

    let (indicator_output, set_indicator_output) = create_signal(OutputKind::Overlay);
    let (indicator_pane, set_indicator_pane) = create_signal::<Option<u32>>(None);
    {
        let panes = pane_layout;
        let indicator_output = indicator_output;
        let indicator_pane = indicator_pane;
        let set_indicator_pane = set_indicator_pane;
        create_effect(move |_| {
            if indicator_output.get() == OutputKind::SeparatePane && indicator_pane.get().is_none()
            {
                set_indicator_pane.set(panes.get().first().map(|p| p.id));
            }
        });
    }

    let add_indicator = {
        #[cfg(target_arch = "wasm32")]
        let handle = handle.clone();
        let pane_layout = pane_layout.clone();
        let indicator_output = indicator_output;
        let indicator_pane = indicator_pane;
        move |_| {
            let target_output = indicator_output.get();
            let target_pane = if target_output == OutputKind::SeparatePane {
                indicator_pane
                    .get()
                    .or_else(|| pane_layout.get().first().map(|p| p.id))
            } else {
                None
            };
            let cfg = IndicatorConfig::with_default_styles(
                IndicatorKind::Sma,
                IndicatorParams::Sma {
                    period: 20,
                    source: SourceField::Close,
                },
                target_output,
                target_pane,
            );
            update_chart(&mut |chart| {
                chart.indicators.push(cfg.clone());
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
        #[cfg(target_arch = "wasm32")]
        let last_cross = last_cross.clone();
        move |_| {
            #[cfg(target_arch = "wasm32")]
            let price = last_cross.get().map(|(_, p)| p).unwrap_or(100.0);
            #[cfg(not(target_arch = "wasm32"))]
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
        #[cfg(target_arch = "wasm32")]
        let last_cross = last_cross.clone();
        move |_| {
            #[cfg(target_arch = "wasm32")]
            let ts = last_cross.get().map(|(t, _)| t).unwrap_or(0_i64);
            #[cfg(not(target_arch = "wasm32"))]
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

    let add_pane = {
        let store = app_ctx.store;
        let set_indicator_pane = set_indicator_pane;
        move |_| {
            let mut new_id: Option<u32> = None;
            store.update(|s| {
                if let Some(chart) = s
                    .state_mut()
                    .layouts
                    .iter_mut()
                    .flat_map(|l| l.charts.iter_mut())
                    .find(|c| c.id == chart_id)
                {
                    let next_id = chart.pane_layout.iter().map(|p| p.id).max().unwrap_or(0) + 1;
                    chart.pane_layout.push(PaneConfig {
                        id: next_id,
                        title: format!("Pane {}", next_id),
                        weight: 1.0,
                    });
                    new_id = Some(next_id);
                }
            });
            if let Some(id) = new_id {
                set_indicator_pane.set(Some(id));
            }
        }
    };

    let remove_pane = {
        let store = app_ctx.store;
        move |pane_id: u32| {
            store.update(|s| {
                if let Some(chart) = s
                    .state_mut()
                    .layouts
                    .iter_mut()
                    .flat_map(|l| l.charts.iter_mut())
                    .find(|c| c.id == chart_id)
                {
                    chart.pane_layout.retain(|p| p.id != pane_id);
                    for ind in chart.indicators.iter_mut() {
                        if ind.pane_id == Some(pane_id) {
                            ind.pane_id = None;
                            ind.output = OutputKind::Overlay;
                        }
                    }
                }
            });
        }
    };

    let set_pane_weight = {
        let store = app_ctx.store;
        move |pane_id: u32, weight: f64| {
            store.update(|s| {
                if let Some(chart) = s
                    .state_mut()
                    .layouts
                    .iter_mut()
                    .flat_map(|l| l.charts.iter_mut())
                    .find(|c| c.id == chart_id)
                {
                    if pane_id == 0 {
                        chart.price_pane_weight = weight as f32;
                    } else if let Some(pane) =
                        chart.pane_layout.iter_mut().find(|p| p.id == pane_id)
                    {
                        pane.weight = weight as f32;
                    }
                }
            });
        }
    };

    let update_input_value = {
        let store = app_ctx.store;
        move |input_id: String, value: String| {
            store.update(|s| {
                if let Some(chart) = s
                    .state_mut()
                    .layouts
                    .iter_mut()
                    .flat_map(|l| l.charts.iter_mut())
                    .find(|c| c.id == chart_id)
                {
                    if let Some(input) = chart.inputs.iter_mut().find(|i| i.id == input_id) {
                        input.value = value.clone();
                    }
                }
            });
        }
    };

    let (scale_mode, set_scale_mode) = create_signal("Linear".to_string());

    view! {
        <div class="chart-cell">
            <div class="chart-banner">
                <div class="pill-row">
                    <span class="pill pill-strong">{move || symbol.get()}</span>
                    <span class="pill pill-soft">{move || timeframe.get()}</span>
                </div>
                <div class="chart-tools">
                    <button class="btn ghost" aria-label="Undo" on:click=move |_| { #[cfg(target_arch = "wasm32")] if let Some(h)=handle.get(){ h.undo(); } }>Undo</button>
                    <button class="btn ghost" aria-label="Redo" on:click=move |_| { #[cfg(target_arch = "wasm32")] if let Some(h)=handle.get(){ h.redo(); } }>Redo</button>
                    <button class="btn ghost" aria-label="Add indicator" on:click=add_indicator>+Ind</button>
                    <button class="btn ghost" aria-label="Add H-line" on:click=add_hline>H</button>
                    <button class="btn ghost" aria-label="Add V-line" on:click=add_vline>V</button>
                    <button class="btn ghost" aria-label="Toggle scale" on:click=move |_| {
                        let next = if scale_mode.get() == "Linear" { "Log".to_string() } else { "Linear".to_string() };
                        let mode = next.clone();
                        set_scale_mode.set(next);
                        #[cfg(target_arch = "wasm32")]
                        if let Some(h) = handle.get() {
                            h.set_scale(&mode);
                        }
                    }>{move || format!("{} scale", scale_mode.get())}</button>
                </div>
                <div class="chart-layout-controls">
                    <div class="pane-row">
                        <span class="label">{"Price height"}</span>
                        <input
                            r#type="range"
                            min="0.2"
                            max="3.0"
                            step="0.1"
                            value=move || format!("{:.2}", price_weight.get())
                            on:input=move |ev| {
                                let v = event_target_value(&ev).parse::<f64>().unwrap_or(1.0);
                                set_pane_weight(0, v.clamp(0.2, 5.0));
                            }
                        />
                    </div>
                    <For
                        each=move || pane_layout.get()
                        key=|pane: &PaneConfig| pane.id
                        children=move |pane: PaneConfig| {
                            let pid = pane.id;
                            view! {
                                <div class="pane-row">
                                    <span class="label">{pane.title.clone()}</span>
                                    <input
                                        r#type="range"
                                        min="0.2"
                                        max="3.0"
                                        step="0.1"
                                        value=move || format!("{:.2}", pane.weight)
                                        on:input=move |ev| {
                                            let v = event_target_value(&ev).parse::<f64>().unwrap_or(1.0);
                                            set_pane_weight(pid, v.clamp(0.2, 5.0));
                                        }
                                    />
                                    <button class="btn ghost" on:click=move |_| remove_pane(pid)>{"Remove"}</button>
                                </div>
                            }
                        }
                    />
                    <div class="pane-row">
                        <button class="btn ghost" on:click=add_pane aria-label="Add pane">{"+ Pane"}</button>
                        <div class="pane-route">
                            <label>{"Indicator output"}</label>
                            <select
                                value=move || {
                                    if indicator_output.get() == OutputKind::SeparatePane {
                                        "pane".to_string()
                                    } else {
                                        "overlay".to_string()
                                    }
                                }
                                on:change=move |ev| {
                                    let v = event_target_value(&ev);
                                    if v == "pane" {
                                        set_indicator_output.set(OutputKind::SeparatePane);
                                    } else {
                                        set_indicator_output.set(OutputKind::Overlay);
                                    }
                                }
                            >
                                <option value="overlay">{"Overlay"}</option>
                                <option value="pane">{"Separate pane"}</option>
                            </select>
                            <select
                                disabled=move || indicator_output.get() != OutputKind::SeparatePane
                                value=move || indicator_pane.get().map(|p| p.to_string()).unwrap_or_else(|| "".into())
                                on:change=move |ev| {
                                    let v = event_target_value(&ev).parse::<u32>().ok();
                                    set_indicator_pane.set(v);
                                }
                            >
                                <option value="">{ "Select pane" }</option>
                                <For
                                    each=move || pane_layout.get()
                                    key=|pane: &PaneConfig| pane.id
                                    children=move |pane: PaneConfig| {
                                        view! { <option value=move || pane.id.to_string()>{pane.title.clone()}</option> }
                                    }
                                />
                            </select>
                        </div>
                    </div>
                    <div class="chart-inputs">
                        <For
                            each=move || input_specs.get()
                            key=|input: &ChartInput| input.id.clone()
                            children=move |input: ChartInput| {
                                let input_id = input.id.clone();
                                match input.kind {
                                    InputKind::Slider { min, max, step } => view! {
                                        <label class="input-row">
                                            <span>{input.label.clone()}</span>
                                            <input
                                                r#type="range"
                                                min=move || min.to_string()
                                                max=move || max.to_string()
                                                step=move || step.to_string()
                                                value=move || input.value.clone()
                                                on:input=move |ev| {
                                                    update_input_value(input_id.clone(), event_target_value(&ev));
                                                }
                                            />
                                        </label>
                                    },
                                    InputKind::Toggle => view! {
                                        <label class="input-row">
                                            <span>{input.label.clone()}</span>
                                            <input
                                                r#type="checkbox"
                                                checked=move || input.value == "true"
                                                on:input=move |ev| {
                                                    let checked = event_target_checked(&ev);
                                                    update_input_value(input_id.clone(), checked.to_string());
                                                }
                                            />
                                        </label>
                                    },
                                    InputKind::Select { options } => view! {
                                        <label class="input-row">
                                            <span>{input.label.clone()}</span>
                                            <select
                                                value=move || input.value.clone()
                                                on:change=move |ev| {
                                                    update_input_value(input_id.clone(), event_target_value(&ev));
                                                }
                                            >
                                                <For
                                                    each=move || options.clone()
                                                    key=|opt: &String| opt.clone()
                                                    children=move |opt: String| {
                                                        view! { <option value=opt.clone()>{opt}</option> }
                                                    }
                                                />
                                            </select>
                                        </label>
                                    },
                                }
                            }
                        />
                    </div>
                </div>
            </div>
            <div class="chart-surface">
                <canvas _ref=_node_ref id=canvas_id class="chart-canvas"></canvas>
            </div>
        </div>
    }
}
