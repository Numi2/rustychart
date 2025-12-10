use crate::{
    chart::ChartView,
    lab::{DiagnosticsPanel, LabPanel},
    state::{provide_app_ctx, provide_link_bus},
    theme::GLOBAL_CSS,
};
use app_shell::{LayoutKind, Theme};
use std::str::FromStr;
use ta_engine::{IndicatorConfig, IndicatorKind, IndicatorParams, OutputKind, SourceField};
use leptos::*;
use leptos_meta::*;
use serde::Deserialize;

#[cfg(target_arch = "wasm32")]
use app_shell::StateStore;
#[cfg(target_arch = "wasm32")]
use gloo_net::http::Request;
#[cfg(target_arch = "wasm32")]
use gloo_timers::future::TimeoutFuture;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::spawn_local;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
use js_sys::{encode_uri_component, Reflect};

#[cfg(target_arch = "wasm32")]
fn read_global(key: &str) -> Option<String> {
    Reflect::get(&js_sys::global(), &JsValue::from_str(key))
        .ok()
        .and_then(|v| v.as_string())
}

fn api_base_default() -> String {
    #[cfg(target_arch = "wasm32")]
    {
        read_global("RUSTYCHART_API_BASE")
            .unwrap_or_else(|| "https://rustychart.fly.dev/api".to_string())
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        "/api".to_string()
    }
}

fn ws_base_default() -> String {
    #[cfg(target_arch = "wasm32")]
    {
        read_global("RUSTYCHART_WS_BASE")
            .unwrap_or_else(|| "wss://rustychart.fly.dev/api/ws".to_string())
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        "/api/ws".to_string()
    }
}

#[derive(Clone, Debug, Deserialize)]
struct SearchHit {
    symbol: String,
    name: Option<String>,
    exchange: Option<String>,
}

#[component]
pub fn App() -> impl IntoView {
    provide_meta_context();

    let ctx = provide_app_ctx(api_base_default(), ws_base_default());
    let _link_bus = provide_link_bus();

    // Watchlist (mutable, clickable)
    let (watchlist, set_watchlist) = create_signal::<Vec<String>>(vec![
        "BTC-USD".into(),
        "ETH-USD".into(),
        "ES=F".into(),
    ]);
    let (new_watch, set_new_watch) = create_signal(String::new());

    let theme_class = create_memo(move |_| {
        ctx.store.with(|s| match s.state().theme {
            Theme::Light => "app-shell light-theme".to_string(),
            _ => "app-shell".to_string(),
        })
    });

    let active_layout_id = create_memo(move |_| {
        ctx.store.with(|s| {
            s.state()
                .active_layout_id
                .or_else(|| s.state().layouts.first().map(|l| l.id))
        })
    });

    let active_timeframe = create_memo(move |_| {
        ctx.store.with(|s| {
            let layout_id = s
                .state()
                .active_layout_id
                .or_else(|| s.state().layouts.first().map(|l| l.id));
            layout_id
                .and_then(|id| {
                    s.state()
                        .layouts
                        .iter()
                        .find(|l| l.id == id)
                        .and_then(|l| l.charts.first())
                        .map(|c| c.timeframe.clone())
                })
                .unwrap_or_else(|| "1m".to_string())
        })
    });

    let toggle_theme = move |_| {
        ctx.store.update(|store| {
            let next = match store.state().theme {
                Theme::Light => Theme::Dark,
                _ => Theme::Light,
            };
            store.state_mut().theme = next;
        });
    };

    // Helpers to mutate the active chart.
    let update_active_chart = {
        let store = ctx.store;
        move |f: &mut dyn FnMut(&mut app_shell::ChartState)| {
            store.update(|s| {
                if let Some(id) = s
                    .state()
                    .active_layout_id
                    .or_else(|| s.state().layouts.first().map(|l| l.id))
                {
                    if let Some(layout) = s.state_mut().layouts.iter_mut().find(|l| l.id == id) {
                        if let Some(chart) = layout.charts.first_mut() {
                            f(chart);
                        }
                    }
                }
            });
        }
    };

    // Symbol search state
    let (search_query, set_search_query) = create_signal(String::new());
    let search_results = create_rw_signal::<Vec<SearchHit>>(Vec::new());

    #[cfg(target_arch = "wasm32")]
    {
        create_effect(move |_| {
            let q = search_query.get();
            if q.trim().len() < 2 {
                search_results.set(Vec::new());
                return;
            }
            let results = search_results.clone();
            let query = q.clone();
            spawn_local(async move {
                TimeoutFuture::new(200).await;
                let url = format!("/api/search?q={}", encode_uri_component(&query));
                if let Ok(resp) = Request::get(&url).send().await {
                    if let Ok(hits) = resp.json::<Vec<SearchHit>>().await {
                        results.set(hits);
                    }
                }
            });
        });
    }

    let apply_symbol = {
        let set_search_query = set_search_query;
        let search_results = search_results;
        move |symbol: String| {
            update_active_chart(&mut |chart| chart.symbol = symbol.clone());
            set_search_query.set(symbol);
            search_results.set(Vec::new());
        }
    };

    let apply_timeframe = {
        move |tf: String| {
            update_active_chart(&mut |chart| chart.timeframe = tf.clone());
        }
    };

    let set_layout_kind = {
        move |kind: LayoutKind| {
            ctx.store.update(|s| {
                if let Some(id) = s
                    .state()
                    .active_layout_id
                    .or_else(|| s.state().layouts.first().map(|l| l.id))
                {
                    if let Some(layout) = s.state_mut().layouts.iter_mut().find(|l| l.id == id) {
                        layout.kind = kind;
                    }
                }
            });
        }
    };

    // Indicator quick-add panel state.
    let (ind_kind, set_ind_kind) = create_signal(IndicatorKind::Sma);
    let (ind_period, set_ind_period) = create_signal(20usize);
    let (ind_fast, set_ind_fast) = create_signal(12usize);
    let (ind_slow, set_ind_slow) = create_signal(26usize);
    let (ind_signal, set_ind_signal) = create_signal(9usize);
    let (ind_stddev, set_ind_stddev) = create_signal(2.0_f64);

    let add_indicator_to_active = move || {
        let kind = ind_kind.get();
        let cfg = match kind {
            IndicatorKind::Sma => IndicatorConfig::with_default_styles(
                kind,
                IndicatorParams::Sma {
                    period: ind_period.get(),
                    source: SourceField::Close,
                },
                OutputKind::Overlay,
                None,
            ),
            IndicatorKind::Ema => IndicatorConfig::with_default_styles(
                kind,
                IndicatorParams::Ema {
                    period: ind_period.get(),
                    source: SourceField::Close,
                },
                OutputKind::Overlay,
                None,
            ),
            IndicatorKind::Rsi => IndicatorConfig::with_default_styles(
                kind,
                IndicatorParams::Rsi {
                    period: ind_period.get(),
                    source: SourceField::Close,
                },
                OutputKind::SeparatePane,
                Some(0),
            ),
            IndicatorKind::Macd => IndicatorConfig::with_default_styles(
                kind,
                IndicatorParams::Macd {
                    fast: ind_fast.get(),
                    slow: ind_slow.get(),
                    signal: ind_signal.get(),
                    source: SourceField::Close,
                },
                OutputKind::SeparatePane,
                Some(0),
            ),
            IndicatorKind::Bbands => IndicatorConfig::with_default_styles(
                kind,
                IndicatorParams::Bbands {
                    period: ind_period.get(),
                    stddev: ind_stddev.get(),
                    source: SourceField::Close,
                },
                OutputKind::Overlay,
                None,
            ),
            IndicatorKind::Atr => IndicatorConfig::with_default_styles(
                kind,
                IndicatorParams::Atr {
                    period: ind_period.get(),
                },
                OutputKind::SeparatePane,
                Some(0),
            ),
            IndicatorKind::Stoch => IndicatorConfig::with_default_styles(
                kind,
                IndicatorParams::Stoch {
                    k_period: ind_fast.get().max(3),
                    d_period: ind_signal.get().max(3),
                },
                OutputKind::SeparatePane,
                Some(0),
            ),
            IndicatorKind::Vwap => IndicatorConfig::with_default_styles(
                kind,
                IndicatorParams::Vwap { reset_each_day: true },
                OutputKind::Overlay,
                None,
            ),
            IndicatorKind::Cci => IndicatorConfig::with_default_styles(
                kind,
                IndicatorParams::Cci {
                    period: ind_period.get(),
                    source: SourceField::Hlc3,
                    constant: 0.015,
                },
                OutputKind::SeparatePane,
                Some(0),
            ),
            IndicatorKind::Vwmo => IndicatorConfig::with_default_styles(
                kind,
                IndicatorParams::Vwmo {
                    period: ind_period.get(),
                    source: SourceField::Close,
                },
                OutputKind::SeparatePane,
                Some(0),
            ),
        };

        update_active_chart(&mut |chart| chart.indicators.push(cfg));
    };

    let clear_indicators = move |_| {
        update_active_chart(&mut |chart| chart.indicators.clear());
    };

    #[cfg(target_arch = "wasm32")]
    {
        use gloo_timers::future::TimeoutFuture;
        use wasm_bindgen_futures::spawn_local;

        // Load persisted state: backend -> localStorage -> default.
        {
            let store = ctx.store;
            let api_base = ctx.api_base;
            spawn_local(async move {
                if let Ok(Some(state)) =
                    app_shell::load_state_from_backend(&api_base.get_untracked(), None).await
                {
                    store.set(StateStore::new(state));
                    return;
                }
                if let Ok(Some(state)) = app_shell::load_state_from_local_storage("rustychart-state")
                {
                    store.set(StateStore::new(state));
                }
            });
        }

        // Debounced persistence to localStorage + backend.
        {
            let store = ctx.store;
            let api_base = ctx.api_base;
            create_effect(move |_| {
                let snapshot = store.with(|s| s.state().clone());
                let api = api_base.get_untracked();
                spawn_local(async move {
                    TimeoutFuture::new(300).await;
                    let _ = app_shell::save_state_to_local_storage("rustychart-state", &snapshot);
                    let _ = app_shell::save_state_to_backend(&api, None, &snapshot).await;
                });
            });
        }
    }

    let layout_view = move || {
        if let Some(id) = active_layout_id.get() {
            let layout = ctx
                .store
                .with(|s| s.state().layouts.iter().find(|l| l.id == id).cloned());
            if layout.is_none() {
                return view! { <div class="chart-grid"></div> };
            }
            let layout = layout.unwrap();
            let (rows, cols) = match layout.kind {
                LayoutKind::Single => (1, 1),
                LayoutKind::Grid { rows, cols } => (rows.max(1) as usize, cols.max(1) as usize),
            };
            let style = format!(
                "grid-template-columns: repeat({cols}, 1fr); grid-template-rows: repeat({rows}, minmax(0, 1fr));"
            );
            let api = ctx.api_base.with(|s| s.clone());
            let ws = ctx.ws_base.with(|s| s.clone());
            return view! {
                <div class="chart-grid" style=style>
                    {layout.charts.into_iter().map(|chart| {
                        view! {
                            <ChartView chart_id=chart.id http_base=api.clone() ws_base=ws.clone() />
                        }
                    }).collect_view()}
                </div>
            };
        }
        view! { <div class="chart-grid"></div> }
    };

    view! {
        <Style>{GLOBAL_CSS}</Style>
        <main class=theme_class>
            <section class="panel topbar flex-between">
                <div class="flex-row" style="gap: 12px;">
                    <div class="flex-row" style="gap: 8px;">
                        <strong style="letter-spacing:0.04em;">RustyChart</strong>
                        <span class="chip">alpha</span>
                    </div>
                    <div style="position: relative;">
                        <input
                            type="text"
                            placeholder="Symbol"
                            style="min-width: 220px;"
                            value=move || search_query.get()
                            on:input=move |ev| set_search_query.set(event_target_value(&ev))
                            on:keydown=move |ev| {
                                if ev.key() == "Enter" {
                                    let val = search_query.get().trim().to_string();
                                    if !val.is_empty() {
                                        apply_symbol(val);
                                    }
                                }
                            }
                        />
                        {move || {
                            let hits = search_results.get();
                            if hits.is_empty() {
                                ().into_view()
                            } else {
                                view! {
                                    <div class="panel" style="position:absolute; top:36px; left:0; z-index:10; min-width:280px; padding:8px;">
                                        {hits.into_iter().map(|h| {
                                            let sym = h.symbol.clone();
                                            let sym_display = sym.clone();
                                            let name = h.name.clone().unwrap_or_default();
                                            let exch = h.exchange.clone().unwrap_or_default();
                                            view! {
                                                <div class="watchlist-item" style="cursor:pointer;" on:click=move |_| apply_symbol(sym.clone())>
                                                    <span>{sym_display}</span>
                                                    <span class="chip">{format!("{exch} {name}")}</span>
                                                </div>
                                            }.into_view()
                                        }).collect_view()}
                                    </div>
                                }.into_view()
                            }
                        }}
                    </div>
                    <select
                        value=move || active_timeframe.get()
                        on:change=move |ev| apply_timeframe(event_target_value(&ev))
                    >
                        <option value="1m">1m</option>
                        <option value="5m">5m</option>
                        <option value="15m">15m</option>
                        <option value="1h">1h</option>
                        <option value="4h">4h</option>
                        <option value="1d">1d</option>
                    </select>
                </div>
                <div class="flex-row" style="gap: 8px;">
                    <select aria-label="Layout presets" on:change=move |ev| {
                        match event_target_value(&ev).as_str() {
                            "single" => set_layout_kind(LayoutKind::Single),
                            "grid2x2" => set_layout_kind(LayoutKind::Grid { rows: 2, cols: 2 }),
                            _ => {}
                        }
                    }>
                        <option value="single">Single</option>
                        <option value="grid2x2">Grid 2x2</option>
                    </select>
                    <button aria-label="Theme toggle" on:click=toggle_theme>Theme</button>
                </div>
            </section>

            <section class="panel sidebar">
                <div class="section-label">Watchlist</div>
                <div class="watchlist">
                    {move || {
                        watchlist.get().into_iter().map(|sym| {
                            let display = sym.clone();
                            view! {
                                <div class="watchlist-item" style="cursor:pointer;" on:click=move |_| apply_symbol(display.clone())>
                                    <span>{display.clone()}</span>
                                </div>
                            }.into_view()
                        }).collect_view()
                    }}
                    <div class="flex-row" style="gap:6px; margin-top:8px;">
                        <input
                            type="text"
                            placeholder="Add symbol"
                            value=move || new_watch.get()
                            on:input=move |ev| set_new_watch.set(event_target_value(&ev))
                            on:keydown=move |ev| {
                                if ev.key() == "Enter" {
                                    let v = new_watch.get().trim().to_string();
                                    if !v.is_empty() {
                                        set_watchlist.update(|list| list.push(v.clone()));
                                        apply_symbol(v);
                                        set_new_watch.set(String::new());
                                    }
                                }
                            }
                        />
                        <button on:click=move |_| {
                            let v = new_watch.get().trim().to_string();
                            if !v.is_empty() {
                                set_watchlist.update(|list| list.push(v.clone()));
                                apply_symbol(v);
                                set_new_watch.set(String::new());
                            }
                        }>Add</button>
                    </div>
                </div>
            </section>

            <section class="panel main">
                {layout_view}
            </section>

            <section class="panel rightbar">
                <div class="section-label">Orders</div>
                <div class="order-ticket">
                    <div class="flex-row flex-between">
                        <span>Side</span>
                        <div class="flex-row" style="gap:4px;">
                            <button>Buy</button><button>Sell</button>
                        </div>
                    </div>
                    <div class="flex-col">
                        <label class="section-label">Qty</label>
                        <input type="number" value="1" />
                    </div>
                    <div class="flex-col">
                        <label class="section-label">Limit</label>
                        <input type="number" placeholder="Price" />
                    </div>
                    <button style="width:100%;">Submit</button>
                </div>

                <div class="section-label" style="margin-top: var(--space-3);">Indicators</div>
                <div class="indicator-panel">
                    <button>+ Add indicator</button>
                    <div class="chip">SMA 20</div>
                    <div class="chip">RSI 14</div>
                    <div class="panel" style="margin-top:8px; padding:8px;">
                        <div class="flex-col" style="gap:6px;">
                            <label class="section-label">Indicator</label>
                            <select on:change=move |ev| {
                                let val = event_target_value(&ev);
                                let kind = IndicatorKind::from_str(&val).unwrap_or(IndicatorKind::Sma);
                                set_ind_kind.set(kind);
                            }>
                                <option value="sma">SMA</option>
                                <option value="ema">EMA</option>
                                <option value="rsi">RSI</option>
                                <option value="macd">MACD</option>
                                <option value="bbands">BBands</option>
                                <option value="atr">ATR</option>
                                <option value="stoch">Stoch</option>
                                <option value="vwap">VWAP</option>
                                <option value="cci">CCI</option>
                                <option value="vwmo">VWMO</option>
                            </select>
                            <div class="flex-row" style="gap:6px;">
                                <input type="number" min="1" value=move || ind_period.get() on:input=move |ev| {
                                    if let Ok(v)=event_target_value(&ev).parse::<usize>() { set_ind_period.set(v.max(1)); }
                                } placeholder="period"/>
                                <input type="number" min="1" value=move || ind_fast.get() on:input=move |ev| {
                                    if let Ok(v)=event_target_value(&ev).parse::<usize>() { set_ind_fast.set(v.max(1)); }
                                } placeholder="fast/k"/>
                                <input type="number" min="1" value=move || ind_slow.get() on:input=move |ev| {
                                    if let Ok(v)=event_target_value(&ev).parse::<usize>() { set_ind_slow.set(v.max(1)); }
                                } placeholder="slow/d"/>
                            </div>
                            <div class="flex-row" style="gap:6px;">
                                <input type="number" step="0.1" value=move || ind_stddev.get() on:input=move |ev| {
                                    if let Ok(v)=event_target_value(&ev).parse::<f64>() { set_ind_stddev.set(v.max(0.1)); }
                                } placeholder="stddev"/>
                                <input type="number" min="1" value=move || ind_signal.get() on:input=move |ev| {
                                    if let Ok(v)=event_target_value(&ev).parse::<usize>() { set_ind_signal.set(v.max(1)); }
                                } placeholder="signal"/>
                            </div>
                            <div class="flex-row" style="gap:6px;">
                                <button on:click=move |_| add_indicator_to_active()>+ Add</button>
                                <button on:click=clear_indicators>Clear</button>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="section-label" style="margin-top: var(--space-3);">Alerts</div>
                <div class="alerts-panel">
                    <div class="chip">Breakout @ 42500</div>
                    <div class="chip">Cross MA</div>
                </div>
                <div class="section-label" style="margin-top: var(--space-3);">Lab</div>
                <LabPanel />
                <DiagnosticsPanel />
            </section>

            <section class="panel bottombar">
                <div class="section-label">Activity</div>
                <div class="flex-row flex-between">
                    <span>PnL</span>
                    <span class="chip">+0.84%</span>
                </div>
                <div class="flex-col">
                    <div class="watchlist-item">
                        <span>Order #123</span>
                        <span class="chip">Filled</span>
                    </div>
                    <div class="watchlist-item">
                        <span>Alert Triggered</span>
                        <span class="chip">09:32:10</span>
                    </div>
                </div>
                <div class="section-label" style="margin-top: var(--space-3);">Regime timeline</div>
                <div class="flex-row" style="gap:4px; width:100%;">
                    <div class="regime-bar" style="flex:2; background:#2e7d32;" title="Trend up"></div>
                    <div class="regime-bar" style="flex:1; background:#c62828;" title="Choppy"></div>
                    <div class="regime-bar" style="flex:3; background:#1565c0;" title="Range"></div>
                </div>
            </section>
        </main>
    }
}

trait CollectView {
    fn collect_view(self) -> View;
}

impl<I> CollectView for I
where
    I: Iterator<Item = View>,
{
    fn collect_view(self) -> View {
        View::from_iter(self)
    }
}
