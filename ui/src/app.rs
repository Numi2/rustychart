use crate::{
    chart::ChartView,
    state::{provide_app_ctx, provide_link_bus},
    theme::GLOBAL_CSS,
};
use app_shell::{ChartInput, ChartState, InputKind, LayoutKind, Theme};
use leptos::*;
use leptos_meta::*;
use script_engine::language::{DiagnosticCode, SourceLang as ScriptSourceLang, SourceSpan};
use script_engine::{analyze_script, CompatibilityReport, CompletionItem, IssueSeverity, Template};
use serde::{Deserialize, Serialize};
use serde_json::{self, json};
use std::collections::HashMap;
use ta_engine::{IndicatorConfig, IndicatorKind, IndicatorParams, OutputKind, SourceField};

#[cfg(target_arch = "wasm32")]
use app_shell::StateStore;
#[cfg(target_arch = "wasm32")]
use gloo_net::http::Request;
#[cfg(target_arch = "wasm32")]
use gloo_timers::future::TimeoutFuture;
#[cfg(target_arch = "wasm32")]
use js_sys::{encode_uri_component, Reflect};
#[cfg(target_arch = "wasm32")]
use ts_core::Candle;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::spawn_local;
#[cfg(target_arch = "wasm32")]
use web_sys::{window, Document, HtmlElement, KeyboardEvent};

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

#[derive(Clone, Debug, Deserialize)]
struct ScanParams {
    ema_len: f32,
    band_width: f32,
    atr_stop_mult: f32,
    atr_target_mult: f32,
    risk_per_trade: f32,
}

#[derive(Clone, Debug, Deserialize)]
struct ScanMetrics {
    params: ScanParams,
    final_equity: f32,
    profit_factor: f32,
    sharpe: f32,
    max_drawdown: f32,
    num_trades: u32,
}

#[derive(Clone, Debug, Deserialize)]
struct ScanResponse {
    metrics: Vec<ScanMetrics>,
    combos: u32,
    bars: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ScriptEntry {
    name: String,
    body: String,
    tag: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RunStatus {
    Idle,
    Running,
    Completed,
}

impl RunStatus {
    fn label(&self) -> &'static str {
        match self {
            RunStatus::Idle => "Idle",
            RunStatus::Running => "Running",
            RunStatus::Completed => "Completed",
        }
    }

    fn tone_class(&self) -> &'static str {
        match self {
            RunStatus::Idle => "status-muted",
            RunStatus::Running => "status-warn",
            RunStatus::Completed => "status-good",
        }
    }
}

#[derive(Clone)]
struct RunSummary {
    id: String,
    title: String,
    pf: f32,
    status: RunStatus,
}

#[derive(Clone)]
struct StrategyEntry {
    name: &'static str,
    tag: &'static str,
    body: &'static str,
}

#[derive(Clone, Copy, Debug, Default)]
struct QuoteSnapshot {
    last: f64,
    change_pct: f64,
}

#[derive(Clone, Debug)]
struct ScriptAssistSnapshot {
    lang: ScriptSourceLang,
    report: Option<CompatibilityReport>,
    artifact_json: Option<String>,
    manifest_json: Option<String>,
    completions: Vec<CompletionItem>,
    templates: Vec<Template>,
}

impl Default for ScriptAssistSnapshot {
    fn default() -> Self {
        Self {
            lang: ScriptSourceLang::PineV5,
            report: None,
            artifact_json: None,
            manifest_json: None,
            completions: Vec::new(),
            templates: Vec::new(),
        }
    }
}

fn lang_label(lang: ScriptSourceLang) -> &'static str {
    match lang {
        ScriptSourceLang::PineV5 => "Pine v5",
        ScriptSourceLang::ThinkScriptSubset => "ThinkScript",
        ScriptSourceLang::NativeDsl => "Native DSL",
    }
}

fn severity_class(sev: IssueSeverity) -> &'static str {
    match sev {
        IssueSeverity::Info => "pill-soft",
        IssueSeverity::Warning => "pill-warn",
        IssueSeverity::Error => "pill-error",
    }
}

fn format_span(span: Option<SourceSpan>) -> String {
    span.map(|s| {
        format!(
            "L{}:{}-{}:{}",
            s.start_line, s.start_col, s.end_line, s.end_col
        )
    })
    .unwrap_or_else(|| "-".into())
}

#[cfg(not(target_arch = "wasm32"))]
#[component]
pub fn App() -> impl IntoView {
    view! { <div>UI available in browser build.</div> }
}

#[cfg(target_arch = "wasm32")]
#[component]
pub fn App() -> impl IntoView {
    provide_meta_context();

    let ctx = provide_app_ctx(api_base_default(), ws_base_default());
    let _link_bus = provide_link_bus();

    let run_history = create_rw_signal::<Vec<RunSummary>>(Vec::new());

    let example_scripts: Vec<StrategyEntry> = vec![
        StrategyEntry {
            name: "Breakout momentum",
            tag: "strategy",
            body: "// Breakout momentum template\n",
        },
        StrategyEntry {
            name: "RSI divergence",
            tag: "indicator",
            body: "// RSI divergence helper\n",
        },
    ];

    let default_my_scripts: Vec<ScriptEntry> = vec![
        ScriptEntry {
            name: "Mean reversion v2".into(),
            tag: "strategy".into(),
            body: "// Mean reversion starter\nfn on_bar(ctx: &mut Ctx) {\n    // price action hooks\n}\n"
                .into(),
        },
        ScriptEntry {
            name: "Liquidity sweep".into(),
            tag: "strategy".into(),
            body: "// Sweep liquidity bands\n".into(),
        },
        ScriptEntry {
            name: "Session VWAP bands".into(),
            tag: "indicator".into(),
            body: "// VWAP bands overlay\n".into(),
        },
    ];

    #[cfg(target_arch = "wasm32")]
    fn load_scripts_from_storage(defaults: Vec<ScriptEntry>) -> Vec<ScriptEntry> {
        if let Some(win) = window() {
            if let Ok(Some(storage)) = win.local_storage() {
                if let Ok(Some(raw)) = storage.get_item("rustychart-scripts") {
                    if let Ok(parsed) = serde_json::from_str::<Vec<ScriptEntry>>(&raw) {
                        return parsed;
                    }
                }
            }
        }
        defaults
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn load_scripts_from_storage(defaults: Vec<ScriptEntry>) -> Vec<ScriptEntry> {
        defaults
    }

    let my_scripts = create_rw_signal(load_scripts_from_storage(default_my_scripts));

    // Watchlist (mutable, clickable)
    let (watchlist, set_watchlist) = create_signal::<Vec<String>>(vec![
        "BTC-USD".into(),
        "ETH-USD".into(),
        "ES=F".into(),
        "NVDA".into(),
    ]);
    let (new_watch, set_new_watch) = create_signal(String::new());

    let theme_class = create_memo(move |_| {
        ctx.store.with(|s| match s.state().theme {
            Theme::Light => "lab-app light-theme".to_string(),
            _ => "lab-app".to_string(),
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

    let active_symbol = create_memo(move |_| {
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
                        .map(|c| c.symbol.clone())
                })
                .unwrap_or_else(|| "BTC-USD".to_string())
        })
    });

    // Active chart (pane) tracking for multi-pane layouts.
    let active_chart_id = create_rw_signal::<Option<u32>>(ctx.store.with(|s| {
        s.state()
            .layouts
            .first()
            .and_then(|l| l.charts.first().map(|c| c.id))
    }));
    {
        let store = ctx.store;
        let active_chart_id = active_chart_id.clone();
        create_effect(move |_| {
            let current = active_chart_id.get();
            let next = store.with(|s| {
                let layouts = &s.state().layouts;
                let active_layout = s
                    .state()
                    .active_layout_id
                    .and_then(|id| layouts.iter().find(|l| l.id == id))
                    .or_else(|| layouts.first());
                active_layout.and_then(|layout| {
                    if let Some(cur) = current {
                        if layout.charts.iter().any(|c| c.id == cur) {
                            return Some(cur);
                        }
                    }
                    layout.charts.first().map(|c| c.id)
                })
            });
            active_chart_id.set(next);
        });
    }

    let (dataset, set_dataset) = create_signal("Live".to_string());
    let (run_status, set_run_status) = create_signal(RunStatus::Idle);
    let (status_note, set_status_note) = create_signal("Idle".to_string());
    let (selected_run, set_selected_run) = create_signal(
        run_history
            .with(|runs| runs.first().map(|r| r.id.clone()))
            .unwrap_or_else(|| "run-001".to_string()),
    );
    let (active_tab, set_active_tab) = create_signal("chart".to_string());
    let (selected_strategy, set_selected_strategy) =
        create_signal::<Option<String>>(my_scripts.with(|s| s.first().map(|e| e.name.clone())));
    let (script_body, set_script_body) = create_signal(
        my_scripts
            .with(|s| s.first().map(|e| e.body.clone()))
            .unwrap_or_else(|| "// Script workspace\n".to_string()),
    );
    let (show_script_drawer, set_show_script_drawer) = create_signal(false);
    let (show_params_drawer, set_show_params_drawer) = create_signal(false);
    let (script_lang, set_script_lang) = create_signal(ScriptSourceLang::PineV5);
    let (assist_state, set_assist_state) =
        create_signal::<ScriptAssistSnapshot>(ScriptAssistSnapshot {
            lang: ScriptSourceLang::PineV5,
            ..Default::default()
        });
    let (share_blob, set_share_blob) = create_signal(String::new());
    let (share_input, set_share_input) = create_signal(String::new());
    let (date_range, set_date_range) = create_signal("90d".to_string());
    let watchlist_quotes = create_rw_signal::<HashMap<String, QuoteSnapshot>>(HashMap::new());

    // Sweep/backtest parameters.
    let (risk_per_trade, set_risk_per_trade) = create_signal(1.0_f32);
    let (slippage_bps, set_slippage_bps) = create_signal(0.0_f32);
    let (per_trade_cost, set_per_trade_cost) = create_signal(0.0_f32);

    // Parameter sweep controls (SMA, RSI, ATR) with toggles and explicit ranges.
    let (sma_value, set_sma_value) = create_signal(20.0_f32);
    let (sma_min, set_sma_min) = create_signal(10.0_f32);
    let (sma_max, set_sma_max) = create_signal(60.0_f32);
    let (sma_step, set_sma_step) = create_signal(5.0_f32);
    let (sma_sweep, set_sma_sweep) = create_signal(true);

    let (rsi_value, set_rsi_value) = create_signal(14.0_f32);
    let (rsi_min, set_rsi_min) = create_signal(7.0_f32);
    let (rsi_max, set_rsi_max) = create_signal(28.0_f32);
    let (rsi_step, set_rsi_step) = create_signal(7.0_f32);
    let (rsi_sweep, set_rsi_sweep) = create_signal(true);

    let (atr_value, set_atr_value) = create_signal(2.0_f32);
    let (atr_min, set_atr_min) = create_signal(1.0_f32);
    let (atr_max, set_atr_max) = create_signal(4.0_f32);
    let (atr_step, set_atr_step) = create_signal(0.5_f32);
    let (atr_sweep, set_atr_sweep) = create_signal(false);

    // Sweep grid settings.
    let (sweep_steps, set_sweep_steps) = create_signal(10_u32);
    let (ema_center, set_ema_center) = create_signal(50.0_f32);
    let (band_center, set_band_center) = create_signal(2.0_f32);
    let (atr_stop_center, set_atr_stop_center) = create_signal(1.0_f32);
    let (atr_target_center, set_atr_target_center) = create_signal(2.0_f32);

    let sweep_combo_count = create_memo(move |_| {
        let count = |on: bool, min: f32, max: f32, step: f32, val: f32| -> usize {
            if !on {
                return 1;
            }
            let step = step.max(0.0001);
            let span = (max - min).max(0.0);
            let steps = (span / step).floor() as usize + 1;
            steps.max(1)
        };
        let c1 = count(
            sma_sweep.get(),
            sma_min.get(),
            sma_max.get(),
            sma_step.get(),
            sma_value.get(),
        );
        let c2 = count(
            rsi_sweep.get(),
            rsi_min.get(),
            rsi_max.get(),
            rsi_step.get(),
            rsi_value.get(),
        );
        let c3 = count(
            atr_sweep.get(),
            atr_min.get(),
            atr_max.get(),
            atr_step.get(),
            atr_value.get(),
        );
        c1.saturating_mul(c2).saturating_mul(c3).max(1)
    });

    // Scan results + errors.
    let (scan_loading, set_scan_loading) = create_signal(false);
    let (scan_error, set_scan_error) = create_signal::<Option<String>>(None);
    let scan_results = create_rw_signal::<Vec<ScanMetrics>>(Vec::new());
    let (focused_metrics, set_focused_metrics) = create_signal::<Option<ScanMetrics>>(None);

    let add_script = {
        let scripts = my_scripts;
        let set_selected = set_selected_strategy;
        let set_body = set_script_body;
        move || {
            let mut name_idx = 1;
            let mut candidate = format!("New script {name_idx}");
            scripts.with(|list| {
                while list.iter().any(|s| s.name == candidate) {
                    name_idx += 1;
                    candidate = format!("New script {name_idx}");
                }
            });
            let entry = ScriptEntry {
                name: candidate.clone(),
                tag: "strategy".into(),
                body: "// New script\n".into(),
            };
            scripts.update(|list| {
                list.insert(0, entry);
                if list.len() > 50 {
                    list.truncate(50);
                }
            });
            set_selected.set(Some(candidate.clone()));
            set_body.set("// New script\n".into());
        }
    };

    let delete_script = {
        let scripts = my_scripts;
        let selected = selected_strategy;
        let set_selected = set_selected_strategy;
        let set_body = set_script_body;
        move || {
            if let Some(sel) = selected.get() {
                scripts.update(|list| {
                    if let Some(pos) = list.iter().position(|s| s.name == sel) {
                        list.remove(pos);
                    }
                });
                let next = scripts.with(|list| list.first().map(|e| e.name.clone()));
                set_selected.set(next.clone());
                if let Some(n) = next {
                    if let Some(body) =
                        scripts.with(|l| l.iter().find(|e| e.name == n).map(|e| e.body.clone()))
                    {
                        set_body.set(body);
                    }
                } else {
                    set_body.set("// Script workspace\n".into());
                }
            }
        }
    };

    let apply_template = {
        let set_body = set_script_body;
        let set_lang = set_script_lang;
        move |tpl: Template| {
            set_lang.set(tpl.source_lang);
            set_body.set(tpl.source.to_string());
        }
    };

    let apply_completion = {
        let set_body = set_script_body;
        move |item: CompletionItem| {
            set_body.update(|body| {
                if !body.ends_with('\n') {
                    body.push('\n');
                }
                body.push_str(&item.insert_text);
                if !body.ends_with('\n') {
                    body.push('\n');
                }
            });
        }
    };

    let import_share = {
        let input = share_input;
        let set_body = set_script_body;
        let set_lang = set_script_lang;
        move || {
            let raw = input.get();
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&raw) {
                if let Some(body) = val.get("body").and_then(|v| v.as_str()) {
                    set_body.set(body.to_string());
                }
                if let Some(lang) = val.get("lang").and_then(|v| v.as_str()) {
                    let l = lang.to_ascii_lowercase();
                    let mapped = if l.contains("think") {
                        Some(ScriptSourceLang::ThinkScriptSubset)
                    } else if l.contains("native") {
                        Some(ScriptSourceLang::NativeDsl)
                    } else {
                        Some(ScriptSourceLang::PineV5)
                    };
                    if let Some(m) = mapped {
                        set_lang.set(m);
                    }
                }
            } else {
                // Fallback: treat as raw script text.
                set_body.set(raw);
            }
        }
    };

    {
        let scripts = my_scripts;
        let selected = selected_strategy;
        let body = script_body;
        create_effect(move |_| {
            let current_body = body.get();
            if let Some(sel) = selected.get() {
                scripts.update(|list| {
                    if let Some(entry) = list.iter_mut().find(|e| e.name == sel) {
                        entry.body = current_body.clone();
                    }
                });
            }
        });
    }

    #[derive(Clone, Copy)]
    enum RunKind {
        Backtest,
        Sweep,
    }

    #[cfg(target_arch = "wasm32")]
    let run_scan = {
        let api_base = ctx.api_base;
        let run_status = set_run_status;
        let status_note = set_status_note;
        let scan_loading = set_scan_loading;
        let scan_error = set_scan_error;
        let scan_results = scan_results;
        let dataset = dataset;
        let active_symbol = active_symbol;
        let active_timeframe = active_timeframe;
        let risk_per_trade = risk_per_trade;
        let slippage_bps = slippage_bps;
        let per_trade_cost = per_trade_cost;
        let sweep_steps = sweep_steps;
        let ema_center = ema_center;
        let band_center = band_center;
        let atr_stop_center = atr_stop_center;
        let atr_target_center = atr_target_center;
        let run_history = run_history;
        let set_selected_run = set_selected_run;

        move |kind: RunKind| {
            scan_loading.set(true);
            scan_error.set(None);
            run_status.set(RunStatus::Running);
            let combos_est = sweep_combo_count.get_untracked();
            status_note.set(match kind {
                RunKind::Backtest => format!("Executing backtest ({} combos)", combos_est),
                RunKind::Sweep => format!("Sweep in progress ({} combos)", combos_est),
            });

            let api = api_base.get_untracked();
            let symbol = active_symbol.get();
            let tf = active_timeframe.get();
            let dataset_val = dataset.get();
            if dataset_val == "Synthetic" {
                scan_error.set(Some("Synthetic dataset unsupported for scan-grid".into()));
                run_status.set(RunStatus::Idle);
                status_note.set("Select Live or Historical".into());
                scan_loading.set(false);
                return;
            }
            let days = match dataset_val.as_str() {
                "Live" => Some(5),
                "Historical" => Some(180),
                _ => Some(30),
            };

            let single = matches!(kind, RunKind::Backtest);
            let range_for =
                |on: bool, val: f32, min: f32, max: f32, step: f32| -> (f32, f32, f32) {
                    if single || !on {
                        (val, val, 0.0001)
                    } else {
                        let clean_step = step.max(0.0001);
                        let lo = min.min(max);
                        let hi = max.max(min + clean_step);
                        (lo.max(0.0001), hi, clean_step)
                    }
                };

            let (ema_min, ema_max, ema_step) = range_for(
                sma_sweep.get_untracked(),
                sma_value.get_untracked(),
                sma_min.get_untracked(),
                sma_max.get_untracked(),
                sma_step.get_untracked(),
            );
            let (band_min, band_max, band_step) = range_for(
                rsi_sweep.get_untracked(),
                rsi_value.get_untracked(),
                rsi_min.get_untracked(),
                rsi_max.get_untracked(),
                rsi_step.get_untracked(),
            );
            let (atr_min, atr_max, atr_step) = range_for(
                atr_sweep.get_untracked(),
                atr_value.get_untracked(),
                atr_min.get_untracked(),
                atr_max.get_untracked(),
                atr_step.get_untracked(),
            );
            let (target_min, target_max, target_step) = (atr_min, atr_max, atr_step);
            let (risk_min, risk_max, risk_step) = range_for(
                false,
                risk_per_trade.get_untracked(),
                risk_per_trade.get_untracked(),
                risk_per_trade.get_untracked(),
                0.0001,
            );

            #[derive(Serialize)]
            struct GridRanges {
                ema_min: f32,
                ema_step: f32,
                ema_max: f32,
                band_min: f32,
                band_step: f32,
                band_max: f32,
                atr_stop_min: f32,
                atr_stop_step: f32,
                atr_stop_max: f32,
                atr_target_min: f32,
                atr_target_step: f32,
                atr_target_max: f32,
                risk_min: f32,
                risk_step: f32,
                risk_max: f32,
            }

            #[derive(Serialize)]
            struct ScanPayload {
                symbol: String,
                tf: Option<String>,
                days: Option<u64>,
                ranges: GridRanges,
                per_trade_cost: Option<f32>,
                slippage_bps: Option<f32>,
            }

            let payload = ScanPayload {
                symbol: symbol.clone(),
                tf: Some(tf.clone()),
                days,
                ranges: GridRanges {
                    ema_min,
                    ema_step,
                    ema_max,
                    band_min,
                    band_step,
                    band_max,
                    atr_stop_min: stop_min,
                    atr_stop_step: stop_step,
                    atr_stop_max: stop_max,
                    atr_target_min: target_min,
                    atr_target_step: target_step,
                    atr_target_max: target_max,
                    risk_min,
                    risk_step,
                    risk_max,
                },
                per_trade_cost: Some(per_trade_cost.get()),
                slippage_bps: Some(slippage_bps.get()),
            };

            spawn_local(async move {
                let url = format!("{}/scan-grid", api);
                let resp = Request::post(&url).json(&payload);
                let resp = match resp {
                    Ok(req) => req.send().await,
                    Err(e) => Err(e),
                };
                match resp {
                    Ok(http) if http.ok() => match http.json::<ScanResponse>().await {
                        Ok(mut body) => {
                            body.metrics.sort_by(|a, b| {
                                b.profit_factor
                                    .partial_cmp(&a.profit_factor)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            });
                            let pf = body
                                .metrics
                                .first()
                                .map(|m| m.profit_factor as f32)
                                .unwrap_or(0.0);
                            scan_results.set(body.metrics.clone());
                            run_status.set(RunStatus::Completed);
                            status_note.set(format!(
                                "{} complete ({} combos, bars: {})",
                                match kind {
                                    RunKind::Backtest => "Backtest",
                                    RunKind::Sweep => "Sweep",
                                },
                                body.combos,
                                body.bars
                            ));
                            let mut new_id = String::new();
                            let best = body.metrics.first().cloned();
                            set_focused_metrics.set(best.clone());
                            run_history.update(|runs| {
                                let next_num = runs.len() + 1;
                                new_id = format!("run-{next_num:03}");
                                runs.insert(
                                    0,
                                    RunSummary {
                                        id: new_id.clone(),
                                        title: format!(
                                            "{} {} {}",
                                            symbol,
                                            tf,
                                            match kind {
                                                RunKind::Backtest => "backtest",
                                                RunKind::Sweep => "sweep",
                                            }
                                        ),
                                        pf,
                                        status: RunStatus::Completed,
                                    },
                                );
                                if runs.len() > 20 {
                                    runs.truncate(20);
                                }
                            });
                            set_selected_run.set(new_id);
                        }
                        Err(e) => {
                            scan_error.set(Some(format!("Parse error: {e}")));
                            run_status.set(RunStatus::Idle);
                            status_note.set("Scan failed".to_string());
                            scan_results.set(Vec::new());
                        }
                    },
                    Ok(http) => {
                        scan_error.set(Some(format!("Scan failed: {}", http.status())));
                        run_status.set(RunStatus::Idle);
                        status_note.set("Scan failed".to_string());
                        scan_results.set(Vec::new());
                    }
                    Err(e) => {
                        scan_error.set(Some(format!("Request error: {e}")));
                        run_status.set(RunStatus::Idle);
                        status_note.set("Scan failed".to_string());
                        scan_results.set(Vec::new());
                    }
                }
                scan_loading.set(false);
            });
        }
    };

    #[cfg(not(target_arch = "wasm32"))]
    let run_scan = {
        let set_status_note = set_status_note;
        let set_run_status = set_run_status;
        move |kind: RunKind| {
            let note = match kind {
                RunKind::Backtest => "Backtest available in browser build",
                RunKind::Sweep => "Sweep available in browser build",
            };
            set_status_note.set(note.to_string());
            set_run_status.set(RunStatus::Completed);
        }
    };

    #[cfg(target_arch = "wasm32")]
    {
        // Global keyboard shortcuts: `/` focuses symbol, Cmd/Ctrl+Enter runs backtest, Esc closes drawers.
        let set_show_script_drawer = set_show_script_drawer.clone();
        let set_show_params_drawer = set_show_params_drawer.clone();
        create_effect(move |_| {
            let Some(win) = window() else {
                return;
            };
            let Some(doc) = win.document() else {
                return;
            };
            let doc_focus = doc.clone();
            let cb = Rc::new(Closure::<dyn FnMut(web_sys::Event)>::wrap(Box::new(
                move |ev: web_sys::Event| {
                    if let Ok(key_ev) = ev.dyn_into::<KeyboardEvent>() {
                        let key = key_ev.key();
                        if key == "/"
                            && !key_ev.ctrl_key()
                            && !key_ev.meta_key()
                            && !key_ev.alt_key()
                        {
                            key_ev.prevent_default();
                            if let Some(el) = doc_focus.get_element_by_id("symbol-search") {
                                if let Some(el) = el.dyn_ref::<HtmlElement>() {
                                    let _ = el.focus();
                                }
                            }
                        }
                        if key == "Enter" && (key_ev.ctrl_key() || key_ev.meta_key()) {
                            key_ev.prevent_default();
                            if let Some(el) = doc_focus.get_element_by_id("run-backtest-btn") {
                                if let Some(el) = el.dyn_ref::<HtmlElement>() {
                                    let _ = el.click();
                                }
                            }
                        }
                        if key == "Escape" {
                            set_show_script_drawer.set(false);
                            set_show_params_drawer.set(false);
                        }
                    }
                },
            )));
            let _ = win
                .add_event_listener_with_callback("keydown", cb.as_ref().as_ref().unchecked_ref());
            on_cleanup({
                let cb = cb.clone();
                move || {
                    if let Some(win) = window() {
                        let _ = win.remove_event_listener_with_callback(
                            "keydown",
                            cb.as_ref().as_ref().unchecked_ref(),
                        );
                    }
                }
            });
        });

        // Persist custom scripts to localStorage.
        let scripts = my_scripts;
        create_effect(move |_| {
            if let Some(win) = window() {
                if let Ok(Some(storage)) = win.local_storage() {
                    if let Ok(json) = serde_json::to_string(&scripts.get()) {
                        let _ = storage.set_item("rustychart-scripts", &json);
                    }
                }
            }
        });
    }

    {
        let body_signal = script_body;
        let lang_signal = script_lang;
        let set_assist = set_assist_state;
        create_effect(move |_| {
            let body = body_signal.get();
            let lang = lang_signal.get();
            if body.trim().is_empty() {
                set_assist.set(ScriptAssistSnapshot {
                    lang,
                    ..Default::default()
                });
                return;
            }
            let analysis = analyze_script(&body, lang);
            let artifact_json = analysis
                .artifact
                .as_ref()
                .and_then(|a| serde_json::to_string_pretty(a).ok());
            let manifest_json = serde_json::to_string_pretty(&analysis.manifest).ok();
            set_assist.set(ScriptAssistSnapshot {
                lang,
                report: Some(analysis.report),
                artifact_json,
                manifest_json,
                completions: analysis.completions,
                templates: analysis.templates,
            });
        });
    }

    {
        let body_signal = script_body;
        let lang_signal = script_lang;
        let set_blob = set_share_blob;
        create_effect(move |_| {
            let body = body_signal.get();
            let lang = lang_signal.get();
            let payload = json!({
                "lang": lang_label(lang),
                "body": body,
            });
            let pretty = serde_json::to_string_pretty(&payload).unwrap_or_else(|_| body.clone());
            set_blob.set(pretty);
        });
    }

    // Helpers to mutate the active chart (tracked by active_chart_id).
    let update_active_chart = {
        let store = ctx.store;
        let active_chart_id = active_chart_id.clone();
        move |f: &mut dyn FnMut(&mut app_shell::ChartState)| {
            store.update(|s| {
                let target = active_chart_id.get_untracked().or_else(|| {
                    s.state()
                        .layouts
                        .first()
                        .and_then(|l| l.charts.first().map(|c| c.id))
                });
                if let Some(target_id) = target {
                    if let Some(chart) = s
                        .state_mut()
                        .layouts
                        .iter_mut()
                        .flat_map(|l| l.charts.iter_mut())
                        .find(|c| c.id == target_id)
                    {
                        f(chart);
                    }
                }
            });
        }
    };

    // Symbol search state
    let (search_query, set_search_query) = create_signal(String::new());
    let search_results = create_rw_signal::<Vec<SearchHit>>(Vec::new());
    let search_loading = create_rw_signal(false);
    let search_error = create_rw_signal::<Option<String>>(None);
    let search_highlight = create_rw_signal::<Option<usize>>(None);

    #[cfg(target_arch = "wasm32")]
    {
        let api_base = ctx.api_base;
        create_effect(move |_| {
            let q = search_query.get();
            if q.trim().len() < 2 {
                search_results.set(Vec::new());
                search_highlight.set(None);
                search_error.set(None);
                search_loading.set(false);
                return;
            }
            let results = search_results.clone();
            let highlight = search_highlight.clone();
            let loading = search_loading.clone();
            let errors = search_error.clone();
            let query = q.clone();
            spawn_local(async move {
                loading.set(true);
                errors.set(None);
                TimeoutFuture::new(200).await;
                let api = api_base.get_untracked();
                let url = format!("{}/search?q={}", api, encode_uri_component(&query));
                match Request::get(&url).send().await {
                    Ok(resp) => match resp.json::<Vec<SearchHit>>().await {
                        Ok(hits) => {
                            highlight.set(if hits.is_empty() { None } else { Some(0) });
                            results.set(hits);
                        }
                        Err(e) => {
                            errors.set(Some(format!("Search failed: {}", e)));
                            results.set(Vec::new());
                            highlight.set(None);
                        }
                    },
                    Err(e) => {
                        errors.set(Some(format!("Search failed: {}", e)));
                        results.set(Vec::new());
                        highlight.set(None);
                    }
                }
                loading.set(false);
            });
        });
    }

    let apply_symbol = {
        move |symbol: String| {
            update_active_chart(&mut |chart| chart.symbol = symbol.clone());
            set_search_query.set(symbol.clone());
            search_results.set(Vec::new());
            search_highlight.set(None);
        }
    };

    let apply_timeframe = {
        move |tf: String| {
            update_active_chart(&mut |chart| chart.timeframe = tf.clone());
        }
    };

    let set_layout_preset = {
        let store = ctx.store;
        move |rows: u8, cols: u8| {
            store.update(|s| {
                let desired = rows.max(1).saturating_mul(cols.max(1));
                let active_id = s.state().active_layout_id;
                let layouts = &mut s.state_mut().layouts;
                let layout = active_id
                    .and_then(|id| layouts.iter_mut().find(|l| l.id == id))
                    .or_else(|| layouts.first_mut());
                if let Some(layout) = layout {
                    let base = layout
                        .charts
                        .first()
                        .cloned()
                        .unwrap_or_else(|| ChartState {
                            id: 1,
                            symbol: "BTC-USD".into(),
                            timeframe: "1m".into(),
                            indicators: Vec::new(),
                            drawings: Vec::new(),
                            link_group: Some("A".into()),
                            orders: Vec::new(),
                            positions: Vec::new(),
                            alerts: Vec::new(),
                            price_pane_weight: 1.0,
                            pane_layout: Vec::new(),
                            pane: 0,
                            height_ratio: 1.0,
                            inputs: Vec::new(),
                        });
                    let mut next_id = layout.charts.iter().map(|c| c.id).max().unwrap_or(0) + 1;
                    layout.kind = if rows == 1 && cols == 1 {
                        LayoutKind::Single
                    } else {
                        LayoutKind::Grid { rows, cols }
                    };
                    if layout.charts.len() > desired as usize {
                        layout.charts.truncate(desired as usize);
                    } else {
                        while layout.charts.len() < desired as usize {
                            let mut clone = base.clone();
                            clone.id = next_id;
                            next_id += 1;
                            clone.link_group = base.link_group.clone().or_else(|| Some("A".into()));
                            clone.pane = layout
                                .charts
                                .len()
                                .checked_div(cols.max(1) as usize)
                                .unwrap_or(0) as u8;
                            if clone.height_ratio <= 0.0 {
                                clone.height_ratio = 1.0;
                            }
                            layout.charts.push(clone);
                        }
                    }
                    for (idx, c) in layout.charts.iter_mut().enumerate() {
                        if c.link_group.is_none() {
                            c.link_group = Some("A".into());
                        }
                        let pane_idx = idx.checked_div(cols.max(1) as usize).unwrap_or(0) as u8;
                        c.pane = pane_idx;
                        if c.height_ratio <= 0.0 {
                            c.height_ratio = 1.0;
                        }
                    }
                }
            });
        }
    };

    let set_layout_single = {
        let preset = set_layout_preset;
        move || preset(1, 1)
    };
    let set_layout_dual = {
        let preset = set_layout_preset;
        move || preset(2, 1)
    };
    let set_layout_quad = {
        let preset = set_layout_preset;
        move || preset(2, 2)
    };

    let add_pane = {
        let store = ctx.store;
        let active_chart_id = active_chart_id.clone();
        move || {
            store.update(|s| {
                let active_id = s.state().active_layout_id;
                let layouts = &mut s.state_mut().layouts;
                let layout = active_id
                    .and_then(|id| layouts.iter_mut().find(|l| l.id == id))
                    .or_else(|| layouts.first_mut());
                if let Some(layout) = layout {
                    let next_id = layout.charts.iter().map(|c| c.id).max().unwrap_or(0) + 1;
                    let next_pane = layout
                        .charts
                        .iter()
                        .map(|c| c.pane)
                        .max()
                        .unwrap_or(0)
                        .saturating_add(1);
                    let mut base = layout
                        .charts
                        .first()
                        .cloned()
                        .unwrap_or_else(|| ChartState {
                            id: 1,
                            symbol: "BTC-USD".into(),
                            timeframe: "1m".into(),
                            indicators: Vec::new(),
                            drawings: Vec::new(),
                            link_group: Some("A".into()),
                            orders: Vec::new(),
                            positions: Vec::new(),
                            alerts: Vec::new(),
                            price_pane_weight: 1.0,
                            pane_layout: Vec::new(),
                            pane: next_pane,
                            height_ratio: 1.0,
                            inputs: Vec::new(),
                        });
                    base.id = next_id;
                    base.pane = next_pane;
                    base.height_ratio = 1.0;
                    layout.kind = LayoutKind::Grid {
                        rows: (next_pane + 1).max(1) as u8,
                        cols: 1,
                    };
                    layout.charts.push(base);
                    active_chart_id.set(Some(next_id));
                }
            });
        }
    };

    let remove_active_pane = {
        let store = ctx.store;
        let active_chart_id = active_chart_id.clone();
        move || {
            store.update(|s| {
                let target = active_chart_id.get_untracked();
                let active_id = s.state().active_layout_id;
                let layouts = &mut s.state_mut().layouts;
                let active_layout = active_id
                    .and_then(|id| layouts.iter_mut().find(|l| l.id == id))
                    .or_else(|| layouts.first_mut());
                if let Some(layout) = active_layout {
                    if let Some(tid) = target {
                        layout.charts.retain(|c| c.id != tid);
                    }
                    // Re-index panes to be dense.
                    let mut sorted = layout.charts.clone();
                    sorted.sort_by_key(|c| c.pane);
                    for (idx, chart) in sorted.iter_mut().enumerate() {
                        chart.pane = idx as u8;
                    }
                    layout.charts = sorted;
                    if layout.charts.is_empty() {
                        layout.charts.push(ChartState {
                            id: 1,
                            symbol: "BTC-USD".into(),
                            timeframe: "1m".into(),
                            indicators: Vec::new(),
                            drawings: Vec::new(),
                            link_group: Some("A".into()),
                            orders: Vec::new(),
                            positions: Vec::new(),
                            alerts: Vec::new(),
                            price_pane_weight: 1.0,
                            pane_layout: Vec::new(),
                            pane: 0,
                            height_ratio: 1.0,
                            inputs: Vec::new(),
                        });
                    }
                    active_chart_id.set(layout.charts.first().map(|c| c.id));
                }
            });
        }
    };

    let set_theme_choice = {
        let store = ctx.store;
        move |theme: Theme| {
            store.update(|s| {
                s.state_mut().theme = theme.clone();
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
                IndicatorParams::Vwap {
                    reset_each_day: true,
                },
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

        update_active_chart(&mut |chart| chart.indicators.push(cfg.clone()));
    };

    let remove_indicator_from_active = {
        move |idx: usize| {
            update_active_chart(&mut |chart| {
                if idx < chart.indicators.len() {
                    chart.indicators.remove(idx);
                }
            });
        }
    };

    let clear_indicators = move |_| {
        update_active_chart(&mut |chart| chart.indicators.clear());
    };

    let move_indicator = {
        move |idx: usize, dir: i32| {
            update_active_chart(&mut |chart| {
                let len = chart.indicators.len();
                if idx < len {
                    let next_idx = (idx as i32 + dir).clamp(0, len as i32 - 1) as usize;
                    if next_idx != idx {
                        chart.indicators.swap(idx, next_idx);
                    }
                }
            });
        }
    };

    let add_input_to_active = {
        move || {
            update_active_chart(&mut |chart| {
                let idx = chart.inputs.len() + 1;
                chart.inputs.push(ChartInput {
                    id: format!("input-{idx}"),
                    label: format!("Input {idx}"),
                    value: "0".into(),
                    kind: InputKind::Slider {
                        min: 0.0,
                        max: 100.0,
                        step: 1.0,
                    },
                });
            });
        }
    };

    let remove_input_from_active = {
        move |idx: usize| {
            update_active_chart(&mut |chart| {
                if idx < chart.inputs.len() {
                    chart.inputs.remove(idx);
                }
            });
        }
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
                if let Ok(Some(state)) =
                    app_shell::load_state_from_local_storage("rustychart-state")
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

    #[cfg(target_arch = "wasm32")]
    {
        let watchlist_quotes = watchlist_quotes.clone();
        let api_base = ctx.api_base;
        create_effect(move |_| {
            let symbols = watchlist.get();
            let tf = active_timeframe.get();
            for sym in symbols {
                let sym_clone = sym.clone();
                let tf_clone = tf.clone();
                let api = api_base.get_untracked();
                let dest = watchlist_quotes.clone();
                spawn_local(async move {
                    let url = format!(
                        "{}/history?tf={}&limit={}&symbol={}",
                        api,
                        encode_uri_component(&tf_clone),
                        2,
                        encode_uri_component(&sym_clone)
                    );
                    let resp = Request::get(&url).send().await;
                    let snapshot = if let Ok(resp) = resp {
                        if resp.ok() {
                            if let Ok(candles) = resp.json::<Vec<Candle>>().await {
                                if let Some(last) = candles.last() {
                                    let last_close = last.close;
                                    let prev_close = candles
                                        .iter()
                                        .rev()
                                        .nth(1)
                                        .map(|c| c.close)
                                        .unwrap_or(last_close);
                                    let change_pct = if prev_close.abs() > f64::EPSILON {
                                        (last_close - prev_close) / prev_close * 100.0
                                    } else {
                                        0.0
                                    };
                                    QuoteSnapshot {
                                        last: last_close,
                                        change_pct,
                                    }
                                } else {
                                    QuoteSnapshot::default()
                                }
                            } else {
                                QuoteSnapshot::default()
                            }
                        } else {
                            QuoteSnapshot::default()
                        }
                    } else {
                        QuoteSnapshot::default()
                    };
                    dest.update(|m| {
                        m.insert(sym_clone, snapshot);
                    });
                });
            }
        });
    }

    let trigger_backtest = { move |_| run_scan(RunKind::Backtest) };

    let trigger_sweep = {
        move |_| {
            set_show_params_drawer.set(true);
            run_scan(RunKind::Sweep);
        }
    };

    let layout_view = move || {
        if let Some(id) = active_layout_id.get() {
            let layout = ctx
                .store
                .with(|s| s.state().layouts.iter().find(|l| l.id == id).cloned());
            if layout.is_none() {
                return view! { <div class="chart-grid"></div> };
            }
            let layout = layout.unwrap();
            let (rows_base, cols) = match layout.kind {
                LayoutKind::Single => (1, 1),
                LayoutKind::Grid { rows, cols } => (rows.max(1) as usize, cols.max(1) as usize),
            };
            let mut pane_heights: Vec<(u8, f32)> = Vec::new();
            for c in layout.charts.iter() {
                if pane_heights.iter().all(|(p, _)| *p != c.pane) {
                    pane_heights.push((c.pane, c.height_ratio.max(0.25)));
                }
            }
            pane_heights.sort_by_key(|(p, _)| *p);
            let use_panes = cols == 1 && !pane_heights.is_empty();
            let rows = if use_panes {
                pane_heights.len()
            } else {
                rows_base
            };
            let row_template = if use_panes {
                pane_heights
                    .iter()
                    .map(|(_, h)| format!("minmax(0, {}fr)", h))
                    .collect::<Vec<_>>()
                    .join(" ")
            } else {
                format!("repeat({rows}, minmax(0, 1fr))")
            };
            let style = format!(
                "grid-template-columns: repeat({cols}, 1fr); grid-template-rows: {row_template};"
            );
            let api = ctx.api_base.with(|s| s.clone());
            let ws = ctx.ws_base.with(|s| s.clone());
            let mut charts = layout.charts;
            charts.sort_by_key(|c| (c.pane, c.id));
            return view! {
                <div class="chart-grid" style=style>
                    {charts.into_iter().map(|chart| {
                        view! { <ChartView chart_id=chart.id http_base=api.clone() ws_base=ws.clone() /> }
                    }).collect_view()}
                </div>
            };
        }
        view! { <div class="chart-grid"></div> }
    };

    let remove_watch = {
        let set_watchlist_handle = set_watchlist;
        move |symbol: String| {
            set_watchlist_handle.update(|list| {
                if let Some(pos) = list.iter().position(|s| s == &symbol) {
                    list.remove(pos);
                }
            });
        }
    };

    view! {
        <Style>{GLOBAL_CSS}</Style>
        <main class=theme_class>
            <div class="lab-shell">
                <header class="panel lab-topbar">
                    <div class="topbar-left">
                        <div class="brand-mark">
                            <span class="brand-title">RustyChart</span>
                            <span class="pill pill-muted">Algo Lab</span>
                            <span class="pill pill-soft">alpha</span>
                        </div>
                    </div>
                    <div class="topbar-center">
                        <div class="input-stack wide">
                            <label class="input-label" for="symbol-search">Symbol</label>
                            <div class="input-wrap">
                                <input
                                    class="input-lg"
                                    id="symbol-search"
                                    name="symbol-search"
                                    type="text"
                                    placeholder="Search symbol..."
                                    value=move || search_query.get()
                                    on:input=move |ev| set_search_query.set(event_target_value(&ev))
                                    on:keydown=move |ev| {
                                        match ev.key().as_str() {
                                            "Enter" => {
                                                if let Some(idx) = search_highlight.get() {
                                                    if let Some(hit) = search_results.get().get(idx) {
                                                        apply_symbol(hit.symbol.clone());
                                                        return;
                                                    }
                                                }
                                                let val = search_query.get().trim().to_string();
                                                if !val.is_empty() {
                                                    apply_symbol(val);
                                                }
                                            }
                                            "ArrowDown" => {
                                                let len = search_results.get().len();
                                                if len > 0 {
                                                    let next = search_highlight
                                                        .get()
                                                        .map(|i| (i + 1).min(len - 1))
                                                        .unwrap_or(0);
                                                    search_highlight.set(Some(next));
                                                    ev.prevent_default();
                                                }
                                            }
                                            "ArrowUp" => {
                                                let len = search_results.get().len();
                                                if len > 0 {
                                                    let next = search_highlight
                                                        .get()
                                                        .map(|i| i.saturating_sub(1))
                                                        .unwrap_or(0);
                                                    search_highlight.set(Some(next.min(len - 1)));
                                                    ev.prevent_default();
                                                }
                                            }
                                            _ => {}
                                        }
                                    }
                                />
                                {move || {
                                    let hits = search_results.get();
                                    let loading = search_loading.get();
                                    let err = search_error.get();
                                    let query_len = search_query.get().trim().len();
                                    if loading {
                                        return view! { <div class="flyout panel"><div class="flyout-meta">Searching...</div></div> }.into_view();
                                    }
                                    if let Some(e) = err.clone() {
                                        return view! { <div class="flyout panel"><div class="flyout-meta">{e}</div></div> }.into_view();
                                    }
                                    if hits.is_empty() && query_len >= 2 {
                                        return view! { <div class="flyout panel"><div class="flyout-meta">No matches</div></div> }.into_view();
                                    }
                                    if hits.is_empty() {
                                        return ().into_view();
                                    }
                                    view! {
                                        <div class="flyout panel">
                                            {hits.into_iter().enumerate().map(|(idx, h)| {
                                                let sym = h.symbol.clone();
                                                let sym_display = sym.clone();
                                                let name = h.name.clone().unwrap_or_default();
                                                let exch = h.exchange.clone().unwrap_or_default();
                                                let is_active = search_highlight.get() == Some(idx);
                                                let row_class = if is_active { "flyout-row active" } else { "flyout-row" };
                                                view! {
                                                    <button class=row_class on:click=move |_| apply_symbol(sym.clone())>
                                                        <span class="flyout-symbol">{sym_display}</span>
                                                        <span class="flyout-meta">{format!("{exch} {name}")}</span>
                                                    </button>
                                                }.into_view()
                                            }).collect_view()}
                                        </div>
                                    }.into_view()
                                }}
                            </div>
                        </div>
                        <div class="topbar-controls">
                            <div class="control-stack">
                                <label class="input-label" for="active-timeframe">Timeframe</label>
                                <select
                                    class="input-compact"
                                    id="active-timeframe"
                                    name="active-timeframe"
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
                            <div class="control-stack">
                                <label class="input-label" for="dataset-select">Dataset</label>
                                <select
                                    class="input-compact"
                                    id="dataset-select"
                                    name="dataset-select"
                                    value=move || dataset.get()
                                    on:change=move |ev| set_dataset.set(event_target_value(&ev))
                                >
                                    <option value="Live">Live</option>
                                    <option value="Historical">Historical</option>
                                </select>
                            </div>
                            <div class="control-stack">
                                <label class="input-label" for="pane-select">Pane</label>
                                <select
                                    class="input-compact"
                                    id="pane-select"
                                    name="pane-select"
                                    value=move || active_chart_id.get().map(|id| id.to_string()).unwrap_or_else(|| "0".into())
                                    on:change=move |ev| {
                                        if let Ok(val) = event_target_value(&ev).parse::<u32>() {
                                            active_chart_id.set(Some(val));
                                        }
                                    }
                                >
                                    {move || {
                                        ctx.store.with(|s| {
                                            let lid = s
                                                .state()
                                                .active_layout_id
                                                .or_else(|| s.state().layouts.first().map(|l| l.id));
                                            let mut opts = Vec::new();
                                            if let Some(layout_id) = lid {
                                                if let Some(layout) =
                                                    s.state().layouts.iter().find(|l| l.id == layout_id)
                                                {
                                                    let mut charts = layout.charts.clone();
                                                    charts.sort_by_key(|c| (c.pane, c.id));
                                                    opts = charts
                                                        .into_iter()
                                                        .map(|c| {
                                                            let label = format!("Pane {} ({})", c.pane + 1, c.symbol);
                                                            view! {
                                                                <option value=c.id.to_string()>{label}</option>
                                                            }
                                                        })
                                                        .collect::<Vec<_>>();
                                                }
                                            }
                                            opts.into_iter().collect_view()
                                        })
                                    }}
                                </select>
                            </div>
                            <div class="control-stack">
                                <label class="input-label" for="pane-height">Pane height</label>
                                <div class="input-wrap">
                                    <input
                                        id="pane-height"
                                        name="pane-height"
                                        class="input-compact"
                                        type="range"
                                        min="0.5"
                                        max="4.0"
                                        step="0.1"
                                        value=move || {
                                            ctx.store.with(|s| {
                                                let target = active_chart_id.get();
                                                let layouts = &s.state().layouts;
                                                let active_layout = s
                                                    .state()
                                                    .active_layout_id
                                                    .and_then(|id| layouts.iter().find(|l| l.id == id))
                                                    .or_else(|| layouts.first());
                                                active_layout
                                                    .and_then(|layout| {
                                                        if let Some(id) = target {
                                                            layout
                                                                .charts
                                                                .iter()
                                                                .find(|c| c.id == id)
                                                                .map(|c| c.height_ratio)
                                                        } else {
                                                            layout.charts.first().map(|c| c.height_ratio)
                                                        }
                                                    })
                                                    .unwrap_or(1.0)
                                                    .to_string()
                                            })
                                        }
                                        on:input=move |ev| {
                                            if let Ok(val) = event_target_value(&ev).parse::<f32>() {
                                                update_active_chart(&mut |chart| {
                                                    chart.height_ratio = val.max(0.25);
                                                });
                                            }
                                        }
                                    />
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="topbar-actions">
                        <button id="run-backtest-btn" class="btn secondary" on:click=trigger_backtest>Run backtest</button>
                        <button id="run-sweep-btn" class="btn primary" on:click=trigger_sweep>Run sweep</button>
                        <div class=format!("status-pill {}", run_status.get().tone_class())>
                            <span class="status-dot"></span>
                            <span>{run_status.get().label()}</span>
                        </div>
                    </div>
                </header>

                <div class="lab-body">
                    <div class="left-column">
                        <div class="panel chart-card">
                            <div class="chart-meta">
                                <div class="pill-row">
                                    <span class="pill pill-strong">{move || active_symbol.get()}</span>
                                    <span class="pill">{move || active_timeframe.get()}</span>
                                    <span class="pill">{move || dataset.get()}</span>
                                    {move || {
                                        selected_strategy
                                            .get()
                                            .map(|s| view! { <span class="pill pill-accent">{s}</span> }.into_view())
                                            .unwrap_or_else(|| ().into_view())
                                    }}
                                </div>
                                <div class="chart-meta-actions">
                                    <button class="btn ghost" on:click=move |_| set_show_script_drawer.set(true)>
                                        Script editor
                                    </button>
                                    <button class="btn ghost" on:click=move |_| add_indicator_to_active()>Quick SMA</button>
                                </div>
                                <div class="chart-meta-actions" style="gap:8px; flex-wrap:wrap;">
                                    {move || {
                                        if scan_loading.get() {
                                            view! { <span class="pill pill-soft">Running...</span> }.into_view()
                                        } else if let Some(err) = scan_error.get() {
                                            view! { <span class="pill pill-down">{err}</span> }.into_view()
                                        } else {
                                            view! { <span class="pill pill-soft">Idle</span> }.into_view()
                                        }
                                    }}
                                    {move || {
                                        let mut results = scan_results.get();
                                        if results.is_empty() {
                                            return ().into_view();
                                        }
                                        results.sort_by(|a, b| {
                                            b.profit_factor
                                                .partial_cmp(&a.profit_factor)
                                                .unwrap_or(std::cmp::Ordering::Equal)
                                        });
                                        results
                                            .first()
                                            .map(|m| {
                                                view! {
                                                    <span class="pill pill-accent">{format!("PF {:.2}", m.profit_factor)}</span>
                                                    <span class="pill pill-soft">{format!("Sharpe {:.2}", m.sharpe)}</span>
                                                }
                                                .into_view()
                                            })
                                            .unwrap_or_else(|| ().into_view())
                                    }}
                                </div>
                            </div>

                            <div class="tab-bar">
                                {["chart", "equity", "heatmap", "runs"].iter().map(|tab| {
                                    let label = match *tab {
                                        "chart" => "Chart",
                                        "equity" => "Equity",
                                        "heatmap" => "Sweep Map",
                                        "runs" => "Runs Table",
                                        _ => "",
                                    };
                                    let tab_id = tab.to_string();
                                    view! {
                                        <button class=format!("pill selectable {}", if active_tab.get()==tab_id { "active" } else { "" }) on:click=move |_| set_active_tab.set(tab_id.clone())>
                                            {label}
                                        </button>
                                    }
                                }).collect_view()}
                            </div>

                            {move || {
                                match active_tab.get().as_str() {
                                    "equity" => {
                                        view! {
                                            <div class="pane resizable-pane">
                                                <div class="pane-header">
                                                    <div>
                                                        <div class="pane-title">Equity curve</div>
                                                        <div class="pane-subtitle">{move || format!("Run {}", selected_run.get())}</div>
                                                    </div>
                                                </div>
                                                <div class="equity-placeholder">
                                                    <div class="equity-line"></div>
                                                    <div class="drawdown-line"></div>
                                                </div>
                                                <div class="pane-kpis">
                                                    {move || {
                                                        let mut res = scan_results.get();
                                                        if let Some(sel) = focused_metrics.get() {
                                                            return view! {
                                                                <span class="pill pill-muted">{format!("PF {:.2}", sel.profit_factor)}</span>
                                                                <span class="pill pill-muted">{format!("Sharpe {:.2}", sel.sharpe)}</span>
                                                                <span class="pill pill-muted">{format!("Trades {}", sel.num_trades)}</span>
                                                            }.into_view();
                                                        }
                                                        if res.is_empty() {
                                                            return view! {
                                                                <span class="pill pill-muted">PF --</span>
                                                                <span class="pill pill-muted">Sharpe --</span>
                                                                <span class="pill pill-muted">Trades --</span>
                                                            }.into_view();
                                                        }
                                                        res.sort_by(|a, b| {
                                                            b.profit_factor
                                                                .partial_cmp(&a.profit_factor)
                                                                .unwrap_or(std::cmp::Ordering::Equal)
                                                        });
                                                        let top = res.first().unwrap();
                                                        view! {
                                                            <span class="pill pill-muted">{format!("PF {:.2}", top.profit_factor)}</span>
                                                            <span class="pill pill-muted">{format!("Sharpe {:.2}", top.sharpe)}</span>
                                                            <span class="pill pill-muted">{format!("Trades {}", top.num_trades)}</span>
                                                        }.into_view()
                                                    }}
                                                </div>
                                            </div>
                                        }.into_view()
                                    }
                                    "heatmap" => {
                                        view! {
                                            <div class="pane resizable-pane">
                                                <div class="pane-header">
                                                    <div>
                                                        <div class="pane-title">Sweep map</div>
                                                        <div class="pane-subtitle">Parameter A vs B</div>
                                                    </div>
                                                    <div class="pane-subtitle">{move || format!("{} runs", sweep_combo_count.get())}</div>
                                                </div>
                                                {move || {
                                                    let results = scan_results.get();
                                                    if results.is_empty() {
                                                        return view! { <div class="pane-empty">Run a sweep to populate the heatmap.</div> }.into_view();
                                                    }
                                                    let mut xs: Vec<f32> = results.iter().map(|m| m.params.ema_len).collect();
                                                    let mut ys: Vec<f32> = results.iter().map(|m| m.params.band_width).collect();
                                                    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                                                    ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                                                    xs.dedup_by(|a, b| (*a - *b).abs() < f32::EPSILON);
                                                    ys.dedup_by(|a, b| (*a - *b).abs() < f32::EPSILON);
                                                    let best_pf = results.iter().map(|m| m.profit_factor).fold(0.0_f32, f32::max);
                                                    let worst_pf = results.iter().map(|m| m.profit_factor).fold(0.0_f32, f32::min);
                                                    let color_for = |pf: f32, best: f32, worst: f32| {
                                                        let hi = if best <= 0.0 { 1.0 } else { best };
                                                        let lo = worst.min(0.0);
                                                        let span = (hi - lo).max(0.0001);
                                                        let t = ((pf - lo) / span).clamp(0.0, 1.0);
                                                        let r = (240.0 * (1.0 - t) + 30.0 * t) as i32;
                                                        let g = (70.0 * (1.0 - t) + 180.0 * t) as i32;
                                                        let b = (70.0 * (1.0 - t) + 80.0 * t) as i32;
                                                        format!("rgba({r},{g},{b},0.9)")
                                                    };
                                                    view! {
                                                        <div class="heatmap-grid">
                                                            <div class="heatmap-legend">
                                                                <span class="pill pill-soft">Rows: SMA length</span>
                                                                <span class="pill pill-soft">Cols: RSI length</span>
                                                            </div>
                                                            <div class="heatmap-body" style=format!("grid-template-columns: repeat({}, minmax(0, 1fr));", xs.len())>
                                                                {ys.iter().flat_map(|y| {
                                                                    let row_val = *y;
                                                                    xs.iter().map(move |x| {
                                                                        let col_val = *x;
                                                                        let cell = results.iter().find(|m| (m.params.ema_len - col_val).abs() < 0.0001 && (m.params.band_width - row_val).abs() < 0.0001);
                                                                        if let Some(m) = cell {
                                                                            let m = m.clone();
                                                                            let color = color_for(m.profit_factor, best_pf, worst_pf);
                                                                            let pf = format!("{:.2}", m.profit_factor);
                                                                            let title = format!(
                                                                                "SMA: {:.0}\nRSI: {:.0}\nPF: {:.2}\nSharpe: {:.2}\nTrades: {}\nMax DD: {:.2}%",
                                                                                m.params.ema_len,
                                                                                m.params.band_width,
                                                                                m.profit_factor,
                                                                                m.sharpe,
                                                                                m.num_trades,
                                                                                m.max_drawdown * 100.0
                                                                            );
                                                                            view! {
                                                                                <button class="heatmap-cell" style=format!("background:{color}") title=title.clone() on:click=move |_| {
                                                                                    set_focused_metrics.set(Some(m.clone()));
                                                                                }>
                                                                                    <span>{pf}</span>
                                                                                </button>
                                                                            }.into_view()
                                                                        } else {
                                                                            view! { <div class="heatmap-cell empty"></div> }.into_view()
                                                                        }
                                                                    })
                                                                }).collect_view()}
                                                            </div>
                                                        </div>
                                                    }.into_view()
                                                }}
                                            </div>
                                        }.into_view()
                                    }
                                    "runs" => {
                                        view! {
                                            <div class="pane resizable-pane">
                                                <div class="pane-header">
                                                    <div class="pane-title">Runs table</div>
                                                    <div class="pane-subtitle">Sorted by PF</div>
                                                </div>
                                                {move || {
                                                    let mut rows = scan_results.get();
                                                    if rows.is_empty() {
                                                        return view! { <div class="pane-empty">No runs yet.</div> }.into_view();
                                                    }
                                                    rows.sort_by(|a, b| b.profit_factor.partial_cmp(&a.profit_factor).unwrap_or(std::cmp::Ordering::Equal));
                                                    view! {
                                                        <div class="runs-table">
                                                            <div class="runs-head">
                                                                <span>Params</span><span>PF</span><span>Sharpe</span><span>Max DD</span><span>Trades</span><span>View</span>
                                                            </div>
                                                            {rows.into_iter().map(|m| {
                                                                let row = m.clone();
                                                                let params = format!("SMA {:.0} | RSI {:.0} | ATR {:.1}", row.params.ema_len, row.params.band_width, row.params.atr_stop_mult);
                                                                let pf = format!("{:.2}", row.profit_factor);
                                                                let sh = format!("{:.2}", row.sharpe);
                                                                let dd = format!("{:.2}%", row.max_drawdown * 100.0);
                                                                let trades = format!("{}", row.num_trades);
                                                                view! {
                                                                    <button class="runs-row" on:click=move |_| {
                                                                        set_focused_metrics.set(Some(row.clone()));
                                                                    }>
                                                                        <span>{params.clone()}</span>
                                                                        <span>{pf}</span>
                                                                        <span>{sh}</span>
                                                                        <span>{dd}</span>
                                                                        <span>{trades}</span>
                                                                        <span class="pill pill-soft">Load</span>
                                                                    </button>
                                                                }
                                                            }).collect_view()}
                                                        </div>
                                                    }.into_view()
                                                }}
                                            </div>
                                        }.into_view()
                                    }
                                    _ => {
                                        view! {
                                            <div class="chart-stage">
                                                {layout_view}
                                                <div class="status-floating">{move || status_note.get()}</div>
                                                <div class="equity-inline">
                                                    <div class="equity-line"></div>
                                                    <div class="drawdown-line"></div>
                                                </div>
                                            </div>
                                        }.into_view()
                                    }
                                }
                            }}
                        </div>
                    </div>
                    <aside class="panel sidebar">
                        <div class="sidebar-section">
                            <div class="section-head">
                                <div class="section-title">Watchlist</div>
                                <div class="section-subtitle">Symbol / Last / Change%</div>
                            </div>
                            <div class="watchlist">
                                {move || {
                                    watchlist.get().into_iter().map(|sym| {
                                        let display = sym.clone();
                                        let display_for_click = display.clone();
                                        let display_for_remove = display.clone();
                                        let quote_map = watchlist_quotes.get();
                                        let snapshot = quote_map.get(&display).cloned();
                                        let price_display = snapshot
                                            .map(|q| format!("{:.2}", q.last))
                                            .unwrap_or_else(|| "--".to_string());
                                        let change = snapshot.map(|q| q.change_pct);
                                        let change_display = change
                                            .map(|c| format!("{c:+.2}%"))
                                            .unwrap_or_else(|| "--%".to_string());
                                        let change_class = match change {
                                            Some(c) if c > 0.0001 => "pill-up",
                                            Some(c) if c < -0.0001 => "pill-down",
                                            _ => "pill-soft",
                                        };
                                        view! {
                                            <div class="list-row compact" on:click=move |_| apply_symbol(display_for_click.clone())>
                                                <div class="list-left">
                                                    <span class="list-title">{display}</span>
                                                </div>
                                                <div class="list-right">
                                                    <span class="pill pill-soft">{price_display}</span>
                                                    <span class=format!("pill {}", change_class)>{change_display}</span>
                                                    <button class="btn ghost" aria-label="Remove symbol" on:click=move |ev| {
                                                        ev.stop_propagation();
                                                        remove_watch(display_for_remove.clone());
                                                    }>Remove</button>
                                                </div>
                                            </div>
                                        }.into_view()
                                    }).collect_view()
                                }}
                            </div>
                            <div class="watchlist-add">
                                <input
                                    class="input-compact"
                                    id="watchlist-add"
                                    name="watchlist-add"
                                    type="text"
                                    placeholder="Add symbol"
                                    value=move || new_watch.get()
                                    on:input=move |ev| set_new_watch.set(event_target_value(&ev))
                                    on:keydown=move |ev| {
                                        if ev.key() == "Enter" {
                                            let v = new_watch.get().trim().to_string();
                                            if !v.is_empty() {
                                                set_watchlist.update(|list| list.push(v.clone()));
                                                set_new_watch.set(String::new());
                                            }
                                        }
                                    }
                                />
                                <button class="btn ghost" on:click=move |_| {
                                    let v = new_watch.get().trim().to_string();
                                    if !v.is_empty() {
                                        set_watchlist.update(|list| list.push(v.clone()));
                                        set_new_watch.set(String::new());
                                    }
                                }>Add</button>
                            </div>
                        </div>

                        <div class="sidebar-section">
                            <div class="section-head">
                                <div class="section-title">Strategies</div>
                                <div class="section-subtitle">Scripts / Examples</div>
                            </div>
                            <div class="pill-row" style="margin-bottom:6px;">
                                <button class="btn ghost" on:click=move |_| add_script()>New</button>
                                <button class="btn ghost" on:click=move |_| delete_script()>Delete</button>
                            </div>
                            <div class="section-subtitle muted">My Scripts</div>
                            <div class="pill-grid">
                                {my_scripts.get().into_iter().map(|s| {
                                    let name = s.name.to_string();
                                    let body = s.body.to_string();
                                    let tag = s.tag.to_string();
                                    let name_for_click = name.clone();
                                    view! {
                                        <button class="pill-card" on:click=move |_| {
                                            set_selected_strategy.set(Some(name_for_click.clone()));
                                            set_script_body.set(body.clone());
                                            set_show_script_drawer.set(true);
                                        }>
                                            <div class="pill-card-title">{name.clone()}</div>
                                            <div class="pill-card-sub">{tag}</div>
                                        </button>
                                    }
                                }).collect_view()}
                            </div>
                            <div class="section-subtitle muted" style="margin-top:8px;">Examples</div>
                            <div class="pill-grid">
                                {example_scripts.iter().map(|s| {
                                    let name = s.name.to_string();
                                    let body = s.body.to_string();
                                    let name_for_click = name.clone();
                                    view! {
                                        <button class="pill-card" on:click=move |_| {
                                            set_selected_strategy.set(Some(name_for_click.clone()));
                                            set_script_body.set(body.clone());
                                            set_show_script_drawer.set(true);
                                        }>
                                            <div class="pill-card-title">{name.clone()}</div>
                                            <div class="pill-card-sub">{s.tag}</div>
                                        </button>
                                    }
                                }).collect_view()}
                            </div>
                        </div>

                <div class="sidebar-section">
                    <div class="section-head">
                        <div class="section-title">Layouts & theme</div>
                        <div class="section-subtitle">Multi-pane + styling</div>
                    </div>
                    <div class="pill-row" style="margin-bottom:6px;">
                        <button class="btn ghost" on:click=move |_| set_layout_single()>Single</button>
                        <button class="btn ghost" on:click=move |_| set_layout_dual()>2 panes</button>
                        <button class="btn ghost" on:click=move |_| set_layout_quad()>4 panes</button>
                    </div>
                    <div class="pill-row" style="margin-bottom:6px;">
                        <button class="btn ghost" on:click=move |_| add_pane()>Add pane</button>
                        <button class="btn ghost" on:click=move |_| remove_active_pane()>Remove active</button>
                    </div>
                    <div class="pill-row">
                        <button class="btn ghost" on:click=move |_| set_theme_choice(Theme::Dark)>Dark</button>
                        <button class="btn ghost" on:click=move |_| set_theme_choice(Theme::Light)>Light</button>
                    </div>
                </div>

                <div class="sidebar-section">
                    <div class="section-head">
                        <div class="section-title">Overlays</div>
                        <div class="section-subtitle">Order + visibility</div>
                    </div>
                    <div class="pill-row" style="margin-bottom:8px;">
                        <button class="btn ghost" on:click=move |_| add_indicator_to_active()>Add overlay</button>
                        <button class="btn ghost" on:click=clear_indicators>Clear all</button>
                    </div>
                    <div class="list-stack">
                        {move || {
                            ctx.store.with(|s| {
                                let layouts = &s.state().layouts;
                                let active_layout = s
                                    .state()
                                    .active_layout_id
                                    .and_then(|id| layouts.iter().find(|l| l.id == id))
                                    .or_else(|| layouts.first());
                                let mut items: Vec<View> = Vec::new();
                                if let Some(layout) = active_layout {
                                    let target = active_chart_id.get();
                                    let chart = layout
                                        .charts
                                        .iter()
                                        .find(|c| Some(c.id) == target)
                                        .or_else(|| layout.charts.first());
                                    if let Some(chart) = chart {
                                        for (idx, cfg) in chart.indicators.iter().enumerate() {
                                            let name = format!("{:?}", cfg.kind);
                                            let idx_copy = idx;
                                            let row: View = view! {
                                                <div class="list-row">
                                                    <div class="list-left">
                                                        <span class="list-title">{name}</span>
                                                        <span class="list-sub">{format!("{:?}", cfg.params)}</span>
                                                    </div>
                                                    <div class="list-right" style="gap:6px;">
                                    <button class="btn ghost" on:click=move |_| move_indicator(idx_copy, -1)>Up</button>
                                    <button class="btn ghost" on:click=move |_| move_indicator(idx_copy, 1)>Down</button>
                                                        <button class="btn ghost" on:click=move |_| remove_indicator_from_active(idx_copy)>Remove</button>
                                                    </div>
                                                </div>
                                            }.into_view();
                                            items.push(row);
                                        }
                                    }
                                }
                                items.into_iter().collect_view()
                            })
                        }}
                    </div>
                </div>

                <div class="sidebar-section">
                    <div class="section-head">
                        <div class="section-title">Pane inputs</div>
                        <div class="section-subtitle">Custom sliders/toggles</div>
                    </div>
                    <div class="pill-row" style="margin-bottom:8px;">
                        <button class="btn ghost" on:click=move |_| add_input_to_active()>+ Input</button>
                    </div>
                    <div class="list-stack">
                        {move || {
                            ctx.store.with(|s| {
                                let layouts = &s.state().layouts;
                                let active_layout = s
                                    .state()
                                    .active_layout_id
                                    .and_then(|id| layouts.iter().find(|l| l.id == id))
                                    .or_else(|| layouts.first());
                                let mut items: Vec<View> = Vec::new();
                                if let Some(layout) = active_layout {
                                    let target = active_chart_id.get();
                                    let chart = layout
                                        .charts
                                        .iter()
                                        .find(|c| Some(c.id) == target)
                                        .or_else(|| layout.charts.first());
                                    if let Some(chart) = chart {
                                        for (idx, input) in chart.inputs.iter().enumerate() {
                                            let idx_copy = idx;
                                            let label = input.label.clone();
                                            let value = input.value.clone();
                                            let input_id = format!("pane-input-{idx_copy}");
                                            let input_name = input_id.clone();
                                            let row: View = match &input.kind {
                                                InputKind::Slider { min, max, step } => {
                                                    view! {
                                                        <div class="list-row">
                                                            <div class="list-left">
                                                                <label class="list-title" for=input_id.clone()>{label.clone()}</label>
                                                                <span class="list-sub">{format!("{}-{} step {}", min, max, step)}</span>
                                                            </div>
                                                            <div class="list-right" style="gap:8px;">
                                                                <input
                                                                    id=input_id.clone()
                                                                    name=input_name.clone()
                                                                    type="range"
                                                                    min=min.to_string()
                                                                    max=max.to_string()
                                                                    step=step.to_string()
                                                                    value=value.clone()
                                                                    on:input=move |ev| {
                                                                        if let Ok(val) = event_target_value(&ev).parse::<f32>() {
                                                                            update_active_chart(&mut |chart| {
                                                                                if let Some(inp) = chart.inputs.get_mut(idx_copy) {
                                                                                    inp.value = format!("{:.3}", val);
                                                                                }
                                                                            });
                                                                        }
                                                                    }
                                                                />
                                                                <span class="pill pill-soft">{value.clone()}</span>
                                                                <button class="btn ghost" on:click=move |_| remove_input_from_active(idx_copy)>Remove</button>
                                                            </div>
                                                        </div>
                                                    }
                                                    .into_view()
                                                }
                                                InputKind::Toggle => {
                                                    let checked = value == "true";
                                                    view! {
                                                        <div class="list-row">
                                                            <div class="list-left">
                                                                <label class="list-title" for=input_id.clone()>{label.clone()}</label>
                                                                <span class="list-sub">Toggle</span>
                                                            </div>
                                                            <div class="list-right" style="gap:8px;">
                                                                <input
                                                                    id=input_id.clone()
                                                                    name=input_name.clone()
                                                                    type="checkbox"
                                                                    checked=checked
                                                                    on:change=move |ev| {
                                                                        let val = event_target_checked(&ev);
                                                                        update_active_chart(&mut |chart| {
                                                                            if let Some(inp) = chart.inputs.get_mut(idx_copy) {
                                                                                inp.value = val.to_string();
                                                                            }
                                                                        });
                                                                    }
                                                                />
                                                                <button class="btn ghost" on:click=move |_| remove_input_from_active(idx_copy)>Remove</button>
                                                            </div>
                                                        </div>
                                                    }
                                                    .into_view()
                                                }
                                                InputKind::Select { options } => {
                                                    let options_clone = options.clone();
                                                    view! {
                                                        <div class="list-row">
                                                            <div class="list-left">
                                                                <label class="list-title" for=input_id.clone()>{label.clone()}</label>
                                                                <span class="list-sub">Select</span>
                                                            </div>
                                                            <div class="list-right" style="gap:8px;">
                                                                <select
                                                                    class="input-compact"
                                                                    id=input_id.clone()
                                                                    name=input_name.clone()
                                                                    value=value.clone()
                                                                    on:change=move |ev| {
                                                                        let val = event_target_value(&ev);
                                                                        update_active_chart(&mut |chart| {
                                                                            if let Some(inp) = chart.inputs.get_mut(idx_copy) {
                                                                                inp.value = val.clone();
                                                                            }
                                                                        });
                                                                    }
                                                                >
                                                                    {options_clone.into_iter().map(|opt| {
                                                                        view! { <option value=opt.clone()>{opt}</option> }
                                                                    }).collect_view()}
                                                                </select>
                                                                <button class="btn ghost" on:click=move |_| remove_input_from_active(idx_copy)>Remove</button>
                                                            </div>
                                                        </div>
                                                    }
                                                    .into_view()
                                                }
                                            };
                                            items.push(row);
                                        }
                                    }
                                }
                                items.into_iter().collect_view()
                            })
                        }}
                    </div>
                </div>

                        <div class="sidebar-section">
                            <div class="section-head">
                                <div class="section-title">Run setup</div>
                                <div class="section-subtitle">Tweak + replay</div>
                            </div>
                            <div class="form-grid">
                                <div class="control-stack">
                                    <label class="input-label" for="tf-run">Timeframe</label>
                                    <select
                                        class="input-compact"
                                        id="tf-run"
                                        name="tf-run"
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
                                <div class="control-stack">
                                    <label class="input-label">Date range</label>
                                    <div class="pill-row">
                                        {["30d", "90d", "1y"].iter().map(|r| {
                                            let r_str = r.to_string();
                                            view! {
                                                <button class=format!("pill selectable {}", if date_range.get()==*r { "active" } else { "" }) on:click=move |_| set_date_range.set(r_str.clone())>
                                                    {r.to_string()}
                                                </button>
                                            }
                                        }).collect_view()}
                                    </div>
                                </div>
                                <div class="control-stack">
                                    <label class="input-label">Parameters</label>
                                    <button class="btn ghost" on:click=move |_| set_show_params_drawer.set(true)>Open editor</button>
                                </div>
                            </div>
                            <div class="section-subtitle muted" style="margin-top:4px;">Recent runs</div>
                            <div class="list-stack">
                                {run_history.get().into_iter().map(|r| {
                                    let rid = r.id.to_string();
                                    let status_class = r.status.tone_class();
                                    let rid_for_click = rid.clone();
                                    view! {
                                        <button class=format!("list-row {}", if selected_run.get()==rid { "active-row" } else { "" }) on:click=move |_| set_selected_run.set(rid_for_click.clone())>
                                            <div class="list-left">
                                                <span class="list-title">{r.title}</span>
                                                <span class="list-sub">{r.id}</span>
                                            </div>
                                            <div class="list-right">
                                                <span class=format!("pill {}", status_class)>{r.status.label()}</span>
                                                <span class="pill pill-soft">{format!("PF {:.2}", r.pf)}</span>
                                            </div>
                                        </button>
                                    }
                                }).collect_view()}
                            </div>
                        </div>
                    </aside>
                </div>

                <div class=format!("drawer {}", if show_script_drawer.get() { "open" } else { "" })>
                    <div class="drawer-header">
                        <div>
                            <div class="pane-title">Script editor</div>
                            <div class="pane-subtitle">Typing assist / templates / sharing</div>
                        </div>
                        <button class="btn ghost" on:click=move |_| set_show_script_drawer.set(false)>Close</button>
                    </div>
                    <div class="drawer-body script-pane">
                        <div class="drawer-main">
                            <div class="pane-subtitle">Language</div>
                            <select
                                class="input-compact"
                                value=move || lang_label(script_lang.get()).to_string()
                                on:change=move |ev| {
                                    let v = event_target_value(&ev).to_ascii_lowercase();
                                    if v.contains("think") {
                                        set_script_lang.set(ScriptSourceLang::ThinkScriptSubset);
                                    } else if v.contains("native") {
                                        set_script_lang.set(ScriptSourceLang::NativeDsl);
                                    } else {
                                        set_script_lang.set(ScriptSourceLang::PineV5);
                                    }
                                }
                            >
                                <option value="Pine v5">Pine v5</option>
                                <option value="ThinkScript">ThinkScript</option>
                                <option value="Native DSL">Native DSL</option>
                            </select>

                            <textarea
                                class="script-area"
                                value=move || script_body.get()
                                on:input=move |ev| set_script_body.set(event_target_value(&ev))
                            ></textarea>

                            <div class="pane-subtitle">Completions</div>
                            <div class="pill-row wrap">
                                {move || {
                                    assist_state
                                        .get()
                                        .completions
                                        .iter()
                                        .take(12)
                                        .map(|c| {
                                            let item = c.clone();
                                            view! {
                                                <button class="pill pill-soft" title={item.detail.clone()} on:click=move |_| apply_completion(item.clone())>
                                                    {item.label.clone()}
                                                </button>
                                            }
                                        })
                                        .collect_view()
                                }}
                            </div>

                            <div class="pane-subtitle">Diagnostics</div>
                            <div class="list-stack">
                                {move || {
                                    if let Some(report) = assist_state.get().report {
                                        let errs = report.issues.iter().filter(|i| matches!(i.severity, IssueSeverity::Error)).count();
                                        let warns = report.issues.iter().filter(|i| matches!(i.severity, IssueSeverity::Warning)).count();
                                        let infos = report.issues.iter().filter(|i| matches!(i.severity, IssueSeverity::Info)).count();
                                        let status_cls = if report.supported && errs == 0 {
                                            "pill-good"
                                        } else {
                                            "pill-error"
                                        };
                                        let status_copy = if report.supported && errs == 0 {
                                            "Ready to run"
                                        } else {
                                            "Needs fixes"
                                        };

                                        return view! {
                                            <div class="list-row compact">
                                                <span class={format!("pill {}", status_cls)}>{status_copy}</span>
                                                <span class="list-sub">{format!("{} errors / {} warnings / {} info", errs, warns, infos)}</span>
                                            </div>
                                            {
                                                if report.issues.is_empty() && report.supported {
                                                    view! { <div class="pane-subtitle muted">No issues detected.</div> }.into_view()
                                                } else {
                                                    report
                                                        .issues
                                                        .iter()
                                                        .map(|i| {
                                                            let cls = severity_class(i.severity);
                                                            let hint = i.hint.clone().unwrap_or_default();
                                                            let code = format!("{:?}", i.code);
                                                            let span = format_span(i.span);
                                                            view! {
                                                                <div class="list-row compact">
                                                                    <span class={format!("pill {}", cls)}>{format!("{:?}", i.severity)}</span>
                                                                    <span class="pill pill-soft">{code}</span>
                                                                    <span class="list-title">{i.message.clone()}</span>
                                                                    <span class="list-sub">{format!("{} - {}", lang_label(script_lang.get()), span)}</span>
                                                                    {(!hint.is_empty()).then(|| view! { <span class="list-sub">{hint.clone()}</span> }.into_view())}
                                                                </div>
                                                            }
                                                        })
                                                        .collect_view()
                                                }
                                            }
                                        }.into_view();
                                    }
                                    view! { <div class="pane-subtitle muted">Type to see diagnostics.</div> }.into_view()
                                }}
                            </div>

                            <div class="pane-subtitle">Templates</div>
                            <div class="pill-row wrap">
                                {move || {
                                    assist_state
                                        .get()
                                        .templates
                                        .iter()
                                        .map(|t| {
                                            let tpl = t.clone();
                                            view! {
                                                <button class="pill pill-soft" title={tpl.description} on:click=move |_| apply_template(tpl.clone())>
                                                    {tpl.name}
                                                </button>
                                            }
                                        })
                                        .collect_view()
                                }}
                            </div>

                            <div class="pane-subtitle">Share / import</div>
                            <div class="share-stack">
                                <label class="input-label">Share blob (lang + script)</label>
                                <textarea class="script-share" readonly value=move || share_blob.get()></textarea>
                                <label class="input-label">Import blob</label>
                                <textarea
                                    class="script-share"
                                    placeholder="Paste shared JSON or raw script"
                                    value=move || share_input.get()
                                    on:input=move |ev| set_share_input.set(event_target_value(&ev))
                                ></textarea>
                                <button class="btn ghost" on:click=move |_| import_share()>Import</button>
                            </div>
                        </div>

                        <div class="drawer-side">
                            <div class="pane-title">Actions</div>
                            <div class="pane-subtitle">Ctrl/Cmd+Enter to run</div>
                            <button class="btn primary" on:click=trigger_backtest>Run backtest</button>
                            <button class="btn ghost" on:click=trigger_sweep>Queue sweep</button>
                            <div class="pane-subtitle" style="margin-top:12px;">Artifacts</div>
                            <details>
                                <summary class="pane-subtitle muted">Compiled artifact</summary>
                                <pre class="artifact-view">{move || assist_state.get().artifact_json.unwrap_or_else(|| "Compile to view artifact".into())}</pre>
                            </details>
                            <details>
                                <summary class="pane-subtitle muted">Manifest</summary>
                                <pre class="artifact-view">{move || assist_state.get().manifest_json.unwrap_or_else(|| "Compile to view manifest".into())}</pre>
                            </details>
                        </div>
                    </div>
                </div>

                <div class=format!("drawer drawer-narrow {}", if show_params_drawer.get() { "open" } else { "" })>
                    <div class="drawer-header">
                        <div>
                            <div class="pane-title">Parameter editor</div>
                            <div class="pane-subtitle">Grouped fields, helper text</div>
                        </div>
                        <button class="btn ghost" on:click=move |_| set_show_params_drawer.set(false)>Close</button>
                    </div>
                    <div class="drawer-body drawer-body-column">
                        <div class="form-grid">
                            <div class="control-stack">
                                <label class="input-label">Date range</label>
                                <div class="pill-row">
                                    {["30d", "90d", "1y"].iter().map(|r| {
                                        let r_str = r.to_string();
                                        view! {
                                            <button class=format!("pill selectable {}", if date_range.get()==*r { "active" } else { "" }) on:click=move |_| set_date_range.set(r_str.clone())>
                                                {r.to_string()}
                                            </button>
                                        }
                                    }).collect_view()}
                                </div>
                            </div>
                            <div class="control-stack">
                                <label class="input-label" for="sweep-steps">Sweep steps</label>
                                <input
                                    class="input-compact"
                                    id="sweep-steps"
                                    name="sweep-steps"
                                    type="number"
                                    min="2"
                                    value=move || sweep_steps.get().to_string()
                                    on:input=move |ev| {
                                        if let Ok(v) = event_target_value(&ev).parse::<usize>() {
                                            set_sweep_steps.set((v.max(2)) as u32);
                                        }
                                    }
                                />
                            </div>
                            <div class="control-stack">
                                <label class="input-label" for="risk-cap">Risk per trade</label>
                                <input
                                    class="input-compact"
                                    id="risk-cap"
                                    name="risk-cap"
                                    type="number"
                                    step="0.1"
                                    value=move || risk_per_trade.get().to_string()
                                    on:input=move |ev| {
                                        if let Ok(v) = event_target_value(&ev).parse::<f32>() {
                                            set_risk_per_trade.set(v.max(0.01));
                                        }
                                    }
                                />
                            </div>
                            <div class="control-stack">
                                <label class="input-label" for="ema-center">EMA len center</label>
                                <input
                                    class="input-compact"
                                    id="ema-center"
                                    name="ema-center"
                                    type="number"
                                    min="1"
                                    value=move || ema_center.get().to_string()
                                    on:input=move |ev| {
                                        if let Ok(v) = event_target_value(&ev).parse::<f32>() {
                                            set_ema_center.set(v.max(1.0));
                                        }
                                    }
                                />
                            </div>
                            <div class="control-stack">
                                <label class="input-label" for="band-center">Band width center</label>
                                <input
                                    class="input-compact"
                                    id="band-center"
                                    name="band-center"
                                    type="number"
                                    step="0.1"
                                    value=move || band_center.get().to_string()
                                    on:input=move |ev| {
                                        if let Ok(v) = event_target_value(&ev).parse::<f32>() {
                                            set_band_center.set(v.max(0.1));
                                        }
                                    }
                                />
                            </div>
                            <div class="control-stack">
                                <label class="input-label" for="stop-center">ATR stop mult</label>
                                <input
                                    class="input-compact"
                                    id="stop-center"
                                    name="stop-center"
                                    type="number"
                                    step="0.1"
                                    value=move || atr_stop_center.get().to_string()
                                    on:input=move |ev| {
                                        if let Ok(v) = event_target_value(&ev).parse::<f32>() {
                                            set_atr_stop_center.set(v.max(0.1));
                                        }
                                    }
                                />
                            </div>
                            <div class="control-stack">
                                <label class="input-label" for="target-center">ATR target mult</label>
                                <input
                                    class="input-compact"
                                    id="target-center"
                                    name="target-center"
                                    type="number"
                                    step="0.1"
                                    value=move || atr_target_center.get().to_string()
                                    on:input=move |ev| {
                                        if let Ok(v) = event_target_value(&ev).parse::<f32>() {
                                            set_atr_target_center.set(v.max(0.1));
                                        }
                                    }
                                />
                            </div>
                            <div class="control-stack">
                                <label class="input-label" for="slip-bps">Slippage (bps)</label>
                                <input
                                    class="input-compact"
                                    id="slip-bps"
                                    name="slip-bps"
                                    type="number"
                                    step="0.1"
                                    value=move || slippage_bps.get().to_string()
                                    on:input=move |ev| {
                                        if let Ok(v) = event_target_value(&ev).parse::<f32>() {
                                            set_slippage_bps.set(v.max(0.0));
                                        }
                                    }
                                />
                            </div>
                            <div class="control-stack">
                                <label class="input-label" for="per-trade-cost">Per-trade cost</label>
                                <input
                                    class="input-compact"
                                    id="per-trade-cost"
                                    name="per-trade-cost"
                                    type="number"
                                    step="0.01"
                                    value=move || per_trade_cost.get().to_string()
                                    on:input=move |ev| {
                                        if let Ok(v) = event_target_value(&ev).parse::<f32>() {
                                            set_per_trade_cost.set(v.max(0.0));
                                        }
                                    }
                                />
                            </div>
                        </div>
                        <div class="drawer-actions">
                            <button class="btn ghost" on:click=move |_| set_show_params_drawer.set(false)>Cancel</button>
                            <button class="btn primary" on:click=trigger_sweep>Queue sweep</button>
                        </div>
                    </div>
                </div>
            </div>
        </main>
    }
}

trait CollectView {
    fn collect_view(self) -> View;
}

impl<I, V> CollectView for I
where
    I: Iterator<Item = V>,
    V: IntoView,
{
    fn collect_view(self) -> View {
        View::from_iter(self.map(|v| v.into_view()))
    }
}
