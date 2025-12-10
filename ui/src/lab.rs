use leptos::*;
use gpu_param_sweep::CostModel;
#[cfg(target_arch = "wasm32")]
use gpu_param_sweep::{Ohlc, SampledParams};
#[cfg(target_arch = "wasm32")]
use scan_engine::cpu_reference_metrics;
use scan_engine::{
    EvaluationWindow, ScanDimension, ScanMetric, ScanRequest, ScanResultPoint, WebGpuScanEngine,
    ScanEngine,
};
use std::rc::Rc;
use ts_core::{Candle, TimeFrame};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::spawn_local;

#[derive(Clone, Debug)]
struct LabPoint {
    coords: Vec<f64>,
    value: f64,
}

#[cfg(target_arch = "wasm32")]
fn synthetic_candles(n: usize, base: f64) -> Vec<Candle> {
    let mut out = Vec::with_capacity(n);
    let mut price = base;
    let tf = TimeFrame::Minutes(1);
    for i in 0..n {
        let ts = i as i64 * tf.duration_ms();
        let change = ((i as f64 * 13.0).sin() * 0.05) as f64;
        let open = price;
        let close = price * (1.0 + change * 0.01);
        let high = open.max(close) + 0.5;
        let low = open.min(close) - 0.5;
        out.push(Candle {
            ts,
            timeframe: tf,
            open,
            high,
            low,
            close,
            volume: 1.0,
        });
        price = close;
    }
    out
}

#[cfg(target_arch = "wasm32")]
async fn run_gpu_scan(metric: ScanMetric) -> Option<Vec<ScanResultPoint>> {
    let engine = WebGpuScanEngine::new(42);
    let candles = synthetic_candles(512, 100.0);
    let dims = vec![
        ScanDimension {
            name: "ema".into(),
            min: 5.0,
            max: 60.0,
            steps: 8,
        },
        ScanDimension {
            name: "band".into(),
            min: 0.5,
            max: 3.0,
            steps: 6,
        },
    ];
    let req = ScanRequest {
        script_or_indicator_id: "ema-band".into(),
        dimensions: dims,
        evaluation_window: EvaluationWindow::LastNBars(400),
        metric,
        cost_model: CostModel::default(),
        candles,
    };

    match engine.scan(req).await {
        Ok(res) => Some(res.points),
        Err(_) => None,
    }
}

#[component]
pub fn LabPanel() -> impl IntoView {
    let metrics = Rc::new(vec![
        (ScanMetric::Sharpe, "Sharpe"),
        (ScanMetric::ProfitFactor, "Profit factor"),
        (ScanMetric::FinalEquity, "Final equity"),
    ]);
    #[cfg(target_arch = "wasm32")]
    let metrics_for_run = metrics.clone();
    let metrics_for_select_change = metrics.clone();
    let metrics_for_select_render = metrics.clone();
    let (metric_idx, set_metric_idx) = create_signal(0usize);
    let (progress, set_progress) = create_signal(0.0f32);
    let (points, set_points) = create_signal::<Vec<LabPoint>>(Vec::new());

    let run = move |_| {
        set_progress.set(0.1);
        #[cfg(target_arch = "wasm32")]
        {
            let metric = metrics_for_run
                .get(metric_idx.get())
                .map(|(m, _)| m.clone())
                .unwrap_or(ScanMetric::Sharpe);
            spawn_local(async move {
                if let Some(res) = run_gpu_scan(metric).await {
                    set_points.set(
                        res.into_iter()
                            .map(|p| LabPoint {
                                coords: p.coords,
                                value: p.metric_value,
                            })
                            .collect(),
                    );
                }
                set_progress.set(1.0);
            });
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            set_points.set(Vec::new());
            set_progress.set(1.0);
        }
    };

    view! {
        <div class="panel lab-panel">
            <div class="section-label">GPU Lab</div>
            <div class="flex-col" style="gap:8px;">
                <label class="section-label">Metric</label>
                <select
                    on:change=move |ev| {
                        if let Ok(idx) = event_target_value(&ev).parse::<usize>() {
                            set_metric_idx.set(
                                idx.min(metrics_for_select_change.len().saturating_sub(1)),
                            );
                        }
                    }
                >
                    {metrics_for_select_render.iter().enumerate().map(|(i, (_, label))| view! {
                        <option value=i>{label.to_string()}</option>
                    }).collect_view()}
                </select>
                <button on:click=run>Run GPU scan</button>
                <div class="section-label">Progress</div>
                <progress max="1.0" value=move || progress.get() as f64></progress>
                <div class="section-label">Top configurations</div>
                <div class="flex-col" style="gap:4px; max-height:200px; overflow:auto;">
                    {move || {
                        let mut sorted = points.get();
                        sorted.sort_by(|a,b| b.value.partial_cmp(&a.value).unwrap_or(std::cmp::Ordering::Equal));
                        sorted.truncate(5);
                        sorted.into_iter().map(|p| {
                            let label = format!("{:.2} -> {:.4}", p.coords.get(0).unwrap_or(&0.0), p.value);
                            view! { <div class="chip">{label}</div> }
                        }).collect_view()
                    }}
                </div>
            </div>
        </div>
    }
}

#[component]
pub fn DiagnosticsPanel() -> impl IntoView {
    let (status, set_status) = create_signal(String::from("Idle"));
    let (delta, set_delta) = create_signal(0.0f64);
    #[cfg(not(target_arch = "wasm32"))]
    let _ = &set_delta;

    let run_diag = move |_| {
        set_status.set("Running".into());
        #[cfg(target_arch = "wasm32")]
        {
            spawn_local(async move {
                let engine = WebGpuScanEngine::new(99);
                let candles = synthetic_candles(256, 101.0);
                let dims = vec![
                    ScanDimension {
                        name: "ema".into(),
                        min: 8.0,
                        max: 16.0,
                        steps: 3,
                    },
                    ScanDimension {
                        name: "band".into(),
                        min: 0.8,
                        max: 1.2,
                        steps: 2,
                    },
                    ScanDimension {
                        name: "stop".into(),
                        min: 1.0,
                        max: 2.0,
                        steps: 2,
                    },
                ];
                let req = ScanRequest {
                    script_or_indicator_id: "diag".into(),
                    dimensions: dims,
                    evaluation_window: EvaluationWindow::LastNBars(200),
                    metric: ScanMetric::Sharpe,
                    cost_model: CostModel::default(),
                    candles: candles.clone(),
                };
                match engine.scan(req).await {
                    Ok(res) => {
                        if let Some(first) = res.points.first() {
                            let params = SampledParams {
                                ema_len: *first.coords.get(0).unwrap_or(&10.0) as f32,
                                band_width: *first.coords.get(1).unwrap_or(&1.0) as f32,
                                atr_stop_mult: *first.coords.get(2).unwrap_or(&1.0) as f32,
                                atr_target_mult: 1.0,
                                risk_per_trade: 0.01,
                            };
                            let ohlc: Vec<Ohlc> = candles
                                .iter()
                                .map(|c| Ohlc {
                                    open: c.open as f32,
                                    high: c.high as f32,
                                    low: c.low as f32,
                                    close: c.close as f32,
                                })
                                .collect();
                            let cpu =
                                cpu_reference_metrics(&ohlc, &params, &CostModel::default());
                            let cpu_metric = cpu.sharpe as f64;
                            let gpu_metric = first.metric_value;
                            set_delta.set((gpu_metric - cpu_metric).abs());
                            set_status.set("Complete".into());
                            return;
                        }
                        set_status.set("No results".into());
                    }
                    Err(e) => set_status.set(format!("Err: {e}")),
                }
            });
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            set_status.set("Unavailable in SSR".into());
        }
    };

    view! {
        <div class="panel lab-panel" style="margin-top: var(--space-2);">
            <div class="section-label">Diagnostics</div>
            <div class="flex-row flex-between">
                <span>Status: {status}</span>
                <span>Δ Sharpe: {move || format!("{:.5}", delta.get())}</span>
            </div>
            <button on:click=run_diag>Run GPU vs CPU check</button>
        </div>
    }
}
