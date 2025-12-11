use leptos::*;
use scan_engine::ScanMetric;
use std::rc::Rc;

#[cfg(target_arch = "wasm32")]
use gpu_param_sweep::{
    BacktestMetrics, CostModel, GpuBacktester, GridConfig, Ohlc, ParamRange, SampledParams,
};
#[cfg(target_arch = "wasm32")]
use scan_engine::{
    cpu_reference_metrics, EvaluationWindow, ScanDimension, ScanEngine, ScanRequest,
    ScanResultPoint, WebGpuScanEngine,
};
#[cfg(target_arch = "wasm32")]
use std::cmp::Ordering;
#[cfg(target_arch = "wasm32")]
use std::collections::HashSet;
#[cfg(target_arch = "wasm32")]
use std::str::FromStr;
#[cfg(target_arch = "wasm32")]
use ts_core::{Candle, TimeFrame};

#[cfg(target_arch = "wasm32")]
use gloo_net::http::Request;
#[cfg(target_arch = "wasm32")]
use js_sys::encode_uri_component;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::spawn_local;

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum FieldKind {
    Ema,
    Band,
    Stop,
    Target,
    Risk,
}

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Debug)]
struct ParamField {
    id: u32,
    label: String,
    values: String,
    kind: FieldKind,
}

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Debug)]
struct SweepRow {
    id: usize,
    params: SampledParams,
    labels: Vec<(String, f64)>,
    metrics: BacktestMetrics,
}

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SortKey {
    Sharpe,
    ProfitFactor,
    FinalEquity,
    MaxDrawdown,
    WinRate,
    Trades,
}

#[derive(Clone, Debug)]
struct LabPoint {
    coords: Vec<f64>,
    value: f64,
}

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Debug)]
struct Trade {
    side: &'static str,
    entry: f64,
    exit: f64,
    pnl: f64,
    r: f64,
    bars: u32,
}

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Debug)]
struct EquityPoint {
    ts: i64,
    equity: f64,
}

#[cfg(target_arch = "wasm32")]
fn default_fields_for_template(key: &str) -> Vec<ParamField> {
    match key {
        "rsi-revert" => vec![
            ParamField {
                id: 1,
                label: "Mean lookback (EMA proxy)".into(),
                values: "10, 20, 30, 40".into(),
                kind: FieldKind::Ema,
            },
            ParamField {
                id: 2,
                label: "Band width".into(),
                values: "0.8, 1.0, 1.2, 1.6".into(),
                kind: FieldKind::Band,
            },
            ParamField {
                id: 3,
                label: "ATR stop".into(),
                values: "1.0, 1.5, 2.0".into(),
                kind: FieldKind::Stop,
            },
            ParamField {
                id: 4,
                label: "ATR target".into(),
                values: "2.0, 2.5, 3.0".into(),
                kind: FieldKind::Target,
            },
            ParamField {
                id: 5,
                label: "Risk per trade %".into(),
                values: "0.25, 0.5, 1.0".into(),
                kind: FieldKind::Risk,
            },
        ],
        "breakout-atr" => vec![
            ParamField {
                id: 1,
                label: "EMA length".into(),
                values: "20, 40, 60, 80".into(),
                kind: FieldKind::Ema,
            },
            ParamField {
                id: 2,
                label: "Band width".into(),
                values: "1.0, 1.5, 2.0, 2.5".into(),
                kind: FieldKind::Band,
            },
            ParamField {
                id: 3,
                label: "ATR stop".into(),
                values: "1.5, 2.0, 2.5".into(),
                kind: FieldKind::Stop,
            },
            ParamField {
                id: 4,
                label: "ATR target".into(),
                values: "2.5, 3.0, 4.0".into(),
                kind: FieldKind::Target,
            },
            ParamField {
                id: 5,
                label: "Risk per trade %".into(),
                values: "0.25, 0.5, 1.0".into(),
                kind: FieldKind::Risk,
            },
        ],
        _ => vec![
            ParamField {
                id: 1,
                label: "EMA length".into(),
                values: "20, 30, 40, 50".into(),
                kind: FieldKind::Ema,
            },
            ParamField {
                id: 2,
                label: "Band width".into(),
                values: "1.0, 1.5, 2.0, 2.5".into(),
                kind: FieldKind::Band,
            },
            ParamField {
                id: 3,
                label: "ATR stop".into(),
                values: "1.5, 2.0, 2.5".into(),
                kind: FieldKind::Stop,
            },
            ParamField {
                id: 4,
                label: "ATR target".into(),
                values: "3.0, 3.5, 4.0".into(),
                kind: FieldKind::Target,
            },
            ParamField {
                id: 5,
                label: "Risk per trade %".into(),
                values: "0.25, 0.5, 1.0".into(),
                kind: FieldKind::Risk,
            },
        ],
    }
}

#[cfg(target_arch = "wasm32")]
fn parse_values(raw: &str) -> Vec<f64> {
    let mut vals: Vec<f64> = raw
        .split(',')
        .filter_map(|v| v.trim().parse::<f64>().ok())
        .collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    vals.dedup();
    vals
}

#[cfg(target_arch = "wasm32")]
fn uniform_step(values: &[f64]) -> Option<f64> {
    if values.len() <= 1 {
        return Some(1.0);
    }
    let step = values[1] - values[0];
    if step.abs() < 1e-9 {
        return None;
    }
    let ok = values
        .windows(2)
        .all(|w| ((w[1] - w[0]) - step).abs() < 1e-6);
    if ok {
        Some(step)
    } else {
        None
    }
}

#[cfg(target_arch = "wasm32")]
fn kind_index(kind: FieldKind) -> usize {
    match kind {
        FieldKind::Ema => 0,
        FieldKind::Band => 1,
        FieldKind::Stop => 2,
        FieldKind::Target => 3,
        FieldKind::Risk => 4,
    }
}

#[cfg(target_arch = "wasm32")]
fn kind_key(kind: FieldKind) -> &'static str {
    match kind {
        FieldKind::Ema => "ema",
        FieldKind::Band => "band",
        FieldKind::Stop => "stop",
        FieldKind::Target => "target",
        FieldKind::Risk => "risk",
    }
}

#[cfg(target_arch = "wasm32")]
fn parse_kind(s: &str) -> FieldKind {
    match s {
        "band" => FieldKind::Band,
        "stop" => FieldKind::Stop,
        "target" => FieldKind::Target,
        "risk" => FieldKind::Risk,
        _ => FieldKind::Ema,
    }
}

#[cfg(target_arch = "wasm32")]
fn decode_combo(idx: usize, counts: &[usize; 5], values: &[Vec<f64>; 5]) -> SampledParams {
    let mut id = idx;
    let ema_idx = id % counts[0];
    id /= counts[0];
    let band_idx = id % counts[1];
    id /= counts[1];
    let stop_idx = id % counts[2];
    id /= counts[2];
    let target_idx = id % counts[3];
    id /= counts[3];
    let risk_idx = id % counts[4];

    SampledParams {
        ema_len: *values[0].get(ema_idx).unwrap_or(&20.0) as f32,
        band_width: *values[1].get(band_idx).unwrap_or(&1.5) as f32,
        atr_stop_mult: *values[2].get(stop_idx).unwrap_or(&1.5) as f32,
        atr_target_mult: *values[3].get(target_idx).unwrap_or(&3.0) as f32,
        risk_per_trade: *values[4].get(risk_idx).unwrap_or(&0.5) as f32,
    }
}

#[cfg(target_arch = "wasm32")]
fn format_pct(v: f64) -> String {
    format!("{:.1}%", v * 100.0)
}

#[cfg(target_arch = "wasm32")]
fn format_dd(v: f64) -> String {
    format!("{:.1}%", v * 100.0)
}

#[cfg(target_arch = "wasm32")]
fn metric_value(row: &SweepRow, key: SortKey) -> f64 {
    match key {
        SortKey::Sharpe => row.metrics.sharpe as f64,
        SortKey::ProfitFactor => row.metrics.profit_factor as f64,
        SortKey::FinalEquity => row.metrics.final_equity as f64,
        SortKey::MaxDrawdown => -(row.metrics.max_drawdown as f64),
        SortKey::WinRate => {
            if row.metrics.num_trades == 0 {
                0.0
            } else {
                row.metrics.win_trades as f64 / row.metrics.num_trades as f64
            }
        }
        SortKey::Trades => row.metrics.num_trades as f64,
    }
}

#[cfg(target_arch = "wasm32")]
fn sort_rows(rows: &mut [SweepRow], key: SortKey, desc: bool) {
    rows.sort_by(|a, b| {
        let lhs = metric_value(a, key);
        let rhs = metric_value(b, key);
        let ord = lhs.partial_cmp(&rhs).unwrap_or(Ordering::Equal);
        if desc {
            ord.reverse()
        } else {
            ord
        }
    });
}

#[cfg(target_arch = "wasm32")]
fn available_kind(existing: &[ParamField]) -> Option<FieldKind> {
    let used: HashSet<FieldKind> = existing.iter().map(|f| f.kind).collect();
    [
        FieldKind::Ema,
        FieldKind::Band,
        FieldKind::Stop,
        FieldKind::Target,
        FieldKind::Risk,
    ]
    .into_iter()
    .find(|k| !used.contains(k))
}

#[cfg(target_arch = "wasm32")]
fn limit_for_timeframe(tf: TimeFrame) -> usize {
    let bars = (60.0 * ts_core::DAY_MS as f64) / (tf.duration_ms() as f64);
    bars.ceil() as usize + 8
}

#[cfg(target_arch = "wasm32")]
fn values_by_kind(fields: &[ParamField]) -> Result<[Vec<f64>; 5], String> {
    let mut out: [Vec<f64>; 5] = [vec![20.0], vec![1.5], vec![1.5], vec![3.0], vec![0.005]];
    for f in fields {
        let vals = parse_values(&f.values);
        if vals.is_empty() {
            return Err(format!("{} has no numeric values", f.label));
        }
        match f.kind {
            FieldKind::Ema => out[0] = vals,
            FieldKind::Band => out[1] = vals,
            FieldKind::Stop => out[2] = vals,
            FieldKind::Target => out[3] = vals,
            FieldKind::Risk => {
                out[4] = vals.into_iter().map(|v| v / 100.0).collect();
            }
        }
    }
    Ok(out)
}

#[cfg(target_arch = "wasm32")]
fn grid_from_values(values: &[Vec<f64>; 5]) -> Result<(GridConfig, [usize; 5]), String> {
    let mut ranges = [
        ParamRange {
            min: 20.0,
            max: 20.0,
            step: 1.0,
        },
        ParamRange {
            min: 1.5,
            max: 1.5,
            step: 0.1,
        },
        ParamRange {
            min: 1.5,
            max: 1.5,
            step: 0.1,
        },
        ParamRange {
            min: 3.0,
            max: 3.0,
            step: 0.1,
        },
        ParamRange {
            min: 0.5,
            max: 0.5,
            step: 0.01,
        },
    ];
    let mut counts = [1usize; 5];
    for i in 0..5 {
        let vals = &values[i];
        counts[i] = vals.len().max(1);
        if vals.len() >= 2 {
            let step = uniform_step(vals)
                .ok_or_else(|| "Values must be evenly spaced (e.g. 20, 30, 40)".to_string())?;
            ranges[i] = ParamRange {
                min: vals.first().copied().unwrap_or(0.0) as f32,
                max: vals.last().copied().unwrap_or(0.0) as f32,
                step: step as f32,
            };
        } else if let Some(v) = vals.first() {
            ranges[i] = ParamRange {
                min: *v as f32,
                max: *v as f32,
                step: 1.0,
            };
        }
    }

    Ok((
        GridConfig {
            ema: ranges[0],
            band: ranges[1],
            atr_stop: ranges[2],
            atr_target: ranges[3],
            risk: ranges[4],
        },
        counts,
    ))
}

#[cfg(target_arch = "wasm32")]
fn ohlc_from_candles(candles: &[Candle]) -> Vec<Ohlc> {
    candles
        .iter()
        .map(|c| Ohlc {
            open: c.open as f32,
            high: c.high as f32,
            low: c.low as f32,
            close: c.close as f32,
        })
        .collect()
}

#[cfg(target_arch = "wasm32")]
fn simulate_equity(
    candles: &[Candle],
    params: &SampledParams,
    cost: &CostModel,
) -> (Vec<EquityPoint>, Vec<Trade>) {
    if candles.len() < 2 {
        return (Vec::new(), Vec::new());
    }
    let mut equity = 1.0_f64;
    let mut peak = 1.0_f64;
    let mut points = Vec::with_capacity(candles.len());
    let mut trades = Vec::new();

    let mut ema = candles[0].close;
    let mut prev_close = ema;
    let mut atr = (candles[0].high - candles[0].low).abs();

    let alpha_ema = 2.0 / (params.ema_len as f64 + 1.0);
    let atr_len = params.ema_len as f64;
    let alpha_atr = 1.0 / atr_len;

    let mut pos_dir = 0.0_f64;
    let mut entry_price = 0.0_f64;
    let mut stop_price = 0.0_f64;
    let mut target_price = 0.0_f64;
    let mut entry_equity = 0.0_f64;
    let mut current_trade_bars = 0_u32;

    for (i, bar) in candles.iter().enumerate().skip(1) {
        let close = bar.close;
        let high = bar.high;
        let low = bar.low;

        if pos_dir != 0.0 {
            current_trade_bars = current_trade_bars.saturating_add(1);
        }

        ema = alpha_ema * close + (1.0 - alpha_ema) * ema;

        let high_low = high - low;
        let high_prev = (high - prev_close).abs();
        let low_prev = (low - prev_close).abs();
        let tr1 = high_low.max(high_prev.max(low_prev));
        atr = (atr * (atr_len - 1.0) + tr1) * alpha_atr;
        prev_close = close;

        let upper = ema + params.band_width as f64 * atr;
        let lower = ema - params.band_width as f64 * atr;

        if pos_dir != 0.0 {
            let mut exit = false;
            let mut exit_price = close;
            let mut win = false;

            if pos_dir > 0.0 {
                if high >= target_price {
                    exit = true;
                    exit_price = target_price;
                    win = true;
                } else if low <= stop_price {
                    exit = true;
                    exit_price = stop_price;
                }
            } else if low <= target_price {
                exit = true;
                exit_price = target_price;
                win = true;
            } else if high >= stop_price {
                exit = true;
                exit_price = stop_price;
            }

            if exit {
                let stop_dist = (entry_price - stop_price).abs().max(1e-6);
                let size = (entry_equity * params.risk_per_trade as f64) / stop_dist;
                let pnl = (exit_price - entry_price) * pos_dir * size;
                let notional = entry_price.abs() * size;
                let slippage = notional * cost.slippage_bps as f64 * 0.0001 * 2.0;
                let pnl_net = pnl - cost.per_trade as f64 - slippage;
                equity += pnl_net;
                let r = pnl_net / entry_equity.max(1e-6);
                trades.push(Trade {
                    side: if pos_dir > 0.0 { "Long" } else { "Short" },
                    entry: entry_price,
                    exit: exit_price,
                    pnl: pnl_net,
                    r,
                    bars: current_trade_bars,
                });

                if equity > peak {
                    peak = equity;
                }

                pos_dir = 0.0;
                current_trade_bars = 0;
            }
        }

        if pos_dir == 0.0 {
            if close > upper {
                pos_dir = 1.0;
                entry_price = close;
                entry_equity = equity;
                stop_price = close - params.atr_stop_mult as f64 * atr;
                target_price = close + params.atr_target_mult as f64 * atr;
            } else if close < lower {
                pos_dir = -1.0;
                entry_price = close;
                entry_equity = equity;
                stop_price = close + params.atr_stop_mult as f64 * atr;
                target_price = close - params.atr_target_mult as f64 * atr;
            }
        }

        points.push(EquityPoint { ts: bar.ts, equity });
    }

    if pos_dir != 0.0 {
        if let Some(last) = candles.last() {
            let close = last.close;
            let stop_dist = (entry_price - stop_price).abs().max(1e-6);
            let size = (entry_equity * params.risk_per_trade as f64) / stop_dist;
            let pnl = (close - entry_price) * pos_dir * size;
            let notional = entry_price.abs() * size;
            let slippage = notional * cost.slippage_bps as f64 * 0.0001 * 2.0;
            let pnl_net = pnl - cost.per_trade as f64 - slippage;
            equity += pnl_net;
            let r = pnl_net / entry_equity.max(1e-6);
            trades.push(Trade {
                side: if pos_dir > 0.0 { "Long" } else { "Short" },
                entry: entry_price,
                exit: close,
                pnl: pnl_net,
                r,
                bars: current_trade_bars,
            });
        }
    }

    (points, trades)
}

#[component]
pub fn ParamSweepPanel() -> impl IntoView {
    #[cfg(not(target_arch = "wasm32"))]
    {
        view! {
            <div class="panel sweep-panel">
                <div class="section-label">Parameter sweep</div>
                <p class="muted" style="margin:0;">Available in browser build (WebGPU).</p>
            </div>
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        let (template, set_template) = create_signal("ema-atr".to_string());
        let (market, set_market) = create_signal("BTC-USD".to_string());
        let (timeframe, set_timeframe) = create_signal("1h".to_string());
        let (fields, set_fields) = create_signal(default_fields_for_template("ema-atr"));
        let (results, set_results) = create_signal::<Vec<SweepRow>>(Vec::new());
        let (selected, set_selected) = create_signal::<Option<usize>>(None);
        let (sort_key, set_sort_key) = create_signal(SortKey::Sharpe);
        let (sort_desc, set_sort_desc) = create_signal(true);
        let (heat_key, set_heat_key) = create_signal(SortKey::Sharpe);
        let (x_axis, set_x_axis) = create_signal(String::new());
        let (y_axis, set_y_axis) = create_signal(String::new());
        let (is_running, set_is_running) = create_signal(false);
        let (error, set_error) = create_signal(String::new());
        let (candles, set_candles) = create_signal::<Vec<Candle>>(Vec::new());

        create_effect(move |_| {
            let f = fields.get();
            if x_axis.get().is_empty() {
                if let Some(first) = f.first() {
                    set_x_axis.set(first.label.clone());
                }
            }
            if y_axis.get().is_empty() && f.len() > 1 {
                if let Some(second) = f.get(1) {
                    set_y_axis.set(second.label.clone());
                }
            }
        });

        let apply_template = move |key: String| {
            set_template.set(key.clone());
            set_fields.set(default_fields_for_template(&key));
            set_results.set(Vec::new());
            set_selected.set(None);
            set_error.set(String::new());
        };

        let run_sweep = move |_| {
            set_is_running.set(true);
            set_error.set(String::new());
            let tf_str = timeframe.get();
            let mkt = market.get();
            let field_state = fields.get();
            spawn_local(async move {
                let tf = TimeFrame::from_str(&tf_str).unwrap_or(TimeFrame::Minutes(5));
                let limit = limit_for_timeframe(tf).min(60_000);
                let url = format!(
                    "/api/history?tf={}&limit={}&symbol={}",
                    encode_uri_component(&tf.name()),
                    limit,
                    encode_uri_component(&mkt)
                );

                let mut label_names = [
                    "EMA length".to_string(),
                    "Band width".to_string(),
                    "ATR stop".to_string(),
                    "ATR target".to_string(),
                    "Risk %".to_string(),
                ];
                for f in &field_state {
                    label_names[kind_index(f.kind)] = f.label.clone();
                }

                let candles: Vec<Candle> = match Request::get(&url).send().await {
                    Ok(resp) if resp.ok() => match resp.json().await {
                        Ok(json) => json,
                        Err(e) => {
                            set_error.set(format!("Failed to parse history: {e}"));
                            set_is_running.set(false);
                            return;
                        }
                    },
                    Ok(resp) => {
                        set_error.set(format!("History request failed: {}", resp.status()));
                        set_is_running.set(false);
                        return;
                    }
                    Err(e) => {
                        set_error.set(format!("History fetch error: {e}"));
                        set_is_running.set(false);
                        return;
                    }
                };

                if candles.len() < 50 {
                    set_error.set("Not enough candles for 60d window".into());
                    set_is_running.set(false);
                    return;
                }

                let values = match values_by_kind(&field_state) {
                    Ok(v) => v,
                    Err(e) => {
                        set_error.set(e);
                        set_is_running.set(false);
                        return;
                    }
                };
                let (grid, counts) = match grid_from_values(&values) {
                    Ok(g) => g,
                    Err(e) => {
                        set_error.set(e);
                        set_is_running.set(false);
                        return;
                    }
                };
                let combos = counts.iter().product::<usize>();
                if combos == 0 || combos > 50_000 {
                    set_error.set("Combination count too large".into());
                    set_is_running.set(false);
                    return;
                }

                let mut tester = match GpuBacktester::new().await {
                    Ok(t) => t,
                    Err(e) => {
                        set_error.set(format!("WebGPU init failed: {e}"));
                        set_is_running.set(false);
                        return;
                    }
                };

                let ohlc = ohlc_from_candles(&candles);
                let cost = CostModel::default();
                let metrics = match tester.run_param_sweep(&ohlc, &grid, cost).await {
                    Ok(res) => res,
                    Err(e) => {
                        set_error.set(format!("GPU sweep failed: {e}"));
                        set_is_running.set(false);
                        return;
                    }
                };

                let mut rows = Vec::with_capacity(metrics.len());
                for (idx, m) in metrics.into_iter().enumerate() {
                    let params = decode_combo(idx, &counts, &values);
                    let labels = vec![
                        (label_names[0].clone(), params.ema_len as f64),
                        (label_names[1].clone(), params.band_width as f64),
                        (label_names[2].clone(), params.atr_stop_mult as f64),
                        (label_names[3].clone(), params.atr_target_mult as f64),
                        (label_names[4].clone(), params.risk_per_trade as f64 * 100.0),
                    ];
                    rows.push(SweepRow {
                        id: idx,
                        params,
                        labels,
                        metrics: m,
                    });
                }
                sort_rows(&mut rows, sort_key.get(), sort_desc.get());
                let first = rows.first().map(|r| r.id);
                set_candles.set(candles);
                set_results.set(rows);
                set_selected.set(first);
                set_is_running.set(false);
            });
        };

        let toggle_sort = move |key: SortKey| {
            let same = sort_key.get() == key;
            let next_desc = if same { !sort_desc.get() } else { true };
            set_sort_key.set(key);
            set_sort_desc.set(next_desc);
            set_results.update(|rows| sort_rows(rows, key, next_desc));
        };

        let add_field = move |_| {
            set_fields.update(|f| {
                if let Some(kind) = available_kind(f) {
                    let next_id = f.iter().map(|p| p.id).max().unwrap_or(0) + 1;
                    f.push(ParamField {
                        id: next_id,
                        label: "New param".into(),
                        values: "1, 2, 3".into(),
                        kind,
                    });
                }
            });
        };

        let reset_axes = move |_| {
            set_x_axis.set(String::new());
            set_y_axis.set(String::new());
        };

        view! {
            <div class="panel sweep-panel">
                <div class="flex-between">
                    <div class="flex-row" style="gap:6px;">
                        <div class="section-label">Parameter sweep</div>
                        <span class="badge-soft">60d window</span>
                        <span class="badge-soft">WebGPU</span>
                    </div>
                    <div class="flex-row" style="gap:6px;">
                        <label class="sr-only" for="sweep-template">Parameter template</label>
                        <select
                            id="sweep-template"
                            name="sweep-template"
                            value=move || template.get()
                            on:change=move |ev| apply_template(event_target_value(&ev))
                        >
                            <option value="ema-atr">EMA + ATR bands</option>
                            <option value="rsi-revert">RSI mean reversion</option>
                            <option value="breakout-atr">Breakout + ATR stop</option>
                        </select>
                        <button on:click=run_sweep disabled=move || is_running.get()>Run sweep</button>
                    </div>
                </div>

                <div class="sweep-grid">
                    <div class="flex-col sweep-params">
                        <div class="section-label">Setup</div>
                        <div class="flex-row" style="gap:6px; flex-wrap: wrap;">
                    <label class="sr-only" for="sweep-market">Market</label>
                            <input
                        id="sweep-market"
                        name="sweep-market"
                                style="flex:1;"
                                value=move || market.get()
                                on:input=move |ev| set_market.set(event_target_value(&ev))
                                placeholder="Market (e.g. BTC-USD)"
                            />
                    <label class="sr-only" for="sweep-timeframe">Timeframe</label>
                            <select
                        id="sweep-timeframe"
                        name="sweep-timeframe"
                                value=move || timeframe.get()
                                on:change=move |ev| set_timeframe.set(event_target_value(&ev))
                            >
                                <option value="1m">1m</option>
                                <option value="5m">5m</option>
                                <option value="15m">15m</option>
                                <option value="1h">1h</option>
                                <option value="4h">4h</option>
                                <option value="1d">1d</option>
                            </select>
                        </div>
                        <div class="section-label" style="margin-top:4px;">Parameters to sweep</div>
                        <div class="flex-col" style="gap:6px;">
                            {move || {
                                fields.get().into_iter().map(|field| {
                                    let fid = field.id;
                                    let kind_id = format!("param-kind-{fid}");
                                    let label_id = format!("param-label-{fid}");
                                    let values_id = format!("param-values-{fid}");
                                    view! {
                                        <div class="param-row">
                                            <label class="sr-only" for=kind_id.clone()>Parameter kind</label>
                                            <select
                                                id=kind_id.clone()
                                                name=kind_id.clone()
                                                value=kind_key(field.kind)
                                                on:change=move |ev| {
                                                    let val = event_target_value(&ev);
                                                    set_fields.update(|f| {
                                                        if let Some(row) = f.iter_mut().find(|r| r.id == fid) {
                                                            row.kind = parse_kind(&val);
                                                        }
                                                    });
                                                }
                                            >
                                                <option value="ema">EMA length</option>
                                                <option value="band">Band width</option>
                                                <option value="stop">ATR stop</option>
                                                <option value="target">ATR target</option>
                                                <option value="risk">Risk %</option>
                                            </select>
                                            <label class="sr-only" for=label_id.clone()>Parameter label</label>
                                            <input
                                                id=label_id.clone()
                                                name=label_id.clone()
                                                style="min-width: 140px;"
                                                value=field.label.clone()
                                                on:input=move |ev| {
                                                    let val = event_target_value(&ev);
                                                    set_fields.update(|f| {
                                                        if let Some(row) = f.iter_mut().find(|r| r.id == fid) {
                                                            row.label = val.clone();
                                                        }
                                                    });
                                                }
                                            />
                                            <label class="sr-only" for=values_id.clone()>Parameter values</label>
                                            <input
                                                id=values_id.clone()
                                                name=values_id.clone()
                                                style="flex:1;"
                                                value=field.values.clone()
                                                on:input=move |ev| {
                                                    let val = event_target_value(&ev);
                                                    set_fields.update(|f| {
                                                        if let Some(row) = f.iter_mut().find(|r| r.id == fid) {
                                                            row.values = val.clone();
                                                        }
                                                    });
                                                }
                                                placeholder="Comma-separated values"
                                            />
                                            <button
                                                aria-label="Delete parameter"
                                                on:click=move |_| set_fields.update(|f| f.retain(|r| r.id != fid))
                                            >x</button>
                                        </div>
                                    }
                                }).collect_view()
                            }}
                            <div class="flex-row" style="gap:6px;">
                                <button on:click=add_field disabled=move || available_kind(&fields.get()).is_none()>+ Add parameter</button>
                                <button class="ghost" on:click=reset_axes>Reset heatmap axes</button>
                            </div>
                        </div>
                        <Show when=move || !error.get().is_empty() fallback=|| ().into_view()>
                            <div class="panel muted" style="border:1px solid var(--negative); color: var(--negative);">
                                {error}
                            </div>
                        </Show>
                        <Show when=move || is_running.get()>
                            <div class="panel muted">Running sweep...</div>
                        </Show>
                    </div>

                    <div class="flex-col sweep-summary">
                        <div class="section-label">Run summary</div>
                        <div class="flex-row" style="gap:6px; flex-wrap:wrap;">
                            <span class="badge-soft">Market: {market}</span>
                            <span class="badge-soft">TF: {timeframe}</span>
                            <span class="badge-soft">Params: {move || fields.get().len()}</span>
                            <span class="badge-soft">Rows: {move || results.get().len()}</span>
                        </div>

                        <div class="heatmap-controls">
                            <div class="section-label">Heatmap</div>
                            <div class="flex-row" style="gap:6px;">
                                <label class="sr-only" for="heatmap-x-axis">X axis</label>
                                <select
                                    id="heatmap-x-axis"
                                    name="heatmap-x-axis"
                                    value=move || x_axis.get()
                                    on:change=move |ev| set_x_axis.set(event_target_value(&ev))
                                >
                                    {move || fields.get().iter().map(|f| view! {
                                        <option value=f.label.clone()>{f.label.clone()}</option>
                                    }).collect_view()}
                                </select>
                                <label class="sr-only" for="heatmap-y-axis">Y axis</label>
                                <select
                                    id="heatmap-y-axis"
                                    name="heatmap-y-axis"
                                    value=move || y_axis.get()
                                    on:change=move |ev| set_y_axis.set(event_target_value(&ev))
                                >
                                    {move || fields.get().iter().map(|f| view! {
                                        <option value=f.label.clone()>{f.label.clone()}</option>
                                    }).collect_view()}
                                </select>
                                <label class="sr-only" for="heatmap-metric">Heatmap metric</label>
                                <select
                                    id="heatmap-metric"
                                    name="heatmap-metric"
                                    value=move || {
                                        match heat_key.get() {
                                            SortKey::Sharpe => "sharpe".to_string(),
                                            SortKey::ProfitFactor => "pf".to_string(),
                                            SortKey::FinalEquity => "equity".to_string(),
                                            SortKey::MaxDrawdown => "dd".to_string(),
                                            SortKey::WinRate => "wr".to_string(),
                                            SortKey::Trades => "trades".to_string(),
                                        }
                                    }
                                    on:change=move |ev| {
                                        let val = event_target_value(&ev);
                                        match val.as_str() {
                                            "pf" => set_heat_key.set(SortKey::ProfitFactor),
                                            "equity" => set_heat_key.set(SortKey::FinalEquity),
                                            "dd" => set_heat_key.set(SortKey::MaxDrawdown),
                                            "wr" => set_heat_key.set(SortKey::WinRate),
                                            "trades" => set_heat_key.set(SortKey::Trades),
                                            _ => set_heat_key.set(SortKey::Sharpe),
                                        }
                                    }
                                >
                                    <option value="sharpe">Sharpe</option>
                                    <option value="pf">Profit factor</option>
                                    <option value="equity">Final equity</option>
                                    <option value="dd">Drawdown</option>
                                    <option value="wr">Win rate</option>
                                    <option value="trades">Trades</option>
                                </select>
                            </div>
                        </div>

                        {move || {
                            let rows = results.get();
                            let x = x_axis.get();
                            let y = y_axis.get();
                            if rows.is_empty() || x.is_empty() || y.is_empty() || x == y {
                                return view! { <div class="panel muted">Run a sweep to populate heatmap.</div> }.into_view();
                            }

                            let mut x_vals: Vec<f64> = rows
                                .iter()
                                .filter_map(|r| r.labels.iter().find(|(k, _)| *k == x).map(|(_, v)| *v))
                                .collect();
                            x_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                            x_vals.dedup();

                            let mut y_vals: Vec<f64> = rows
                                .iter()
                                .filter_map(|r| r.labels.iter().find(|(k, _)| *k == y).map(|(_, v)| *v))
                                .collect();
                            y_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                            y_vals.dedup();

                            let key = heat_key.get();
                            let mut min_v = f64::MAX;
                            let mut max_v = f64::MIN;
                            for r in rows.iter() {
                                let v = metric_value(r, key);
                                if v < min_v { min_v = v; }
                                if v > max_v { max_v = v; }
                            }
                            let span = (max_v - min_v).max(1e-6);
                            let template = format!("grid-template-columns: 100px repeat({}, minmax(64px, 1fr));", x_vals.len());

                            view! {
                                <div class="heatmap-grid" style=template>
                                    <div></div>
                                    {x_vals.iter().map(|v| view! { <div class="heatmap-axis">{format!("{v:.2}")}</div> }).collect_view()}
                                    {y_vals.iter().map(|yv| {
                                        view! {
                                            <div class="heatmap-axis">{format!("{yv:.2}")}</div>
                                            {x_vals.iter().map(|xv| {
                                                let cell = rows.iter().find(|r| {
                                                    r.labels.iter().find(|(k, v)| *k == x && (*v - xv).abs() < 1e-6).is_some()
                                                        && r.labels.iter().find(|(k, v)| *k == y && (*v - yv).abs() < 1e-6).is_some()
                                                });
                                                if let Some(row) = cell {
                                                    let raw = metric_value(row, key);
                                                    let t = (raw - min_v) / span;
                                                    let alpha = 0.2 + t * 0.7;
                                                    let color = if key == SortKey::MaxDrawdown {
                                                        format!("rgba(240, 99, 92, {alpha:.2})")
                                                    } else {
                                                        format!("rgba(92, 176, 255, {alpha:.2})")
                                                    };
                                                    view! {
                                                        <div class="heat-cell" style=format!("background:{color};")>
                                                            {format!("{raw:.2}")}
                                                        </div>
                                                    }
                                                } else {
                                                    view! { <div class="heat-cell muted">-</div> }
                                                }
                                            }).collect_view()}
                                        }
                                    }).collect_view()}
                                </div>
                            }.into_view()
                        }}
                    </div>
                </div>

                <div class="section-label" style="margin-top:4px;">Results</div>
                <div class="results-table">
                    <div class="table-scroll">
                        <table class="sweep-table">
                            <thead>
                                <tr>
                                    <th>Params</th>
                                    <th on:click=move |_| toggle_sort(SortKey::Sharpe)>Sharpe</th>
                                    <th on:click=move |_| toggle_sort(SortKey::ProfitFactor)>PF</th>
                                    <th on:click=move |_| toggle_sort(SortKey::FinalEquity)>Final Eq</th>
                                    <th on:click=move |_| toggle_sort(SortKey::MaxDrawdown)>Max DD</th>
                                    <th on:click=move |_| toggle_sort(SortKey::WinRate)>Win%</th>
                                    <th on:click=move |_| toggle_sort(SortKey::Trades)>Trades</th>
                                </tr>
                            </thead>
                            <tbody>
                                {move || {
                                    let mut rows = results.get();
                                    sort_rows(&mut rows, sort_key.get(), sort_desc.get());
                                    rows.into_iter().map(|row| {
                                        let is_selected = selected.get() == Some(row.id);
                                        let row_id = row.id;
                                        let win_rate = if row.metrics.num_trades == 0 {
                                            0.0
                                        } else {
                                            row.metrics.win_trades as f64 / row.metrics.num_trades as f64
                                        };
                                        let params_label = row.labels.iter().map(|(k,v)| format!("{k}: {v:.2}")).collect::<Vec<_>>().join(" • ");
                                        view! {
                                            <tr
                                                class=move || if is_selected { "active-row" } else { "" }
                                                on:click=move |_| set_selected.set(Some(row_id))
                                            >
                                                <td>{params_label}</td>
                                                <td>{format!("{:.2}", row.metrics.sharpe)}</td>
                                                <td>{format!("{:.2}", row.metrics.profit_factor)}</td>
                                                <td>{format!("{:.0}", row.metrics.final_equity)}</td>
                                                <td>{format_dd(row.metrics.max_drawdown as f64)}</td>
                                                <td>{format_pct(win_rate)}</td>
                                                <td>{row.metrics.num_trades}</td>
                                            </tr>
                                        }
                                    }).collect_view()
                                }}
                            </tbody>
                        </table>
                    </div>
                </div>

                <div class="selection-panel">
                    {move || {
                        let rows = results.get();
                        if rows.is_empty() {
                            return view! { <div class="panel muted">Run the sweep to see per-combo details.</div> }.into_view();
                        }
                        let picked = selected.get().and_then(|id| rows.iter().find(|r| r.id == id).cloned());
                        if let Some(row) = picked {
                            let (curve, trades) = simulate_equity(&candles.get(), &row.params, &CostModel::default());
                            let win_rate = if row.metrics.num_trades == 0 {
                                0.0
                            } else {
                                row.metrics.win_trades as f64 / row.metrics.num_trades as f64
                            };
                            let max_val = curve.iter().map(|p| p.equity).fold(f64::MIN, f64::max);
                            let min_val = curve.iter().map(|p| p.equity).fold(f64::MAX, f64::min);
                            let span = (max_val - min_val).max(1.0);
                            return view! {
                                <div class="flex-col" style="gap:6px;">
                                    <div class="section-label">Selected configuration</div>
                                    <div class="flex-row" style="gap:6px; flex-wrap:wrap;">
                                        {row.labels.iter().map(|(k,v)| view! { <span class="badge-soft">{format!("{k}: {v:.2}")}</span> }).collect_view()}
                                    </div>
                                    <div class="flex-row" style="gap:10px; flex-wrap:wrap;">
                                        <div class="metric-card">
                                            <div class="section-label">Sharpe</div>
                                            <strong>{format!("{:.2}", row.metrics.sharpe)}</strong>
                                        </div>
                                        <div class="metric-card">
                                            <div class="section-label">PF</div>
                                            <strong>{format!("{:.2}", row.metrics.profit_factor)}</strong>
                                        </div>
                                        <div class="metric-card">
                                            <div class="section-label">Max DD</div>
                                            <strong>{format_dd(row.metrics.max_drawdown as f64)}</strong>
                                        </div>
                                        <div class="metric-card">
                                            <div class="section-label">Win%</div>
                                            <strong>{format_pct(win_rate)}</strong>
                                        </div>
                                    </div>
                                    <div class="section-label">Equity curve (normalized)</div>
                                    <div class="sparkline">
                                        {curve.iter().map(|p| {
                                            let t = (p.equity - min_val) / span;
                                            let h = (t * 60.0).max(4.0);
                                            view! { <div class="spark-bar" style=format!("height:{h}px")></div> }
                                        }).collect_view()}
                                    </div>
                                    <div class="section-label">Trades</div>
                                    <div class="table-scroll" style="max-height:140px;">
                                        <table class="sweep-table">
                                            <thead>
                                                <tr>
                                                    <th>Side</th>
                                                    <th>Entry</th>
                                                    <th>Exit</th>
                                                    <th>PnL</th>
                                                    <th>R</th>
                                                    <th>Bars</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {trades.into_iter().map(|t| {
                                                    view! {
                                                        <tr>
                                                            <td>{t.side}</td>
                                                            <td>{format!("{:.2}", t.entry)}</td>
                                                            <td>{format!("{:.2}", t.exit)}</td>
                                                            <td>{format!("{:.2}", t.pnl)}</td>
                                                            <td>{format!("{:.2}", t.r)}</td>
                                                            <td>{t.bars}</td>
                                                        </tr>
                                                    }
                                                }).collect_view()}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            }.into_view();
                        }
                        view! { <div class="panel muted">Select a row to inspect its stats.</div> }.into_view()
                    }}
                </div>
            </div>
        }
    }
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
                <label class="section-label" for="gpu-metric">Metric</label>
                <select
                    id="gpu-metric"
                    name="gpu-metric"
                    value=move || metric_idx.get().to_string()
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
                        sorted
                            .into_iter()
                            .map(|p| {
                                let label = format!(
                                    "{:.2} -> {:.4}",
                                    p.coords.first().copied().unwrap_or(0.0),
                                    p.value
                                );
                                view! { <div class="chip">{label}</div> }
                            })
                            .collect_view()
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
                                ema_len: first.coords.first().copied().unwrap_or(10.0) as f32,
                                band_width: first.coords.get(1).copied().unwrap_or(1.0) as f32,
                                atr_stop_mult: first.coords.get(2).copied().unwrap_or(1.0) as f32,
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
                            let cpu = cpu_reference_metrics(&ohlc, &params, &CostModel::default());
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
            <button on:click=run_diag disabled=cfg!(not(target_arch = "wasm32"))>Run GPU vs CPU check</button>
        </div>
    }
}
