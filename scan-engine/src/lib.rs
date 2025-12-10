use async_trait::async_trait;
use futures::{
    channel::mpsc::{unbounded, UnboundedReceiver},
    future::BoxFuture,
    FutureExt,
};
use gpu_param_sweep::{
    BacktestMetrics, CostModel, GridConfig, GpuBacktester, Ohlc, ParamRange, SampledParams,
};
use rand::{rngs::StdRng, seq::IteratorRandom, SeedableRng};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use ts_core::{Candle, Timestamp};

/// Dimensional definition for a parameter sweep.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScanDimension {
    pub name: String,
    pub min: f64,
    pub max: f64,
    pub steps: usize,
}

/// Metric options that can be produced by the scanning engine.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ScanMetric {
    Sharpe,
    ProfitFactor,
    FinalEquity,
    MaxDrawdown,
    HitRate,
    WinRate,
    Trades,
}

/// Which part of the series to evaluate.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EvaluationWindow {
    FullSeries,
    LastNBars(usize),
    LastDays(u32),
    DateRange { start: Timestamp, end: Timestamp },
    Regimes(Vec<u8>),
}

/// A full scan request that is portable across CPU/GPU engines.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScanRequest {
    pub script_or_indicator_id: String,
    pub dimensions: Vec<ScanDimension>,
    pub evaluation_window: EvaluationWindow,
    pub metric: ScanMetric,
    #[serde(default)]
    pub cost_model: CostModel,
    pub candles: Vec<Candle>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScanResultPoint {
    pub coords: Vec<f64>,
    pub metric_value: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScanResult {
    pub request: ScanRequest,
    pub points: Vec<ScanResultPoint>,
}

/// Incremental chunk for streaming visual updates.
#[derive(Clone, Debug)]
pub struct ScanResultChunk {
    pub progress: f32,
    pub partial: Vec<ScanResultPoint>,
}

pub type ScanStream = UnboundedReceiver<ScanResultChunk>;

#[derive(Error, Debug)]
pub enum ScanError {
    #[error("invalid request: {0}")]
    Invalid(String),
    #[error("gpu error: {0}")]
    Gpu(String),
    #[error("cpu validation failed: {0}")]
    Validation(String),
}

#[async_trait]
pub trait ScanEngine: Send + Sync {
    async fn scan(&self, request: ScanRequest) -> Result<ScanResult, ScanError>;

    /// Streaming variant that emits partial progress and returns the full result
    /// when complete. The returned future should be awaited by the caller to
    /// obtain the final `ScanResult`.
    async fn scan_streaming(
        &self,
        request: ScanRequest,
    ) -> Result<(ScanStream, BoxFuture<'static, Result<ScanResult, ScanError>>), ScanError>;
}

/// WebGPU-backed implementation using the existing `gpu_param_sweep` crate,
/// with a lightweight CPU validation loop to ensure deterministic correctness.
#[derive(Clone, Default)]
pub struct WebGpuScanEngine {
    seed: u64,
}

impl WebGpuScanEngine {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    fn slice_window<'a>(&self, request: &ScanRequest, candles: &'a [Candle]) -> &'a [Candle] {
        match &request.evaluation_window {
            EvaluationWindow::FullSeries => candles,
            EvaluationWindow::LastNBars(n) => {
                let len = candles.len();
                if *n >= len {
                    candles
                } else {
                    &candles[len - n..]
                }
            }
            EvaluationWindow::LastDays(days) => {
                if candles.is_empty() {
                    return candles;
                }
                let end = candles.last().unwrap().ts;
                let start = end - (ts_core::DAY_MS * *days as i64);
                let start_idx = candles.iter().position(|c| c.ts >= start).unwrap_or(0);
                &candles[start_idx..]
            }
            EvaluationWindow::DateRange { start, end } => {
                let start_idx = candles.iter().position(|c| c.ts >= *start).unwrap_or(0);
                let end_idx = candles
                    .iter()
                    .position(|c| c.ts >= *end)
                    .unwrap_or(candles.len());
                &candles[start_idx..end_idx]
            }
            EvaluationWindow::Regimes(_) => candles, // regime slicing handled upstream
        }
    }

    fn to_ohlc_f32(&self, candles: &[Candle]) -> Vec<Ohlc> {
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

    fn grid_from_request(&self, req: &ScanRequest) -> Result<GridConfig, ScanError> {
        if req.dimensions.is_empty() || req.dimensions.len() > 3 {
            return Err(ScanError::Invalid(
                "dimensions must contain 1–3 entries".to_string(),
            ));
        }

        let mut ranges = [
            ParamRange {
                min: 10.0,
                max: 10.0,
                step: 1.0,
            }, // ema
            ParamRange {
                min: 1.0,
                max: 1.0,
                step: 1.0,
            }, // band
            ParamRange {
                min: 1.0,
                max: 1.0,
                step: 1.0,
            }, // stop
            ParamRange {
                min: 1.0,
                max: 1.0,
                step: 1.0,
            }, // target
            ParamRange {
                min: 0.01,
                max: 0.01,
                step: 0.01,
            }, // risk
        ];

        for (i, d) in req.dimensions.iter().enumerate() {
            let step = if d.steps <= 1 {
                1.0
            } else {
                ((d.max - d.min) / ((d.steps as f64 - 1.0).max(1.0))) as f32
            };
            let range = ParamRange {
                min: d.min as f32,
                max: d.max as f32,
                step,
            };
            match i {
                0 => ranges[0] = range,
                1 => ranges[1] = range,
                2 => ranges[2] = range,
                _ => {}
            }
        }

        Ok(GridConfig {
            ema: ranges[0],
            band: ranges[1],
            atr_stop: ranges[2],
            atr_target: ranges[3],
            risk: ranges[4],
        })
    }

    fn metric_value(metric: &ScanMetric, m: &BacktestMetrics) -> f64 {
        match metric {
            ScanMetric::Sharpe => m.sharpe as f64,
            ScanMetric::ProfitFactor => m.profit_factor as f64,
            ScanMetric::FinalEquity => m.final_equity as f64,
            ScanMetric::MaxDrawdown => m.max_drawdown as f64,
            ScanMetric::HitRate | ScanMetric::WinRate => {
                if m.num_trades == 0 {
                    0.0
                } else {
                    m.win_trades as f64 / m.num_trades as f64
                }
            }
            ScanMetric::Trades => m.num_trades as f64,
        }
    }

    fn coords_from_params(&self, dims: &[ScanDimension], params: &SampledParams) -> Vec<f64> {
        dims.iter()
            .enumerate()
            .map(|(i, _)| match i {
                0 => params.ema_len as f64,
                1 => params.band_width as f64,
                2 => params.atr_stop_mult as f64,
                _ => 0.0,
            })
            .collect()
    }

    fn to_points(
        &self,
        dims: &[ScanDimension],
        metric: &ScanMetric,
        gpu_results: &[BacktestMetrics],
    ) -> Vec<ScanResultPoint> {
        gpu_results
            .iter()
            .map(|g| ScanResultPoint {
                coords: self.coords_from_params(dims, &g.params),
                metric_value: Self::metric_value(metric, g),
            })
            .collect()
    }

    fn validate_subset(
        &self,
        metric: &ScanMetric,
        gpu_results: &[BacktestMetrics],
        ohlc: &[Ohlc],
        cost_model: &CostModel,
    ) -> Result<(), ScanError> {
        if gpu_results.is_empty() || ohlc.len() < 2 {
            return Ok(());
        }
        let mut rng = StdRng::seed_from_u64(self.seed);
        let sample_size = gpu_results.len().min(5);
        for gpu_m in gpu_results.iter().choose_multiple(&mut rng, sample_size) {
            let cpu_m = cpu_reference_metrics(ohlc, &gpu_m.params, cost_model);
            let gv = Self::metric_value(metric, gpu_m);
            let cv = Self::metric_value(metric, &cpu_m);
            let diff = (gv - cv).abs();
            let tol = 1e-3;
            if diff > tol && !(gv.is_nan() && cv.is_nan()) {
                return Err(ScanError::Validation(format!(
                    "metric mismatch gpu {gv:.6} vs cpu {cv:.6}"
                )));
            }
        }
        Ok(())
    }
}

#[async_trait]
impl ScanEngine for WebGpuScanEngine {
    async fn scan(&self, request: ScanRequest) -> Result<ScanResult, ScanError> {
        let windowed = self.slice_window(&request, &request.candles).to_vec();
        if windowed.is_empty() {
            return Err(ScanError::Invalid("no candles to scan".into()));
        }
        let ohlc = self.to_ohlc_f32(&windowed);
        let grid = self.grid_from_request(&request)?;
        let cost_model = request.cost_model;

        let mut gpu = GpuBacktester::new()
            .await
            .map_err(|e| ScanError::Gpu(e.to_string()))?;

        let gpu_results = gpu
            .run_param_sweep(&ohlc, &grid, cost_model)
            .await
            .map_err(|e| ScanError::Gpu(e.to_string()))?;

        self.validate_subset(&request.metric, &gpu_results, &ohlc, &cost_model)?;

        let points = self.to_points(&request.dimensions, &request.metric, &gpu_results);
        Ok(ScanResult { request, points })
    }

    async fn scan_streaming(
        &self,
        request: ScanRequest,
    ) -> Result<(ScanStream, BoxFuture<'static, Result<ScanResult, ScanError>>), ScanError> {
        let (tx, rx) = unbounded();
        let engine = self.clone();
        let fut = async move {
            let result = engine.scan(request).await;
            if let Ok(ref res) = result {
                let _ = tx.unbounded_send(ScanResultChunk {
                    progress: 1.0,
                    partial: res.points.clone(),
                });
            }
            result
        }
        .boxed();
        Ok((rx, fut))
    }
}

/// CPU reference implementation that mirrors the WGSL kernel for validation and
/// CPU-only fallback.
/// CPU reference for diagnostics and validation.
pub fn cpu_reference_metrics(
    ohlc: &[Ohlc],
    params: &SampledParams,
    cost_model: &CostModel,
) -> BacktestMetrics {
    if ohlc.len() < 2 {
        return BacktestMetrics {
            params: params.clone(),
            final_equity: 1.0,
            profit_factor: 0.0,
            sharpe: 0.0,
            max_drawdown: 0.0,
            num_trades: 0,
            win_trades: 0,
            loss_trades: 0,
            total_trade_bars: 0,
            bars_with_position: 0,
            average_trade_bars: 0.0,
            exposure: 0.0,
        };
    }

    let ema_len = params.ema_len.max(1.0);
    let band_width = params.band_width.max(0.0);
    let stop_mult = params.atr_stop_mult.max(0.01);
    let target_mult = params.atr_target_mult.max(0.01);
    let risk = params.risk_per_trade.max(0.00001);

    let alpha_ema = 2.0 / (ema_len + 1.0);
    let atr_len = ema_len;
    let alpha_atr = 1.0 / atr_len;

    let mut ema = ohlc[0].close;
    let mut prev_close = ema;
    let mut atr = (ohlc[0].high - ohlc[0].low).abs();

    let mut equity = 1.0f32;
    let mut equity_peak = 1.0f32;
    let mut max_dd = 0.0f32;

    let mut pos_dir = 0.0f32;
    let mut entry_price = 0.0f32;
    let mut stop_price = 0.0f32;
    let mut target_price = 0.0f32;
    let mut entry_equity = 0.0f32;

    let mut trades = 0u32;
    let mut win_trades = 0u32;
    let mut loss_trades = 0u32;
    let mut gross_profit = 0.0f32;
    let mut gross_loss = 0.0f32;
    let mut mean_r = 0.0f32;
    let mut m2_r = 0.0f32;
    let mut total_trade_bars = 0u32;
    let mut current_trade_bars = 0u32;
    let mut bars_with_position = 0u32;

    for bar in &ohlc[1..] {
        let close = bar.close;
        let high = bar.high;
        let low = bar.low;

        if pos_dir != 0.0 {
            current_trade_bars += 1;
            bars_with_position += 1;
        }

        ema = alpha_ema * close + (1.0 - alpha_ema) * ema;

        let high_low = high - low;
        let high_prev = (high - prev_close).abs();
        let low_prev = (low - prev_close).abs();
        let tr1 = high_low.max(high_prev.max(low_prev));
        atr = (atr * (atr_len - 1.0) + tr1) * alpha_atr;
        prev_close = close;

        let upper = ema + band_width * atr;
        let lower = ema - band_width * atr;

        if pos_dir != 0.0 {
            let mut exit = false;
            let mut exit_price = close;
            if pos_dir > 0.0 {
                if high >= target_price {
                    exit = true;
                    exit_price = target_price;
                    win_trades += 1;
                } else if low <= stop_price {
                    exit = true;
                    exit_price = stop_price;
                    loss_trades += 1;
                }
            } else if low <= target_price {
                exit = true;
                exit_price = target_price;
                win_trades += 1;
            } else if high >= stop_price {
                exit = true;
                exit_price = stop_price;
                loss_trades += 1;
            }

            if exit {
                let stop_dist = (entry_price - stop_price).abs().max(1e-6);
                let size = (entry_equity * risk) / stop_dist;
                let pnl = (exit_price - entry_price) * pos_dir * size;
                let notional = entry_price.abs() * size;
                let slippage_cost = notional * cost_model.slippage_bps * 0.0001 * 2.0;
                let pnl_net = pnl - cost_model.per_trade - slippage_cost;
                equity += pnl_net;
                trades += 1;

                let eq_before = entry_equity.max(1e-6);
                let r = pnl_net / eq_before;
                let t = trades as f32;
                let delta = r - mean_r;
                mean_r += delta / t;
                let delta2 = r - mean_r;
                m2_r += delta * delta2;

                if pnl_net > 0.0 {
                    gross_profit += pnl_net;
                } else {
                    gross_loss += -pnl_net;
                }

                if equity > equity_peak {
                    equity_peak = equity;
                } else {
                    let dd = (equity_peak - equity) / equity_peak.max(1e-6);
                    if dd > max_dd {
                        max_dd = dd;
                    }
                }

                total_trade_bars += current_trade_bars;
                current_trade_bars = 0;
                pos_dir = 0.0;
            }
        }

        if pos_dir == 0.0 {
            if close > upper {
                pos_dir = 1.0;
                entry_price = close;
                entry_equity = equity;
                stop_price = close - stop_mult * atr;
                target_price = close + target_mult * atr;
            } else if close < lower {
                pos_dir = -1.0;
                entry_price = close;
                entry_equity = equity;
                stop_price = close + stop_mult * atr;
                target_price = close - target_mult * atr;
            }
        }
    }

    if pos_dir != 0.0 {
        let last = ohlc.last().unwrap();
        let stop_dist = (entry_price - stop_price).abs().max(1e-6);
        let size = (entry_equity * risk) / stop_dist;
        let pnl = (last.close - entry_price) * pos_dir * size;
        let notional = entry_price.abs() * size;
        let slippage_cost = notional * cost_model.slippage_bps * 0.0001 * 2.0;
        let pnl_net = pnl - cost_model.per_trade - slippage_cost;
        equity += pnl_net;
        trades += 1;

        let eq_before = entry_equity.max(1e-6);
        let r = pnl_net / eq_before;
        let t = trades as f32;
        let delta = r - mean_r;
        mean_r += delta / t;
        let delta2 = r - mean_r;
        m2_r += delta * delta2;

        if pnl_net > 0.0 {
            gross_profit += pnl_net;
        } else {
            gross_loss += -pnl_net;
        }

        let dd = if equity_peak > 0.0 {
            (equity_peak - equity) / equity_peak.max(1e-6)
        } else {
            0.0
        };
        if dd > max_dd {
            max_dd = dd;
        }
        total_trade_bars += current_trade_bars;
    }

    let profit_factor = if gross_loss > 0.0 {
        gross_profit / gross_loss
    } else {
        0.0
    };
    let sharpe = if trades > 1 {
        let variance = m2_r / (trades as f32 - 1.0);
        if variance > 0.0 {
            mean_r / variance.sqrt()
        } else {
            0.0
        }
    } else {
        0.0
    };
    let avg_trade_bars = if trades > 0 {
        total_trade_bars as f32 / trades as f32
    } else {
        0.0
    };
    let exposure = if !ohlc.is_empty() {
        bars_with_position as f32 / ohlc.len() as f32
    } else {
        0.0
    };

    BacktestMetrics {
        params: params.clone(),
        final_equity: equity,
        profit_factor,
        sharpe,
        max_drawdown: max_dd,
        num_trades: trades,
        win_trades,
        loss_trades,
        total_trade_bars,
        bars_with_position,
        average_trade_bars: avg_trade_bars,
        exposure,
    }
}
