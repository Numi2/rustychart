use serde::{Deserialize, Serialize};
use std::{collections::VecDeque, fmt, str::FromStr};
use ts_core::{Candle, HasTimestamp, TimeFrame, TimeSeries, Timestamp, DAY_MS};

pub type IndicatorId = u64;

/// Which candle field to use as input.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceField {
    Open,
    High,
    Low,
    Close,
    Hlc3,
    Ohlc4,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseSourceFieldError;

impl fmt::Display for ParseSourceFieldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("unknown source field")
    }
}

impl std::error::Error for ParseSourceFieldError {}

impl FromStr for SourceField {
    type Err = ParseSourceFieldError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match normalized(s).as_str() {
            "open" => Ok(SourceField::Open),
            "high" => Ok(SourceField::High),
            "low" => Ok(SourceField::Low),
            "close" => Ok(SourceField::Close),
            "hlc3" => Ok(SourceField::Hlc3),
            "ohlc4" => Ok(SourceField::Ohlc4),
            _ => Err(ParseSourceFieldError),
        }
    }
}

impl SourceField {
    pub fn value(&self, c: &Candle) -> f64 {
        match self {
            SourceField::Open => c.open,
            SourceField::High => c.high,
            SourceField::Low => c.low,
            SourceField::Close => c.close,
            SourceField::Hlc3 => (c.high + c.low + c.close) / 3.0,
            SourceField::Ohlc4 => (c.open + c.high + c.low + c.close) / 4.0,
        }
    }
}

/// Supported indicator kinds.
///
/// Outputs:
/// - Sma/Ema: single line (overlay)
/// - Rsi/Atr: single line (separate pane)
/// - Macd: macd, signal, histogram (separate pane)
/// - Bbands: mid/upper/lower (overlay)
/// - Stoch: %K, %D (separate pane)
/// - Vwap: anchored VWAP (overlay)
/// - Cci: commodity channel index (separate pane)
/// - Vwmo: volume-weighted moving oscillator/average (separate pane)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IndicatorKind {
    Sma,
    Ema,
    Rsi,
    Macd,
    Bbands,
    Atr,
    Stoch,
    Vwap,
    Cci,
    Vwmo,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseIndicatorKindError;

impl fmt::Display for ParseIndicatorKindError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("unknown indicator kind")
    }
}

impl std::error::Error for ParseIndicatorKindError {}

impl FromStr for IndicatorKind {
    type Err = ParseIndicatorKindError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match normalized(s).as_str() {
            "sma" | "ma" | "simple_ma" => Ok(IndicatorKind::Sma),
            "ema" => Ok(IndicatorKind::Ema),
            "rsi" => Ok(IndicatorKind::Rsi),
            "macd" => Ok(IndicatorKind::Macd),
            "bbands" | "bollinger" | "bollinger_bands" => Ok(IndicatorKind::Bbands),
            "atr" => Ok(IndicatorKind::Atr),
            "stoch" | "stochastic" => Ok(IndicatorKind::Stoch),
            "vwap" => Ok(IndicatorKind::Vwap),
            "cci" => Ok(IndicatorKind::Cci),
            "vwmo" | "vwma" | "vwmo_avg" => Ok(IndicatorKind::Vwmo),
            _ => Err(ParseIndicatorKindError),
        }
    }
}

/// Where to draw the indicator.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum OutputKind {
    Overlay,
    SeparatePane,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseOutputKindError;

impl fmt::Display for ParseOutputKindError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("unknown output kind")
    }
}

impl std::error::Error for ParseOutputKindError {}

impl FromStr for OutputKind {
    type Err = ParseOutputKindError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match normalized(s).as_str() {
            "overlay" | "main" | "price" => Ok(OutputKind::Overlay),
            "pane" | "panel" | "separate" | "separate_pane" => Ok(OutputKind::SeparatePane),
            _ => Err(ParseOutputKindError),
        }
    }
}

/// Parameters for supported indicators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndicatorParams {
    /// Simple moving average of a candle source over `period`.
    Sma { period: usize, source: SourceField },
    /// Exponential moving average of a candle source over `period`.
    Ema { period: usize, source: SourceField },
    /// Relative Strength Index (Wilder) of `period`.
    Rsi { period: usize, source: SourceField },
    /// MACD: fast/slow EMAs and signal EMA on the diff.
    Macd {
        fast: usize,
        slow: usize,
        signal: usize,
        source: SourceField,
    },
    /// Bollinger Bands: middle SMA, upper/lower by `stddev` over `period`.
    Bbands {
        period: usize,
        stddev: f64,
        source: SourceField,
    },
    /// Average True Range (Wilder) over `period`.
    Atr { period: usize },
    /// Stochastic oscillator: %K over `k_period`, %D SMA over `d_period`.
    Stoch { k_period: usize, d_period: usize },
    /// Volume-weighted average price; resets by session/day if enabled.
    Vwap { reset_each_day: bool },
    /// Commodity Channel Index over `period` with scaling `constant` (typically 0.015).
    Cci {
        period: usize,
        source: SourceField,
        constant: f64,
    },
    /// Volume-weighted moving oscillator/average over `period`.
    Vwmo { period: usize, source: SourceField },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LinePattern {
    Solid,
    Dashed,
    Dotted,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineStyle {
    pub color: String,
    pub width: f64,
    pub pattern: LinePattern,
}

/// Config for an indicator instance (but no runtime state).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndicatorConfig {
    pub kind: IndicatorKind,
    pub params: IndicatorParams,
    pub output: OutputKind,
    /// For separate-pane indicators, identifies which pane they belong to.
    pub pane_id: Option<u32>,
    pub line_styles: Vec<LineStyle>,
}

impl IndicatorConfig {
    pub fn with_default_styles(
        kind: IndicatorKind,
        params: IndicatorParams,
        output: OutputKind,
        pane_id: Option<u32>,
    ) -> Self {
        let dim = indicator_output_dimension(kind);
        let line_styles = default_line_styles(kind, dim);
        Self {
            kind,
            params,
            output,
            pane_id,
            line_styles,
        }
    }
}

/// Single indicator sample for arbitrary-dimension output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndicatorSample {
    pub ts: Timestamp,
    pub values: Vec<f64>,
}

impl HasTimestamp for IndicatorSample {
    fn ts(&self) -> Timestamp {
        self.ts
    }
}

pub type IndicatorSeries = TimeSeries<IndicatorSample>;

/// Indicator computation engine – stateful, incremental.
pub trait IndicatorEngine {
    fn kind(&self) -> IndicatorKind;
    fn output_dimension(&self) -> usize;
    fn reset(&mut self);

    fn apply_history(&mut self, candles: &[Candle]) -> Vec<IndicatorSample> {
        self.reset();
        let mut out = Vec::new();
        for c in candles {
            if let Some(s) = self.apply_incremental(c) {
                out.push(s);
            }
        }
        out
    }

    fn apply_incremental(&mut self, candle: &Candle) -> Option<IndicatorSample>;
}

// ---------- helper for EMA smoothing ----------------------------------------

struct EmaSub {
    period: usize,
    alpha: f64,
    window: VecDeque<f64>,
    sum: f64,
    count: usize,
    ema: Option<f64>,
}

impl EmaSub {
    fn new(period: usize) -> Self {
        let alpha = 2.0 / (period as f64 + 1.0);
        Self {
            period,
            alpha,
            window: VecDeque::new(),
            sum: 0.0,
            count: 0,
            ema: None,
        }
    }

    fn reset(&mut self) {
        self.window.clear();
        self.sum = 0.0;
        self.count = 0;
        self.ema = None;
    }

    /// Returns EMA once enough history is accumulated.
    fn next(&mut self, value: f64) -> Option<f64> {
        self.count += 1;
        self.window.push_back(value);
        self.sum += value;
        if self.window.len() > self.period {
            if let Some(old) = self.window.pop_front() {
                self.sum -= old;
            }
        }

        if self.count < self.period {
            return None;
        }

        if self.ema.is_none() {
            let ema = self.sum / self.period as f64;
            self.ema = Some(ema);
            Some(ema)
        } else {
            let prev = self.ema.unwrap();
            let ema = self.alpha * value + (1.0 - self.alpha) * prev;
            self.ema = Some(ema);
            Some(ema)
        }
    }
}

// ---------- individual indicator engines -----------------------------------

struct SmaEngine {
    period: usize,
    source: SourceField,
    window: VecDeque<f64>,
    sum: f64,
}

impl SmaEngine {
    fn new(period: usize, source: SourceField) -> Self {
        Self {
            period,
            source,
            window: VecDeque::new(),
            sum: 0.0,
        }
    }
}

impl IndicatorEngine for SmaEngine {
    fn kind(&self) -> IndicatorKind {
        IndicatorKind::Sma
    }

    fn output_dimension(&self) -> usize {
        1
    }

    fn reset(&mut self) {
        self.window.clear();
        self.sum = 0.0;
    }

    fn apply_incremental(&mut self, candle: &Candle) -> Option<IndicatorSample> {
        let v = self.source.value(candle);
        self.window.push_back(v);
        self.sum += v;
        if self.window.len() > self.period {
            if let Some(old) = self.window.pop_front() {
                self.sum -= old;
            }
        }
        if self.window.len() == self.period {
            let ma = self.sum / self.period as f64;
            Some(IndicatorSample {
                ts: candle.ts,
                values: vec![ma],
            })
        } else {
            None
        }
    }
}

struct EmaEngine {
    source: SourceField,
    ema: EmaSub,
}

impl EmaEngine {
    fn new(period: usize, source: SourceField) -> Self {
        Self {
            source,
            ema: EmaSub::new(period),
        }
    }
}

impl IndicatorEngine for EmaEngine {
    fn kind(&self) -> IndicatorKind {
        IndicatorKind::Ema
    }

    fn output_dimension(&self) -> usize {
        1
    }

    fn reset(&mut self) {
        self.ema.reset();
    }

    fn apply_incremental(&mut self, candle: &Candle) -> Option<IndicatorSample> {
        let v = self.source.value(candle);
        self.ema.next(v).map(|ema| IndicatorSample {
            ts: candle.ts,
            values: vec![ema],
        })
    }
}

struct RsiEngine {
    period: usize,
    source: SourceField,
    prev_price: Option<f64>,
    gain_sum: f64,
    loss_sum: f64,
    count: usize,
    avg_gain: Option<f64>,
    avg_loss: Option<f64>,
}

impl RsiEngine {
    fn new(period: usize, source: SourceField) -> Self {
        Self {
            period,
            source,
            prev_price: None,
            gain_sum: 0.0,
            loss_sum: 0.0,
            count: 0,
            avg_gain: None,
            avg_loss: None,
        }
    }
}

impl IndicatorEngine for RsiEngine {
    fn kind(&self) -> IndicatorKind {
        IndicatorKind::Rsi
    }

    fn output_dimension(&self) -> usize {
        1
    }

    fn reset(&mut self) {
        self.prev_price = None;
        self.gain_sum = 0.0;
        self.loss_sum = 0.0;
        self.count = 0;
        self.avg_gain = None;
        self.avg_loss = None;
    }

    fn apply_incremental(&mut self, candle: &Candle) -> Option<IndicatorSample> {
        let price = self.source.value(candle);
        let mut result = None;

        if let Some(prev) = self.prev_price {
            let delta = price - prev;
            let gain = delta.max(0.0);
            let loss = (-delta).max(0.0);

            if self.count < self.period {
                self.gain_sum += gain;
                self.loss_sum += loss;
                self.count += 1;

                if self.count == self.period {
                    let avg_gain = self.gain_sum / self.period as f64;
                    let avg_loss = self.loss_sum / self.period as f64;
                    self.avg_gain = Some(avg_gain);
                    self.avg_loss = Some(avg_loss);

                    let rsi = if avg_loss == 0.0 {
                        100.0
                    } else {
                        let rs = avg_gain / avg_loss;
                        100.0 - 100.0 / (1.0 + rs)
                    };
                    result = Some(IndicatorSample {
                        ts: candle.ts,
                        values: vec![rsi],
                    });
                }
            } else {
                let period_f = self.period as f64;
                let mut avg_gain = self.avg_gain.unwrap_or(0.0);
                let mut avg_loss = self.avg_loss.unwrap_or(0.0);

                avg_gain = (avg_gain * (period_f - 1.0) + gain) / period_f;
                avg_loss = (avg_loss * (period_f - 1.0) + loss) / period_f;

                self.avg_gain = Some(avg_gain);
                self.avg_loss = Some(avg_loss);

                let rsi = if avg_loss == 0.0 {
                    100.0
                } else {
                    let rs = avg_gain / avg_loss;
                    100.0 - 100.0 / (1.0 + rs)
                };

                result = Some(IndicatorSample {
                    ts: candle.ts,
                    values: vec![rsi],
                });
            }
        }

        self.prev_price = Some(price);
        result
    }
}

struct MacdEngine {
    source: SourceField,
    ema_fast: EmaSub,
    ema_slow: EmaSub,
    signal: EmaSub,
}

impl MacdEngine {
    fn new(fast: usize, slow: usize, signal: usize, source: SourceField) -> Self {
        Self {
            source,
            ema_fast: EmaSub::new(fast),
            ema_slow: EmaSub::new(slow),
            signal: EmaSub::new(signal),
        }
    }
}

impl IndicatorEngine for MacdEngine {
    fn kind(&self) -> IndicatorKind {
        IndicatorKind::Macd
    }

    fn output_dimension(&self) -> usize {
        3 // macd line, signal line, histogram
    }

    fn reset(&mut self) {
        self.ema_fast.reset();
        self.ema_slow.reset();
        self.signal.reset();
    }

    fn apply_incremental(&mut self, candle: &Candle) -> Option<IndicatorSample> {
        let v = self.source.value(candle);
        let fast = self.ema_fast.next(v)?;
        let slow = self.ema_slow.next(v)?;
        let macd = fast - slow;

        if let Some(signal) = self.signal.next(macd) {
            let hist = macd - signal;
            Some(IndicatorSample {
                ts: candle.ts,
                values: vec![macd, signal, hist],
            })
        } else {
            None
        }
    }
}

struct BbandsEngine {
    period: usize,
    stddev_mult: f64,
    source: SourceField,
    window: VecDeque<f64>,
    sum: f64,
    sum_sq: f64,
}

impl BbandsEngine {
    fn new(period: usize, stddev_mult: f64, source: SourceField) -> Self {
        Self {
            period,
            stddev_mult,
            source,
            window: VecDeque::new(),
            sum: 0.0,
            sum_sq: 0.0,
        }
    }
}

impl IndicatorEngine for BbandsEngine {
    fn kind(&self) -> IndicatorKind {
        IndicatorKind::Bbands
    }

    fn output_dimension(&self) -> usize {
        3 // middle, upper, lower
    }

    fn reset(&mut self) {
        self.window.clear();
        self.sum = 0.0;
        self.sum_sq = 0.0;
    }

    fn apply_incremental(&mut self, candle: &Candle) -> Option<IndicatorSample> {
        let v = self.source.value(candle);
        self.window.push_back(v);
        self.sum += v;
        self.sum_sq += v * v;

        if self.window.len() > self.period {
            if let Some(old) = self.window.pop_front() {
                self.sum -= old;
                self.sum_sq -= old * old;
            }
        }

        if self.window.len() == self.period {
            let n = self.period as f64;
            let mean = self.sum / n;
            let var = (self.sum_sq / n - mean * mean).max(0.0);
            let stddev = var.sqrt();
            let upper = mean + self.stddev_mult * stddev;
            let lower = mean - self.stddev_mult * stddev;

            Some(IndicatorSample {
                ts: candle.ts,
                values: vec![mean, upper, lower],
            })
        } else {
            None
        }
    }
}

struct AtrEngine {
    period: usize,
    prev_close: Option<f64>,
    tr_sum: f64,
    count: usize,
    atr: Option<f64>,
}

impl AtrEngine {
    fn new(period: usize) -> Self {
        Self {
            period,
            prev_close: None,
            tr_sum: 0.0,
            count: 0,
            atr: None,
        }
    }
}

impl IndicatorEngine for AtrEngine {
    fn kind(&self) -> IndicatorKind {
        IndicatorKind::Atr
    }

    fn output_dimension(&self) -> usize {
        1
    }

    fn reset(&mut self) {
        self.prev_close = None;
        self.tr_sum = 0.0;
        self.count = 0;
        self.atr = None;
    }

    fn apply_incremental(&mut self, candle: &Candle) -> Option<IndicatorSample> {
        let tr = if let Some(prev_close) = self.prev_close {
            let h_l = candle.high - candle.low;
            let h_pc = (candle.high - prev_close).abs();
            let l_pc = (candle.low - prev_close).abs();
            h_l.max(h_pc).max(l_pc)
        } else {
            candle.high - candle.low
        };

        self.prev_close = Some(candle.close);

        let mut result = None;

        if self.count < self.period {
            self.tr_sum += tr;
            self.count += 1;

            if self.count == self.period {
                let atr = self.tr_sum / self.period as f64;
                self.atr = Some(atr);
                result = Some(IndicatorSample {
                    ts: candle.ts,
                    values: vec![atr],
                });
            }
        } else {
            let period_f = self.period as f64;
            let prev_atr = self.atr.unwrap_or(tr);
            let atr = (prev_atr * (period_f - 1.0) + tr) / period_f;
            self.atr = Some(atr);
            result = Some(IndicatorSample {
                ts: candle.ts,
                values: vec![atr],
            });
        }

        result
    }
}

struct StochEngine {
    k_period: usize,
    d_period: usize,
    highs: VecDeque<f64>,
    lows: VecDeque<f64>,
    k_history: VecDeque<f64>,
}

impl StochEngine {
    fn new(k_period: usize, d_period: usize) -> Self {
        Self {
            k_period,
            d_period,
            highs: VecDeque::with_capacity(k_period + 1),
            lows: VecDeque::with_capacity(k_period + 1),
            k_history: VecDeque::with_capacity(d_period + 1),
        }
    }
}

impl IndicatorEngine for StochEngine {
    fn kind(&self) -> IndicatorKind {
        IndicatorKind::Stoch
    }

    fn output_dimension(&self) -> usize {
        2 // %K and %D
    }

    fn reset(&mut self) {
        self.highs.clear();
        self.lows.clear();
        self.k_history.clear();
    }

    fn apply_incremental(&mut self, candle: &Candle) -> Option<IndicatorSample> {
        self.highs.push_back(candle.high);
        self.lows.push_back(candle.low);
        if self.highs.len() > self.k_period {
            self.highs.pop_front();
        }
        if self.lows.len() > self.k_period {
            self.lows.pop_front();
        }

        if self.highs.len() < self.k_period || self.lows.len() < self.k_period {
            return None;
        }

        let highest = self.highs.iter().cloned().fold(f64::MIN, f64::max);
        let lowest = self.lows.iter().cloned().fold(f64::MAX, f64::min);
        let range = (highest - lowest).max(1e-12);
        let k = ((candle.close - lowest) / range * 100.0).clamp(0.0, 100.0);

        self.k_history.push_back(k);
        if self.k_history.len() > self.d_period {
            self.k_history.pop_front();
        }
        let d = if self.k_history.len() == self.d_period {
            self.k_history.iter().sum::<f64>() / self.d_period as f64
        } else {
            f64::NAN
        };

        Some(IndicatorSample {
            ts: candle.ts,
            values: vec![k, d],
        })
    }
}

struct VwapEngine {
    reset_each_day: bool,
    cur_day: Option<i64>,
    pv_sum: f64,
    vol_sum: f64,
}

impl VwapEngine {
    fn new(reset_each_day: bool) -> Self {
        Self {
            reset_each_day,
            cur_day: None,
            pv_sum: 0.0,
            vol_sum: 0.0,
        }
    }
}

impl IndicatorEngine for VwapEngine {
    fn kind(&self) -> IndicatorKind {
        IndicatorKind::Vwap
    }

    fn output_dimension(&self) -> usize {
        1
    }

    fn reset(&mut self) {
        self.cur_day = None;
        self.pv_sum = 0.0;
        self.vol_sum = 0.0;
    }

    fn apply_incremental(&mut self, candle: &Candle) -> Option<IndicatorSample> {
        if self.reset_each_day {
            let day = candle.ts.div_euclid(DAY_MS);
            if self.cur_day != Some(day) {
                self.cur_day = Some(day);
                self.pv_sum = 0.0;
                self.vol_sum = 0.0;
            }
        }

        let typical = (candle.high + candle.low + candle.close) / 3.0;
        self.pv_sum += typical * candle.volume;
        self.vol_sum += candle.volume.max(1e-12);
        let vwap = self.pv_sum / self.vol_sum;

        Some(IndicatorSample {
            ts: candle.ts,
            values: vec![vwap],
        })
    }
}

struct CciEngine {
    period: usize,
    source: SourceField,
    constant: f64,
    window: VecDeque<f64>,
}

impl CciEngine {
    fn new(period: usize, source: SourceField, constant: f64) -> Self {
        Self {
            period,
            source,
            constant,
            window: VecDeque::with_capacity(period + 1),
        }
    }
}

impl IndicatorEngine for CciEngine {
    fn kind(&self) -> IndicatorKind {
        IndicatorKind::Cci
    }

    fn output_dimension(&self) -> usize {
        1
    }

    fn reset(&mut self) {
        self.window.clear();
    }

    fn apply_incremental(&mut self, candle: &Candle) -> Option<IndicatorSample> {
        let tp = self.source.value(candle);
        self.window.push_back(tp);
        if self.window.len() > self.period {
            self.window.pop_front();
        }
        if self.window.len() < self.period {
            return None;
        }
        let mean = self.window.iter().sum::<f64>() / self.period as f64;
        let dev = self.window.iter().map(|v| (v - mean).abs()).sum::<f64>() / self.period as f64;
        let cci = if dev.abs() < 1e-12 {
            0.0
        } else {
            (tp - mean) / (self.constant * dev)
        };
        Some(IndicatorSample {
            ts: candle.ts,
            values: vec![cci],
        })
    }
}

struct VwmoEngine {
    period: usize,
    source: SourceField,
    window: VecDeque<(f64, f64)>, // (price, volume)
}

impl VwmoEngine {
    fn new(period: usize, source: SourceField) -> Self {
        Self {
            period,
            source,
            window: VecDeque::with_capacity(period + 1),
        }
    }
}

impl IndicatorEngine for VwmoEngine {
    fn kind(&self) -> IndicatorKind {
        IndicatorKind::Vwmo
    }

    fn output_dimension(&self) -> usize {
        1
    }

    fn reset(&mut self) {
        self.window.clear();
    }

    fn apply_incremental(&mut self, candle: &Candle) -> Option<IndicatorSample> {
        let price = self.source.value(candle);
        self.window.push_back((price, candle.volume));
        if self.window.len() > self.period {
            self.window.pop_front();
        }
        if self.window.len() < self.period {
            return None;
        }
        let mut pv_sum = 0.0;
        let mut v_sum = 0.0;
        for (p, v) in &self.window {
            pv_sum += *p * *v;
            v_sum += *v;
        }
        let vwmo = pv_sum / v_sum.max(1e-12);
        Some(IndicatorSample {
            ts: candle.ts,
            values: vec![vwmo],
        })
    }
}

// ---------- factory & defaults ----------------------------------------------

fn indicator_output_dimension(kind: IndicatorKind) -> usize {
    match kind {
        IndicatorKind::Macd | IndicatorKind::Bbands => 3,
        IndicatorKind::Stoch => 2,
        _ => 1,
    }
}

fn default_line_styles(kind: IndicatorKind, dim: usize) -> Vec<LineStyle> {
    let mut out = Vec::with_capacity(dim);
    match kind {
        IndicatorKind::Sma => {
            out.push(LineStyle {
                color: "#7ee0ff".to_string(),
                width: 1.8,
                pattern: LinePattern::Solid,
            });
        }
        IndicatorKind::Ema => {
            out.push(LineStyle {
                color: "#ff8ba7".to_string(),
                width: 1.8,
                pattern: LinePattern::Solid,
            });
        }
        IndicatorKind::Rsi => {
            out.push(LineStyle {
                color: "#7dd3fc".to_string(),
                width: 1.6,
                pattern: LinePattern::Solid,
            });
        }
        IndicatorKind::Macd => {
            out.push(LineStyle {
                color: "#4ade80".to_string(),
                width: 1.8,
                pattern: LinePattern::Solid,
            });
            out.push(LineStyle {
                color: "#f472b6".to_string(),
                width: 1.6,
                pattern: LinePattern::Solid,
            });
            out.push(LineStyle {
                color: "#94a3b8".to_string(),
                width: 1.2,
                pattern: LinePattern::Dotted,
            });
        }
        IndicatorKind::Bbands => {
            out.push(LineStyle {
                color: "#60a5fa".to_string(),
                width: 1.2,
                pattern: LinePattern::Solid,
            });
            out.push(LineStyle {
                color: "#93c5fd".to_string(),
                width: 1.0,
                pattern: LinePattern::Dashed,
            });
            out.push(LineStyle {
                color: "#93c5fd".to_string(),
                width: 1.0,
                pattern: LinePattern::Dashed,
            });
        }
        IndicatorKind::Atr => {
            out.push(LineStyle {
                color: "#fbbf24".to_string(),
                width: 1.6,
                pattern: LinePattern::Solid,
            });
        }
        IndicatorKind::Stoch => {
            out.push(LineStyle {
                color: "#c084fc".to_string(),
                width: 1.4,
                pattern: LinePattern::Solid,
            });
            out.push(LineStyle {
                color: "#facc15".to_string(),
                width: 1.2,
                pattern: LinePattern::Dashed,
            });
        }
        IndicatorKind::Vwap => {
            out.push(LineStyle {
                color: "#8b5cf6".to_string(),
                width: 1.8,
                pattern: LinePattern::Solid,
            });
        }
        IndicatorKind::Cci => {
            out.push(LineStyle {
                color: "#22d3ee".to_string(),
                width: 1.6,
                pattern: LinePattern::Solid,
            });
        }
        IndicatorKind::Vwmo => {
            out.push(LineStyle {
                color: "#38bdf8".to_string(),
                width: 1.7,
                pattern: LinePattern::Solid,
            });
        }
    }

    // If dim > declared ones (shouldn't happen), clone last.
    while out.len() < dim {
        let last = out.last().cloned().unwrap_or(LineStyle {
            color: "#ffffff".to_string(),
            width: 1.0,
            pattern: LinePattern::Solid,
        });
        out.push(last);
    }

    out.truncate(dim);
    out
}

fn create_engine(kind: IndicatorKind, params: &IndicatorParams) -> Box<dyn IndicatorEngine> {
    match (kind, params) {
        (IndicatorKind::Sma, IndicatorParams::Sma { period, source }) => {
            Box::new(SmaEngine::new(*period, *source))
        }
        (IndicatorKind::Ema, IndicatorParams::Ema { period, source }) => {
            Box::new(EmaEngine::new(*period, *source))
        }
        (IndicatorKind::Rsi, IndicatorParams::Rsi { period, source }) => {
            Box::new(RsiEngine::new(*period, *source))
        }
        (
            IndicatorKind::Macd,
            IndicatorParams::Macd {
                fast,
                slow,
                signal,
                source,
            },
        ) => Box::new(MacdEngine::new(*fast, *slow, *signal, *source)),
        (
            IndicatorKind::Bbands,
            IndicatorParams::Bbands {
                period,
                stddev,
                source,
            },
        ) => Box::new(BbandsEngine::new(*period, *stddev, *source)),
        (IndicatorKind::Atr, IndicatorParams::Atr { period }) => Box::new(AtrEngine::new(*period)),
        (IndicatorKind::Stoch, IndicatorParams::Stoch { k_period, d_period }) => {
            Box::new(StochEngine::new(*k_period, *d_period))
        }
        (IndicatorKind::Vwap, IndicatorParams::Vwap { reset_each_day }) => {
            Box::new(VwapEngine::new(*reset_each_day))
        }
        (
            IndicatorKind::Cci,
            IndicatorParams::Cci {
                period,
                source,
                constant,
            },
        ) => Box::new(CciEngine::new(*period, *source, *constant)),
        (IndicatorKind::Vwmo, IndicatorParams::Vwmo { period, source }) => {
            Box::new(VwmoEngine::new(*period, *source))
        }
        _ => panic!("indicator params mismatch for {kind:?}"),
    }
}

/// One running indicator instance: config + engine + time-series.
pub struct IndicatorInstance {
    pub id: IndicatorId,
    pub config: IndicatorConfig,
    pub series: IndicatorSeries,
    engine: Box<dyn IndicatorEngine>,
}

impl IndicatorInstance {
    pub fn output_dimension(&self) -> usize {
        self.engine.output_dimension()
    }

    pub fn series(&self) -> &IndicatorSeries {
        &self.series
    }
}

/// Manager for all indicators on a single chart / symbol / timeframe.
pub struct IndicatorManager {
    base_timeframe: TimeFrame,
    indicators: Vec<IndicatorInstance>,
    next_id: IndicatorId,
}

impl IndicatorManager {
    pub fn new(base_timeframe: TimeFrame) -> Self {
        Self {
            base_timeframe,
            indicators: Vec::new(),
            next_id: 1,
        }
    }

    pub fn base_timeframe(&self) -> TimeFrame {
        self.base_timeframe
    }

    pub fn indicators(&self) -> &[IndicatorInstance] {
        &self.indicators
    }

    /// Add an indicator and build its series from existing history.
    pub fn add_indicator(&mut self, config: IndicatorConfig, history: &[Candle]) -> IndicatorId {
        let id = self.next_id;
        self.next_id += 1;

        let mut engine = create_engine(config.kind, &config.params);
        let samples = engine.apply_history(history);

        let mut series = IndicatorSeries::new();
        series.append_batch(samples);

        let instance = IndicatorInstance {
            id,
            config,
            series,
            engine,
        };
        self.indicators.push(instance);
        id
    }

    pub fn remove_indicator(&mut self, id: IndicatorId) {
        if let Some(idx) = self.indicators.iter().position(|i| i.id == id) {
            self.indicators.remove(idx);
        }
    }

    pub fn set_line_styles(&mut self, id: IndicatorId, styles: Vec<LineStyle>) {
        if let Some(inst) = self.indicators.iter_mut().find(|i| i.id == id) {
            let dim = inst.output_dimension();
            if styles.len() == dim {
                inst.config.line_styles = styles;
            }
        }
    }

    pub fn on_new_candle(&mut self, candle: &Candle) {
        for inst in &mut self.indicators {
            if let Some(sample) = inst.engine.apply_incremental(candle) {
                inst.series.append(sample);
            }
        }
    }

    /// Rebuild all indicators from scratch using history.
    pub fn rebuild_all(&mut self, history: &[Candle]) {
        for inst in &mut self.indicators {
            inst.series = IndicatorSeries::new();
            inst.engine.reset();
            let samples = inst.engine.apply_history(history);
            inst.series.append_batch(samples);
        }
    }

    /// Clear series data but keep configs.
    pub fn clear(&mut self) {
        for inst in &mut self.indicators {
            inst.series = IndicatorSeries::new();
            inst.engine.reset();
        }
    }
}

/// Default output location if caller doesn't specify.
pub fn default_output_for(kind: IndicatorKind) -> OutputKind {
    match kind {
        IndicatorKind::Sma | IndicatorKind::Ema | IndicatorKind::Bbands | IndicatorKind::Vwap => {
            OutputKind::Overlay
        }
        IndicatorKind::Rsi
        | IndicatorKind::Macd
        | IndicatorKind::Atr
        | IndicatorKind::Stoch
        | IndicatorKind::Cci
        | IndicatorKind::Vwmo => OutputKind::SeparatePane,
    }
}

fn normalized(input: &str) -> String {
    input.trim().to_ascii_lowercase()
}

// ---------- Adaptive wrappers -------------------------------------------------

/// Parameter template tied to a specific regime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeParamTemplate {
    pub regime: usize,
    pub params: IndicatorParams,
}

/// Adaptive indicator that swaps parameters based on the active market regime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveIndicatorConfig {
    pub base: IndicatorConfig,
    pub templates: Vec<RegimeParamTemplate>,
    /// Blend window in bars for smoothing transitions (placeholder; smoothing is handled upstream).
    pub blend_window: usize,
}

impl AdaptiveIndicatorConfig {
    /// Resolve an indicator configuration for the provided regime id.
    pub fn resolve_for_regime(&self, regime: usize) -> IndicatorConfig {
        if let Some(t) = self.templates.iter().find(|t| t.regime == regime) {
            let mut cfg = self.base.clone();
            cfg.params = t.params.clone();
            cfg
        } else {
            self.base.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_candle(ts: i64, close: f64) -> Candle {
        Candle {
            ts,
            timeframe: TimeFrame::Minutes(1),
            open: close,
            high: close + 1.0,
            low: close - 1.0,
            close,
            volume: 1.0,
        }
    }

    #[test]
    fn stoch_outputs_two_lines() {
        let params = IndicatorParams::Stoch {
            k_period: 3,
            d_period: 3,
        };
        let mut mgr = IndicatorManager::new(TimeFrame::Minutes(1));
        let config = IndicatorConfig::with_default_styles(
            IndicatorKind::Stoch,
            params,
            OutputKind::SeparatePane,
            Some(0),
        );
        let candles = (0..5)
            .map(|i| mk_candle(i as i64 * 60_000, 100.0 + i as f64))
            .collect::<Vec<_>>();
        let _ = mgr.add_indicator(config, &candles);
        let series = mgr.indicators()[0].series();
        let last = series.last().unwrap();
        assert_eq!(last.values.len(), 2);
        assert!(last.values[0].is_finite());
    }

    #[test]
    fn vwap_resets_by_day_when_enabled() {
        let params = IndicatorParams::Vwap {
            reset_each_day: true,
        };
        let mut mgr = IndicatorManager::new(TimeFrame::Minutes(1));
        let config = IndicatorConfig::with_default_styles(
            IndicatorKind::Vwap,
            params,
            OutputKind::Overlay,
            None,
        );
        let mut candles = Vec::new();
        for i in 0..3 {
            candles.push(Candle {
                ts: i * 60_000,
                timeframe: TimeFrame::Minutes(1),
                open: 100.0,
                high: 101.0,
                low: 99.0,
                close: 100.0 + i as f64,
                volume: 1.0,
            });
        }
        for i in 0..3 {
            candles.push(Candle {
                ts: DAY_MS + i * 60_000,
                timeframe: TimeFrame::Minutes(1),
                open: 200.0,
                high: 201.0,
                low: 199.0,
                close: 200.0 + i as f64,
                volume: 1.0,
            });
        }
        let _ = mgr.add_indicator(config, &candles);
        let series = mgr.indicators()[0].series();
        assert!(series.len() >= 6);
        let first_day_last = series.as_slice()[2].values[0];
        let second_day_first = series.as_slice()[3].values[0];
        assert!(second_day_first > first_day_last);
    }

    #[test]
    fn cci_generates_values() {
        let params = IndicatorParams::Cci {
            period: 3,
            source: SourceField::Hlc3,
            constant: 0.015,
        };
        let mut mgr = IndicatorManager::new(TimeFrame::Minutes(1));
        let config = IndicatorConfig::with_default_styles(
            IndicatorKind::Cci,
            params,
            OutputKind::SeparatePane,
            Some(0),
        );
        let candles = (0..4)
            .map(|i| Candle {
                ts: i * 60_000,
                timeframe: TimeFrame::Minutes(1),
                open: 100.0 + i as f64,
                high: 101.0 + i as f64,
                low: 99.0 + i as f64,
                close: 100.0 + i as f64,
                volume: 1.0,
            })
            .collect::<Vec<_>>();
        let _ = mgr.add_indicator(config, &candles);
        let series = mgr.indicators()[0].series();
        assert!(!series.is_empty());
        assert!(series.last().unwrap().values[0].is_finite());
    }

    #[test]
    fn vwmo_generates_values() {
        let params = IndicatorParams::Vwmo {
            period: 3,
            source: SourceField::Close,
        };
        let mut mgr = IndicatorManager::new(TimeFrame::Minutes(1));
        let config = IndicatorConfig::with_default_styles(
            IndicatorKind::Vwmo,
            params,
            OutputKind::SeparatePane,
            Some(0),
        );
        let candles = (0..4)
            .map(|i| Candle {
                ts: i * 60_000,
                timeframe: TimeFrame::Minutes(1),
                open: 100.0,
                high: 101.0,
                low: 99.0,
                close: 100.0 + i as f64,
                volume: 1.0 + i as f64,
            })
            .collect::<Vec<_>>();
        let _ = mgr.add_indicator(config, &candles);
        let series = mgr.indicators()[0].series();
        assert!(!series.is_empty());
        assert!(series.last().unwrap().values[0].is_finite());
    }
}
