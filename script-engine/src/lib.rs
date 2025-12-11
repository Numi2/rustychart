use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};
use ts_core::{Candle, TimeFrame};

pub mod ai;
pub mod assist;
pub mod compat;
pub mod incremental;
pub mod js_facade;
pub mod language;
pub mod manifest;
pub mod docs;
pub mod parser;
pub mod remediation;
pub mod sandbox;
pub mod templates;
pub mod wasm;

use crate::language::SourceLang;

pub use ai::{generate_script, AiPromptRequest, AiSuggestion};
pub use assist::{analyze_script, CompletionItem, ScriptAnalysis};
pub use compat::{
    translate_pine, translate_thinkscript, CompatibilityIssue, CompatibilityReport, IssueSeverity,
    TranslationOutput, UnifiedIr,
};
pub use incremental::{ExprSnapshot, IncrementalRunner, ScriptCheckpoint, SignalSnapshot};
pub use js_facade::{validate_script, JsScriptHandle, ScriptArtifact};
pub use language::SourceLang as ScriptSourceLang;
pub use manifest::{
    InputKind, InputParam, Manifest, ManifestCapabilities, OutputKind, OutputSpec, Permission,
};
pub use docs::{builtin_docs, BuiltinDoc};
pub use parser::{pine, think};
pub use remediation::{suggest_fixes, FixSuggestion};
pub use sandbox::{HostEnvironment, InstanceAdapter, SandboxLimits, SandboxedScript};
pub use templates::{find_template, templates_for_lang, Template};
pub use wasm::{compile_artifact, instantiate_from_artifact, WasmArtifact, WasmSandbox, WasmTrap};

/// Primitive sources a script expression can reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Source {
    Price(PriceField),
    Series(String), // reference to a previously computed plot series
    Input(String),  // user-defined parameter
    Const(f64),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PriceField {
    Open,
    High,
    Low,
    Close,
    Volume,
}

/// Expression AST for script formulas.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Expr {
    Src(Source),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Gt(Box<Expr>, Box<Expr>),         // boolean -> 1.0 or 0.0
    Lt(Box<Expr>, Box<Expr>),         // boolean -> 1.0 or 0.0
    And(Box<Expr>, Box<Expr>),        // logical and on numeric booleans
    Or(Box<Expr>, Box<Expr>),         // logical or on numeric booleans
    CrossOver(Box<Expr>, Box<Expr>),  // emits 1.0 only on crossing up
    CrossUnder(Box<Expr>, Box<Expr>), // emits 1.0 only on crossing down
    Sma {
        period: usize,
        src: Box<Expr>,
    },
    Ema {
        period: usize,
        src: Box<Expr>,
    },
    Rsi {
        period: usize,
        src: Box<Expr>,
    },
    Macd {
        fast: usize,
        slow: usize,
        signal: usize,
        src: Box<Expr>,
    },
    IfGt {
        left: Box<Expr>,
        right: Box<Expr>,
        then_expr: Box<Expr>,
        else_expr: Box<Expr>,
    },
}

/// Signal definitions that produce alert events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignalSpec {
    CrossOver { a: Expr, b: Expr },
    CrossUnder { a: Expr, b: Expr },
    Greater { a: Expr, b: Expr },
}

/// Script specification (portable and serializable).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScriptSpec {
    pub name: String,
    /// Parameters with defaults.
    pub inputs: HashMap<String, f64>,
    /// Plot outputs (named series).
    pub plots: HashMap<String, Expr>,
    /// Signals keyed by name.
    pub signals: HashMap<String, SignalSpec>,
}

#[derive(Debug, Clone)]
struct SmaState {
    period: usize,
    window: VecDeque<f64>,
    sum: f64,
}

impl SmaState {
    fn new(period: usize) -> Self {
        Self {
            period,
            window: VecDeque::with_capacity(period + 1),
            sum: 0.0,
        }
    }

    fn push(&mut self, v: f64) -> f64 {
        self.window.push_back(v);
        self.sum += v;
        if self.window.len() > self.period {
            if let Some(old) = self.window.pop_front() {
                self.sum -= old;
            }
        }
        self.sum / (self.window.len() as f64)
    }
}

#[derive(Debug, Clone)]
struct EmaState {
    alpha: f64,
    initialized: bool,
    value: f64,
}

impl EmaState {
    fn new(period: usize) -> Self {
        let alpha = 2.0 / (period as f64 + 1.0);
        Self {
            alpha,
            initialized: false,
            value: 0.0,
        }
    }

    fn push(&mut self, v: f64) -> f64 {
        if !self.initialized {
            self.value = v;
            self.initialized = true;
        } else {
            self.value = self.alpha * v + (1.0 - self.alpha) * self.value;
        }
        self.value
    }
}

#[derive(Debug, Clone)]
struct RsiState {
    period: usize,
    avg_gain: f64,
    avg_loss: f64,
    initialized: bool,
    prev_close: Option<f64>,
}

impl RsiState {
    fn new(period: usize) -> Self {
        Self {
            period,
            avg_gain: 0.0,
            avg_loss: 0.0,
            initialized: false,
            prev_close: None,
        }
    }

    fn push(&mut self, close: f64) -> f64 {
        let prev = match self.prev_close.replace(close) {
            Some(p) => p,
            None => return 50.0,
        };
        let change = close - prev;
        let gain = if change > 0.0 { change } else { 0.0 };
        let loss = if change < 0.0 { -change } else { 0.0 };

        if !self.initialized {
            self.avg_gain =
                (self.avg_gain * (self.period as f64 - 1.0) + gain) / (self.period as f64);
            self.avg_loss =
                (self.avg_loss * (self.period as f64 - 1.0) + loss) / (self.period as f64);
            if self.prev_close.is_some() && self.avg_gain + self.avg_loss > 0.0 {
                self.initialized = true;
            }
        } else {
            self.avg_gain =
                (self.avg_gain * (self.period as f64 - 1.0) + gain) / (self.period as f64);
            self.avg_loss =
                (self.avg_loss * (self.period as f64 - 1.0) + loss) / (self.period as f64);
        }

        if self.avg_loss == 0.0 {
            100.0
        } else {
            let rs = self.avg_gain / self.avg_loss;
            100.0 - (100.0 / (1.0 + rs))
        }
    }
}

#[derive(Debug, Clone)]
struct MacdState {
    fast: EmaState,
    slow: EmaState,
    signal: EmaState,
}

impl MacdState {
    fn new(fast: usize, slow: usize, signal: usize) -> Self {
        Self {
            fast: EmaState::new(fast),
            slow: EmaState::new(slow),
            signal: EmaState::new(signal),
        }
    }

    fn push(&mut self, v: f64) -> (f64, f64, f64) {
        let fast_v = self.fast.push(v);
        let slow_v = self.slow.push(v);
        let macd = fast_v - slow_v;
        let signal = self.signal.push(macd);
        let hist = macd - signal;
        (macd, signal, hist)
    }
}

#[derive(Debug, Clone)]
enum CompiledExpr {
    Src(Source),
    Add(Box<CompiledExpr>, Box<CompiledExpr>),
    Sub(Box<CompiledExpr>, Box<CompiledExpr>),
    Mul(Box<CompiledExpr>, Box<CompiledExpr>),
    Div(Box<CompiledExpr>, Box<CompiledExpr>),
    Gt(Box<CompiledExpr>, Box<CompiledExpr>),
    Lt(Box<CompiledExpr>, Box<CompiledExpr>),
    And(Box<CompiledExpr>, Box<CompiledExpr>),
    Or(Box<CompiledExpr>, Box<CompiledExpr>),
    CrossOver {
        left: Box<CompiledExpr>,
        right: Box<CompiledExpr>,
        last: Option<(f64, f64)>,
    },
    CrossUnder {
        left: Box<CompiledExpr>,
        right: Box<CompiledExpr>,
        last: Option<(f64, f64)>,
    },
    Sma {
        state: SmaState,
        src: Box<CompiledExpr>,
    },
    Ema {
        state: EmaState,
        src: Box<CompiledExpr>,
    },
    Rsi {
        state: RsiState,
        src: Box<CompiledExpr>,
    },
    Macd {
        state: MacdState,
        src: Box<CompiledExpr>,
    },
    IfGt {
        left: Box<CompiledExpr>,
        right: Box<CompiledExpr>,
        then_expr: Box<CompiledExpr>,
        else_expr: Box<CompiledExpr>,
    },
}

impl CompiledExpr {
    fn bool_to_num(v: bool) -> f64 {
        if v {
            1.0
        } else {
            0.0
        }
    }

    fn eval(
        &mut self,
        candle: &Candle,
        plots: &HashMap<String, f64>,
        inputs: &HashMap<String, f64>,
    ) -> f64 {
        match self {
            CompiledExpr::Src(Source::Const(v)) => *v,
            CompiledExpr::Src(Source::Input(name)) => *inputs.get(name).unwrap_or(&f64::NAN),
            CompiledExpr::Src(Source::Series(name)) => *plots.get(name).unwrap_or(&f64::NAN),
            CompiledExpr::Src(Source::Price(field)) => match field {
                PriceField::Open => candle.open,
                PriceField::High => candle.high,
                PriceField::Low => candle.low,
                PriceField::Close => candle.close,
                PriceField::Volume => candle.volume,
            },
            CompiledExpr::Add(a, b) => {
                a.eval(candle, plots, inputs) + b.eval(candle, plots, inputs)
            }
            CompiledExpr::Sub(a, b) => {
                a.eval(candle, plots, inputs) - b.eval(candle, plots, inputs)
            }
            CompiledExpr::Mul(a, b) => {
                a.eval(candle, plots, inputs) * b.eval(candle, plots, inputs)
            }
            CompiledExpr::Div(a, b) => {
                let denom = b.eval(candle, plots, inputs);
                if denom.abs() < 1e-12 {
                    f64::NAN
                } else {
                    a.eval(candle, plots, inputs) / denom
                }
            }
            CompiledExpr::Gt(a, b) => {
                Self::bool_to_num(a.eval(candle, plots, inputs) > b.eval(candle, plots, inputs))
            }
            CompiledExpr::Lt(a, b) => {
                Self::bool_to_num(a.eval(candle, plots, inputs) < b.eval(candle, plots, inputs))
            }
            CompiledExpr::And(a, b) => {
                let av = a.eval(candle, plots, inputs);
                let bv = b.eval(candle, plots, inputs);
                Self::bool_to_num(av != 0.0 && bv != 0.0)
            }
            CompiledExpr::Or(a, b) => {
                let av = a.eval(candle, plots, inputs);
                let bv = b.eval(candle, plots, inputs);
                Self::bool_to_num(av != 0.0 || bv != 0.0)
            }
            CompiledExpr::CrossOver { left, right, last } => {
                let l = left.eval(candle, plots, inputs);
                let r = right.eval(candle, plots, inputs);
                let prev = *last;
                *last = Some((l, r));
                if let Some((pl, pr)) = prev {
                    Self::bool_to_num((pl - pr) <= 0.0 && (l - r) > 0.0)
                } else {
                    0.0
                }
            }
            CompiledExpr::CrossUnder { left, right, last } => {
                let l = left.eval(candle, plots, inputs);
                let r = right.eval(candle, plots, inputs);
                let prev = *last;
                *last = Some((l, r));
                if let Some((pl, pr)) = prev {
                    Self::bool_to_num((pl - pr) >= 0.0 && (l - r) < 0.0)
                } else {
                    0.0
                }
            }
            CompiledExpr::Sma { state, src } => {
                let v = src.eval(candle, plots, inputs);
                state.push(v)
            }
            CompiledExpr::Ema { state, src } => {
                let v = src.eval(candle, plots, inputs);
                state.push(v)
            }
            CompiledExpr::Rsi { state, src } => {
                let v = src.eval(candle, plots, inputs);
                state.push(v)
            }
            CompiledExpr::Macd { state, src } => {
                let v = src.eval(candle, plots, inputs);
                let (macd, _signal, _hist) = state.push(v);
                macd
            }
            CompiledExpr::IfGt {
                left,
                right,
                then_expr,
                else_expr,
            } => {
                if left.eval(candle, plots, inputs) > right.eval(candle, plots, inputs) {
                    then_expr.eval(candle, plots, inputs)
                } else {
                    else_expr.eval(candle, plots, inputs)
                }
            }
        }
    }

    fn snapshot(&self) -> ExprSnapshot {
        match self {
            CompiledExpr::Src(_) => ExprSnapshot::Stateless,
            CompiledExpr::Add(a, b)
            | CompiledExpr::Sub(a, b)
            | CompiledExpr::Mul(a, b)
            | CompiledExpr::Div(a, b)
            | CompiledExpr::Gt(a, b)
            | CompiledExpr::Lt(a, b)
            | CompiledExpr::And(a, b)
            | CompiledExpr::Or(a, b) => {
                let _ = (a, b); // stateless arithmetic/logical
                ExprSnapshot::Stateless
            }
            CompiledExpr::CrossOver { last, .. } | CompiledExpr::CrossUnder { last, .. } => {
                ExprSnapshot::Cross { last: *last }
            }
            CompiledExpr::Sma { state, .. } => ExprSnapshot::Sma {
                period: state.period,
                window: state.window.iter().cloned().collect(),
                sum: state.sum,
            },
            CompiledExpr::Ema { state, .. } => ExprSnapshot::Ema {
                alpha: state.alpha,
                initialized: state.initialized,
                value: state.value,
            },
            CompiledExpr::Rsi { state, .. } => ExprSnapshot::Rsi {
                period: state.period,
                avg_gain: state.avg_gain,
                avg_loss: state.avg_loss,
                initialized: state.initialized,
                prev_close: state.prev_close,
            },
            CompiledExpr::Macd { state, .. } => ExprSnapshot::Macd {
                fast: Box::new(ExprSnapshot::Ema {
                    alpha: state.fast.alpha,
                    initialized: state.fast.initialized,
                    value: state.fast.value,
                }),
                slow: Box::new(ExprSnapshot::Ema {
                    alpha: state.slow.alpha,
                    initialized: state.slow.initialized,
                    value: state.slow.value,
                }),
                signal: Box::new(ExprSnapshot::Ema {
                    alpha: state.signal.alpha,
                    initialized: state.signal.initialized,
                    value: state.signal.value,
                }),
            },
            CompiledExpr::IfGt {
                then_expr,
                else_expr,
                left,
                right,
            } => {
                let _ = (then_expr, else_expr, left, right);
                ExprSnapshot::Stateless
            }
        }
    }

    fn restore(&mut self, snap: &ExprSnapshot) {
        match (self, snap) {
            (CompiledExpr::CrossOver { last, .. }, ExprSnapshot::Cross { last: saved })
            | (CompiledExpr::CrossUnder { last, .. }, ExprSnapshot::Cross { last: saved }) => {
                *last = *saved;
            }
            (
                CompiledExpr::Sma { state, .. },
                ExprSnapshot::Sma {
                    period,
                    window,
                    sum,
                },
            ) => {
                state.period = *period;
                state.window = window.iter().cloned().collect();
                state.sum = *sum;
            }
            (
                CompiledExpr::Ema { state, .. },
                ExprSnapshot::Ema {
                    alpha,
                    initialized,
                    value,
                },
            ) => {
                state.alpha = *alpha;
                state.initialized = *initialized;
                state.value = *value;
            }
            (
                CompiledExpr::Rsi { state, .. },
                ExprSnapshot::Rsi {
                    period,
                    avg_gain,
                    avg_loss,
                    initialized,
                    prev_close,
                },
            ) => {
                state.period = *period;
                state.avg_gain = *avg_gain;
                state.avg_loss = *avg_loss;
                state.initialized = *initialized;
                state.prev_close = *prev_close;
            }
            (CompiledExpr::Macd { state, .. }, ExprSnapshot::Macd { fast, slow, signal }) => {
                if let ExprSnapshot::Ema {
                    alpha,
                    initialized,
                    value,
                } = **fast
                {
                    state.fast.alpha = alpha;
                    state.fast.initialized = initialized;
                    state.fast.value = value;
                }
                if let ExprSnapshot::Ema {
                    alpha,
                    initialized,
                    value,
                } = **slow
                {
                    state.slow.alpha = alpha;
                    state.slow.initialized = initialized;
                    state.slow.value = value;
                }
                if let ExprSnapshot::Ema {
                    alpha,
                    initialized,
                    value,
                } = **signal
                {
                    state.signal.alpha = alpha;
                    state.signal.initialized = initialized;
                    state.signal.value = value;
                }
            }
            _ => {}
        }
    }
}

#[derive(Debug, Clone)]
enum CompiledSignal {
    CrossOver {
        a: CompiledExpr,
        b: CompiledExpr,
        last: Option<(f64, f64)>,
    },
    CrossUnder {
        a: CompiledExpr,
        b: CompiledExpr,
        last: Option<(f64, f64)>,
    },
    Greater {
        a: CompiledExpr,
        b: CompiledExpr,
    },
}

#[derive(Debug, Clone)]
struct CompiledPlot {
    name: String,
    expr: CompiledExpr,
}

/// Compiled and stateful script instance; safe to run incrementally per new candle.
pub struct ScriptInstance {
    spec_name: String,
    timeframe: TimeFrame,
    inputs: HashMap<String, f64>,
    plots: Vec<CompiledPlot>,
    signals: HashMap<String, CompiledSignal>,
    source_lang: Option<SourceLang>,
    plots_buf: HashMap<String, f64>,
    trigger_buf: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ScriptResult {
    pub plots: HashMap<String, f64>,
    pub triggered_signals: Vec<String>,
}

pub struct ScriptEngine;

impl ScriptEngine {
    pub fn compile(
        timeframe: TimeFrame,
        spec: ScriptSpec,
        overrides: HashMap<String, f64>,
    ) -> ScriptInstance {
        Self::compile_with_lang(timeframe, spec, overrides, None)
    }

    pub fn compile_with_lang(
        timeframe: TimeFrame,
        spec: ScriptSpec,
        overrides: HashMap<String, f64>,
        source_lang: Option<SourceLang>,
    ) -> ScriptInstance {
        let mut inputs = spec.inputs.clone();
        for (k, v) in overrides {
            inputs.insert(k, v);
        }

        let plots = spec.plots;
        let ordered_plots = order_plots(&plots)
            .into_iter()
            .map(|(k, expr)| CompiledPlot {
                name: k,
                expr: compile_expr(expr),
            })
            .collect();

        let signals = spec
            .signals
            .into_iter()
            .map(|(k, s)| (k, compile_signal(s)))
            .collect();

        ScriptInstance {
            spec_name: spec.name,
            timeframe,
            inputs,
            plots: ordered_plots,
            signals,
            source_lang,
            plots_buf: HashMap::new(),
            trigger_buf: Vec::new(),
        }
    }
}

impl ScriptInstance {
    /// Process a new candle; returns current plot values and any signals fired on this bar.
    pub fn on_candle(&mut self, candle: &Candle) -> ScriptResult {
        // Reuse buffers to reduce allocations across candles.
        self.plots_buf.clear();
        self.trigger_buf.clear();
        self.plots_buf.reserve(self.plots.len());
        self.trigger_buf.reserve(self.signals.len());

        // Evaluate plots in a deterministic, dependency-aware order.
        for plot in self.plots.iter_mut() {
            let v = plot.expr.eval(candle, &self.plots_buf, &self.inputs);
            self.plots_buf.insert(plot.name.clone(), v);
        }

        for (name, sig) in self.signals.iter_mut() {
            if sig.fire(candle, &self.plots_buf, &self.inputs) {
                self.trigger_buf.push(name.clone());
            }
        }

        ScriptResult {
            plots: self.plots_buf.clone(),
            triggered_signals: self.trigger_buf.clone(),
        }
    }

    /// Process a batch of candles sequentially (incremental evaluation, append-only).
    pub fn on_candles(&mut self, candles: &[Candle]) -> Vec<ScriptResult> {
        candles.iter().map(|c| self.on_candle(c)).collect()
    }

    pub fn name(&self) -> &str {
        &self.spec_name
    }

    pub fn timeframe(&self) -> TimeFrame {
        self.timeframe
    }

    pub fn set_source_lang(&mut self, lang: SourceLang) {
        self.source_lang = Some(lang);
    }

    pub fn source_lang(&self) -> Option<SourceLang> {
        self.source_lang
    }

    pub fn snapshot_states(
        &self,
    ) -> (
        HashMap<String, ExprSnapshot>,
        HashMap<String, SignalSnapshot>,
    ) {
        let plot_states = self
            .plots
            .iter()
            .map(|p| (p.name.clone(), p.expr.snapshot()))
            .collect();
        let signal_states = self
            .signals
            .iter()
            .map(|(k, v)| (k.clone(), v.snapshot()))
            .collect();
        (plot_states, signal_states)
    }

    pub fn restore_states(
        &mut self,
        plot_states: &HashMap<String, ExprSnapshot>,
        signal_states: &HashMap<String, SignalSnapshot>,
    ) {
        for (name, snap) in plot_states {
            if let Some(plot) = self.plots.iter_mut().find(|p| &p.name == name) {
                plot.expr.restore(snap);
            }
        }
        for (name, snap) in signal_states {
            if let Some(sig) = self.signals.get_mut(name) {
                sig.restore(snap);
            }
        }
    }
}

impl CompiledSignal {
    fn fire(
        &mut self,
        candle: &Candle,
        plots: &HashMap<String, f64>,
        inputs: &HashMap<String, f64>,
    ) -> bool {
        match self {
            CompiledSignal::CrossOver { a, b, last } => {
                let av = a.eval(candle, plots, inputs);
                let bv = b.eval(candle, plots, inputs);
                let prev = *last;
                *last = Some((av, bv));
                if let Some((pa, pb)) = prev {
                    (pa - pb) <= 0.0 && (av - bv) > 0.0
                } else {
                    false
                }
            }
            CompiledSignal::CrossUnder { a, b, last } => {
                let av = a.eval(candle, plots, inputs);
                let bv = b.eval(candle, plots, inputs);
                let prev = *last;
                *last = Some((av, bv));
                if let Some((pa, pb)) = prev {
                    (pa - pb) >= 0.0 && (av - bv) < 0.0
                } else {
                    false
                }
            }
            CompiledSignal::Greater { a, b } => {
                let av = a.eval(candle, plots, inputs);
                let bv = b.eval(candle, plots, inputs);
                av > bv
            }
        }
    }

    fn snapshot(&self) -> SignalSnapshot {
        match self {
            CompiledSignal::CrossOver { last, .. } | CompiledSignal::CrossUnder { last, .. } => {
                SignalSnapshot::Cross { last: *last }
            }
            CompiledSignal::Greater { .. } => SignalSnapshot::Stateless,
        }
    }

    fn restore(&mut self, snap: &SignalSnapshot) {
        match (self, snap) {
            (CompiledSignal::CrossOver { last, .. }, SignalSnapshot::Cross { last: saved })
            | (CompiledSignal::CrossUnder { last, .. }, SignalSnapshot::Cross { last: saved }) => {
                *last = *saved;
            }
            _ => {}
        }
    }
}

fn compile_expr(expr: Expr) -> CompiledExpr {
    match expr {
        Expr::Src(s) => CompiledExpr::Src(s),
        Expr::Add(a, b) => {
            CompiledExpr::Add(Box::new(compile_expr(*a)), Box::new(compile_expr(*b)))
        }
        Expr::Sub(a, b) => {
            CompiledExpr::Sub(Box::new(compile_expr(*a)), Box::new(compile_expr(*b)))
        }
        Expr::Mul(a, b) => {
            CompiledExpr::Mul(Box::new(compile_expr(*a)), Box::new(compile_expr(*b)))
        }
        Expr::Div(a, b) => {
            CompiledExpr::Div(Box::new(compile_expr(*a)), Box::new(compile_expr(*b)))
        }
        Expr::Gt(a, b) => CompiledExpr::Gt(Box::new(compile_expr(*a)), Box::new(compile_expr(*b))),
        Expr::Lt(a, b) => CompiledExpr::Lt(Box::new(compile_expr(*a)), Box::new(compile_expr(*b))),
        Expr::And(a, b) => {
            CompiledExpr::And(Box::new(compile_expr(*a)), Box::new(compile_expr(*b)))
        }
        Expr::Or(a, b) => CompiledExpr::Or(Box::new(compile_expr(*a)), Box::new(compile_expr(*b))),
        Expr::CrossOver(a, b) => CompiledExpr::CrossOver {
            left: Box::new(compile_expr(*a)),
            right: Box::new(compile_expr(*b)),
            last: None,
        },
        Expr::CrossUnder(a, b) => CompiledExpr::CrossUnder {
            left: Box::new(compile_expr(*a)),
            right: Box::new(compile_expr(*b)),
            last: None,
        },
        Expr::Sma { period, src } => CompiledExpr::Sma {
            state: SmaState::new(period),
            src: Box::new(compile_expr(*src)),
        },
        Expr::Ema { period, src } => CompiledExpr::Ema {
            state: EmaState::new(period),
            src: Box::new(compile_expr(*src)),
        },
        Expr::Rsi { period, src } => CompiledExpr::Rsi {
            state: RsiState::new(period),
            src: Box::new(compile_expr(*src)),
        },
        Expr::Macd {
            fast,
            slow,
            signal,
            src,
        } => CompiledExpr::Macd {
            state: MacdState::new(fast, slow, signal),
            src: Box::new(compile_expr(*src)),
        },
        Expr::IfGt {
            left,
            right,
            then_expr,
            else_expr,
        } => CompiledExpr::IfGt {
            left: Box::new(compile_expr(*left)),
            right: Box::new(compile_expr(*right)),
            then_expr: Box::new(compile_expr(*then_expr)),
            else_expr: Box::new(compile_expr(*else_expr)),
        },
    }
}

fn compile_signal(spec: SignalSpec) -> CompiledSignal {
    match spec {
        SignalSpec::CrossOver { a, b } => CompiledSignal::CrossOver {
            a: compile_expr(a),
            b: compile_expr(b),
            last: None,
        },
        SignalSpec::CrossUnder { a, b } => CompiledSignal::CrossUnder {
            a: compile_expr(a),
            b: compile_expr(b),
            last: None,
        },
        SignalSpec::Greater { a, b } => CompiledSignal::Greater {
            a: compile_expr(a),
            b: compile_expr(b),
        },
    }
}

fn collect_series_deps(expr: &Expr, deps: &mut HashSet<String>) {
    match expr {
        Expr::Src(Source::Series(name)) => {
            deps.insert(name.clone());
        }
        Expr::Add(a, b)
        | Expr::Sub(a, b)
        | Expr::Mul(a, b)
        | Expr::Div(a, b)
        | Expr::Gt(a, b)
        | Expr::Lt(a, b)
        | Expr::And(a, b)
        | Expr::Or(a, b)
        | Expr::CrossOver(a, b)
        | Expr::CrossUnder(a, b) => {
            collect_series_deps(a, deps);
            collect_series_deps(b, deps);
        }
        Expr::Sma { src, .. }
        | Expr::Ema { src, .. }
        | Expr::Rsi { src, .. }
        | Expr::Macd { src, .. } => {
            collect_series_deps(src, deps);
        }
        Expr::IfGt {
            left,
            right,
            then_expr,
            else_expr,
        } => {
            collect_series_deps(left, deps);
            collect_series_deps(right, deps);
            collect_series_deps(then_expr, deps);
            collect_series_deps(else_expr, deps);
        }
        Expr::Src(_) => {}
    }
}

/// Produce a deterministic, dependency-respecting order of plots.
/// Cycles fall back to lexicographic order to remain deterministic.
fn order_plots(plots: &HashMap<String, Expr>) -> Vec<(String, Expr)> {
    let mut ordered = Vec::new();
    if plots.is_empty() {
        return ordered;
    }

    let mut names: Vec<String> = plots.keys().cloned().collect();
    names.sort();

    let name_set: HashSet<String> = names.iter().cloned().collect();
    let mut indegree: HashMap<String, usize> = names.iter().map(|n| (n.clone(), 0)).collect();
    let mut edges: HashMap<String, Vec<String>> = HashMap::new();

    for name in &names {
        let mut deps = HashSet::new();
        if let Some(expr) = plots.get(name) {
            collect_series_deps(expr, &mut deps);
        }
        for dep in deps {
            if name_set.contains(&dep) {
                edges.entry(dep.clone()).or_default().push(name.clone());
                *indegree.entry(name.clone()).or_insert(0) += 1;
            }
        }
    }

    // Kahn's algorithm with lexicographic tiebreaker.
    let mut queue: Vec<String> = indegree
        .iter()
        .filter_map(|(name, &deg)| if deg == 0 { Some(name.clone()) } else { None })
        .collect();
    queue.sort_by(|a, b| b.cmp(a));

    while let Some(name) = queue.pop() {
        if let Some(expr) = plots.get(&name) {
            ordered.push((name.clone(), expr.clone()));
        }
        if let Some(children) = edges.get(&name) {
            for child in children {
                if let Some(entry) = indegree.get_mut(child) {
                    *entry -= 1;
                    if *entry == 0 {
                        queue.push(child.clone());
                        queue.sort_by(|a, b| b.cmp(a));
                    }
                }
            }
        }
    }

    if ordered.len() != plots.len() {
        // Cycle detected; fall back to deterministic lexicographic order.
        ordered.clear();
        for name in names {
            if let Some(expr) = plots.get(&name) {
                ordered.push((name.clone(), expr.clone()));
            }
        }
    }

    ordered
}

/// Example helper to construct a basic EMA script from code.
pub fn ema_script(name: &str, period: usize) -> ScriptSpec {
    let mut inputs = HashMap::new();
    inputs.insert("period".to_string(), period as f64);
    let mut plots = HashMap::new();
    plots.insert(
        "ema".to_string(),
        Expr::Ema {
            period,
            src: Box::new(Expr::Src(Source::Price(PriceField::Close))),
        },
    );
    ScriptSpec {
        name: name.to_string(),
        inputs,
        plots,
        signals: HashMap::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_candle(ts: i64, close: f64) -> Candle {
        Candle {
            ts,
            timeframe: TimeFrame::Minutes(1),
            open: close,
            high: close,
            low: close,
            close,
            volume: 1.0,
        }
    }

    #[test]
    fn incremental_runner_processes_new_bars_only() {
        let mut plots = HashMap::new();
        plots.insert(
            "close".to_string(),
            Expr::Src(Source::Price(PriceField::Close)),
        );
        let spec = ScriptSpec {
            name: "close-plot".to_string(),
            inputs: HashMap::new(),
            plots,
            signals: HashMap::new(),
        };
        let instance = ScriptEngine::compile(TimeFrame::Minutes(1), spec, HashMap::new());
        let mut runner = incremental::IncrementalRunner::new(instance);
        let candles = vec![sample_candle(0, 10.0), sample_candle(60_000, 11.0)];
        let first_run = runner.apply_delta(&candles);
        assert_eq!(first_run.len(), 2);
        assert_eq!(first_run[0].plots.get("close"), Some(&10.0));
        assert_eq!(first_run[1].plots.get("close"), Some(&11.0));

        // Re-applying same candles should produce zero new results since checkpoint advanced.
        let second_run = runner.apply_delta(&candles);
        assert!(second_run.is_empty());
    }

    #[test]
    fn incremental_runner_accepts_appended_slices() {
        let mut plots = HashMap::new();
        plots.insert(
            "close".to_string(),
            Expr::Src(Source::Price(PriceField::Close)),
        );
        let spec = ScriptSpec {
            name: "close-plot".to_string(),
            inputs: HashMap::new(),
            plots,
            signals: HashMap::new(),
        };
        let instance = ScriptEngine::compile(TimeFrame::Minutes(1), spec, HashMap::new());
        let mut runner = incremental::IncrementalRunner::new(instance);
        let candles = vec![sample_candle(0, 10.0), sample_candle(60_000, 11.0)];

        let first = runner.apply_delta(&candles);
        assert_eq!(first.len(), 2);

        // Provide only the newly appended bars (no history); should still process them.
        let appended_only = vec![sample_candle(120_000, 12.0)];
        let second = runner.apply_delta(&appended_only);
        assert_eq!(second.len(), 1);
        assert_eq!(second[0].plots.get("close"), Some(&12.0));
    }

    #[test]
    fn manifest_derives_from_spec() {
        let spec = ema_script("ema20", 20);
        let manifest = manifest::Manifest::from_spec(&spec, SourceLang::PineV5);
        assert_eq!(manifest.name, "ema20");
        assert_eq!(manifest.inputs.len(), 1);
        assert_eq!(manifest.outputs.len(), 1);
        assert!(manifest.capabilities.pine_compatible);
    }

    #[test]
    fn compat_translation_stubs_report_unsupported() {
        let pine = compat::translate_pine("study('x')");
        let ts = compat::translate_thinkscript("plot x = close;");
        assert!(pine.report.supported || !pine.report.issues.is_empty());
        assert!(ts.report.supported || !ts.report.issues.is_empty());
    }

    #[test]
    fn plot_evaluation_respects_dependencies() {
        let mut plots = HashMap::new();
        // Insert dependent plot first to ensure ordering is handled by engine.
        plots.insert(
            "derived".to_string(),
            Expr::Add(
                Box::new(Expr::Src(Source::Series("base".to_string()))),
                Box::new(Expr::Src(Source::Const(1.0))),
            ),
        );
        plots.insert(
            "base".to_string(),
            Expr::Src(Source::Price(PriceField::Close)),
        );

        let spec = ScriptSpec {
            name: "dep-plot".to_string(),
            inputs: HashMap::new(),
            plots,
            signals: HashMap::new(),
        };
        let mut instance = ScriptEngine::compile(TimeFrame::Minutes(1), spec, HashMap::new());
        let candle = sample_candle(0, 10.0);
        let result = instance.on_candle(&candle);
        assert_eq!(result.plots.get("base"), Some(&10.0));
        assert_eq!(result.plots.get("derived"), Some(&(10.0 + 1.0)));
    }

    #[test]
    fn pine_parser_extracts_plot_and_input() {
        let src = r#"//@version=6
indicator("EMA Demo", overlay=true)
len = input.int(20)
src = close
ema1 = ta.ema(src, len)
plot(ema1, title="ema")
alertcondition(crossover(close, ema1), "cross up", "cross up")
"#;
        let parsed = compat::translate_pine(src);
        assert!(
            parsed.report.supported,
            "issues: {:?}",
            parsed.report.issues
        );
        assert_eq!(parsed.ir.spec.inputs.get("len"), Some(&20.0));
        assert!(parsed.ir.spec.plots.contains_key("ema"));
        assert_eq!(parsed.ir.spec.signals.len(), 1);
    }

    #[test]
    fn thinkscript_parser_extracts_plot() {
        let src = r#"
input len = 10;
def ema1 = ExpAverage(close, len);
plot p = ema1;
alert up = crossover(close, ema1);
"#;
        let parsed = compat::translate_thinkscript(src);
        assert!(
            parsed.report.supported,
            "issues: {:?}",
            parsed.report.issues
        );
        assert!(parsed.ir.spec.plots.contains_key("p"));
    }

    #[test]
    fn js_facade_reports_translation_status() {
        let ok_src = r#"//@version=6
indicator("Test")
plot(close)
"#;
        let ok = js_facade::JsScriptHandle::from_pine(ok_src);
        assert!(ok.is_ok(), "expected ok translation");
        let (_handle, report) = ok.unwrap();
        assert!(report.supported);

        let unsupported_src = r#"//@version=6
indicator("OnlyTitle")
"#;
        let err = js_facade::JsScriptHandle::from_pine(unsupported_src);
        assert!(err.is_err(), "expected unsupported translation");
    }

    #[test]
    fn ai_prompt_selects_template_and_runs_compat() {
        let req = ai::AiPromptRequest {
            prompt: "give me a macd crossover with alerts".to_string(),
            target_lang: SourceLang::PineV5,
        };
        let suggestion = ai::generate_script(req);
        assert!(suggestion.template_used.is_some());
        assert!(
            suggestion.compatibility.supported,
            "expected supported script, issues: {:?}",
            suggestion.compatibility.issues
        );
        assert!(suggestion.confidence > 0.5);
    }

    #[test]
    fn remediation_suggests_version_and_plot() {
        let src = r#"indicator("no version")
ema1 = ta.ema(close, 10)
"#;
        let translation = compat::translate_pine(src);
        let fixes = crate::suggest_fixes(src, SourceLang::PineV5, &translation.report);
        let titles: Vec<_> = fixes.iter().map(|f| f.title.as_str()).collect();
        assert!(titles.contains(&"Add //@version=6"));
        assert!(titles.contains(&"Add default plot"));
    }

    #[test]
    fn templates_exist_for_languages() {
        let pine = crate::templates_for_lang(SourceLang::PineV5);
        let think = crate::templates_for_lang(SourceLang::ThinkScriptSubset);
        assert!(pine.len() >= 3);
        assert!(think.len() >= 3);
    }

    #[test]
    fn validation_harness_passes_supported_script() {
        let src = r#"//@version=6
indicator("Valid")
plot(close)
"#;
        let result = crate::validate_script(src, SourceLang::PineV5);
        assert!(result.is_ok(), "expected ok: {result:?}");
    }

    #[test]
    fn validation_harness_rejects_empty_script() {
        let src = r#"//@version=6
indicator("Empty")
"#;
        let result = crate::validate_script(src, SourceLang::PineV5);
        assert!(result.is_err(), "expected err");
    }

    #[test]
    fn pine_translation_reports_missing_version() {
        let src = r#"indicator("NoVersion")"#;
        let translation = compat::translate_pine(src);
        assert!(!translation.report.supported);
        let codes: Vec<_> = translation
            .report
            .issues
            .iter()
            .map(|i| i.code)
            .collect();
        assert!(
            codes.contains(&crate::language::DiagnosticCode::MissingVersion),
            "expected MissingVersion code, got {codes:?}"
        );
    }

    #[test]
    fn pine_translation_reports_missing_outputs() {
        let src = r#"//@version=6
indicator("NoOutputs")
"#;
        let translation = compat::translate_pine(src);
        assert!(!translation.report.supported);
        let codes: Vec<_> = translation
            .report
            .issues
            .iter()
            .map(|i| i.code)
            .collect();
        assert!(
            codes.contains(&crate::language::DiagnosticCode::MissingPlotOrSignal),
            "expected MissingPlotOrSignal code, got {codes:?}"
        );
    }

    #[test]
    fn checkpoint_roundtrip_restores_progress() {
        let spec = ema_script("ema", 3);
        let mut runner = incremental::IncrementalRunner::new(ScriptEngine::compile(
            TimeFrame::Minutes(1),
            spec.clone(),
            HashMap::new(),
        ));
        let candles = vec![sample_candle(0, 10.0), sample_candle(60_000, 11.0)];
        let first = runner.apply_delta(&candles[..1]);
        assert_eq!(first.len(), 1);
        let checkpoint = runner.checkpoint().clone();

        let mut runner2 = incremental::IncrementalRunner::from_checkpoint(
            ScriptEngine::compile(TimeFrame::Minutes(1), spec, HashMap::new()),
            checkpoint,
        );
        let remaining = runner2.apply_delta(&candles);
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].plots.get("ema"), Some(&10.5));
    }

    #[test]
    fn artifact_serialization_roundtrip() {
        let src = r#"//@version=6
indicator("Test")
plot(close)
"#;
        let translation = compat::translate_pine(src);
        let artifact = js_facade::artifact_from_translation(&translation).unwrap();
        let parsed_manifest: manifest::Manifest =
            serde_json::from_str(&artifact.manifest_json).unwrap();
        assert_eq!(parsed_manifest.name, "Test");
        let wasm_artifact: crate::wasm::WasmArtifact =
            serde_json::from_str(&artifact.artifact_json).unwrap();
        assert_eq!(wasm_artifact.source_lang, language::SourceLang::PineV5);
    }

    #[test]
    fn runner_handles_thousand_candles() {
        let mut plots = HashMap::new();
        plots.insert(
            "ema".to_string(),
            Expr::Ema {
                period: 20,
                src: Box::new(Expr::Src(Source::Price(PriceField::Close))),
            },
        );
        let spec = ScriptSpec {
            name: "ema-load".into(),
            inputs: HashMap::new(),
            plots,
            signals: HashMap::new(),
        };
        let mut runner =
            incremental::IncrementalRunner::new(ScriptEngine::compile(TimeFrame::Minutes(1), spec, HashMap::new()));
        let candles: Vec<_> = (0..1024)
            .map(|i| sample_candle(i as i64 * 60_000, 100.0 + i as f64 * 0.01))
            .collect();
        let results = runner.apply_delta(&candles);
        assert_eq!(results.len(), 1024);
        assert!(results.last().unwrap().plots.contains_key("ema"));
    }
}
