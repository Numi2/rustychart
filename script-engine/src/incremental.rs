use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use ts_core::Candle;

use crate::{ScriptInstance, ScriptResult};

/// Serializable snapshot of expression state for checkpointing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExprSnapshot {
    Stateless,
    Sma {
        period: usize,
        window: Vec<f64>,
        sum: f64,
    },
    Ema {
        alpha: f64,
        initialized: bool,
        value: f64,
    },
    Rsi {
        period: usize,
        avg_gain: f64,
        avg_loss: f64,
        initialized: bool,
        prev_close: Option<f64>,
    },
    Macd {
        fast: Box<ExprSnapshot>,
        slow: Box<ExprSnapshot>,
        signal: Box<ExprSnapshot>,
    },
    Cross {
        last: Option<(f64, f64)>,
    },
}

/// Serializable snapshot of signal state (for crossover memory).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignalSnapshot {
    Stateless,
    Cross { last: Option<(f64, f64)> },
}

/// Captures where we left off processing a series so we can resume without replaying history.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ScriptCheckpoint {
    pub last_bar_index: usize,
    pub plot_states: HashMap<String, ExprSnapshot>,
    pub signal_states: HashMap<String, SignalSnapshot>,
}

/// Helper for running a script incrementally over appended bars.
pub struct IncrementalRunner {
    pub instance: ScriptInstance,
    pub checkpoint: ScriptCheckpoint,
}

impl IncrementalRunner {
    pub fn new(instance: ScriptInstance) -> Self {
        Self {
            instance,
            checkpoint: ScriptCheckpoint::default(),
        }
    }

    pub fn from_checkpoint(instance: ScriptInstance, checkpoint: ScriptCheckpoint) -> Self {
        let mut runner = Self {
            instance,
            checkpoint,
        };
        runner.instance.restore_states(
            &runner.checkpoint.plot_states,
            &runner.checkpoint.signal_states,
        );
        runner
    }

    /// Apply only the new bars (deltas) to the script, returning results for each new bar.
    ///
    /// Behavior:
    /// - If the caller passes the full history again, only the unprocessed suffix is executed.
    /// - If the caller passes only newly appended bars, all provided bars are executed.
    ///
    /// This allows both "full history" and "append-only" calling patterns without duplication.
    pub fn apply_delta(&mut self, candles: &[Candle]) -> Vec<ScriptResult> {
        let mut results = Vec::with_capacity(candles.len().saturating_sub(self.checkpoint.last_bar_index));
        // If callers provide the full history again, skip the prefix we already processed.
        // If they provide only the newly appended bars, process them all.
        let processed_so_far = self.checkpoint.last_bar_index;
        let start_idx = if candles.len() >= processed_so_far {
            processed_so_far
        } else {
            0
        };

        for (idx, candle) in candles.iter().enumerate().skip(start_idx) {
            let res = self.instance.on_candle(candle);
            results.push(res);
            let processed = processed_so_far + (idx + 1 - start_idx);
            self.checkpoint.last_bar_index = processed;
        }

        // refresh snapshots
        let (plots, signals) = self.instance.snapshot_states();
        self.checkpoint.plot_states = plots;
        self.checkpoint.signal_states = signals;
        results
    }

    /// Reset checkpoint to force full replay (used after configuration change).
    pub fn reset(&mut self) {
        self.checkpoint = ScriptCheckpoint::default();
    }

    pub fn checkpoint(&self) -> &ScriptCheckpoint {
        &self.checkpoint
    }
}
