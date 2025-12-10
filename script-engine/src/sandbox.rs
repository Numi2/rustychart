use serde::{Deserialize, Serialize};
use ts_core::Candle;

use crate::{ScriptInstance, ScriptResult};

/// Sandbox limits enforced per execution to avoid runaway scripts.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SandboxLimits {
    pub max_fuel: u64,
    pub max_heap_bytes: u64,
}

impl Default for SandboxLimits {
    fn default() -> Self {
        Self {
            max_fuel: 50_000,
            max_heap_bytes: 2 * 1024 * 1024,
        }
    }
}

/// Host callbacks exposed to sandboxed scripts; this is intentionally thin and deterministic.
pub trait HostEnvironment {
    fn now_ms(&self) -> i64;
    fn read_candle(&self, lookback: usize) -> Option<Candle>;
    fn read_series(&self, name: &str, lookback: usize) -> Option<f64>;
    fn emit_plot(&mut self, name: &str, value: f64);
    fn emit_signal(&mut self, name: &str);
    fn log(&mut self, msg: &str);
}

/// Engine API a sandbox needs to expose to host.
pub trait SandboxedScript {
    fn on_candle(&mut self, candle: &Candle, host: &mut dyn HostEnvironment) -> ScriptResult;
}

/// Adapter to run an already-compiled `ScriptInstance` inside a sandbox abstraction.
pub struct InstanceAdapter<'a, H: HostEnvironment> {
    pub inner: ScriptInstance,
    pub host: &'a mut H,
}

impl<'a, H: HostEnvironment> InstanceAdapter<'a, H> {
    pub fn on_candle(&mut self, candle: &Candle) -> ScriptResult {
        let result = self.inner.on_candle(candle);
        for (name, value) in &result.plots {
            self.host.emit_plot(name, *value);
        }
        for sig in &result.triggered_signals {
            self.host.emit_signal(sig);
        }
        result
    }
}
