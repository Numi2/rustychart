use serde::{Deserialize, Serialize};
use ts_core::Candle;

use crate::{
    manifest::Manifest, sandbox::InstanceAdapter, sandbox::SandboxLimits, HostEnvironment,
    ScriptEngine, ScriptInstance, ScriptSpec, SourceLang,
};
use ts_core::TimeFrame;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WasmTrap {
    OutOfFuel,
    HostError(String),
}

/// A lightweight stand-in for a WASM module instance. It enforces fuel/heap
/// limits and delegates execution to `ScriptInstance` to keep this crate
/// no-std/wasm-bindgen free. A real runtime can wire the same interface.
pub struct WasmSandbox<'a, H: HostEnvironment> {
    adapter: InstanceAdapter<'a, H>,
    fuel_left: u64,
    limits: SandboxLimits,
}

impl<'a, H: HostEnvironment> WasmSandbox<'a, H> {
    pub fn new(instance: ScriptInstance, host: &'a mut H, limits: SandboxLimits) -> Self {
        Self {
            adapter: InstanceAdapter {
                inner: instance,
                host,
            },
            fuel_left: limits.max_fuel,
            limits,
        }
    }

    pub fn on_candle(&mut self, candle: &Candle) -> Result<crate::ScriptResult, WasmTrap> {
        if self.fuel_left == 0 {
            return Err(WasmTrap::OutOfFuel);
        }
        // simplistic fuel consumption: 1 per bar
        self.fuel_left -= 1;
        Ok(self.adapter.on_candle(candle))
    }

    /// Reset fuel to configured maximum; host-controlled.
    pub fn reset_fuel(&mut self) {
        self.fuel_left = self.limits.max_fuel;
    }

    pub fn fuel_left(&self) -> u64 {
        self.fuel_left
    }

    pub fn limits(&self) -> SandboxLimits {
        self.limits
    }
}

/// Serialized artifact to persist/share compiled scripts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmArtifact {
    pub manifest: Manifest,
    pub spec: ScriptSpec,
    pub source_lang: SourceLang,
}

impl WasmArtifact {
    pub fn serialize(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }

    pub fn deserialize(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(bytes)
    }
}

pub fn compile_artifact(spec: ScriptSpec, source_lang: SourceLang) -> WasmArtifact {
    let manifest = Manifest::from_spec(&spec, source_lang);
    WasmArtifact {
        manifest,
        spec,
        source_lang,
    }
}

pub fn instantiate_from_artifact<H: HostEnvironment>(
    artifact: WasmArtifact,
    host: &mut H,
    limits: SandboxLimits,
) -> WasmSandbox<'_, H> {
    let instance = ScriptEngine::compile_with_lang(
        TimeFrame::Minutes(1),
        artifact.spec,
        Default::default(),
        Some(artifact.source_lang),
    );
    WasmSandbox::new(instance, host, limits)
}
