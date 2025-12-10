pub mod app;
pub mod chart;
pub mod state;
pub mod theme;
pub mod perf;
pub mod lab;

pub use app::App;
pub use perf::{default_scenarios, make_perf_state, PerfScenario};

#[cfg(all(any(feature = "csr", feature = "hydrate"), target_arch = "wasm32"))]
use leptos::*;
#[cfg(all(any(feature = "csr", feature = "hydrate"), target_arch = "wasm32"))]
use wasm_bindgen::prelude::*;

#[cfg(all(feature = "csr", target_arch = "wasm32"))]
#[wasm_bindgen]
pub fn start() {
    console_error_panic_hook::set_once();
    leptos::mount_to_body(|| view! { <App/> });
}

#[cfg(all(feature = "hydrate", target_arch = "wasm32"))]
#[wasm_bindgen]
pub fn hydrate() {
    console_error_panic_hook::set_once();
    leptos::mount_to_body(|| view! { <App/> });
}

#[cfg(all(any(feature = "csr", feature = "hydrate"), target_arch = "wasm32"))]
#[wasm_bindgen]
pub fn perf_state_json(name: &str) -> Result<String, JsValue> {
    let scenarios = default_scenarios();
    if let Some(s) = scenarios.iter().find(|s| s.name == name) {
        serde_json::to_string(&make_perf_state(s)).map_err(|e| JsValue::from_str(&e.to_string()))
    } else {
        Err(JsValue::from_str("scenario not found"))
    }
}
