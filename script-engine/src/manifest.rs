use serde::{Deserialize, Serialize};

use crate::language::SourceLang;
use crate::ScriptSpec;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InputKind {
    Number,
    Bool,
    Text,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputParam {
    pub name: String,
    pub kind: InputKind,
    pub default: serde_json::Value,
    pub label: Option<String>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub step: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum OutputKind {
    Plot,
    Signal,
    Order,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputSpec {
    pub name: String,
    pub kind: OutputKind,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Permission {
    MultiTimeframe,
    StrategyOrders,
    ExternalSecurity,
}

/// Versioned manifest used for artifact packaging and UI surface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub name: String,
    pub description: Option<String>,
    pub version: String,
    pub source_lang: SourceLang,
    pub tags: Vec<String>,
    pub inputs: Vec<InputParam>,
    pub outputs: Vec<OutputSpec>,
    pub permissions: Vec<Permission>,
    /// Compatibility flags useful when importing third-party scripts.
    pub capabilities: ManifestCapabilities,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ManifestCapabilities {
    pub pine_compatible: bool,
    pub thinkscript_compatible: bool,
}

impl Manifest {
    pub fn from_spec(spec: &ScriptSpec, source_lang: SourceLang) -> Self {
        let inputs = spec
            .inputs
            .iter()
            .map(|(name, default)| InputParam {
                name: name.clone(),
                kind: InputKind::Number,
                default: serde_json::json!(default),
                label: None,
                min: None,
                max: None,
                step: None,
            })
            .collect();

        let outputs = spec
            .plots
            .keys()
            .map(|name| OutputSpec {
                name: name.clone(),
                kind: OutputKind::Plot,
                description: None,
            })
            .collect();

        let capabilities = ManifestCapabilities {
            pine_compatible: matches!(source_lang, SourceLang::PineV5),
            thinkscript_compatible: matches!(source_lang, SourceLang::ThinkScriptSubset),
        };

        Manifest {
            name: spec.name.clone(),
            description: None,
            version: "0.1.0".to_string(),
            source_lang,
            tags: Vec::new(),
            inputs,
            outputs,
            permissions: Vec::new(),
            capabilities,
        }
    }
}
