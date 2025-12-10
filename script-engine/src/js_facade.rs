use serde::{Deserialize, Serialize};
use ts_core::Candle;

use crate::{
    compat, incremental::IncrementalRunner, language::SourceLang, ScriptEngine, ScriptSpec,
};

/// Payload used at JS/TS boundary (serde-friendly).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScriptArtifact {
    pub manifest_json: String,
    pub compat_report_json: String,
    pub artifact_json: String,
}

/// A lightweight facade for browser-side usage; we keep it pure Rust so it can
/// be bound via wasm-bindgen or other bridges without pulling dependencies here.
pub struct JsScriptHandle {
    pub(crate) runner: IncrementalRunner,
}

impl JsScriptHandle {
    pub fn from_pine(
        source: &str,
    ) -> Result<(Self, compat::CompatibilityReport), compat::CompatibilityReport> {
        let translation = compat::translate_pine(source);
        let report = translation.report.clone();
        if !translation.report.supported {
            return Err(translation.report);
        }
        let mut instance = ScriptEngine::compile(
            ts_core::TimeFrame::Minutes(1),
            translation.ir.spec,
            Default::default(),
        );
        instance.set_source_lang(SourceLang::PineV5);
        Ok((
            Self {
                runner: IncrementalRunner::new(instance),
            },
            report,
        ))
    }

    pub fn from_thinkscript(
        source: &str,
    ) -> Result<(Self, compat::CompatibilityReport), compat::CompatibilityReport> {
        let translation = compat::translate_thinkscript(source);
        let report = translation.report.clone();
        if !translation.report.supported {
            return Err(translation.report);
        }
        let mut instance = ScriptEngine::compile(
            ts_core::TimeFrame::Minutes(1),
            translation.ir.spec,
            Default::default(),
        );
        instance.set_source_lang(SourceLang::ThinkScriptSubset);
        Ok((
            Self {
                runner: IncrementalRunner::new(instance),
            },
            report,
        ))
    }

    pub fn from_spec(spec: ScriptSpec) -> Self {
        let instance =
            ScriptEngine::compile(ts_core::TimeFrame::Minutes(1), spec, Default::default());
        Self {
            runner: IncrementalRunner::new(instance),
        }
    }

    pub fn on_candles(&mut self, candles: &[Candle]) -> Vec<crate::ScriptResult> {
        self.runner.apply_delta(candles)
    }

    pub fn from_artifact_json(json: &str) -> Result<Self, serde_json::Error> {
        let artifact: crate::wasm::WasmArtifact = serde_json::from_str(json)?;
        let instance = ScriptEngine::compile_with_lang(
            ts_core::TimeFrame::Minutes(1),
            artifact.spec,
            Default::default(),
            Some(artifact.source_lang),
        );
        Ok(Self {
            runner: IncrementalRunner::new(instance),
        })
    }
}

pub fn artifact_from_translation(
    translation: &compat::TranslationOutput,
) -> Result<ScriptArtifact, serde_json::Error> {
    let wasm_artifact =
        crate::wasm::compile_artifact(translation.ir.spec.clone(), translation.report.source_lang);
    Ok(ScriptArtifact {
        manifest_json: serde_json::to_string(&wasm_artifact.manifest)?,
        compat_report_json: serde_json::to_string(&translation.report)?,
        artifact_json: serde_json::to_string(&wasm_artifact)?,
    })
}

/// Validate a raw script source; returns an artifact and report if supported,
/// or the compatibility report detailing issues.
pub fn validate_script(
    source: &str,
    lang: SourceLang,
) -> Result<(ScriptArtifact, compat::CompatibilityReport), compat::CompatibilityReport> {
    let translation = match lang {
        SourceLang::PineV5 => compat::translate_pine(source),
        SourceLang::ThinkScriptSubset => compat::translate_thinkscript(source),
        SourceLang::NativeDsl => compat::TranslationOutput {
            ir: compat::UnifiedIr {
                manifest: crate::manifest::Manifest::from_spec(
                    &crate::ema_script("ema", 20),
                    SourceLang::NativeDsl,
                ),
                spec: crate::ema_script("ema", 20),
            },
            report: compat::CompatibilityReport {
                source_lang: SourceLang::NativeDsl,
                supported: true,
                issues: Vec::new(),
            },
        },
    };
    if !translation.report.supported {
        return Err(translation.report);
    }
    let artifact =
        artifact_from_translation(&translation).map_err(|_| compat::CompatibilityReport {
            source_lang: lang,
            supported: false,
            issues: vec![compat::CompatibilityIssue {
                message: "Serialization failed".to_string(),
                severity: compat::IssueSeverity::Error,
                location: None,
            }],
        })?;
    Ok((artifact, translation.report))
}
