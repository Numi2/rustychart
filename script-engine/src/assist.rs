use serde::Serialize;

use crate::{
    builtin_docs,
    compat::{self, CompatibilityReport},
    language::SourceLang,
    manifest::Manifest,
    templates::{templates_for_lang, Template},
    wasm::{compile_artifact, WasmArtifact},
};

/// Lightweight completion entry consumable by UI editors.
#[derive(Debug, Clone, Serialize)]
pub struct CompletionItem {
    pub label: String,
    pub detail: String,
    pub insert_text: String,
    pub doc: String,
    pub group: String,
}

/// Aggregate analysis of a script: translation report plus helpful extras.
#[derive(Debug, Clone, Serialize)]
pub struct ScriptAnalysis {
    pub report: CompatibilityReport,
    pub manifest: Manifest,
    pub artifact: Option<WasmArtifact>,
    pub completions: Vec<CompletionItem>,
    pub templates: Vec<Template>,
}

/// Produce an analysis payload that feels LSP-like for frontends:
/// - translation + diagnostics
/// - manifest + optional artifact for execution
/// - completions derived from builtin docs
/// - curated templates for the selected language
pub fn analyze_script(source: &str, lang: SourceLang) -> ScriptAnalysis {
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

    let completions = builtin_docs(lang)
        .into_iter()
        .map(|doc| CompletionItem {
            label: doc.name.to_string(),
            detail: doc.signature.to_string(),
            insert_text: doc.example.to_string(),
            doc: doc.description.to_string(),
            group: doc.group.to_string(),
        })
        .collect();

    let artifact = if translation.report.supported {
        Some(compile_artifact(
            translation.ir.spec.clone(),
            translation.report.source_lang,
        ))
    } else {
        None
    };

    ScriptAnalysis {
        report: translation.report,
        manifest: translation.ir.manifest,
        artifact,
        completions,
        templates: templates_for_lang(lang),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analysis_surfaces_completions_and_manifest() {
        let src = r#"//@version=6
indicator("Test", overlay=true)
plot(close)"#;
        let result = analyze_script(src, SourceLang::PineV5);
        assert!(result.report.supported);
        assert!(!result.completions.is_empty());
        assert_eq!(result.manifest.name, "Test");
        assert!(result.artifact.is_some());
        assert!(result.templates.iter().any(|t| t.name == "starter"));
    }

    #[test]
    fn analysis_handles_unsupported_source() {
        let bad_src = "study()";
        let result = analyze_script(bad_src, SourceLang::PineV5);
        assert!(
            !result.report.supported || !result.report.issues.is_empty(),
            "expected issues for unsupported script"
        );
    }
}
