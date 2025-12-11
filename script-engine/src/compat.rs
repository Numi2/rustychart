use serde::{Deserialize, Serialize};

use crate::language::SourceLang;
use crate::{manifest::Manifest, parser, ScriptSpec};

pub use crate::language::{DiagnosticCode, IssueSeverity, SourceSpan};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityIssue {
    pub code: DiagnosticCode,
    pub message: String,
    pub severity: IssueSeverity,
    pub span: Option<SourceSpan>,
    pub hint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityReport {
    pub source_lang: SourceLang,
    pub supported: bool,
    pub issues: Vec<CompatibilityIssue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedIr {
    pub manifest: Manifest,
    pub spec: ScriptSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationOutput {
    pub ir: UnifiedIr,
    pub report: CompatibilityReport,
}

pub fn translate_pine(source: &str) -> TranslationOutput {
    parser::pine::translate(source)
}

pub fn translate_thinkscript(source: &str) -> TranslationOutput {
    parser::think::translate(source)
}
