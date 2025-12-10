use serde::{Deserialize, Serialize};

use crate::language::SourceLang;
use crate::{manifest::Manifest, parser, ScriptSpec};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueSeverity {
    Info,
    Warning,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityIssue {
    pub message: String,
    pub severity: IssueSeverity,
    pub location: Option<(usize, usize)>,
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
