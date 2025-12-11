use serde::{Deserialize, Serialize};

/// Script source language variants we aim to support/ingest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceLang {
    PineV5,
    ThinkScriptSubset,
    NativeDsl,
}

impl SourceLang {
    pub fn name(&self) -> &'static str {
        match self {
            SourceLang::PineV5 => "pine-v5",
            SourceLang::ThinkScriptSubset => "thinkscript-subset",
            SourceLang::NativeDsl => "native-dsl",
        }
    }
}

/// Severity used across translation and runtime diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueSeverity {
    Info,
    Warning,
    Error,
}

/// Coarse typing for script expressions; keeps diagnostics human-friendly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValueType {
    Number,
    Boolean,
    Unknown,
}

/// A byte-span aligned to line/column positions, used for UI highlighting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceSpan {
    pub start_line: usize,
    pub start_col: usize,
    pub end_line: usize,
    pub end_col: usize,
}

impl SourceSpan {
    pub fn single_line(line: usize, start_col: usize, end_col: usize) -> Self {
        Self {
            start_line: line,
            start_col,
            end_line: line,
            end_col,
        }
    }
}

/// Machine-readable error codes so UIs can map to tips/remediations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiagnosticCode {
    MissingVersion,
    MissingPlotOrSignal,
    UnsupportedFunction,
    UnsupportedExpression,
    MalformedAssignment,
    UnknownIdentifier,
    ParseError,
    SerializationFailed,
}

/// Rich diagnostic information that can be surfaced to users.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostic {
    pub code: DiagnosticCode,
    pub message: String,
    pub severity: IssueSeverity,
    pub span: Option<SourceSpan>,
    pub hint: Option<String>,
    pub inferred_type: Option<ValueType>,
}
