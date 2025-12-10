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
