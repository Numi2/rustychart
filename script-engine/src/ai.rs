use serde::{Deserialize, Serialize};

use crate::compat::{self, CompatibilityReport};
use crate::language::SourceLang;
use crate::remediation::{suggest_fixes, FixSuggestion};
use crate::templates::{find_template, templates_for_lang};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiPromptRequest {
    pub prompt: String,
    pub target_lang: SourceLang,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiSuggestion {
    pub script_text: String,
    pub explanation: String,
    pub compatibility: CompatibilityReport,
    pub confidence: f32,
    pub template_used: Option<String>,
    pub quick_fixes: Vec<FixSuggestion>,
    pub issues_summary: Vec<String>,
}

/// Rule-based AI pipeline: selects a template by intent, runs compatibility,
/// and returns quick-fixes to keep the loop tight without external models.
pub fn generate_script(req: AiPromptRequest) -> AiSuggestion {
    let (template_source, template_name) = select_template(&req.prompt, req.target_lang);
    let translation = match req.target_lang {
        SourceLang::PineV5 => compat::translate_pine(template_source),
        SourceLang::ThinkScriptSubset => compat::translate_thinkscript(template_source),
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

    let confidence = if translation.report.supported {
        0.9
    } else {
        0.5
    };
    let issues_summary = translation
        .report
        .issues
        .iter()
        .map(|i| i.message.clone())
        .collect();
    let quick_fixes = suggest_fixes(template_source, req.target_lang, &translation.report);

    AiSuggestion {
        script_text: template_source.to_string(),
        explanation: format!(
            "Generated using '{}' template inferred from prompt: {}",
            template_name, req.prompt
        ),
        compatibility: translation.report,
        confidence,
        template_used: Some(template_name.to_string()),
        quick_fixes,
        issues_summary,
    }
}

fn select_template(prompt: &str, lang: SourceLang) -> (&str, &str) {
    let lower = prompt.to_ascii_lowercase();
    let name = if lower.contains("crossover") {
        "crossover-alert"
    } else if lower.contains("macd") {
        "macd"
    } else if lower.contains("rsi") {
        "rsi"
    } else if lower.contains("ema") || lower.contains("moving average") {
        "ema"
    } else {
        // fallback to first template for the language
        templates_for_lang(lang)
            .first()
            .map(|t| t.name)
            .unwrap_or("ema")
    };
    if let Some(tpl) = find_template(lang, name) {
        (tpl.source, tpl.name)
    } else {
        // very small fallback per language
        match lang {
            SourceLang::PineV5 => (
                "//@version=6\nindicator(\"Auto\", overlay=true)\nplot(close)",
                "fallback-pine",
            ),
            SourceLang::ThinkScriptSubset => ("plot p = close;", "fallback-thinkscript"),
            SourceLang::NativeDsl => ("plot close", "fallback-native"),
        }
    }
}
