use crate::compat::CompatibilityReport;
use crate::language::SourceLang;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FixSuggestion {
    pub title: String,
    pub description: String,
    pub applies_to: SourceLang,
    pub patched_source: String,
    pub diff: String,
}

pub fn suggest_fixes(
    source: &str,
    lang: SourceLang,
    report: &CompatibilityReport,
) -> Vec<FixSuggestion> {
    match lang {
        SourceLang::PineV5 => suggest_pine(source, report),
        SourceLang::ThinkScriptSubset => suggest_thinkscript(source, report),
        SourceLang::NativeDsl => Vec::new(),
    }
}

fn suggest_pine(source: &str, report: &CompatibilityReport) -> Vec<FixSuggestion> {
    let mut fixes = Vec::new();
    if !source.contains("//@version=6")
        || report
            .issues
            .iter()
            .any(|i| i.message.contains("Missing //@version=6"))
    {
        let patched = format!("//@version=6\n{}", source.trim_start());
        fixes.push(FixSuggestion {
            title: "Add //@version=6".to_string(),
            description: "Declare Pine v6 version directive at the top of the script.".to_string(),
            applies_to: SourceLang::PineV5,
            diff: make_diff(source, &patched),
            patched_source: patched,
        });
    }

    if report
        .issues
        .iter()
        .any(|i| i.message.contains("No plots or signals found"))
    {
        let patched = format!("{}\nplot(close, title=\"auto-plot\")\n", source.trim_end());
        fixes.push(FixSuggestion {
            title: "Add default plot".to_string(),
            description: "Insert a simple plot(close) so the script produces output.".to_string(),
            applies_to: SourceLang::PineV5,
            diff: make_diff(source, &patched),
            patched_source: patched,
        });
    }

    fixes
}

fn suggest_thinkscript(source: &str, report: &CompatibilityReport) -> Vec<FixSuggestion> {
    let mut fixes = Vec::new();
    if report
        .issues
        .iter()
        .any(|i| i.message.contains("No plots or signals found"))
    {
        let patched = format!("{}\nplot p = close;\n", source.trim_end());
        fixes.push(FixSuggestion {
            title: "Add default plot".to_string(),
            description: "Insert a simple plot of close so the script produces output.".to_string(),
            applies_to: SourceLang::ThinkScriptSubset,
            diff: make_diff(source, &patched),
            patched_source: patched,
        });
    }
    fixes
}

fn make_diff(before: &str, after: &str) -> String {
    let mut diff = String::new();
    diff.push_str("--- before\n+++ after\n");
    diff.push_str("@@\n");
    let before_lines: Vec<&str> = before.lines().collect();
    let after_lines: Vec<&str> = after.lines().collect();
    let max_len = before_lines.len().max(after_lines.len());
    for idx in 0..max_len {
        match (before_lines.get(idx), after_lines.get(idx)) {
            (Some(a), Some(b)) if a == b => {
                diff.push_str(&format!(" {}|{}\n", idx + 1, a));
            }
            (Some(a), Some(b)) => {
                diff.push_str(&format!("-{}|{}\n", idx + 1, a));
                diff.push_str(&format!("+{}|{}\n", idx + 1, b));
            }
            (Some(a), None) => diff.push_str(&format!("-{}|{}\n", idx + 1, a)),
            (None, Some(b)) => diff.push_str(&format!("+{}|{}\n", idx + 1, b)),
            (None, None) => {}
        }
    }
    diff
}
