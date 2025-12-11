use crate::language::SourceLang;
use serde::Serialize;

/// Simple, curated script templates to speed up authoring.
#[derive(Debug, Clone, Serialize)]
pub struct Template {
    pub name: &'static str,
    pub description: &'static str,
    pub source_lang: SourceLang,
    pub source: &'static str,
    pub tags: &'static [&'static str],
}

pub fn templates_for_lang(lang: SourceLang) -> Vec<Template> {
    match lang {
        SourceLang::PineV5 => pine_templates(),
        SourceLang::ThinkScriptSubset => think_templates(),
        SourceLang::NativeDsl => Vec::new(),
    }
}

pub fn find_template(lang: SourceLang, name: &str) -> Option<Template> {
    templates_for_lang(lang)
        .into_iter()
        .find(|t| t.name.eq_ignore_ascii_case(name))
}

fn pine_templates() -> Vec<Template> {
    vec![
        Template {
            name: "starter",
            description: "Starter template with version, plot, alert",
            source_lang: SourceLang::PineV5,
            tags: &["starter", "onboarding", "alert"],
            source: r#"//@version=6
indicator("Starter", overlay=true)
len = input.int(20, "Length")
src = close
ema1 = ta.ema(src, len)
plot(ema1, title="ema")
alertcondition(ta.crossover(src, ema1), "Cross Up", "Price crossed above EMA")
"#,
        },
        Template {
            name: "ema",
            description: "Single EMA with length input and plot",
            source_lang: SourceLang::PineV5,
            tags: &["ema", "trend", "ma"],
            source: r#"//@version=6
indicator("EMA", overlay=true)
len = input.int(20, "Length")
src = close
ema1 = ta.ema(src, len)
plot(ema1, title="ema")
"#,
        },
        Template {
            name: "rsi",
            description: "RSI plot with length input",
            source_lang: SourceLang::PineV5,
            tags: &["rsi", "momentum"],
            source: r#"//@version=6
indicator("RSI", overlay=false)
len = input.int(14, "Length")
src = close
rsi1 = ta.rsi(src, len)
plot(rsi1, title="rsi")
"#,
        },
        Template {
            name: "macd",
            description: "MACD (12,26,9) with histogram",
            source_lang: SourceLang::PineV5,
            tags: &["macd", "trend", "momentum"],
            source: r#"//@version=6
indicator("MACD", overlay=false)
fast = input.int(12, "Fast")
slow = input.int(26, "Slow")
signal = input.int(9, "Signal")
macd = ta.ema(close, fast) - ta.ema(close, slow)
sig = ta.ema(macd, signal)
hist = macd - sig
plot(macd, title="macd")
plot(sig, title="signal")
plot(hist, title="hist", style=plot.style_columns)
"#,
        },
        Template {
            name: "crossover-alert",
            description: "Alert when price crosses above EMA",
            source_lang: SourceLang::PineV5,
            tags: &["alert", "crossover", "ema"],
            source: r#"//@version=6
indicator("EMA Cross Alert", overlay=true)
len = input.int(20, "Length")
ema1 = ta.ema(close, len)
plot(ema1, title="ema")
alertcondition(ta.crossover(close, ema1), "Cross Up", "Price crossed above EMA")
"#,
        },
    ]
}

fn think_templates() -> Vec<Template> {
    vec![
        Template {
            name: "starter",
            description: "Starter template with plot and alert",
            source_lang: SourceLang::ThinkScriptSubset,
            tags: &["starter", "onboarding", "alert"],
            source: r#"
input len = 20;
def ema1 = ExpAverage(close, len);
plot ema = ema1;
alert crossUp = crossover(close, ema1);
"#,
        },
        Template {
            name: "ema",
            description: "Single EMA with length input and plot",
            source_lang: SourceLang::ThinkScriptSubset,
            tags: &["ema", "trend", "ma"],
            source: r#"
input len = 20;
def ema1 = ExpAverage(close, len);
plot ema = ema1;
"#,
        },
        Template {
            name: "rsi",
            description: "RSI plot with length input",
            source_lang: SourceLang::ThinkScriptSubset,
            tags: &["rsi", "momentum"],
            source: r#"
input len = 14;
def rsi1 = RSI(close, len);
plot rsi = rsi1;
"#,
        },
        Template {
            name: "macd",
            description: "MACD (12,26,9) with histogram",
            source_lang: SourceLang::ThinkScriptSubset,
            tags: &["macd", "trend", "momentum"],
            source: r#"
input fast = 12;
input slow = 26;
input signal = 9;
def macd = ExpAverage(close, fast) - ExpAverage(close, slow);
def sig = ExpAverage(macd, signal);
def hist = macd - sig;
plot macdLine = macd;
plot signalLine = sig;
plot histogram = hist;
"#,
        },
        Template {
            name: "crossover-alert",
            description: "Alert when price crosses above EMA",
            source_lang: SourceLang::ThinkScriptSubset,
            tags: &["alert", "crossover", "ema"],
            source: r#"
input len = 20;
def ema1 = ExpAverage(close, len);
plot ema = ema1;
alert up = crossover(close, ema1);
"#,
        },
    ]
}
