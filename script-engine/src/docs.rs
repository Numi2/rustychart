use serde::Serialize;

use crate::language::SourceLang;

/// Lightweight documentation surface for built-in functions and keywords.
#[derive(Debug, Clone, Serialize)]
pub struct BuiltinDoc {
    pub name: &'static str,
    pub signature: &'static str,
    pub description: &'static str,
    pub example: &'static str,
    pub group: &'static str,
}

/// Return docs for a given source language. Keep this small and static so it can
/// be shipped to the frontend for completions and inline help without allocating
/// at runtime.
pub fn builtin_docs(lang: SourceLang) -> Vec<BuiltinDoc> {
    match lang {
        SourceLang::PineV5 => pine_docs(),
        SourceLang::ThinkScriptSubset => think_docs(),
        SourceLang::NativeDsl => native_docs(),
    }
}

fn pine_docs() -> Vec<BuiltinDoc> {
    vec![
        BuiltinDoc {
            name: "ta.ema",
            signature: "ta.ema(src, length)",
            description: "Exponential moving average of a series.",
            example: "ema1 = ta.ema(close, 21)",
            group: "trend",
        },
        BuiltinDoc {
            name: "ta.sma",
            signature: "ta.sma(src, length)",
            description: "Simple moving average of a series.",
            example: "sma1 = ta.sma(close, 50)",
            group: "trend",
        },
        BuiltinDoc {
            name: "ta.rsi",
            signature: "ta.rsi(src, length)",
            description: "Relative Strength Index oscillator.",
            example: "rsi1 = ta.rsi(close, 14)",
            group: "momentum",
        },
        BuiltinDoc {
            name: "ta.macd",
            signature: "ta.macd(src, fastlen, slowlen, siglen)",
            description: "MACD triple EMA momentum.",
            example: "macd_val = ta.macd(close, 12, 26, 9)",
            group: "momentum",
        },
        BuiltinDoc {
            name: "ta.crossover",
            signature: "ta.crossover(a, b)",
            description: "True when a crosses above b this bar.",
            example: "bull = ta.crossover(close, ta.ema(close, 21))",
            group: "logic",
        },
        BuiltinDoc {
            name: "ta.crossunder",
            signature: "ta.crossunder(a, b)",
            description: "True when a crosses under b this bar.",
            example: "bear = ta.crossunder(close, ta.ema(close, 21))",
            group: "logic",
        },
    ]
}

fn think_docs() -> Vec<BuiltinDoc> {
    vec![
        BuiltinDoc {
            name: "ExpAverage",
            signature: "ExpAverage(src, length)",
            description: "Exponential moving average (EMA).",
            example: "def ema1 = ExpAverage(close, 21);",
            group: "trend",
        },
        BuiltinDoc {
            name: "Average",
            signature: "Average(src, length)",
            description: "Simple moving average (SMA).",
            example: "def sma1 = Average(close, 50);",
            group: "trend",
        },
        BuiltinDoc {
            name: "RSI",
            signature: "RSI(src, length)",
            description: "Relative Strength Index oscillator.",
            example: "def rsi1 = RSI(close, 14);",
            group: "momentum",
        },
        BuiltinDoc {
            name: "MACD",
            signature: "MACD(src, fastlen, slowlen, siglen)",
            description: "MACD momentum and histogram.",
            example: "def macd = MACD(close, 12, 26, 9);",
            group: "momentum",
        },
        BuiltinDoc {
            name: "crossover",
            signature: "crossover(a, b)",
            description: "True when a crosses above b this bar.",
            example: "def bull = crossover(close, ema1);",
            group: "logic",
        },
        BuiltinDoc {
            name: "crossunder",
            signature: "crossunder(a, b)",
            description: "True when a crosses under b this bar.",
            example: "def bear = crossunder(close, ema1);",
            group: "logic",
        },
    ]
}

fn native_docs() -> Vec<BuiltinDoc> {
    vec![
        BuiltinDoc {
            name: "ema",
            signature: "ema(src, length)",
            description: "Exponential moving average.",
            example: "ema(close, 34)",
            group: "trend",
        },
        BuiltinDoc {
            name: "sma",
            signature: "sma(src, length)",
            description: "Simple moving average.",
            example: "sma(close, 50)",
            group: "trend",
        },
        BuiltinDoc {
            name: "rsi",
            signature: "rsi(src, length)",
            description: "RSI oscillator returning 0-100.",
            example: "rsi(close, 14)",
            group: "momentum",
        },
        BuiltinDoc {
            name: "macd",
            signature: "macd(src, fast, slow, signal)",
            description: "MACD line minus signal line.",
            example: "macd(close, 12, 26, 9)",
            group: "momentum",
        },
        BuiltinDoc {
            name: "cross_over",
            signature: "cross_over(a, b)",
            description: "1.0 only on crossing a > b.",
            example: "cross_over(close, ema(close, 21))",
            group: "logic",
        },
        BuiltinDoc {
            name: "cross_under",
            signature: "cross_under(a, b)",
            description: "1.0 only on crossing a < b.",
            example: "cross_under(close, ema(close, 21))",
            group: "logic",
        },
    ]
}
