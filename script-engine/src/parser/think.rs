use regex::Regex;
use std::collections::HashMap;

use crate::{
    compat::{CompatibilityIssue, CompatibilityReport, TranslationOutput, UnifiedIr},
    language::{DiagnosticCode, IssueSeverity, SourceLang, SourceSpan},
    manifest::Manifest,
    Expr, PriceField, ScriptSpec, SignalSpec, Source,
};

pub fn translate(source: &str) -> TranslationOutput {
    let mut parser = ThinkParser::new();
    let parsed = parser.parse(source);
    let supported = parsed
        .as_ref()
        .map(|spec| !parser.has_errors() && (!spec.plots.is_empty() || !spec.signals.is_empty()))
        .unwrap_or(false);
    let spec = parsed.unwrap_or_else(|| ScriptSpec {
        name: "thinkscript-unsupported".to_string(),
        inputs: HashMap::new(),
        plots: HashMap::new(),
        signals: HashMap::new(),
    });

    let manifest = Manifest::from_spec(&spec, SourceLang::ThinkScriptSubset);
    let report = CompatibilityReport {
        source_lang: SourceLang::ThinkScriptSubset,
        supported,
        issues: parser.issues,
    };

    TranslationOutput {
        ir: UnifiedIr { manifest, spec },
        report,
    }
}

struct ThinkParser {
    issues: Vec<CompatibilityIssue>,
    env: HashMap<String, Expr>,
}

impl ThinkParser {
    fn new() -> Self {
        Self {
            issues: Vec::new(),
            env: HashMap::new(),
        }
    }

    fn has_errors(&self) -> bool {
        self.issues
            .iter()
            .any(|i| i.severity == IssueSeverity::Error)
    }

    fn warn(&mut self, code: DiagnosticCode, msg: &str, line: Option<usize>, hint: Option<&str>) {
        self.issues.push(CompatibilityIssue {
            code,
            message: msg.to_string(),
            severity: IssueSeverity::Warning,
            span: line.map(|l| SourceSpan::single_line(l, 0, 0)),
            hint: hint.map(|h| h.to_string()),
        });
    }

    fn error(&mut self, code: DiagnosticCode, msg: &str, line: Option<usize>, hint: Option<&str>) {
        self.issues.push(CompatibilityIssue {
            code,
            message: msg.to_string(),
            severity: IssueSeverity::Error,
            span: line.map(|l| SourceSpan::single_line(l, 0, 0)),
            hint: hint.map(|h| h.to_string()),
        });
    }

    fn parse(&mut self, src: &str) -> Option<ScriptSpec> {
        let name = "think-script".to_string();
        let mut inputs: HashMap<String, f64> = HashMap::new();
        let mut plots: HashMap<String, Expr> = HashMap::new();
        let mut signals: HashMap<String, SignalSpec> = HashMap::new();

        let input_re = Regex::new(r#"(?i)^input\s+(\w+)\s*=\s*([0-9\.]+)"#).unwrap();
        let def_re = Regex::new(r#"(?i)^def\s+(\w+)\s*=\s*(.+)$"#).unwrap();
        let plot_re = Regex::new(r#"(?i)^plot\s+(\w+)\s*=\s*(.+)$"#).unwrap();
        let alert_re = Regex::new(r#"(?i)^alert\s+(\w+)\s*=\s*(.+)$"#).unwrap();
        let declare_re = Regex::new(r#"(?i)^declare\s+upper|lower"#).unwrap();

        for (idx, raw_line) in src.lines().enumerate() {
            let line = raw_line.trim();
            let line = line.trim_end_matches(';');
            if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
                continue;
            }
            if declare_re.is_match(line) {
                continue;
            }

            if let Some(cap) = input_re.captures(line) {
                let var = cap[1].to_string();
                let def: f64 = cap[2].parse().unwrap_or(0.0);
                inputs.insert(var.clone(), def);
                self.env.insert(var.clone(), Expr::Src(Source::Input(var)));
                continue;
            }

            if let Some(cap) = def_re.captures(line) {
                let var = cap[1].to_string();
                let rhs = cap[2].trim();
                if let Some(expr) = self.parse_expr(rhs) {
                    self.env.insert(var, expr);
                } else {
                    self.warn(
                        DiagnosticCode::UnsupportedExpression,
                        "Unsupported def expression",
                        Some(idx + 1),
                        Some("Simplify the expression or use supported functions"),
                    );
                }
                continue;
            }

            if let Some(cap) = plot_re.captures(line) {
                let name_cap = cap[1].to_string();
                let rhs = cap[2].trim();
                let expr = self.parse_expr(rhs).unwrap_or_else(|| {
                    self.warn(
                        DiagnosticCode::UnsupportedExpression,
                        "Unsupported plot expression, defaulting to NaN",
                        Some(idx + 1),
                        Some("Simplify the expression to a supported form"),
                    );
                    Expr::Src(Source::Const(f64::NAN))
                });
                plots.insert(name_cap, expr);
                continue;
            }

            if let Some(cap) = alert_re.captures(line) {
                let name_cap = cap[1].to_string();
                let rhs = cap[2].trim();
                if let Some(expr) = self.parse_expr(rhs) {
                    signals.insert(
                        name_cap,
                        SignalSpec::Greater {
                            a: expr.clone(),
                            b: Expr::Src(Source::Const(0.5)),
                        },
                    );
                } else {
                    self.warn(
                        DiagnosticCode::UnsupportedExpression,
                        "Unsupported alert expression",
                        Some(idx + 1),
                        Some("Alerts must resolve to true/false (crossover/crossunder)"),
                    );
                }
                continue;
            }

            // Fallback: attempt assignment with '='
            if let Some((lhs, rhs)) = line.split_once('=') {
                let lhs = lhs.trim();
                let rhs = rhs.trim();
                if let Some(expr) = self.parse_expr(rhs) {
                    self.env.insert(lhs.to_string(), expr);
                } else {
                    self.warn(
                        DiagnosticCode::UnsupportedExpression,
                        "Ignored assignment; unsupported expression",
                        Some(idx + 1),
                        Some("Try Average/ExpAverage/RSI or basic math"),
                    );
                }
            } else {
                self.warn(
                    DiagnosticCode::ParseError,
                    "Ignored line (unsupported construct)",
                    Some(idx + 1),
                    Some("Remove unsupported constructs or comment them"),
                );
            }
        }

        if plots.is_empty() && signals.is_empty() {
            self.error(
                DiagnosticCode::MissingPlotOrSignal,
                "No plots or signals found; script produces no outputs",
                None,
                Some("Add at least one plot or alert"),
            );
        }

        Some(ScriptSpec {
            name,
            inputs,
            plots,
            signals,
        })
    }

    fn parse_expr(&mut self, text: &str) -> Option<Expr> {
        let trimmed = text.trim().trim_end_matches(';');
        if let Ok(num) = trimmed.parse::<f64>() {
            return Some(Expr::Src(Source::Const(num)));
        }
        match trimmed.to_ascii_lowercase().as_str() {
            "open" => return Some(Expr::Src(Source::Price(PriceField::Open))),
            "high" => return Some(Expr::Src(Source::Price(PriceField::High))),
            "low" => return Some(Expr::Src(Source::Price(PriceField::Low))),
            "close" => return Some(Expr::Src(Source::Price(PriceField::Close))),
            "volume" => return Some(Expr::Src(Source::Price(PriceField::Volume))),
            _ => {}
        }
        if let Some(bound) = self.env.get(trimmed) {
            return Some(bound.clone());
        }

        // Function-like calls: Average(close, length), ExpAverage, RSI
        if let Some(expr) = self.parse_call(trimmed) {
            return Some(expr);
        }

        // Comparators and basic arithmetic
        if trimmed.contains('>')
            || trimmed.contains('<')
            || trimmed.contains('+')
            || trimmed.contains('-')
        {
            return self.parse_binary(trimmed);
        }

        if trimmed
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$')
        {
            self.warn(
                DiagnosticCode::UnknownIdentifier,
                &format!("Unknown identifier `{trimmed}`"),
                None,
                Some("Define it via `def` or `input` before use"),
            );
        }

        None
    }

    fn parse_call(&mut self, text: &str) -> Option<Expr> {
        let open = text.find('(')?;
        let close = text.rfind(')')?;
        if close <= open {
            return None;
        }
        let func = text[..open].trim();
        let args_str = &text[open + 1..close];
        let args: Vec<&str> = args_str.split(',').map(|s| s.trim()).collect();

        match func.to_ascii_lowercase().as_str() {
            "average" => {
                if args.len() >= 2 {
                    let src = self.parse_expr(args[0])?;
                    let period = args[1].parse::<usize>().ok()?;
                    Some(Expr::Sma {
                        period,
                        src: Box::new(src),
                    })
                } else {
                    None
                }
            }
            "exponentialaverage" | "expaverage" => {
                if args.len() >= 2 {
                    let src = self.parse_expr(args[0])?;
                    let period = args[1].parse::<usize>().ok()?;
                    Some(Expr::Ema {
                        period,
                        src: Box::new(src),
                    })
                } else {
                    None
                }
            }
            "rsi" => {
                if args.len() >= 2 {
                    let src = self.parse_expr(args[0])?;
                    let period = args[1].parse::<usize>().ok()?;
                    Some(Expr::Rsi {
                        period,
                        src: Box::new(src),
                    })
                } else {
                    None
                }
            }
            "crossover" => {
                if args.len() >= 2 {
                    let a = self.parse_expr(args[0])?;
                    let b = self.parse_expr(args[1])?;
                    Some(Expr::CrossOver(Box::new(a), Box::new(b)))
                } else {
                    None
                }
            }
            "crossunder" => {
                if args.len() >= 2 {
                    let a = self.parse_expr(args[0])?;
                    let b = self.parse_expr(args[1])?;
                    Some(Expr::CrossUnder(Box::new(a), Box::new(b)))
                } else {
                    None
                }
            }
            _ => {
                self.warn(
                    DiagnosticCode::UnsupportedFunction,
                    &format!("Unsupported function call: {func}"),
                    None,
                    Some("Supported: Average/ExpAverage/RSI, crossover/crossunder"),
                );
                None
            }
        }
    }

    fn parse_binary(&mut self, text: &str) -> Option<Expr> {
        if let Some((l, r)) = text.split_once('>') {
            let left = self.parse_expr(l.trim())?;
            let right = self.parse_expr(r.trim())?;
            return Some(Expr::Gt(Box::new(left), Box::new(right)));
        }
        if let Some((l, r)) = text.split_once('<') {
            let left = self.parse_expr(l.trim())?;
            let right = self.parse_expr(r.trim())?;
            return Some(Expr::Lt(Box::new(left), Box::new(right)));
        }
        if let Some((l, r)) = text.split_once('+') {
            let left = self.parse_expr(l.trim())?;
            let right = self.parse_expr(r.trim())?;
            return Some(Expr::Add(Box::new(left), Box::new(right)));
        }
        if let Some((l, r)) = text.split_once('-') {
            let left = self.parse_expr(l.trim())?;
            let right = self.parse_expr(r.trim())?;
            return Some(Expr::Sub(Box::new(left), Box::new(right)));
        }
        None
    }
}
