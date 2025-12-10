use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{
    compat::{
        CompatibilityIssue, CompatibilityReport, IssueSeverity, TranslationOutput, UnifiedIr,
    },
    language::SourceLang,
    manifest::Manifest,
    Expr, PriceField, ScriptSpec, SignalSpec, Source,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ParsedScript {
    name: String,
    inputs: HashMap<String, f64>,
    plots: HashMap<String, Expr>,
    signals: HashMap<String, SignalSpec>,
}

pub fn translate(source: &str) -> TranslationOutput {
    let mut parser = PineParser::new();
    let parsed = parser.parse(source);
    let supported = parsed
        .as_ref()
        .map(|spec| !parser.has_errors() && (!spec.plots.is_empty() || !spec.signals.is_empty()))
        .unwrap_or(false);
    let spec = parsed.unwrap_or_else(|| ScriptSpec {
        name: "pine-unsupported".to_string(),
        inputs: HashMap::new(),
        plots: HashMap::new(),
        signals: HashMap::new(),
    });

    let manifest = Manifest::from_spec(&spec, SourceLang::PineV5);
    let report = CompatibilityReport {
        source_lang: SourceLang::PineV5,
        supported,
        issues: parser.issues,
    };

    TranslationOutput {
        ir: UnifiedIr { manifest, spec },
        report,
    }
}

struct PineParser {
    issues: Vec<CompatibilityIssue>,
    env: HashMap<String, Expr>,
    version_seen: bool,
    version_line: Option<usize>,
}

fn first_arg(arglist: &str) -> String {
    let mut depth = 0usize;
    let mut buf = String::new();
    for ch in arglist.chars() {
        match ch {
            '(' => {
                depth += 1;
                buf.push(ch);
            }
            ')' => {
                if depth == 0 {
                    break;
                }
                depth -= 1;
                buf.push(ch);
            }
            ',' if depth == 0 => break,
            _ => buf.push(ch),
        }
    }
    buf.trim().trim_end_matches(')').to_string()
}

impl PineParser {
    fn new() -> Self {
        Self {
            issues: Vec::new(),
            env: HashMap::new(),
            version_seen: false,
            version_line: None,
        }
    }

    fn has_errors(&self) -> bool {
        self.issues
            .iter()
            .any(|i| i.severity == IssueSeverity::Error)
    }

    fn warn(&mut self, msg: &str, line: Option<usize>) {
        self.issues.push(CompatibilityIssue {
            message: msg.to_string(),
            severity: IssueSeverity::Warning,
            location: line.map(|l| (l, 0)),
        });
    }

    fn error(&mut self, msg: &str, line: Option<usize>) {
        self.issues.push(CompatibilityIssue {
            message: msg.to_string(),
            severity: IssueSeverity::Error,
            location: line.map(|l| (l, 0)),
        });
    }

    fn parse(&mut self, src: &str) -> Option<ScriptSpec> {
        let mut name = "pine-script".to_string();
        let mut inputs: HashMap<String, f64> = HashMap::new();
        let mut plots: HashMap<String, Expr> = HashMap::new();
        let mut signals: HashMap<String, SignalSpec> = HashMap::new();

        let indicator_re =
            Regex::new(r#"(?i)indicator\s*\(\s*"([^"]+)""#).expect("regex indicator");
        let strategy_re = Regex::new(r#"(?i)strategy\s*\(\s*"([^"]+)""#).expect("regex strategy");
        let version_re = Regex::new(r"//@version\s*=?\s*6").expect("regex version");
        let input_re = Regex::new(r#"(?i)^(\w+)\s*=\s*input\.[a-z]+\s*\(\s*([0-9\.]+)"#).unwrap();
        let plot_re = Regex::new(r"(?i)plot\s*\(\s*(.+)\)").unwrap();
        let plot_title_re = Regex::new(r#"title\s*=\s*"([^"]+)""#).unwrap();
        let alert_re = Regex::new(r"(?i)alertcondition\s*\(\s*(.+)\)").unwrap();
        let alert_title_re = Regex::new(r#"(?i)"([^"]+)""#).unwrap();

        for (idx, raw_line) in src.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with("//") {
                continue;
            }
            if version_re.is_match(line) {
                self.version_seen = true;
                self.version_line = Some(idx + 1);
                continue;
            }
            if let Some(cap) = indicator_re.captures(line) {
                name = cap[1].to_string();
                continue;
            }
            if let Some(cap) = strategy_re.captures(line) {
                name = cap[1].to_string();
                continue;
            }
            if let Some(cap) = input_re.captures(line) {
                let var = cap[1].to_string();
                let def: f64 = cap[2].parse().unwrap_or(0.0);
                inputs.insert(var.clone(), def);
                self.env.insert(var.clone(), Expr::Src(Source::Input(var)));
                continue;
            }

            if let Some(cap) = plot_re.captures(line) {
                let expr_text = first_arg(cap[1].trim());
                let title = plot_title_re
                    .captures(line)
                    .and_then(|c| c.get(1).map(|m| m.as_str().to_string()))
                    .unwrap_or_else(|| format!("plot{}", plots.len() + 1));
                let expr = self.parse_expr(&expr_text).unwrap_or_else(|| {
                    self.warn(
                        "Unsupported plot expression, defaulting to NaN",
                        Some(idx + 1),
                    );
                    Expr::Src(Source::Const(f64::NAN))
                });
                plots.insert(title, expr);
                continue;
            }

            if let Some(cap) = alert_re.captures(line) {
                let cond = first_arg(cap[1].trim());
                let title = alert_title_re
                    .captures_iter(line)
                    .nth(1)
                    .map(|c| c[1].to_string())
                    .unwrap_or_else(|| format!("alert{}", signals.len() + 1));
                let expr = self.parse_expr(&cond).unwrap_or_else(|| {
                    self.warn(
                        "Unsupported alert expression, defaulting to false",
                        Some(idx + 1),
                    );
                    Expr::Src(Source::Const(0.0))
                });
                let sig = SignalSpec::Greater {
                    a: expr.clone(),
                    b: Expr::Src(Source::Const(0.5)),
                };
                signals.insert(title, sig);
                continue;
            }

            // Assignment line: foo = <expr>
            if let Some((lhs, rhs)) = line.split_once('=') {
                let lhs = lhs.trim();
                let rhs = rhs.trim();
                if lhs.is_empty() || rhs.is_empty() {
                    self.warn("Malformed assignment", Some(idx + 1));
                    continue;
                }
                if let Some(expr) = self.parse_expr(rhs) {
                    self.env.insert(lhs.to_string(), expr);
                } else {
                    self.warn(
                        "Unsupported expression in assignment; skipped",
                        Some(idx + 1),
                    );
                }
                continue;
            }

            self.warn("Ignored line (unsupported construct)", Some(idx + 1));
        }

        if !self.version_seen {
            self.warn(
                "Missing //@version=6; proceeding with subset parsing",
                Some(1),
            );
        }

        if plots.is_empty() && signals.is_empty() {
            self.error(
                "No plots or signals found; script produces no outputs",
                None,
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
        let cleaned = text.trim().trim_end_matches(';');

        // Try simple literals
        if let Ok(num) = cleaned.parse::<f64>() {
            return Some(Expr::Src(Source::Const(num)));
        }
        // Price fields
        let lowered = cleaned.to_ascii_lowercase();
        match lowered.as_str() {
            "open" => return Some(Expr::Src(Source::Price(PriceField::Open))),
            "high" => return Some(Expr::Src(Source::Price(PriceField::High))),
            "low" => return Some(Expr::Src(Source::Price(PriceField::Low))),
            "close" => return Some(Expr::Src(Source::Price(PriceField::Close))),
            "volume" => return Some(Expr::Src(Source::Price(PriceField::Volume))),
            _ => {}
        }
        // Environment binding
        if let Some(bound) = self.env.get(cleaned) {
            return Some(bound.clone());
        }

        // Function calls: ta.sma(x, 20)
        if let Some(expr) = self.parse_call(cleaned) {
            return Some(expr);
        }

        // Binary operations for simple comparators.
        if cleaned.contains('>')
            || cleaned.contains('<')
            || cleaned.contains('+')
            || cleaned.contains('-')
        {
            return self.parse_binary(cleaned);
        }

        None
    }

    fn parse_call(&mut self, text: &str) -> Option<Expr> {
        // pattern func(arg1, arg2)
        let open = text.find('(')?;
        let close = text.rfind(')')?;
        if close <= open {
            return None;
        }
        let func = text[..open].trim();
        let args_str = &text[open + 1..close];
        let args: Vec<&str> = args_str.split(',').map(|s| s.trim()).collect();

        match func {
            "ta.sma" | "sma" => {
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
            "ta.ema" | "ema" => {
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
            "ta.rsi" | "rsi" => {
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
            "ta.crossover" | "crossover" => {
                if args.len() >= 2 {
                    let left = self.parse_expr(args[0])?;
                    let right = self.parse_expr(args[1])?;
                    Some(Expr::CrossOver(Box::new(left), Box::new(right)))
                } else {
                    None
                }
            }
            "ta.crossunder" | "crossunder" => {
                if args.len() >= 2 {
                    let left = self.parse_expr(args[0])?;
                    let right = self.parse_expr(args[1])?;
                    Some(Expr::CrossUnder(Box::new(left), Box::new(right)))
                } else {
                    None
                }
            }
            _ => {
                self.warn(&format!("Unsupported function call: {func}"), None);
                None
            }
        }
    }

    fn parse_binary(&mut self, text: &str) -> Option<Expr> {
        // Very small parser: check for > or < or + or -
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
