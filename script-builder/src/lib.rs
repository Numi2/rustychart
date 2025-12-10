use std::collections::HashMap;

use script_engine::{Expr, ScriptEngine, ScriptInstance, ScriptResult, ScriptSpec, SignalSpec, Source, PriceField};
use ts_core::TimeFrame;

/// High-level building blocks to drive a visual builder UI.
#[derive(Debug, Clone)]
pub struct IndicatorBlock {
    pub id: String,
    pub label: String,
    pub description: String,
    pub params: HashMap<String, f64>,
    pub expr: Expr,
}

#[derive(Debug, Clone)]
pub struct ScoreRule {
    pub label: String,
    pub description: String,
    pub condition: Expr, // boolean -> 1.0/0.0
    pub weight: f64,
}

#[derive(Debug, Clone)]
pub struct ActionRule {
    pub name: String,
    pub description: String,
    pub when_score_gte: f64,
    pub signal: String, // "buy" | "sell" | "exit"
}

#[derive(Debug, Clone)]
pub struct VisualStrategy {
    pub name: String,
    pub timeframe: TimeFrame,
    pub indicators: Vec<IndicatorBlock>,
    pub score_rules: Vec<ScoreRule>,
    pub actions: Vec<ActionRule>,
}

impl VisualStrategy {
    /// Compile the visual representation into a ScriptInstance ready for incremental evaluation.
    pub fn compile(self) -> ScriptInstance {
        let mut plots: HashMap<String, Expr> = HashMap::new();
        for ind in &self.indicators {
            plots.insert(ind.id.clone(), ind.expr.clone());
        }

        // score = sum(weight * condition_boolean)
        let mut score_expr: Option<Expr> = None;
        for rule in &self.score_rules {
            let term = Expr::Mul(
                Box::new(Expr::Src(Source::Const(rule.weight))),
                Box::new(rule.condition.clone()),
            );
            score_expr = Some(if let Some(cur) = score_expr {
                Expr::Add(Box::new(cur), Box::new(term))
            } else {
                term
            });
        }
        if let Some(se) = score_expr {
            plots.insert("score".to_string(), se);
        }

        // Signals based on score thresholds.
        let mut signals = HashMap::new();
        for action in &self.actions {
            let cond = Expr::Gt(
                Box::new(Expr::Src(Source::Series("score".to_string()))),
                Box::new(Expr::Src(Source::Const(action.when_score_gte))),
            );
            signals.insert(
                action.signal.clone(),
                SignalSpec::Greater {
                    a: Expr::Src(Source::Series("score".to_string())),
                    b: Expr::Src(Source::Const(action.when_score_gte)),
                },
            );
            // Also keep a plot that can be used to render action markers (1 on trigger else 0).
            plots.insert(
                format!("sig_{}", action.signal),
                Expr::Mul(Box::new(cond), Box::new(Expr::Src(Source::Const(1.0)))),
            );
        }

        let spec = ScriptSpec {
            name: self.name.clone(),
            inputs: HashMap::new(),
            plots,
            signals,
        };

        ScriptEngine::compile(self.timeframe, spec, HashMap::new())
    }
}

/// Some beginner-friendly presets that the UI can show as templates.
pub fn preset_trend_follow(timeframe: TimeFrame) -> VisualStrategy {
    let fast = IndicatorBlock {
        id: "ema_fast".to_string(),
        label: "EMA Fast".to_string(),
        description: "Short-term EMA (trend)".to_string(),
        params: [("period".to_string(), 12.0)].into_iter().collect(),
        expr: Expr::Ema {
            period: 12,
            src: Box::new(Expr::Src(Source::Price(PriceField::Close))),
        },
    };
    let slow = IndicatorBlock {
        id: "ema_slow".to_string(),
        label: "EMA Slow".to_string(),
        description: "Long-term EMA (trend)".to_string(),
        params: [("period".to_string(), 26.0)].into_iter().collect(),
        expr: Expr::Ema {
            period: 26,
            src: Box::new(Expr::Src(Source::Price(PriceField::Close))),
        },
    };

    let score_rules = vec![
        ScoreRule {
            label: "Fast above Slow".to_string(),
            description: "Uptrend bias".to_string(),
            condition: Expr::Gt(
                Box::new(Expr::Src(Source::Series("ema_fast".to_string()))),
                Box::new(Expr::Src(Source::Series("ema_slow".to_string()))),
            ),
            weight: 1.0,
        },
        ScoreRule {
            label: "Bullish crossover".to_string(),
            description: "Fresh momentum".to_string(),
            condition: Expr::CrossOver(
                Box::new(Expr::Src(Source::Series("ema_fast".to_string()))),
                Box::new(Expr::Src(Source::Series("ema_slow".to_string()))),
            ),
            weight: 2.0,
        },
    ];

    let actions = vec![
        ActionRule {
            name: "Enter Long".to_string(),
            description: "Buy when score high".to_string(),
            when_score_gte: 2.0,
            signal: "buy".to_string(),
        },
        ActionRule {
            name: "Exit Long".to_string(),
            description: "Exit when score weakens".to_string(),
            when_score_gte: 1.0,
            signal: "exit".to_string(),
        },
        ActionRule {
            name: "Enter Short".to_string(),
            description: "Sell when score negative".to_string(),
            when_score_gte: -2.0,
            signal: "sell".to_string(),
        },
    ];

    VisualStrategy {
        name: "Trend Follow (EMA)".to_string(),
        timeframe,
        indicators: vec![fast, slow],
        score_rules,
        actions,
    }
}

/// Simple mean reversion preset using RSI.
pub fn preset_mean_revert(timeframe: TimeFrame) -> VisualStrategy {
    let rsi = IndicatorBlock {
        id: "rsi".to_string(),
        label: "RSI".to_string(),
        description: "Momentum oscillator".to_string(),
        params: [("period".to_string(), 14.0)].into_iter().collect(),
        expr: Expr::Rsi {
            period: 14,
            src: Box::new(Expr::Src(Source::Price(PriceField::Close))),
        },
    };

    let score_rules = vec![
        ScoreRule {
            label: "Oversold".to_string(),
            description: "RSI < 30".to_string(),
            condition: Expr::Lt(
                Box::new(Expr::Src(Source::Series("rsi".to_string()))),
                Box::new(Expr::Src(Source::Const(30.0))),
            ),
            weight: 2.0,
        },
        ScoreRule {
            label: "Overbought".to_string(),
            description: "RSI > 70".to_string(),
            condition: Expr::Gt(
                Box::new(Expr::Src(Source::Series("rsi".to_string()))),
                Box::new(Expr::Src(Source::Const(70.0))),
            ),
            weight: -2.0,
        },
    };

    let actions = vec![
        ActionRule {
            name: "Enter Long".to_string(),
            description: "Buy oversold".to_string(),
            when_score_gte: 2.0,
            signal: "buy".to_string(),
        },
        ActionRule {
            name: "Enter Short".to_string(),
            description: "Sell overbought".to_string(),
            when_score_gte: -2.0,
            signal: "sell".to_string(),
        },
        ActionRule {
            name: "Exit".to_string(),
            description: "Flatten when neutral".to_string(),
            when_score_gte: 0.0,
            signal: "exit".to_string(),
        },
    ];

    VisualStrategy {
        name: "Mean Revert (RSI)".to_string(),
        timeframe,
        indicators: vec![rsi],
        score_rules,
        actions,
    }
}

