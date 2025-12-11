use app_shell::{
    AlertState, AppState, ChartState, DrawingKind, DrawingState, LayoutKind, LayoutState,
    OrderState, Theme,
};
use ta_engine::{IndicatorConfig, IndicatorKind, IndicatorParams, OutputKind, SourceField};

#[derive(Debug, Clone)]
pub struct PerfScenario {
    pub name: &'static str,
    pub charts: usize,
    pub overlays: usize,
    pub drawings: usize,
    pub orders: usize,
    pub alerts: usize,
}

pub fn default_scenarios() -> Vec<PerfScenario> {
    vec![
        PerfScenario {
            name: "single-100k",
            charts: 1,
            overlays: 6,
            drawings: 12,
            orders: 6,
            alerts: 6,
        },
        PerfScenario {
            name: "grid-2x2",
            charts: 4,
            overlays: 4,
            drawings: 20,
            orders: 8,
            alerts: 8,
        },
        PerfScenario {
            name: "mobile-1x2",
            charts: 2,
            overlays: 3,
            drawings: 10,
            orders: 4,
            alerts: 4,
        },
    ]
}

pub fn make_perf_state(s: &PerfScenario) -> AppState {
    let charts = build_charts(s);
    let (rows, cols) = grid_for(charts.len());
    let layout = LayoutState {
        id: 1,
        kind: LayoutKind::Grid {
            rows: rows as u8,
            cols: cols as u8,
        },
        charts,
    };

    AppState {
        theme: Theme::Dark,
        layouts: vec![layout],
        active_layout_id: Some(1),
    }
}

fn grid_for(n: usize) -> (usize, usize) {
    let r = (n as f64).sqrt().ceil() as usize;
    let c = ((n as f64) / r as f64).ceil() as usize;
    (r.max(1), c.max(1))
}

fn build_charts(s: &PerfScenario) -> Vec<ChartState> {
    let mut charts = Vec::new();
    for i in 0..s.charts.max(1) {
        let mut indicators = Vec::new();
        for k in 0..s.overlays {
            let cfg = IndicatorConfig::with_default_styles(
                match k % 3 {
                    0 => IndicatorKind::Sma,
                    1 => IndicatorKind::Ema,
                    _ => IndicatorKind::Rsi,
                },
                IndicatorParams::Sma {
                    period: 10 + k * 2,
                    source: SourceField::Close,
                },
                OutputKind::Overlay,
                None,
            );
            indicators.push(cfg);
        }

        let mut drawings = Vec::new();
        for d in 0..s.drawings {
            drawings.push(DrawingState {
                id: d as u64 + 1,
                kind: if d % 4 == 0 {
                    DrawingKind::HorizontalLine
                } else if d % 4 == 1 {
                    DrawingKind::VerticalLine
                } else if d % 4 == 2 {
                    DrawingKind::TrendLine
                } else {
                    DrawingKind::Rectangle
                },
                ts1: 1_700_000_000_000 + (d as i64) * 60_000,
                price1: 100.0 + d as f64,
                ts2: Some(1_700_000_000_000 + (d as i64 + 1) * 60_000),
                price2: Some(101.0 + d as f64),
                color: "#4da3ff".into(),
                width: 1.0,
            });
        }

        let mut orders = Vec::new();
        for o in 0..s.orders {
            orders.push(OrderState {
                id: format!("ord-{i}-{o}"),
                side: if o % 2 == 0 { "buy" } else { "sell" }.into(),
                price: 100.0 + o as f64,
                qty: 1.0 + (o as f64) * 0.1,
                label: "perf".into(),
                stop_price: Some(99.0),
                take_profit_price: Some(102.0),
            });
        }

        let mut alerts = Vec::new();
        for a in 0..s.alerts {
            alerts.push(AlertState {
                id: format!("al-{i}-{a}"),
                ts: 1_700_000_000_000 + a as i64 * 120_000,
                price: Some(100.0 + a as f64),
                label: "alert".into(),
                fired: a % 2 == 0,
            });
        }

        charts.push(ChartState {
            id: i as u32 + 1,
            symbol: format!("PERF{}", i + 1),
            timeframe: "1m".into(),
            indicators,
            drawings,
            link_group: Some("perf".into()),
            orders,
            positions: Vec::new(),
            alerts,
            price_pane_weight: 1.0,
            pane_layout: Vec::new(),
            pane: 0,
            height_ratio: 1.0,
            inputs: Vec::new(),
        });
    }
    charts
}
