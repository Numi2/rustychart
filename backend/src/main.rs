use std::cmp::Ordering;
use std::env;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use app_shell::AppState;
use axum::extract::ws::{Message as WsMessage, WebSocket, WebSocketUpgrade};
use axum::extract::{FromRef, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, get_service};
use axum::{Json, Router};
use chrono::{DateTime, Utc};
use data_feed::DataEvent;
use futures_util::{SinkExt, StreamExt};
use leptos::get_configuration;
use leptos_axum::{generate_route_list, LeptosRoutes};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use serde_json::json;
use tokio::time::interval;
use tower_http::services::ServeDir;
use ts_core::{Candle, SeriesStore, Tick, TimeFrame, Timestamp};

#[derive(Clone)]
struct FeedState {
    store: Arc<Mutex<SeriesStore>>,
    base_tf: TimeFrame,
}

#[derive(Clone)]
struct ServerState {
    feed: FeedState,
    leptos_options: leptos::LeptosOptions,
    layout_store: Arc<Mutex<Option<AppState>>>,
}

impl FromRef<ServerState> for leptos::LeptosOptions {
    fn from_ref(state: &ServerState) -> leptos::LeptosOptions {
        state.leptos_options.clone()
    }
}

#[derive(Debug, serde::Deserialize)]
struct HistoryParams {
    tf: String,
    from: Option<Timestamp>,
    to: Option<Timestamp>,
    limit: Option<usize>,
    symbol: Option<String>,
    provider: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct MarketParams {
    symbol: Option<String>,
    days: Option<u64>,
    interval: Option<String>,
    provider: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct SearchParams {
    q: Option<String>,
    limit: Option<usize>,
}

#[derive(Debug, Serialize)]
struct SearchHit {
    symbol: String,
    name: Option<String>,
    exchange: Option<String>,
    quote_type: Option<String>,
    score: Option<f64>,
}

#[derive(Debug, Serialize)]
struct OhlcvBar {
    ts: i64,
    date: String,
    time: String,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: u64,
}

#[derive(Debug, serde::Deserialize)]
struct YahooResponse {
    chart: Option<YahooChart>,
}

#[derive(Debug, serde::Deserialize)]
struct YahooSearchResponse {
    quotes: Option<Vec<YahooSearchQuote>>,
}

#[derive(Debug, serde::Deserialize)]
struct YahooSearchQuote {
    symbol: Option<String>,
    shortname: Option<String>,
    longname: Option<String>,
    exchdisp: Option<String>,
    #[serde(rename = "quoteType")]
    quote_type: Option<String>,
    score: Option<f64>,
}

#[derive(Debug, serde::Deserialize)]
struct YahooChart {
    result: Option<Vec<YahooResult>>,
}

#[derive(Debug, serde::Deserialize)]
struct YahooResult {
    timestamp: Option<Vec<i64>>,
    indicators: Option<Indicators>,
}

#[derive(Debug, serde::Deserialize)]
struct Indicators {
    quote: Option<Vec<Quote>>,
}

#[derive(Debug, Default, serde::Deserialize)]
struct Quote {
    open: Option<Vec<Option<f64>>>,
    high: Option<Vec<Option<f64>>>,
    low: Option<Vec<Option<f64>>>,
    close: Option<Vec<Option<f64>>>,
    volume: Option<Vec<Option<u64>>>,
}

impl Quote {
    fn value_at(series: &Option<Vec<Option<f64>>>, idx: usize) -> Option<f64> {
        series.as_ref().and_then(|v| v.get(idx)).and_then(|v| *v)
    }

    fn open_value(&self, idx: usize) -> Option<f64> {
        Self::value_at(&self.open, idx)
    }

    fn high_value(&self, idx: usize) -> Option<f64> {
        Self::value_at(&self.high, idx)
    }

    fn low_value(&self, idx: usize) -> Option<f64> {
        Self::value_at(&self.low, idx)
    }

    fn close_value(&self, idx: usize) -> Option<f64> {
        Self::value_at(&self.close, idx)
    }

    fn volume_value(&self, idx: usize) -> Option<u64> {
        self.volume
            .as_ref()
            .and_then(|v| v.get(idx))
            .and_then(|v| *v)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MarketProvider {
    Yahoo,
    Coinbase,
}

impl MarketProvider {
    fn from_str(value: Option<&str>) -> Self {
        match value.map(|v| v.to_ascii_lowercase()) {
            Some(v) if v == "coinbase" || v == "cb" => MarketProvider::Coinbase,
            _ => MarketProvider::Yahoo,
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            MarketProvider::Yahoo => "yahoo",
            MarketProvider::Coinbase => "coinbase",
        }
    }
}

fn normalize_coinbase_symbol(symbol: &str) -> String {
    if symbol.contains('-') {
        return symbol.to_ascii_uppercase();
    }
    if symbol.len() > 3 {
        let (base, quote) = symbol.split_at(symbol.len() - 3);
        format!("{}-{}", base.to_ascii_uppercase(), quote.to_ascii_uppercase())
    } else {
        symbol.to_ascii_uppercase()
    }
}

// Allows overriding the default provider without changing clients.
const DEFAULT_PROVIDER_ENV: &str = "MARKET_DATA_PROVIDER";
// Lets deployments point to a different Coinbase-compatible base URL.
const COINBASE_API_URL_ENV: &str = "COINBASE_API_URL";
const DEFAULT_SYMBOL_ENV: &str = "MARKET_SYMBOL";

type MarketResponse = Json<serde_json::Value>;

fn round_two(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

fn coinbase_granularity(interval: &str) -> Option<u64> {
    match interval {
        "1m" => Some(60),
        "5m" => Some(300),
        "15m" => Some(900),
        "1h" => Some(3_600),
        "6h" => Some(21_600),
        "1d" => Some(86_400),
        _ => None,
    }
}

fn map_coinbase_candles(mut candles: Vec<[f64; 6]>) -> Vec<OhlcvBar> {
    candles.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(Ordering::Equal));

    let mut data = Vec::with_capacity(candles.len());

    for entry in candles {
        let ts = entry[0] as i64;
        let Some(dt) = DateTime::<Utc>::from_timestamp(ts, 0) else {
            continue;
        };

        let date = dt.date_naive().format("%Y-%m-%d").to_string();
        let time = dt.time().format("%H:%M").to_string();

        let low = entry[1];
        let high = entry[2];
        let open = entry[3];
        let close = entry[4];
        let volume = entry[5].round() as u64;

        data.push(OhlcvBar {
            ts: ts * 1_000,
            date,
            time,
            open: round_two(open),
            high: round_two(high),
            low: round_two(low),
            close: round_two(close),
            volume,
        });
    }

    data
}

async fn fetch_coinbase_history(
    symbol: &str,
    tf: TimeFrame,
    limit: usize,
) -> Option<Vec<Candle>> {
    let product = normalize_coinbase_symbol(symbol);
    let granularity = coinbase_granularity(&tf.name())?;
    let base = env::var(COINBASE_API_URL_ENV)
        .unwrap_or_else(|_| "https://api.exchange.coinbase.com".to_string());
    let url = format!("{base}/products/{product}/candles?granularity={granularity}&limit={limit}");

    let resp = reqwest::Client::new().get(&url).send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let mut candles: Vec<[f64; 6]> = resp.json().await.ok()?;
    candles.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(Ordering::Equal));

    let mapped = candles
        .into_iter()
        .filter_map(|entry| {
            if entry.len() != 6 {
                return None;
            }
            let ts = (entry[0] as i64) * 1_000;
            Some(Candle {
                ts,
                timeframe: tf,
                open: entry[3],
                high: entry[2],
                low: entry[1],
                close: entry[4],
                volume: entry[5],
            })
        })
        .collect::<Vec<_>>();
    Some(mapped)
}

async fn fetch_yahoo_history(
    symbol: &str,
    tf: TimeFrame,
    days: u64,
) -> Option<Vec<Candle>> {
    let interval = tf.name();
    let symbol = symbol.to_string();
    let end_date = Utc::now().timestamp();
    let start_date = end_date - (days as i64) * 24 * 60 * 60;

    let url = format!(
        "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={start_date}&period2={end_date}&interval={interval}&includePrePost=true"
    );

    let client = reqwest::Client::new();
    let response = client.get(&url).send().await.ok()?;
    if !response.status().is_success() {
        return None;
    }
    let parsed: YahooResponse = response.json().await.ok()?;
    let result = parsed.chart?.result?.into_iter().next()?;
    let timestamps = result.timestamp.unwrap_or_default();
    let quote = result
        .indicators
        .and_then(|i| i.quote)
        .and_then(|mut q| q.pop())
        .unwrap_or_default();

    let mut candles = Vec::new();
    for (idx, ts) in timestamps.iter().enumerate() {
        let open = quote.open_value(idx)?;
        let high = quote.high_value(idx)?;
        let low = quote.low_value(idx)?;
        let close = quote.close_value(idx)?;
        let vol = quote.volume_value(idx).unwrap_or(0) as f64;
        candles.push(Candle {
            ts: ts * 1_000,
            timeframe: tf,
            open: round_two(open),
            high: round_two(high),
            low: round_two(low),
            close: round_two(close),
            volume: vol,
        });
    }
    Some(candles)
}

async fn fetch_remote_history(
    symbol: &str,
    tf: TimeFrame,
    limit: usize,
    provider: MarketProvider,
) -> Option<Vec<Candle>> {
    let mut candles = match provider {
        MarketProvider::Yahoo => fetch_yahoo_history(symbol, tf, 30).await,
        MarketProvider::Coinbase => fetch_coinbase_history(symbol, tf, limit).await,
    }?;

    candles.sort_by_key(|c| c.ts);
    if candles.len() > limit {
        candles = candles.split_off(candles.len().saturating_sub(limit));
    }
    Some(candles)
}

#[derive(Debug, serde::Deserialize)]
struct CoinbaseTicker {
    price: Option<String>,
    size: Option<String>,
    volume: Option<String>,
    time: Option<String>,
}

async fn coinbase_ticker_tick(symbol: &str) -> Result<Option<Tick>, reqwest::Error> {
    let product = normalize_coinbase_symbol(symbol);
    let url = format!(
        "{}/products/{product}/ticker",
        env::var(COINBASE_API_URL_ENV)
            .unwrap_or_else(|_| "https://api.exchange.coinbase.com".to_string())
    );
    let resp = reqwest::Client::new().get(&url).send().await?;
    if !resp.status().is_success() {
        return Ok(None);
    }
    let body: CoinbaseTicker = resp.json().await?;
    let price = body.price.as_deref().and_then(|p| p.parse::<f64>().ok());
    let volume = body
        .size
        .as_deref()
        .or(body.volume.as_deref())
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0);
    let ts = if let Some(t) = body.time {
        DateTime::parse_from_rfc3339(&t)
            .map(|dt| dt.timestamp_millis())
            .unwrap_or_else(|_| Utc::now().timestamp_millis())
    } else {
        Utc::now().timestamp_millis()
    };

    Ok(price.map(|p| Tick {
        ts,
        price: p,
        volume,
    }))
}

#[tokio::main]
async fn main() {
    let conf = get_configuration(None).await.expect("load leptos config");
    let leptos_options = conf.leptos_options;

    let base_tf = TimeFrame::Minutes(1);
    let store = SeriesStore::new(base_tf);

    let feed_state = FeedState {
        store: Arc::new(Mutex::new(store)),
        base_tf,
    };

    let server_state = ServerState {
        feed: feed_state.clone(),
        leptos_options: leptos_options.clone(),
        layout_store: Arc::new(Mutex::new(None)),
    };

    let live_state = feed_state.clone();
    tokio::spawn(async move {
        // Keep a lightweight live ticker loop running so WebSocket
        // clients get fresh prints (Coinbase REST ticker -> ticks).
        run_live_feed(live_state).await;
    });

    let leptos_routes = generate_route_list(ui::App);

    let app = Router::new()
        .route("/api/history", get(history_handler))
        .route("/api/market-data", get(market_data_handler))
        .route("/api/search", get(search_handler))
        .route("/api/ws", get(ws_handler))
        .route("/api/layout", get(load_layout).post(save_layout))
        .leptos_routes(&server_state, leptos_routes, ui::App)
        .fallback_service(
            get_service(ServeDir::new(leptos_options.site_root.clone()))
                .handle_error(|_| async { StatusCode::INTERNAL_SERVER_ERROR }),
        )
        .with_state(server_state.clone());

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080")
        .await
        .expect("bind 0.0.0.0:8080");
    println!(
        "Backend listening on http://0.0.0.0:8080 (UI + API); site_addr: {}",
        leptos_options.site_addr
    );
    axum::serve(listener, app).await.expect("server failed");
}

async fn run_live_feed(state: FeedState) {
    // Currently driven by Coinbase REST ticker for simplicity (no extra WS deps).
    // Falls back to a very light simulator if Coinbase is unreachable.
    let symbol = env::var(DEFAULT_SYMBOL_ENV).unwrap_or_else(|_| "BTCUSD".to_string());
    let provider = MarketProvider::from_str(env::var(DEFAULT_PROVIDER_ENV).ok().as_deref());
    let mut fallback_rng = StdRng::from_entropy();
    let mut fallback_price: f64 = 100.0;

    let mut timer = interval(Duration::from_secs(2));

    loop {
        timer.tick().await;

        let tick = match provider {
            MarketProvider::Coinbase => coinbase_ticker_tick(&symbol).await.unwrap_or(None),
            MarketProvider::Yahoo => None,
        };

        if let Some(tick) = tick {
            let _ = state.store.lock().unwrap().on_tick(tick);
            continue;
        }

        // Lightweight fallback simulator so charts still move.
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;
        let delta: f64 = fallback_rng.gen_range(-0.5..0.5);
        fallback_price = (fallback_price + delta).max(0.1_f64);
        let tick = Tick {
            ts: now_ms,
            price: fallback_price,
            volume: fallback_rng.gen_range(0.1..5.0),
        };
        let _ = state.store.lock().unwrap().on_tick(tick);
    }
}

async fn history_handler(
    State(state): State<ServerState>,
    Query(params): Query<HistoryParams>,
) -> impl IntoResponse {
    let tf = TimeFrame::from_str(&params.tf).unwrap_or(TimeFrame::Minutes(1));
    let limit = params.limit.unwrap_or(1_000);
    let symbol = params
        .symbol
        .clone()
        .unwrap_or_else(|| env::var(DEFAULT_SYMBOL_ENV).unwrap_or_else(|_| "BTCUSD".to_string()));

    let env_provider = env::var(DEFAULT_PROVIDER_ENV).ok();
    let provider =
        MarketProvider::from_str(params.provider.as_deref().or(env_provider.as_deref()));

    // Try remote history first, fallback to in-memory simulator store.
    let mut candles =
        fetch_remote_history(&symbol, tf, limit, provider).await.unwrap_or_default();

    // Cache into the in-memory store so subsequent requests (or derived TFs) are quick.
    if !candles.is_empty() && tf == state.feed.base_tf {
        let mut store = state.feed.store.lock().unwrap();
        store.ensure_timeframe(tf);
        store.on_base_history_batch(candles.clone(), false);
    }

    if candles.is_empty() {
        candles = {
            let mut store = state.feed.store.lock().unwrap();
            store.ensure_timeframe(tf);
            let series = store.series(tf);
            if series.is_empty() {
                Vec::new()
            } else if let (Some(from), Some(to)) = (params.from, params.to) {
                series
                    .range(from, to)
                    .iter()
                    .rev()
                    .take(limit)
                    .cloned()
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect()
            } else {
                series
                    .as_slice()
                    .iter()
                    .rev()
                    .take(limit)
                    .cloned()
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect()
            }
        };
    }

    Json(candles)
}

/// Dispatches market data requests to Yahoo or Coinbase.
/// Provider precedence: `provider` query param, then `MARKET_DATA_PROVIDER` env, default Yahoo.
async fn market_data_handler(Query(params): Query<MarketParams>) -> MarketResponse {
    let symbol = params.symbol.clone();
    let days = params.days.unwrap_or(59);
    let interval = params.interval.clone().unwrap_or_else(|| "5m".to_string());

    let env_provider = env::var(DEFAULT_PROVIDER_ENV).ok();
    let provider = MarketProvider::from_str(params.provider.as_deref().or(env_provider.as_deref()));

    match provider {
        MarketProvider::Yahoo => yahoo_market_data(symbol, days, interval).await,
        MarketProvider::Coinbase => coinbase_market_data(symbol, days, interval).await,
    }
}

/// Lightweight instrument search using Yahoo Finance autocomplete.
async fn search_handler(Query(params): Query<SearchParams>) -> impl IntoResponse {
    let q = params.q.unwrap_or_default().trim().to_string();
    if q.len() < 2 {
        return Json(Vec::<SearchHit>::new());
    }
    let limit = params.limit.unwrap_or(8).max(1).min(20);

    let url = format!(
        "https://query2.finance.yahoo.com/v1/finance/search?q={q}&quotesCount={limit}&newsCount=0"
    );
    let client = reqwest::Client::new();
    let resp = match client.get(&url).send().await {
        Ok(resp) if resp.status().is_success() => resp,
        _ => return Json(Vec::<SearchHit>::new()),
    };
    let parsed: YahooSearchResponse = match resp.json().await {
        Ok(json) => json,
        Err(_) => return Json(Vec::<SearchHit>::new()),
    };

    let hits = parsed
        .quotes
        .unwrap_or_default()
        .into_iter()
        .filter_map(|q| {
            let symbol = q.symbol?;
            let name = q.shortname.or(q.longname);
            Some(SearchHit {
                symbol,
                name,
                exchange: q.exchdisp,
                quote_type: q.quote_type,
                score: q.score,
            })
        })
        .collect::<Vec<_>>();

    Json(hits)
}

async fn yahoo_market_data(
    symbol: Option<String>,
    days: u64,
    interval: String,
) -> MarketResponse {
    let symbol = symbol.unwrap_or_else(|| "ES=F".to_string());

    let end_date = Utc::now().timestamp();
    let start_date = end_date - (days as i64) * 24 * 60 * 60;

    let url = format!(
        "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={start_date}&period2={end_date}&interval={interval}&includePrePost=true"
    );

    let client = reqwest::Client::new();
    let response = client
        .get(&url)
        .header(
            reqwest::header::USER_AGENT,
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        )
        .send()
        .await;

    let response = match response {
        Ok(resp) if resp.status().is_success() => resp,
        _ => {
            return Json(json!({
                "error": "Failed to fetch market data",
                "fallback": true
            }))
        }
    };

    let parsed: YahooResponse = match response.json().await {
        Ok(json) => json,
        Err(_) => {
            return Json(json!({
                "error": "Failed to parse market data",
                "fallback": true
            }))
        }
    };

    let Some(result) = parsed
        .chart
        .and_then(|c| c.result)
        .and_then(|mut r| r.pop())
    else {
        return Json(json!({
            "error": "No data available",
            "fallback": true
        }));
    };

    let timestamps = result.timestamp.unwrap_or_default();
    let quote = result
        .indicators
        .and_then(|i| i.quote)
        .and_then(|mut q| q.pop())
        .unwrap_or_default();

    let mut data: Vec<OhlcvBar> = Vec::with_capacity(timestamps.len());

    for (idx, ts) in timestamps.iter().enumerate() {
        let (Some(open), Some(high), Some(low), Some(close)) = (
            quote.open_value(idx),
            quote.high_value(idx),
            quote.low_value(idx),
            quote.close_value(idx),
        ) else {
            continue;
        };

        let Some(dt) = DateTime::<Utc>::from_timestamp(*ts, 0) else {
            continue;
        };

        let date = dt.date_naive().format("%Y-%m-%d").to_string();
        let time = dt.time().format("%H:%M").to_string();

        data.push(OhlcvBar {
            ts: ts * 1_000,
            date,
            time,
            open: round_two(open),
            high: round_two(high),
            low: round_two(low),
            close: round_two(close),
            volume: quote.volume_value(idx).unwrap_or(0),
        });
    }

    Json(json!({
        "data": data,
        "symbol": symbol.trim_end_matches("=F"),
        "lastUpdated": Utc::now().to_rfc3339(),
        "isDelayed": true,
        "bars": data.len(),
        "provider": MarketProvider::Yahoo.as_str(),
    }))
}

async fn coinbase_market_data(
    symbol: Option<String>,
    days: u64,
    interval: String,
) -> MarketResponse {
    let product = symbol.unwrap_or_else(|| "BTC-USD".to_string());
    let granularity = match coinbase_granularity(&interval) {
        Some(g) => g,
        None => {
            return Json(json!({
                "error": "Unsupported interval for Coinbase",
                "fallback": true
            }))
        }
    };

    let end_dt = Utc::now();
    let start_dt = end_dt - chrono::Duration::seconds((days as i64) * 24 * 60 * 60);

    let url = format!(
        "{}/products/{}/candles?granularity={}&start={}&end={}",
        env::var(COINBASE_API_URL_ENV)
            .unwrap_or_else(|_| "https://api.exchange.coinbase.com".to_string()),
        product,
        granularity,
        start_dt.to_rfc3339(),
        end_dt.to_rfc3339()
    );

    let client = reqwest::Client::new();
    let response = client
        .get(&url)
        .header(
            reqwest::header::USER_AGENT,
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        )
        .send()
        .await;

    let response = match response {
        Ok(resp) if resp.status().is_success() => resp,
        _ => {
            return Json(json!({
                "error": "Failed to fetch market data",
                "fallback": true
            }))
        }
    };

    let parsed: Vec<[f64; 6]> = match response.json().await {
        Ok(json) => json,
        Err(_) => {
            return Json(json!({
                "error": "Failed to parse market data",
                "fallback": true
            }))
        }
    };

    let data = map_coinbase_candles(parsed);

    Json(json!({
        "data": data,
        "symbol": product,
        "lastUpdated": Utc::now().to_rfc3339(),
        "isDelayed": true,
        "bars": data.len(),
        "provider": MarketProvider::Coinbase.as_str(),
        "isCrypto": true
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coinbase_granularity_map() {
        assert_eq!(coinbase_granularity("1m"), Some(60));
        assert_eq!(coinbase_granularity("5m"), Some(300));
        assert_eq!(coinbase_granularity("15m"), Some(900));
        assert_eq!(coinbase_granularity("1h"), Some(3_600));
        assert_eq!(coinbase_granularity("bad"), None);
    }

    #[test]
    fn map_coinbase_candles_sorts_and_formats() {
        let candles = vec![
            [1_700_000_300.0, 12.0, 14.0, 13.0, 13.555, 900.4],
            [1_700_000_000.0, 10.0, 12.0, 11.0, 11.499, 1000.0],
        ];

        let bars = map_coinbase_candles(candles);
        assert_eq!(bars.len(), 2);
        assert_eq!(bars[0].ts, 1_700_000_000_000);
        assert_eq!(bars[0].open, 11.0);
        assert_eq!(bars[0].close, 11.5);
        assert_eq!(bars[1].high, 14.0);
        assert_eq!(bars[1].volume, 900);
        assert!(bars[0].time <= bars[1].time);
    }
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<ServerState>,
    Query(params): Query<WsParams>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ws(socket, state, params))
}

#[derive(Debug, serde::Deserialize, Clone)]
struct WsParams {
    symbol: Option<String>,
    tf: Option<String>,
    provider: Option<String>,
}

async fn handle_ws(stream: WebSocket, _state: ServerState, params: WsParams) {
    let (mut sender, mut receiver) = stream.split();

    // Default values if client omits query params.
    let symbol = params
        .symbol
        .clone()
        .unwrap_or_else(|| env::var(DEFAULT_SYMBOL_ENV).unwrap_or_else(|_| "BTCUSD".to_string()));
    let tf = params
        .tf
        .as_deref()
        .and_then(TimeFrame::from_str)
        .unwrap_or(TimeFrame::Minutes(1));
    let env_provider = env::var(DEFAULT_PROVIDER_ENV).ok();
    let provider =
        MarketProvider::from_str(params.provider.as_deref().or(env_provider.as_deref()));

    // Per-connection local store so we can aggregate ticks into candles.
    let mut store = SeriesStore::new(tf);

    let mut ticker_timer = interval(Duration::from_secs(2));

    loop {
        tokio::select! {
            _ = ticker_timer.tick() => {
                let tick = match provider {
                    MarketProvider::Coinbase => coinbase_ticker_tick(&symbol).await.unwrap_or(None),
                    MarketProvider::Yahoo => None, // Yahoo does not provide live ticks here.
                };

                if let Some(tick) = tick {
                    let event_json = if let Some(candle) = store.on_tick(tick) {
                        serde_json::to_string(&DataEvent::LiveCandle(candle))
                    } else {
                        serde_json::to_string(&DataEvent::LiveTick(tick))
                    };
                    if let Ok(text) = event_json {
                        if sender.send(WsMessage::Text(text)).await.is_err() {
                            break;
                        }
                    }
                }
            }
            msg = receiver.next() => {
                match msg {
                    Some(Ok(_)) => {
                        // Ignore client messages in this example.
                    }
                    Some(Err(_)) | None => break,
                }
            }
        }
    }
}

async fn load_layout(State(state): State<ServerState>) -> impl IntoResponse {
    let layout = state.layout_store.lock().unwrap().clone();
    Json(layout)
}

async fn save_layout(
    State(state): State<ServerState>,
    Json(payload): Json<AppState>,
) -> impl IntoResponse {
    *state.layout_store.lock().unwrap() = Some(payload);
    StatusCode::NO_CONTENT
}
