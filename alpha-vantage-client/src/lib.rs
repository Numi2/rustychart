use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{NaiveDate, NaiveDateTime};
use reqwest::Url;
use serde::Deserialize;
use serde_json::{Map, Value};
use thiserror::Error;
use tokio::sync::Mutex;
use ts_core::{Candle, TimeFrame, Timestamp};

const DEFAULT_BASE_URL: &str = "https://www.alphavantage.co";
const CACHE_TTL_COMPACT: Duration = Duration::from_secs(60 * 3);
const CACHE_TTL_FULL: Duration = Duration::from_secs(60 * 45);
const CACHE_TTL_SEARCH: Duration = Duration::from_secs(60 * 60);

#[derive(Debug, Clone)]
pub struct AlphaVantageConfig {
    pub api_key: String,
    pub base_url: String,
    pub max_requests_per_minute: u32,
}

impl AlphaVantageConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            max_requests_per_minute: 5,
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn with_limit(mut self, max_requests_per_minute: u32) -> Self {
        self.max_requests_per_minute = max_requests_per_minute.max(1);
        self
    }
}

impl Default for AlphaVantageConfig {
    fn default() -> Self {
        let api_key = std::env::var("ALPHAVANTAGE_API_KEY").unwrap_or_default();
        Self {
            api_key,
            base_url: DEFAULT_BASE_URL.to_string(),
            max_requests_per_minute: 5,
        }
    }
}

#[derive(Debug, Error)]
pub enum AlphaVantageError {
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("missing Alpha Vantage API key")]
    MissingApiKey,
    #[error("rate limited by local guard or remote API")]
    RateLimited,
    #[error("alpha vantage api error: {0}")]
    ApiError(String),
    #[error("parse error: {0}")]
    ParseError(String),
    #[error("unsupported interval: {0}")]
    UnsupportedInterval(String),
    #[error("unsupported timeframe: {0:?}")]
    UnsupportedTimeFrame(TimeFrame),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputSize {
    Compact,
    Full,
}

impl OutputSize {
    fn as_str(&self) -> &'static str {
        match self {
            OutputSize::Compact => "compact",
            OutputSize::Full => "full",
        }
    }

    fn ttl(&self) -> Duration {
        match self {
            OutputSize::Compact => CACHE_TTL_COMPACT,
            OutputSize::Full => CACHE_TTL_FULL,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct SymbolMatch {
    #[serde(rename = "1. symbol")]
    pub symbol: String,
    #[serde(rename = "2. name")]
    pub name: Option<String>,
    #[serde(rename = "3. type")]
    pub instrument_type: Option<String>,
    #[serde(rename = "4. region")]
    pub region: Option<String>,
    #[serde(rename = "8. currency")]
    pub currency: Option<String>,
    #[serde(rename = "9. matchScore")]
    pub match_score: Option<String>,
}

#[derive(Clone)]
pub struct AlphaVantageClient {
    config: AlphaVantageConfig,
    http: reqwest::Client,
    rate_limiter: Arc<Mutex<RateLimiter>>,
    cache: Arc<Mutex<AlphaCache>>,
}

impl AlphaVantageClient {
    pub fn new(config: AlphaVantageConfig) -> Result<Self, AlphaVantageError> {
        if config.api_key.trim().is_empty() {
            return Err(AlphaVantageError::MissingApiKey);
        }
        let http = reqwest::Client::builder()
            .user_agent("rustychart-alpha-client/0.1")
            .build()?;
        Ok(Self {
            rate_limiter: Arc::new(Mutex::new(RateLimiter::new(config.max_requests_per_minute))),
            cache: Arc::new(Mutex::new(AlphaCache::new())),
            config,
            http,
        })
    }

    pub fn from_env() -> Result<Self, AlphaVantageError> {
        let cfg = AlphaVantageConfig::default();
        Self::new(cfg)
    }

    pub async fn get_time_series_intraday(
        &self,
        symbol: &str,
        interval: &str,
        outputsize: OutputSize,
    ) -> Result<Vec<Candle>, AlphaVantageError> {
        let timeframe = alpha_interval_to_timeframe(interval)
            .ok_or_else(|| AlphaVantageError::UnsupportedInterval(interval.to_string()))?;
        let cache_key = format!("intraday:{symbol}:{interval}:{}", outputsize.as_str());
        if let Some(hit) = self
            .cache
            .lock()
            .await
            .get_series(&cache_key, outputsize.ttl())
        {
            return Ok(hit);
        }

        let query = [
            ("function", "TIME_SERIES_INTRADAY"),
            ("symbol", symbol),
            ("interval", interval),
            ("datatype", "json"),
            ("outputsize", outputsize.as_str()),
        ];
        let json = self.fetch(&query).await?;
        let series_key = find_series_key(&json, "Time Series").ok_or_else(|| {
            AlphaVantageError::ParseError("missing intraday time series in response".to_string())
        })?;
        let candles = extract_candles(&json, &series_key, timeframe)?;
        self.cache
            .lock()
            .await
            .insert_series(cache_key, candles.clone(), outputsize.ttl());
        Ok(candles)
    }

    pub async fn get_time_series_daily(
        &self,
        symbol: &str,
        outputsize: OutputSize,
    ) -> Result<Vec<Candle>, AlphaVantageError> {
        self.fetch_daily_impl(symbol, outputsize, false).await
    }

    pub async fn get_time_series_daily_adjusted(
        &self,
        symbol: &str,
        outputsize: OutputSize,
    ) -> Result<Vec<Candle>, AlphaVantageError> {
        self.fetch_daily_impl(symbol, outputsize, true).await
    }

    async fn fetch_daily_impl(
        &self,
        symbol: &str,
        outputsize: OutputSize,
        adjusted: bool,
    ) -> Result<Vec<Candle>, AlphaVantageError> {
        let cache_key = format!(
            "daily:{}:{}:{}",
            if adjusted { "adj" } else { "raw" },
            symbol,
            outputsize.as_str()
        );
        if let Some(hit) = self
            .cache
            .lock()
            .await
            .get_series(&cache_key, outputsize.ttl())
        {
            return Ok(hit);
        }

        let function = if adjusted {
            "TIME_SERIES_DAILY_ADJUSTED"
        } else {
            "TIME_SERIES_DAILY"
        };
        let query = [
            ("function", function),
            ("symbol", symbol),
            ("datatype", "json"),
            ("outputsize", outputsize.as_str()),
        ];
        let json = self.fetch(&query).await?;
        let series_key = find_series_key(&json, "Time Series").ok_or_else(|| {
            AlphaVantageError::ParseError("missing daily time series in response".to_string())
        })?;
        let candles = extract_candles(&json, &series_key, TimeFrame::Days(1))?;
        self.cache
            .lock()
            .await
            .insert_series(cache_key, candles.clone(), outputsize.ttl());
        Ok(candles)
    }

    pub async fn symbol_search(
        &self,
        keywords: &str,
    ) -> Result<Vec<SymbolMatch>, AlphaVantageError> {
        let cache_key = format!("search:{keywords}");
        if let Some(hit) = self
            .cache
            .lock()
            .await
            .get_search(&cache_key, CACHE_TTL_SEARCH)
        {
            return Ok(hit);
        }
        let query = [
            ("function", "SYMBOL_SEARCH"),
            ("keywords", keywords),
            ("datatype", "json"),
        ];
        let json = self.fetch(&query).await?;
        let matches = json
            .get("bestMatches")
            .and_then(|v| v.as_array())
            .ok_or_else(|| AlphaVantageError::ParseError("missing bestMatches".to_string()))?
            .iter()
            .filter_map(|v| serde_json::from_value::<SymbolMatch>(v.clone()).ok())
            .collect::<Vec<_>>();
        self.cache
            .lock()
            .await
            .insert_search(cache_key, matches.clone(), CACHE_TTL_SEARCH);
        Ok(matches)
    }

    async fn fetch(&self, params: &[(&str, &str)]) -> Result<Value, AlphaVantageError> {
        self.rate_limiter.lock().await.try_acquire()?;
        let mut url = Url::parse(&self.config.base_url)
            .map_err(|e| AlphaVantageError::ApiError(e.to_string()))?;
        url.set_path("query");
        let mut request = self.http.get(url);
        for (k, v) in params {
            request = request.query(&[(*k, *v)]);
        }
        request = request.query(&[("apikey", self.config.api_key.as_str())]);
        let resp = request.send().await?;
        if resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(AlphaVantageError::RateLimited);
        }
        let json: Value = resp.json().await?;
        if let Some(msg) = json
            .get("Note")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
        {
            return Err(AlphaVantageError::ApiError(msg));
        }
        if let Some(msg) = json
            .get("Error Message")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
        {
            return Err(AlphaVantageError::ApiError(msg));
        }
        if let Some(msg) = json
            .get("Information")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
        {
            return Err(AlphaVantageError::ApiError(msg));
        }
        Ok(json)
    }
}

fn find_series_key(json: &Value, starts_with: &str) -> Option<String> {
    json.as_object()
        .and_then(|obj| obj.keys().find(|k| k.starts_with(starts_with)).cloned())
}

fn extract_candles(
    json: &Value,
    series_key: &str,
    timeframe: TimeFrame,
) -> Result<Vec<Candle>, AlphaVantageError> {
    let series_obj = json
        .get(series_key)
        .and_then(Value::as_object)
        .ok_or_else(|| {
            AlphaVantageError::ParseError("time series missing or invalid".to_string())
        })?;

    let mut candles = Vec::with_capacity(series_obj.len());
    for (ts_str, bar) in series_obj {
        if let Some(c) = parse_bar(ts_str, bar, timeframe)? {
            candles.push(c);
        }
    }
    candles.sort_by_key(|c| c.ts);
    Ok(candles)
}

fn parse_bar(
    ts_str: &str,
    value: &Value,
    timeframe: TimeFrame,
) -> Result<Option<Candle>, AlphaVantageError> {
    let obj = match value.as_object() {
        Some(o) => o,
        None => {
            return Err(AlphaVantageError::ParseError(format!(
                "bar value not object for ts {ts_str}"
            )))
        }
    };
    let ts = parse_timestamp(ts_str)
        .ok_or_else(|| AlphaVantageError::ParseError(format!("invalid timestamp {ts_str}")))?;
    let open = parse_number(obj, &["1. open"])?;
    let high = parse_number(obj, &["2. high"])?;
    let low = parse_number(obj, &["3. low"])?;
    let close = parse_number(obj, &["4. close"])?;
    let volume = parse_number(obj, &["5. volume", "6. volume"])?;
    Ok(Some(Candle {
        ts,
        timeframe,
        open,
        high,
        low,
        close,
        volume,
    }))
}

fn parse_number(obj: &Map<String, Value>, keys: &[&str]) -> Result<f64, AlphaVantageError> {
    for key in keys {
        if let Some(val) = obj.get(*key) {
            if let Some(n) = val.as_str().and_then(|s| s.parse::<f64>().ok()) {
                return Ok(n);
            }
            if let Some(n) = val.as_f64() {
                return Ok(n);
            }
        }
    }
    Err(AlphaVantageError::ParseError(format!(
        "missing numeric field {keys:?}"
    )))
}

fn parse_timestamp(ts_str: &str) -> Option<Timestamp> {
    if let Ok(dt) = NaiveDateTime::parse_from_str(ts_str, "%Y-%m-%d %H:%M:%S") {
        return Some(dt.and_utc().timestamp_millis());
    }
    if let Ok(d) = NaiveDate::parse_from_str(ts_str, "%Y-%m-%d") {
        let dt = d.and_hms_opt(0, 0, 0)?;
        return Some(dt.and_utc().timestamp_millis());
    }
    None
}

pub fn alpha_interval_to_timeframe(interval: &str) -> Option<TimeFrame> {
    match interval {
        "1min" => Some(TimeFrame::Minutes(1)),
        "5min" => Some(TimeFrame::Minutes(5)),
        "15min" => Some(TimeFrame::Minutes(15)),
        "30min" => Some(TimeFrame::Minutes(30)),
        "60min" => Some(TimeFrame::Minutes(60)),
        _ => None,
    }
}

pub fn timeframe_to_alpha_interval(tf: TimeFrame) -> Option<&'static str> {
    match tf {
        TimeFrame::Minutes(1) => Some("1min"),
        TimeFrame::Minutes(5) => Some("5min"),
        TimeFrame::Minutes(15) => Some("15min"),
        TimeFrame::Minutes(30) => Some("30min"),
        TimeFrame::Minutes(60) | TimeFrame::Hours(1) => Some("60min"),
        _ => None,
    }
}

struct RateLimiter {
    capacity: f64,
    tokens: f64,
    refill_per_sec: f64,
    last_refill: Instant,
}

impl RateLimiter {
    fn new(max_per_minute: u32) -> Self {
        let capacity = max_per_minute as f64;
        let refill_per_sec = capacity / 60.0;
        Self {
            capacity,
            tokens: capacity,
            refill_per_sec,
            last_refill: Instant::now(),
        }
    }

    fn try_acquire(&mut self) -> Result<(), AlphaVantageError> {
        self.refill();
        if self.tokens < 1.0 {
            return Err(AlphaVantageError::RateLimited);
        }
        self.tokens -= 1.0;
        Ok(())
    }

    fn refill(&mut self) {
        let elapsed = self.last_refill.elapsed().as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_per_sec).min(self.capacity);
        self.last_refill = Instant::now();
    }
}

#[derive(Default)]
struct AlphaCache {
    series: HashMap<String, CachedSeries>,
    search: HashMap<String, CachedSearch>,
}

impl AlphaCache {
    fn new() -> Self {
        Self::default()
    }

    fn get_series(&mut self, key: &str, ttl: Duration) -> Option<Vec<Candle>> {
        let now = Instant::now();
        if let Some(hit) = self.series.get(key) {
            if now.duration_since(hit.stored_at) <= ttl {
                return Some(hit.candles.clone());
            }
        }
        None
    }

    fn insert_series(&mut self, key: String, candles: Vec<Candle>, _ttl: Duration) {
        self.series.insert(
            key,
            CachedSeries {
                stored_at: Instant::now(),
                candles,
            },
        );
    }

    fn get_search(&mut self, key: &str, ttl: Duration) -> Option<Vec<SymbolMatch>> {
        let now = Instant::now();
        if let Some(hit) = self.search.get(key) {
            if now.duration_since(hit.stored_at) <= ttl {
                return Some(hit.matches.clone());
            }
        }
        None
    }

    fn insert_search(&mut self, key: String, matches: Vec<SymbolMatch>, _ttl: Duration) {
        self.search.insert(
            key,
            CachedSearch {
                stored_at: Instant::now(),
                matches,
            },
        );
    }
}

struct CachedSeries {
    stored_at: Instant,
    candles: Vec<Candle>,
}

struct CachedSearch {
    stored_at: Instant,
    matches: Vec<SymbolMatch>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interval_mapping_round_trip() {
        let pairs = vec![
            ("1min", TimeFrame::Minutes(1)),
            ("5min", TimeFrame::Minutes(5)),
            ("15min", TimeFrame::Minutes(15)),
            ("30min", TimeFrame::Minutes(30)),
            ("60min", TimeFrame::Minutes(60)),
        ];
        for (interval, tf) in pairs {
            assert_eq!(alpha_interval_to_timeframe(interval), Some(tf));
            assert_eq!(timeframe_to_alpha_interval(tf), Some(interval));
        }
        assert!(alpha_interval_to_timeframe("bad").is_none());
        assert!(timeframe_to_alpha_interval(TimeFrame::Days(1)).is_none());
    }

    #[test]
    fn parse_intraday_sample() {
        let json: Value = serde_json::from_str(
            r#"{
                "Time Series (5min)": {
                    "2024-03-01 20:00:00": {
                        "1. open": "10.0",
                        "2. high": "11.0",
                        "3. low": "9.5",
                        "4. close": "10.5",
                        "5. volume": "1500"
                    },
                    "2024-03-01 20:05:00": {
                        "1. open": "10.5",
                        "2. high": "10.7",
                        "3. low": "10.3",
                        "4. close": "10.6",
                        "5. volume": "800"
                    }
                }
            }"#,
        )
        .unwrap();
        let candles = extract_candles(&json, "Time Series (5min)", TimeFrame::Minutes(5)).unwrap();
        assert_eq!(candles.len(), 2);
        assert!(candles[0].ts < candles[1].ts);
        assert_eq!(candles[0].open, 10.0);
        assert_eq!(candles[1].volume, 800.0);
    }

    #[test]
    fn parse_daily_sample() {
        let json: Value = serde_json::from_str(
            r#"{
                "Time Series (Daily)": {
                    "2024-02-29": {
                        "1. open": "100.0",
                        "2. high": "110.0",
                        "3. low": "90.0",
                        "4. close": "105.0",
                        "5. volume": "12345"
                    }
                }
            }"#,
        )
        .unwrap();
        let candles = extract_candles(&json, "Time Series (Daily)", TimeFrame::Days(1)).unwrap();
        assert_eq!(candles.len(), 1);
        assert_eq!(candles[0].close, 105.0);
        assert_eq!(candles[0].timeframe, TimeFrame::Days(1));
    }

    #[tokio::test]
    async fn rate_limiter_blocks_when_exhausted() {
        let mut limiter = RateLimiter::new(1);
        limiter.try_acquire().unwrap();
        let err = limiter.try_acquire().unwrap_err();
        assert!(matches!(err, AlphaVantageError::RateLimited));
    }

    #[tokio::test]
    async fn integration_fetch_daily_if_key_present() -> Result<(), Box<dyn std::error::Error>> {
        if std::env::var("ALPHAVANTAGE_API_KEY").is_err() {
            return Ok(()); // skip when no key configured
        }
        let client = AlphaVantageClient::from_env()?;
        let candles = client
            .get_time_series_daily("IBM", OutputSize::Compact)
            .await?;
        assert!(!candles.is_empty());
        Ok(())
    }
}
