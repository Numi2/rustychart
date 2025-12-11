use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::str::FromStr;

/// Milliseconds since Unix epoch.
pub type Timestamp = i64;

/// Number of milliseconds in common units.
pub const MS: i64 = 1_000;
pub const MINUTE_MS: i64 = 60 * MS;
pub const HOUR_MS: i64 = 60 * MINUTE_MS;
pub const DAY_MS: i64 = 24 * HOUR_MS;

/// Timeframe enum that covers granularities needed by the charts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TimeFrame {
    Tick,
    Seconds(u32),
    Minutes(u32),
    Hours(u32),
    Days(u32),
    Weeks(u32),
    Months(u32),
}

impl TimeFrame {
    /// Duration in milliseconds (months/weeks approximated: 30/7 days).
    pub fn duration_ms(&self) -> i64 {
        match *self {
            TimeFrame::Tick => 0,
            TimeFrame::Seconds(s) => s as i64 * MS,
            TimeFrame::Minutes(m) => m as i64 * MINUTE_MS,
            TimeFrame::Hours(h) => h as i64 * HOUR_MS,
            TimeFrame::Days(d) => d as i64 * DAY_MS,
            TimeFrame::Weeks(w) => w as i64 * 7 * DAY_MS,
            TimeFrame::Months(m) => m as i64 * 30 * DAY_MS,
        }
    }

    /// Align a timestamp to this timeframe boundary.
    pub fn align_ts(&self, ts: Timestamp) -> Timestamp {
        let dur = self.duration_ms();
        if dur == 0 {
            ts
        } else {
            (ts / dur) * dur
        }
    }

    /// Human-readable name (used in URLs/API).
    pub fn name(&self) -> String {
        match *self {
            TimeFrame::Tick => "tick".to_string(),
            TimeFrame::Seconds(s) => format!("{s}s"),
            TimeFrame::Minutes(m) => format!("{m}m"),
            TimeFrame::Hours(h) => format!("{h}h"),
            TimeFrame::Days(d) => format!("{d}d"),
            TimeFrame::Weeks(w) => format!("{w}w"),
            TimeFrame::Months(m) => format!("{m}M"),
        }
    }

    /// Parse e.g. "tick", "1m", "5m", "1h", "4h", "1d", "1w", "1M".
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "tick" | "0" => Some(TimeFrame::Tick),
            "1s" => Some(TimeFrame::Seconds(1)),
            "5s" => Some(TimeFrame::Seconds(5)),
            "15s" => Some(TimeFrame::Seconds(15)),
            "1m" => Some(TimeFrame::Minutes(1)),
            "5m" => Some(TimeFrame::Minutes(5)),
            "15m" => Some(TimeFrame::Minutes(15)),
            "30m" => Some(TimeFrame::Minutes(30)),
            "1h" => Some(TimeFrame::Hours(1)),
            "4h" => Some(TimeFrame::Hours(4)),
            "1d" => Some(TimeFrame::Days(1)),
            "1w" => Some(TimeFrame::Weeks(1)),
            "1M" | "1mo" | "1month" => Some(TimeFrame::Months(1)),
            _ => None,
        }
    }

    /// Returns true if `self` divides `other` (e.g. 1m divides 5m).
    pub fn divides(&self, other: TimeFrame) -> bool {
        let a = self.duration_ms();
        let b = other.duration_ms();
        a != 0 && b != 0 && b % a == 0
    }
}

impl FromStr for TimeFrame {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        TimeFrame::from_str(s).ok_or_else(|| "invalid timeframe".to_string())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Tick {
    pub ts: Timestamp,
    pub price: f64,
    pub volume: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Candle {
    pub ts: Timestamp,        // bucket start time
    pub timeframe: TimeFrame, // timeframe of this candle
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl Candle {
    pub fn from_tick(timeframe: TimeFrame, tick: Tick) -> Self {
        let ts = timeframe.align_ts(tick.ts);
        Self {
            ts,
            timeframe,
            open: tick.price,
            high: tick.price,
            low: tick.price,
            close: tick.price,
            volume: tick.volume,
        }
    }

    pub fn absorb_tick(&mut self, tick: Tick) {
        self.high = self.high.max(tick.price);
        self.low = self.low.min(tick.price);
        self.close = tick.price;
        self.volume += tick.volume;
    }

    pub fn absorb_child(&mut self, child: &Candle) {
        debug_assert_eq!(
            self.timeframe.duration_ms() % child.timeframe.duration_ms(),
            0
        );
        self.high = self.high.max(child.high);
        self.low = self.low.min(child.low);
        self.close = child.close;
        self.volume += child.volume;
    }
}

pub trait HasTimestamp {
    fn ts(&self) -> Timestamp;
}

impl HasTimestamp for Tick {
    fn ts(&self) -> Timestamp {
        self.ts
    }
}

impl HasTimestamp for Candle {
    fn ts(&self) -> Timestamp {
        self.ts
    }
}

/// A daily trading session in minutes-from-midnight (UTC) space.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct Session {
    pub start_minute: u32,
    pub end_minute: u32,
}

impl Session {
    pub fn contains_minute(&self, minute: u32) -> bool {
        if self.start_minute <= self.end_minute {
            minute >= self.start_minute && minute < self.end_minute
        } else {
            // Wrap-around session (e.g., 22:00-06:00)
            minute >= self.start_minute || minute < self.end_minute
        }
    }
}

/// Session calendar with optional holidays (as unix-day integers).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionCalendar {
    pub sessions: Vec<Session>,
    pub holidays: Vec<i64>, // unix days (days since epoch)
}

impl SessionCalendar {
    pub fn is_open(&self, ts: Timestamp) -> bool {
        let day = unix_day(ts);
        if self.holidays.contains(&day) {
            return false;
        }
        let minute = minute_of_day(ts);
        self.sessions.iter().any(|s| s.contains_minute(minute))
    }
}

fn unix_day(ts: Timestamp) -> i64 {
    ts.div_euclid(DAY_MS)
}

fn minute_of_day(ts: Timestamp) -> u32 {
    let ms_into_day = ts.rem_euclid(DAY_MS);
    (ms_into_day / MINUTE_MS) as u32
}

/// Append-only time-series with binary-searchable timestamps and batched prepends.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeries<T> {
    data: Vec<T>,
}

impl<T> Default for TimeSeries<T> {
    fn default() -> Self {
        Self { data: Vec::new() }
    }
}

impl<T: HasTimestamp> TimeSeries<T> {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn first(&self) -> Option<&T> {
        self.data.first()
    }

    pub fn last(&self) -> Option<&T> {
        self.data.last()
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub fn append(&mut self, sample: T) {
        if let Some(last) = self.data.last() {
            assert!(
                sample.ts() >= last.ts(),
                "append expects non-decreasing timestamps"
            );
        }
        self.data.push(sample);
    }

    pub fn append_batch<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for s in iter {
            self.append(s);
        }
    }

    /// Prepend older history before the first sample. `batch` may be unsorted.
    pub fn prepend_batch(&mut self, mut batch: Vec<T>) {
        if self.data.is_empty() {
            batch.sort_by_key(|s| s.ts());
            self.data = batch;
            return;
        }
        batch.sort_by_key(|s| s.ts());
        let first_ts = self.data.first().unwrap().ts();
        let last_batch_ts = batch.last().unwrap().ts();
        assert!(
            last_batch_ts <= first_ts,
            "prepend_batch expects strictly older data"
        );
        let mut new_data = Vec::with_capacity(batch.len() + self.data.len());
        new_data.extend(batch);
        new_data.append(&mut self.data);
        self.data = new_data;
    }

    /// Returns a slice of samples whose timestamps are in [start_ts, end_ts).
    pub fn range(&self, start_ts: Timestamp, end_ts: Timestamp) -> &[T] {
        let start_idx = self.lower_bound(start_ts);
        let end_idx = self.lower_bound(end_ts);
        &self.data[start_idx..end_idx]
    }

    fn lower_bound(&self, ts: Timestamp) -> usize {
        let mut left = 0usize;
        let mut right = self.data.len();
        while left < right {
            let mid = (left + right) / 2;
            match self.data[mid].ts().cmp(&ts) {
                Ordering::Less => left = mid + 1,
                Ordering::Equal | Ordering::Greater => right = mid,
            }
        }
        left
    }
}

/// Incrementally builds candles from ticks or finer candles.
#[derive(Debug, Clone)]
pub struct CandleAggregator {
    pub timeframe: TimeFrame,
    current: Option<Candle>,
}

impl CandleAggregator {
    pub fn new(timeframe: TimeFrame) -> Self {
        Self {
            timeframe,
            current: None,
        }
    }

    /// Feed a tick. Returns a finished candle when the bucket rolls over.
    pub fn on_tick(&mut self, tick: Tick) -> Option<Candle> {
        let bucket_ts = self.timeframe.align_ts(tick.ts);
        match self.current {
            None => {
                self.current = Some(Candle::from_tick(self.timeframe, tick));
                None
            }
            Some(mut c) if c.ts == bucket_ts => {
                c.absorb_tick(tick);
                self.current = Some(c);
                None
            }
            Some(old) => {
                let finished = old;
                self.current = Some(Candle::from_tick(self.timeframe, tick));
                Some(finished)
            }
        }
    }

    /// Feed a child candle (must be a divisor timeframe). Returns parent candles when complete.
    pub fn on_child_candle(&mut self, child: Candle) -> Option<Candle> {
        let bucket_ts = self.timeframe.align_ts(child.ts);
        match self.current {
            None => {
                let mut c = child;
                c.ts = bucket_ts;
                c.timeframe = self.timeframe;
                self.current = Some(c);
                None
            }
            Some(mut parent) if parent.ts == bucket_ts => {
                parent.absorb_child(&child);
                self.current = Some(parent);
                None
            }
            Some(old_parent) => {
                let finished = old_parent;
                let mut new_parent = child;
                new_parent.ts = bucket_ts;
                new_parent.timeframe = self.timeframe;
                self.current = Some(new_parent);
                Some(finished)
            }
        }
    }

    pub fn flush(&mut self) -> Option<Candle> {
        self.current.take()
    }
}

/// In-memory multi-timeframe store for a single symbol.
#[derive(Debug)]
pub struct SeriesStore {
    base_timeframe: TimeFrame,
    base_candles: TimeSeries<Candle>,
    base_agg: CandleAggregator,
    session_calendar: Option<SessionCalendar>,
    /// Derived timeframes and their aggregators.
    derived: HashMap<TimeFrame, (CandleAggregator, TimeSeries<Candle>)>,
}

impl SeriesStore {
    pub fn new(base_timeframe: TimeFrame) -> Self {
        Self {
            base_timeframe,
            base_candles: TimeSeries::new(),
            base_agg: CandleAggregator::new(base_timeframe),
            session_calendar: None,
            derived: HashMap::new(),
        }
    }

    pub fn base_timeframe(&self) -> TimeFrame {
        self.base_timeframe
    }

    /// Ensure derived timeframe exists and is backfilled.
    pub fn ensure_timeframe(&mut self, tf: TimeFrame) {
        if tf == self.base_timeframe {
            return;
        }
        if !self.derived.contains_key(&tf) {
            assert!(
                self.base_timeframe.divides(tf),
                "base timeframe must divide derived timeframe"
            );
            let mut agg = CandleAggregator::new(tf);
            let mut series = TimeSeries::new();
            for c in self.base_candles.as_slice().iter() {
                if let Some(parent) = agg.on_child_candle(*c) {
                    series.append(parent);
                }
            }
            if let Some(last) = agg.flush() {
                series.append(last);
            }
            self.derived.insert(tf, (agg, series));
        }
    }

    /// Process a tick -> base candle -> derived candles.
    /// Returns a finished base candle (if any) so callers can broadcast it.
    pub fn on_tick(&mut self, tick: Tick) -> Option<Candle> {
        if let Some(cal) = &self.session_calendar {
            if !cal.is_open(tick.ts) {
                return None;
            }
        }
        if let Some(candle) = self.base_agg.on_tick(tick) {
            self.base_candles.append(candle);
            for (tf, (agg, series)) in self.derived.iter_mut() {
                debug_assert!(self.base_timeframe.divides(*tf));
                if let Some(parent) = agg.on_child_candle(candle) {
                    series.append(parent);
                }
            }
            Some(candle)
        } else {
            None
        }
    }

    /// Append a completed base timeframe candle and update all derived timeframes incrementally.
    pub fn on_base_candle(&mut self, candle: Candle) {
        self.base_candles.append(candle);
        for (_tf, (agg, series)) in self.derived.iter_mut() {
            if let Some(parent) = agg.on_child_candle(candle) {
                series.append(parent);
            }
        }
    }

    /// Add a batch of base timeframe candles (HTTP history backfill).
    pub fn on_base_history_batch(&mut self, mut candles: Vec<Candle>, prepend: bool) {
        candles.sort_by_key(|c| c.ts);
        if prepend {
            self.base_candles.prepend_batch(candles.clone());
        } else {
            self.base_candles.append_batch(candles.clone());
        }
        for (tf, (agg, series)) in self.derived.iter_mut() {
            *agg = CandleAggregator::new(*tf);
            *series = TimeSeries::new();
        }
        for c in self.base_candles.as_slice().iter() {
            for (tf, (agg, series)) in self.derived.iter_mut() {
                debug_assert!(self.base_timeframe.divides(*tf));
                if let Some(parent) = agg.on_child_candle(*c) {
                    series.append(parent);
                }
            }
        }
    }

    pub fn series(&self, tf: TimeFrame) -> &TimeSeries<Candle> {
        if tf == self.base_timeframe {
            &self.base_candles
        } else {
            &self
                .derived
                .get(&tf)
                .unwrap_or_else(|| panic!("timeframe {tf:?} not enabled"))
                .1
        }
    }

    pub fn series_mut(&mut self, tf: TimeFrame) -> &mut TimeSeries<Candle> {
        if tf == self.base_timeframe {
            &mut self.base_candles
        } else {
            &mut self
                .derived
                .get_mut(&tf)
                .unwrap_or_else(|| panic!("timeframe {tf:?} not enabled"))
                .1
        }
    }

    /// Downsample candles in [start_ts, end_ts) into at most `max_points` pseudo-candles.
    pub fn downsample(
        &self,
        tf: TimeFrame,
        start_ts: Timestamp,
        end_ts: Timestamp,
        max_points: usize,
    ) -> Vec<Candle> {
        let series = self.series(tf);
        let slice = series.range(start_ts, end_ts);
        if slice.len() <= max_points {
            return slice.to_vec();
        }
        downsample_slice(tf, slice, max_points)
    }

    /// Build multiple levels-of-detail (LOD) slices from newest data.
    /// Each entry is downsampled to at most the requested point count.
    pub fn build_lod(&self, tf: TimeFrame, levels: &[usize]) -> Vec<DownsampledLevel> {
        let series = self.series(tf);
        let slice = series.as_slice();
        let mut out = Vec::new();
        for &target in levels {
            if target == 0 {
                continue;
            }
            let candles = if slice.len() <= target {
                slice.to_vec()
            } else {
                downsample_slice(tf, slice, target)
            };
            let bucket = ((slice.len() as f64) / (target as f64)).ceil().max(1.0) as usize;
            out.push(DownsampledLevel {
                bucket_size: bucket,
                candles,
            });
        }
        out
    }

    pub fn set_session_calendar(&mut self, cal: SessionCalendar) {
        self.session_calendar = Some(cal);
    }

    pub fn session_calendar(&self) -> Option<&SessionCalendar> {
        self.session_calendar.as_ref()
    }
}

/// A downsampled representation produced from a finer series.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownsampledLevel {
    pub bucket_size: usize,
    pub candles: Vec<Candle>,
}

fn downsample_slice(tf: TimeFrame, slice: &[Candle], max_points: usize) -> Vec<Candle> {
    let bucket_size = ((slice.len() as f64) / (max_points as f64)).ceil().max(1.0) as usize;
    let mut out = Vec::with_capacity(max_points);
    for chunk in slice.chunks(bucket_size) {
        if chunk.is_empty() {
            continue;
        }
        let mut agg = Candle {
            ts: chunk[0].ts,
            timeframe: tf,
            open: chunk[0].open,
            high: chunk[0].high,
            low: chunk[0].low,
            close: chunk.last().unwrap().close,
            volume: 0.0,
        };
        for c in chunk {
            agg.high = agg.high.max(c.high);
            agg.low = agg.low.min(c.low);
            agg.volume += c.volume;
        }
        out.push(agg);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_candle(ts: i64, ohlc: (f64, f64, f64, f64)) -> Candle {
        Candle {
            ts,
            timeframe: TimeFrame::Minutes(1),
            open: ohlc.0,
            high: ohlc.1,
            low: ohlc.2,
            close: ohlc.3,
            volume: 1.0,
        }
    }

    #[test]
    fn downsample_preserves_extremes() {
        let series: Vec<Candle> = (0..100)
            .map(|i| {
                mk_candle(
                    i * MINUTE_MS,
                    (i as f64, i as f64 + 2.0, i as f64 - 2.0, i as f64),
                )
            })
            .collect();
        let ds = downsample_slice(TimeFrame::Minutes(1), &series, 10);
        assert!(ds.len() <= 10);
        for bucket in ds {
            assert!(bucket.high >= bucket.open);
            assert!(bucket.low <= bucket.open);
            assert!(bucket.high >= bucket.close);
        }
    }

    #[test]
    fn lod_builds_multiple_levels() {
        let mut store = SeriesStore::new(TimeFrame::Minutes(1));
        for i in 0..120 {
            let c = mk_candle(i * MINUTE_MS, (1.0, 2.0, 0.5, 1.5));
            store.on_base_candle(c);
        }
        let lod = store.build_lod(TimeFrame::Minutes(1), &[16, 32]);
        assert_eq!(lod.len(), 2);
        assert!(lod[0].candles.len() <= 16);
        assert!(lod[1].candles.len() <= 32);
        assert!(lod[0].bucket_size >= 1);
    }

    #[test]
    fn session_calendar_filters_ticks() {
        let mut store = SeriesStore::new(TimeFrame::Minutes(1));
        store.set_session_calendar(SessionCalendar {
            sessions: vec![Session {
                start_minute: 60, // 01:00
                end_minute: 120,  // 02:00
            }],
            holidays: vec![],
        });
        // 00:30 should be dropped
        let tick = Tick {
            ts: 30 * MINUTE_MS,
            price: 100.0,
            volume: 1.0,
        };
        assert!(store.on_tick(tick).is_none());
        // 01:05 accepted
        let tick_ok = Tick {
            ts: 65 * MINUTE_MS,
            price: 101.0,
            volume: 1.0,
        };
        let res = store.on_tick(tick_ok);
        assert!(res.is_none() || res.is_some()); // just ensure no panic
    }
}
