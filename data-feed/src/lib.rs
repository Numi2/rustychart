use serde::{Deserialize, Serialize};
use std::pin::Pin;

use futures_core::Stream;
use ts_core::{Candle, SeriesStore, Tick, TimeFrame, Timestamp};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataEvent {
    /// Historical candles (HTTP backfill).
    HistoryBatch {
        timeframe: TimeFrame,
        candles: Vec<Candle>,
        prepend: bool,
    },
    /// Raw live tick (WebSocket).
    LiveTick(Tick),
    /// Pre-aggregated live candle (optional).
    LiveCandle(Candle),
    /// Explicit data gap (optional – gaps are typically implicit).
    Gap { from: Timestamp, to: Timestamp },
    /// Reset symbol state.
    Reset,
}

/// Consumer interface for feed events.
pub trait DataSink {
    fn on_event(&mut self, event: DataEvent);
}

/// A store + aggregator for a single symbol, consuming `DataEvent`s into a `SeriesStore`.
pub struct FeedStore {
    symbol: String,
    base_timeframe: TimeFrame,
    store: SeriesStore,
}

impl FeedStore {
    pub fn new(symbol: impl Into<String>, base_timeframe: TimeFrame) -> Self {
        let store = SeriesStore::new(base_timeframe);
        Self {
            symbol: symbol.into(),
            base_timeframe,
            store,
        }
    }

    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    pub fn base_timeframe(&self) -> TimeFrame {
        self.base_timeframe
    }

    pub fn store(&self) -> &SeriesStore {
        &self.store
    }

    pub fn store_mut(&mut self) -> &mut SeriesStore {
        &mut self.store
    }
}

impl DataSink for FeedStore {
    fn on_event(&mut self, event: DataEvent) {
        match event {
            DataEvent::HistoryBatch {
                timeframe,
                mut candles,
                prepend,
            } => {
                assert_eq!(
                    timeframe.duration_ms(),
                    self.base_timeframe.duration_ms(),
                    "history batch timeframe must match base timeframe"
                );
                let batch = std::mem::take(&mut candles);
                self.store.on_base_history_batch(batch, prepend);
            }
            DataEvent::LiveTick(tick) => {
                let _ = self.store.on_tick(tick);
            }
            DataEvent::LiveCandle(c) => {
                // Incremental base candle update (already completed candle).
                self.store.on_base_candle(c);
            }
            DataEvent::Gap { .. } => {
                // Gaps are implicit via missing buckets.
            }
            DataEvent::Reset => {
                self.store = SeriesStore::new(self.base_timeframe);
            }
        }
    }
}

/// Abstract data source: concrete implementations live in platform-specific crates.
pub type DataStream<E> = Pin<Box<dyn Stream<Item = Result<DataEvent, E>> + 'static>>;

pub trait DataSource {
    type Error;

    /// Subscribe to a symbol at a base timeframe, optionally starting from `from` timestamp.
    fn subscribe(
        &self,
        symbol: &str,
        base_timeframe: TimeFrame,
        from: Option<Timestamp>,
    ) -> DataStream<Self::Error>;
}
