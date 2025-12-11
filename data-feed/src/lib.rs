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

/// Alpha Vantage-backed data source that performs historical fetches and optional polling.
#[cfg(feature = "alpha")]
pub mod alpha {
    use alpha_vantage_client::{
        timeframe_to_alpha_interval, AlphaVantageClient, AlphaVantageError, OutputSize,
    };
    use async_stream::try_stream;
    use tokio::time::sleep;

    use super::{DataEvent, DataSource, DataStream};
    use std::time::Duration;
    use ts_core::{Candle, TimeFrame, Timestamp};

    #[derive(Clone)]
    pub struct AlphaVantageSource {
        client: AlphaVantageClient,
        poll_interval: Duration,
        outputsize: OutputSize,
        poll_after_first: bool,
    }

    impl AlphaVantageSource {
        pub fn new(client: AlphaVantageClient) -> Self {
            Self {
                client,
                poll_interval: Duration::from_secs(60),
                outputsize: OutputSize::Compact,
                poll_after_first: true,
            }
        }

        pub fn with_poll_interval(mut self, interval: Duration) -> Self {
            self.poll_interval = interval;
            self
        }

        pub fn with_outputsize(mut self, outputsize: OutputSize) -> Self {
            self.outputsize = outputsize;
            self
        }

        pub fn disable_polling(mut self) -> Self {
            self.poll_after_first = false;
            self
        }
    }

    impl DataSource for AlphaVantageSource {
        type Error = AlphaVantageError;

        fn subscribe(
            &self,
            symbol: &str,
            base_timeframe: TimeFrame,
            from: Option<Timestamp>,
        ) -> DataStream<Self::Error> {
            let client = self.client.clone();
            let symbol = symbol.to_string();
            let poll_interval = self.poll_interval;
            let outputsize = self.outputsize;
            let poll_after_first = self.poll_after_first;

            Box::pin(try_stream! {
                let mut last_seen = from.unwrap_or(i64::MIN);
                let mut first_cycle = true;

                loop {
                    let candles: Vec<Candle> = match base_timeframe {
                        TimeFrame::Days(1) => client
                            .get_time_series_daily_adjusted(&symbol, outputsize)
                            .await?,
                        TimeFrame::Minutes(_) | TimeFrame::Hours(1) => {
                            let interval = timeframe_to_alpha_interval(base_timeframe)
                                .ok_or(AlphaVantageError::UnsupportedTimeFrame(base_timeframe))?;
                            client
                                .get_time_series_intraday(&symbol, interval, outputsize)
                                .await?
                        }
                        _ => {
                            Err(AlphaVantageError::UnsupportedTimeFrame(base_timeframe))?
                        }
                    };

                    let mut filtered: Vec<Candle> = candles
                        .into_iter()
                        .filter(|c| c.ts > last_seen)
                        .collect();
                    filtered.sort_by_key(|c| c.ts);

                    if let Some(max_ts) = filtered.last().map(|c| c.ts) {
                        last_seen = max_ts;
                    }

                    if !filtered.is_empty() {
                        yield DataEvent::HistoryBatch {
                            timeframe: base_timeframe,
                            candles: filtered,
                            prepend: false,
                        };
                    }

                    if !poll_after_first && !first_cycle {
                        break;
                    }
                    first_cycle = false;
                    sleep(poll_interval).await;
                }
            })
        }
    }
}
