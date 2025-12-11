use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;
use ts_core::{Candle, Timestamp};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FeatureVector {
    pub realized_vol: f64,
    pub atr: f64,
    pub trend_slope: f64,
    pub gap_rate: f64,
    pub range_ratio: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegimeLabel {
    pub start: Timestamp,
    pub end: Timestamp,
    pub regime: usize,
    pub feature: FeatureVector,
}

#[derive(Error, Debug)]
pub enum RegimeError {
    #[error("not enough samples")]
    NotEnoughSamples,
}

/// Extract summary features over a rolling window of candles.
pub fn extract_features(window: &[Candle]) -> Result<FeatureVector, RegimeError> {
    if window.len() < 2 {
        return Err(RegimeError::NotEnoughSamples);
    }

    let mut returns = Vec::with_capacity(window.len() - 1);
    let mut ranges = Vec::with_capacity(window.len());
    let mut gaps = 0usize;

    let mut prev_close = window[0].close;
    for c in window {
        ranges.push((c.high - c.low).max(0.0));
        let ret = ((c.close / prev_close).ln()).clamp(-0.2, 0.2);
        returns.push(ret);
        if (c.open - prev_close).abs() / prev_close.max(1e-6) > 0.003 {
            gaps += 1;
        }
        prev_close = c.close;
    }
    returns.remove(0); // first ret was synthetic

    let mean_ret = returns.iter().sum::<f64>() / returns.len() as f64;
    let var =
        returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>() / returns.len().max(1) as f64;
    let realized_vol = var.sqrt();

    let mut atr_sum = 0.0;
    let mut prev_close = window[0].close;
    for c in window {
        let h_l = c.high - c.low;
        let h_pc = (c.high - prev_close).abs();
        let l_pc = (c.low - prev_close).abs();
        atr_sum += h_l.max(h_pc).max(l_pc);
        prev_close = c.close;
    }
    let atr = atr_sum / window.len().max(1) as f64;

    // simple slope via least squares on index vs close
    let n = window.len() as f64;
    let mean_x = (n - 1.0) / 2.0;
    let mean_y = window.iter().map(|c| c.close).sum::<f64>() / n;
    let mut num = 0.0;
    let mut den = 0.0;
    for (i, c) in window.iter().enumerate() {
        let x = i as f64;
        num += (x - mean_x) * (c.close - mean_y);
        den += (x - mean_x).powi(2);
    }
    let trend_slope = if den.abs() < 1e-9 { 0.0 } else { num / den };

    let gap_rate = gaps as f64 / window.len().max(1) as f64;
    let avg_close = window.iter().map(|c| c.close).sum::<f64>() / window.len() as f64;
    let avg_range = ranges.iter().sum::<f64>() / window.len().max(1) as f64;
    let range_ratio = if avg_close.abs() < 1e-9 {
        0.0
    } else {
        avg_range / avg_close.abs()
    };

    Ok(FeatureVector {
        realized_vol,
        atr,
        trend_slope,
        gap_rate,
        range_ratio,
    })
}

#[derive(Clone, Debug)]
pub struct RegimeClusterer {
    pub k: usize,
    pub centroids: Vec<FeatureVector>,
    pub max_iters: usize,
    seed: u64,
}

impl RegimeClusterer {
    pub fn new(k: usize, seed: u64) -> Self {
        Self {
            k: k.max(1),
            centroids: Vec::new(),
            max_iters: 20,
            seed,
        }
    }

    pub fn fit(&mut self, data: &[FeatureVector]) -> Vec<usize> {
        if data.is_empty() {
            self.centroids.clear();
            return Vec::new();
        }
        let mut rng = StdRng::seed_from_u64(self.seed);
        self.centroids = data
            .choose_multiple(&mut rng, self.k.min(data.len()))
            .cloned()
            .collect();

        let mut labels = vec![0usize; data.len()];

        for _ in 0..self.max_iters {
            // assign
            for (i, fv) in data.iter().enumerate() {
                labels[i] = self.closest_centroid(fv);
            }

            // recompute centroids
            let mut sums = vec![
                FeatureVector {
                    realized_vol: 0.0,
                    atr: 0.0,
                    trend_slope: 0.0,
                    gap_rate: 0.0,
                    range_ratio: 0.0,
                };
                self.centroids.len()
            ];
            let mut counts = vec![0usize; self.centroids.len()];
            for (fv, &label) in data.iter().zip(labels.iter()) {
                let acc = &mut sums[label];
                acc.realized_vol += fv.realized_vol;
                acc.atr += fv.atr;
                acc.trend_slope += fv.trend_slope;
                acc.gap_rate += fv.gap_rate;
                acc.range_ratio += fv.range_ratio;
                counts[label] += 1;
            }
            for (i, c) in self.centroids.iter_mut().enumerate() {
                let cnt = counts[i].max(1) as f64;
                c.realized_vol = sums[i].realized_vol / cnt;
                c.atr = sums[i].atr / cnt;
                c.trend_slope = sums[i].trend_slope / cnt;
                c.gap_rate = sums[i].gap_rate / cnt;
                c.range_ratio = sums[i].range_ratio / cnt;
            }
        }
        labels
    }

    pub fn predict(&self, fv: &FeatureVector) -> usize {
        self.closest_centroid(fv)
    }

    fn closest_centroid(&self, fv: &FeatureVector) -> usize {
        let mut best = 0usize;
        let mut best_dist = f64::MAX;
        for (i, c) in self.centroids.iter().enumerate() {
            let d = squared_dist(c, fv);
            if d < best_dist {
                best_dist = d;
                best = i;
            }
        }
        best
    }
}

fn squared_dist(a: &FeatureVector, b: &FeatureVector) -> f64 {
    (a.realized_vol - b.realized_vol).powi(2)
        + (a.atr - b.atr).powi(2)
        + (a.trend_slope - b.trend_slope).powi(2)
        + (a.gap_rate - b.gap_rate).powi(2)
        + (a.range_ratio - b.range_ratio).powi(2)
}

/// Track best-performing parameter sets per regime.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegimeParamPerformance {
    pub regime: usize,
    pub indicator_or_strategy: String,
    pub params: Vec<f64>,
    pub metric_value: f64,
}

#[derive(Default)]
pub struct RegimeStore {
    capacity: usize,
    slots: HashMap<usize, Vec<RegimeParamPerformance>>,
}

impl RegimeStore {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            slots: HashMap::new(),
        }
    }

    pub fn insert(&mut self, record: RegimeParamPerformance) {
        let bucket = self.slots.entry(record.regime).or_default();
        bucket.push(record);
        bucket.sort_by(|a, b| b.metric_value.partial_cmp(&a.metric_value).unwrap());
        if bucket.len() > self.capacity {
            bucket.truncate(self.capacity);
        }
    }

    pub fn best(&self, regime: usize) -> &[RegimeParamPerformance] {
        self.slots.get(&regime).map(|v| v.as_slice()).unwrap_or(&[])
    }
}
