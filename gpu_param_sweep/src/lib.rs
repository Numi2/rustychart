pub fn name() -> &'static str {
    "gpu_param_sweep"
}
// GPU param-sweep engine usable from native or WASM (Leptos CSR).
// Minimal Leptos wiring (CSR) snippet kept here for quick copy/paste.

use anyhow::{anyhow, Result};
use bytemuck::{Pod, Zeroable};
use futures::channel::oneshot;
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

pub const DEFAULT_WORKGROUP_SIZE: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct Ohlc {
    pub open: f32,
    pub high: f32,
    pub low: f32,
    pub close: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct ParamGridUniform {
    pub ema_min: f32,
    pub ema_step: f32,
    pub ema_count: u32,

    pub band_min: f32,
    pub band_step: f32,
    pub band_count: u32,

    pub stop_min: f32,
    pub stop_step: f32,
    pub stop_count: u32,

    pub target_min: f32,
    pub target_step: f32,
    pub target_count: u32,

    pub risk_min: f32,
    pub risk_step: f32,
    pub risk_count: u32,

    pub num_bars: u32,
    pub num_combos: u32,

    pub cost_per_trade: f32,
    pub slippage_bps: f32,
    pub _pad0: u32,
    pub _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Serialize, Deserialize)]
pub struct CostModel {
    /// Flat currency cost applied per round-trip trade.
    pub per_trade: f32,
    /// Slippage in basis points applied on notional per side (entry + exit).
    pub slippage_bps: f32,
    pub _pad0: f32,
    pub _pad1: f32,
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            per_trade: 0.0,
            slippage_bps: 0.0,
            _pad0: 0.0,
            _pad1: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct GpuResult {
    pub final_equity: f32,
    pub profit_factor: f32,
    pub sharpe: f32,
    pub max_drawdown: f32,
    pub num_trades: u32,
    pub win_trades: u32,
    pub loss_trades: u32,
    pub total_trade_bars: u32,
    pub bars_with_position: u32,
    pub _pad: u32,
    pub _pad2: u32,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ParamRange {
    pub min: f32,
    pub max: f32,
    pub step: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GridConfig {
    pub ema: ParamRange,
    pub band: ParamRange,
    pub atr_stop: ParamRange,
    pub atr_target: ParamRange,
    pub risk: ParamRange,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SampledParams {
    pub ema_len: f32,
    pub band_width: f32,
    pub atr_stop_mult: f32,
    pub atr_target_mult: f32,
    pub risk_per_trade: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BacktestMetrics {
    pub params: SampledParams,
    pub final_equity: f32,
    pub profit_factor: f32,
    pub sharpe: f32,
    pub max_drawdown: f32,
    pub num_trades: u32,
    pub win_trades: u32,
    pub loss_trades: u32,
    pub total_trade_bars: u32,
    pub bars_with_position: u32,
    pub average_trade_bars: f32,
    pub exposure: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WindowMetrics {
    pub start_idx: usize,
    pub end_idx: usize, // exclusive
    pub metrics: Vec<BacktestMetrics>,
}

pub struct GpuBacktester {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    ohlc_buffer: Option<wgpu::Buffer>,
    result_buffer: Option<wgpu::Buffer>,
    readback_buffer: Option<wgpu::Buffer>,
    bind_group: Option<wgpu::BindGroup>,
    ohlc_capacity: u64,
    result_capacity: u64,
    workgroup_size: u32,
}

impl GpuBacktester {
    /// Create a compute-only wgpu device + pipeline (works on native & wasm).
    pub async fn new() -> Result<Self> {
        Self::with_workgroup_size(DEFAULT_WORKGROUP_SIZE).await
    }

    /// Same as `new` but lets you override the workgroup size (must match shader override).
    pub async fn with_workgroup_size(workgroup_size: u32) -> Result<Self> {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .map_err(|e| anyhow!("request_adapter failed: {e:?}"))?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("param-sweep-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: wgpu::ExperimentalFeatures::default(),
                trace: wgpu::Trace::default(),
            })
            .await?;

        let shader_src = include_str!("shader.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("param-sweep-shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("param-grid-uniform"),
            size: std::mem::size_of::<ParamGridUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("param-sweep-bgl"),
                entries: &[
                    // OHLC storage buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // ParamGrid uniform
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Results storage buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("param-sweep-pipeline-layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("param-sweep-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("WORKGROUP_SIZE", workgroup_size as f64)],
                ..Default::default()
            },
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            uniform_buffer,
            ohlc_buffer: None,
            result_buffer: None,
            readback_buffer: None,
            bind_group: None,
            ohlc_capacity: 0,
            result_capacity: 0,
            workgroup_size,
        })
    }

    /// Run the GPU param sweep and return metrics for every combo.
    pub async fn run_param_sweep(
        &mut self,
        ohlc: &[Ohlc],
        grid_cfg: &GridConfig,
        cost_model: CostModel,
    ) -> Result<Vec<BacktestMetrics>> {
        if ohlc.is_empty() {
            return Err(anyhow!("No OHLC data"));
        }

        let num_bars = ohlc.len() as u32;

        let ema_count = sample_count(&grid_cfg.ema)?;
        let band_count = sample_count(&grid_cfg.band)?;
        let stop_count = sample_count(&grid_cfg.atr_stop)?;
        let target_count = sample_count(&grid_cfg.atr_target)?;
        let risk_count = sample_count(&grid_cfg.risk)?;

        let num_combos =
            checked_product(&[ema_count, band_count, stop_count, target_count, risk_count])?;

        let ohlc_size = aligned_size(ohlc.len(), std::mem::size_of::<Ohlc>() as u64);
        let result_size = aligned_size(num_combos as usize, std::mem::size_of::<GpuResult>() as u64);

        let uniform = ParamGridUniform {
            ema_min: grid_cfg.ema.min,
            ema_step: grid_cfg.ema.step,
            ema_count,

            band_min: grid_cfg.band.min,
            band_step: grid_cfg.band.step,
            band_count,

            stop_min: grid_cfg.atr_stop.min,
            stop_step: grid_cfg.atr_stop.step,
            stop_count,

            target_min: grid_cfg.atr_target.min,
            target_step: grid_cfg.atr_target.step,
            target_count,

            risk_min: grid_cfg.risk.min,
            risk_step: grid_cfg.risk.step,
            risk_count,

            num_bars,
            num_combos,

            cost_per_trade: cost_model.per_trade,
            slippage_bps: cost_model.slippage_bps,
            _pad0: 0,
            _pad1: 0,
        };

        // Ensure/reuse GPU buffers
        self.ensure_ohlc_buffer(ohlc_size)?;
        self.ensure_result_buffers(result_size)?;
        self.ensure_bind_group();

        // Upload data
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
        self.queue
            .write_buffer(self.ohlc_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(ohlc));

        let mut encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("param-sweep-encoder"),
                });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("param-sweep-pass"),
                timestamp_writes: None,
            });

            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);

            let wgs = self.workgroup_size.max(1);
            let workgroups = num_combos.div_ceil(wgs);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            self.result_buffer.as_ref().unwrap(),
            0,
            self.readback_buffer.as_ref().unwrap(),
            0,
            result_size,
        );

        self.queue.submit(Some(encoder.finish()));

        // Map readback buffer
        let slice = self.readback_buffer.as_ref().unwrap().slice(..result_size);
        let (tx, rx) = oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res.map_err(|e| anyhow!("map_async failed: {e:?}")));
        });
        rx.await.unwrap_or_else(|_| Err(anyhow!("map_async receiver dropped")))?;

        let data = slice.get_mapped_range();
        let gpu_results: &[GpuResult] = bytemuck::cast_slice(&data);

        let counts = [ema_count, band_count, stop_count, target_count, risk_count];

        let mut out = Vec::with_capacity(num_combos as usize);
        for (idx, g) in gpu_results.iter().enumerate() {
            let idx = idx as u32;
            let indices = decode_index(idx, counts);
            let params = indices_to_params(indices, grid_cfg);
            let avg_trade_bars = if g.num_trades > 0 {
                g.total_trade_bars as f32 / g.num_trades as f32
            } else {
                0.0
            };
            let exposure = if num_bars > 0 {
                g.bars_with_position as f32 / num_bars as f32
            } else {
                0.0
            };

            out.push(BacktestMetrics {
                params,
                final_equity: g.final_equity,
                profit_factor: g.profit_factor,
                sharpe: g.sharpe,
                max_drawdown: g.max_drawdown,
                num_trades: g.num_trades,
                win_trades: g.win_trades,
                loss_trades: g.loss_trades,
                total_trade_bars: g.total_trade_bars,
                bars_with_position: g.bars_with_position,
                average_trade_bars: avg_trade_bars,
                exposure,
            });
        }

        drop(data);
        self.readback_buffer.as_ref().unwrap().unmap();

        Ok(out)
    }
}

// Helpers
fn aligned_size(len: usize, elem_size: u64) -> u64 {
    let raw = len as u64 * elem_size;
    raw.div_ceil(16) * 16
}

fn sample_count(range: &ParamRange) -> Result<u32> {
    if range.step <= 0.0 {
        return Err(anyhow!("Step must be > 0"));
    }
    if range.max < range.min {
        return Err(anyhow!("max < min in range"));
    }
    let span = (range.max - range.min) / range.step;
    let n = span.floor() as u32 + 1;
    if n == 0 {
        return Err(anyhow!("Zero samples in range"));
    }
    Ok(n)
}

fn checked_product(counts: &[u32]) -> Result<u32> {
    let mut acc: u64 = 1;
    for &c in counts {
        acc = acc
            .checked_mul(c as u64)
            .ok_or_else(|| anyhow!("param grid too large"))?;
    }
    if acc > u32::MAX as u64 {
        return Err(anyhow!("param grid too large for u32 indexing"));
    }
    Ok(acc as u32)
}

pub fn grid_counts(grid_cfg: &GridConfig) -> Result<[u32; 5]> {
    Ok([
        sample_count(&grid_cfg.ema)?,
        sample_count(&grid_cfg.band)?,
        sample_count(&grid_cfg.atr_stop)?,
        sample_count(&grid_cfg.atr_target)?,
        sample_count(&grid_cfg.risk)?,
    ])
}

#[derive(Clone, Copy)]
struct Indices {
    ema_idx: u32,
    band_idx: u32,
    stop_idx: u32,
    target_idx: u32,
    risk_idx: u32,
}

fn decode_index(index: u32, counts: [u32; 5]) -> Indices {
    let [ema_c, band_c, stop_c, target_c, risk_c] = counts;
    let mut idx = index;

    let ema_idx = idx % ema_c;
    idx /= ema_c;

    let band_idx = idx % band_c;
    idx /= band_c;

    let stop_idx = idx % stop_c;
    idx /= stop_c;

    let target_idx = idx % target_c;
    idx /= target_c;

    let risk_idx = idx % risk_c;

    Indices {
        ema_idx,
        band_idx,
        stop_idx,
        target_idx,
        risk_idx,
    }
}

fn encode_index(
    ema_idx: u32,
    band_idx: u32,
    stop_idx: u32,
    target_idx: u32,
    risk_idx: u32,
    counts: [u32; 5],
) -> u32 {
    let [ema_c, band_c, stop_c, target_c, _risk_c] = counts;
    ((((risk_idx * target_c + target_idx) * stop_c + stop_idx) * band_c + band_idx) * ema_c)
        + ema_idx
}

fn indices_to_params(indices: Indices, grid: &GridConfig) -> SampledParams {
    SampledParams {
        ema_len: grid.ema.min + grid.ema.step * indices.ema_idx as f32,
        band_width: grid.band.min + grid.band.step * indices.band_idx as f32,
        atr_stop_mult: grid.atr_stop.min + grid.atr_stop.step * indices.stop_idx as f32,
        atr_target_mult: grid.atr_target.min
            + grid.atr_target.step * indices.target_idx as f32,
        risk_per_trade: grid.risk.min + grid.risk.step * indices.risk_idx as f32,
    }
}

impl GpuBacktester {
    /// Simple synthetic data generator useful for testing/benchmarks.
    pub fn generate_dummy_ohlc(n: usize) -> Vec<Ohlc> {
        let mut out = Vec::with_capacity(n);
        let mut price = 100.0_f32;
        let mut seed: u32 = 0x1234abcd;
        for _ in 0..n {
            let open = price;
            // Very small LCG for deterministic pseudo-randomness without extra deps.
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let r = (seed >> 9) as f32 * (1.0 / (1u32 << 23) as f32); // in [0,1)
            let change = (r - 0.5) * 2.0;
            let close = open + change;
            let high = open.max(close) + 0.5;
            let low = open.min(close) - 0.5;
            price = close;
            out.push(Ohlc {
                open,
                high,
                low,
                close,
            });
        }
        out
    }

    fn ensure_ohlc_buffer(&mut self, size: u64) -> Result<()> {
        if self.ohlc_capacity >= size && self.ohlc_buffer.is_some() {
            return Ok(());
        }
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ohlc-buffer"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.ohlc_buffer = Some(buffer);
        self.ohlc_capacity = size;
        self.bind_group = None;
        Ok(())
    }

    fn ensure_result_buffers(&mut self, size: u64) -> Result<()> {
        if self.result_capacity >= size
            && self.result_buffer.is_some()
            && self.readback_buffer.is_some()
        {
            return Ok(());
        }

        let result_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("results-storage"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("results-readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.result_buffer = Some(result_buffer);
        self.readback_buffer = Some(readback_buffer);
        self.result_capacity = size;
        self.bind_group = None;
        Ok(())
    }

    fn ensure_bind_group(&mut self) {
        if self.bind_group.is_some() {
            return;
        }

        let ohlc = self.ohlc_buffer.as_ref().expect("ohlc buffer missing");
        let result = self.result_buffer.as_ref().expect("result buffer missing");

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("param-sweep-bdg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ohlc.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: result.as_entire_binding(),
                },
            ],
        });

        self.bind_group = Some(bind_group);
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct WindowResult {
    pub window_index: usize,
    pub start_idx: usize,
    pub end_idx: usize,
    pub metrics: Vec<BacktestMetrics>,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ExperimentSpec {
    pub name: String,
    pub strategy_id: String,
    pub data_set_id: String,
    pub windows: Vec<(usize, usize)>,
    pub grid: GridConfig,
    pub cost_model: CostModel,
    pub code_hash: String,
    pub seed: Option<u64>,
    pub notes: String,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ExperimentResult {
    pub spec: ExperimentSpec,
    pub window_results: Vec<WindowResult>,
}

pub struct ExperimentRunner {
    backtester: GpuBacktester,
}

impl ExperimentRunner {
    pub async fn new() -> Result<Self> {
        Ok(Self {
            backtester: GpuBacktester::new().await?,
        })
    }

    pub async fn run_experiment(
        &mut self,
        ohlc_full: &[Ohlc],
        spec: ExperimentSpec,
    ) -> Result<ExperimentResult> {
        let mut window_results = Vec::new();

        for (idx, (start, end)) in spec.windows.iter().enumerate() {
            if *end > ohlc_full.len() || *start >= *end {
                continue;
            }
            let slice = &ohlc_full[*start..*end];
            let metrics = self
                .backtester
                .run_param_sweep(slice, &spec.grid, spec.cost_model)
                .await?;

            window_results.push(WindowResult {
                window_index: idx,
                start_idx: *start,
                end_idx: *end,
                metrics,
            });
        }

        Ok(ExperimentResult { spec, window_results })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PersistenceStats {
    pub pf_above_threshold_fraction: f32,
    pub sharpe_above_threshold_fraction: f32,
    pub worst_window_pf: f32,
    pub worst_window_sharpe: f32,
}

pub fn compute_persistence(
    experiment: &ExperimentResult,
    pf_threshold: f32,
    sharpe_threshold: f32,
) -> Vec<PersistenceStats> {
    let num_combos = experiment
        .window_results
        .first()
        .map(|w| w.metrics.len())
        .unwrap_or(0);

    let mut stats = vec![
        PersistenceStats {
            pf_above_threshold_fraction: 0.0,
            sharpe_above_threshold_fraction: 0.0,
            worst_window_pf: f32::INFINITY,
            worst_window_sharpe: f32::INFINITY,
        };
        num_combos
    ];

    let num_windows = experiment.window_results.len() as f32;
    if num_windows == 0.0 {
        return stats;
    }

    for window in &experiment.window_results {
        for (i, m) in window.metrics.iter().enumerate() {
            let s = &mut stats[i];

            if m.profit_factor >= pf_threshold {
                s.pf_above_threshold_fraction += 1.0 / num_windows;
            }
            if m.sharpe >= sharpe_threshold {
                s.sharpe_above_threshold_fraction += 1.0 / num_windows;
            }
            if m.profit_factor < s.worst_window_pf {
                s.worst_window_pf = m.profit_factor;
            }
            if m.sharpe < s.worst_window_sharpe {
                s.worst_window_sharpe = m.sharpe;
            }
        }
    }

    stats
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NeighborRobustness {
    pub score: f32,
    pub neighbor_mean: f32,
    pub neighbor_std: f32,
}

pub fn compute_neighbor_robustness(
    metrics: &[BacktestMetrics],
    counts: [u32; 5],
) -> Vec<NeighborRobustness> {
    let mut out = Vec::with_capacity(metrics.len());
    let [ema_c, band_c, stop_c, target_c, risk_c] = counts;

    for (flat_idx, m) in metrics.iter().enumerate() {
        let indices = decode_index(flat_idx as u32, counts);
        let mut vals = Vec::with_capacity(32);
        for &d_ema in &[-1_i32, 0, 1] {
            for &d_band in &[-1_i32, 0, 1] {
                for &d_stop in &[-1_i32, 0, 1] {
                    for &d_target in &[-1_i32, 0, 1] {
                        for &d_risk in &[-1_i32, 0, 1] {
                            if d_ema == 0 && d_band == 0 && d_stop == 0 && d_target == 0 && d_risk == 0 {
                                continue;
                            }

                            let ema_i = indices.ema_idx as i32 + d_ema;
                            let band_i = indices.band_idx as i32 + d_band;
                            let stop_i = indices.stop_idx as i32 + d_stop;
                            let target_i = indices.target_idx as i32 + d_target;
                            let risk_i = indices.risk_idx as i32 + d_risk;

                            if ema_i < 0
                                || band_i < 0
                                || stop_i < 0
                                || target_i < 0
                                || risk_i < 0
                                || ema_i >= ema_c as i32
                                || band_i >= band_c as i32
                                || stop_i >= stop_c as i32
                                || target_i >= target_c as i32
                                || risk_i >= risk_c as i32
                            {
                                continue;
                            }

                            let idx = encode_index(
                                ema_i as u32,
                                band_i as u32,
                                stop_i as u32,
                                target_i as u32,
                                risk_i as u32,
                                counts,
                            );
                            vals.push(metrics[idx as usize].profit_factor);
                        }
                    }
                }
            }
        }

        if vals.is_empty() {
            out.push(NeighborRobustness {
                score: 0.0,
                neighbor_mean: 0.0,
                neighbor_std: 0.0,
            });
            continue;
        }

        let mut mean = 0.0_f32;
        for v in &vals {
            mean += *v;
        }
        mean /= vals.len() as f32;

        let mut var = 0.0_f32;
        for v in &vals {
            let d = *v - mean;
            var += d * d;
        }
        var /= vals.len() as f32;
        let std = var.sqrt();

        let stability = 1.0 / (1.0 + (m.profit_factor - mean).abs());
        let smoothness = 1.0 / (1.0 + std);
        let score = (stability + smoothness) * 0.5;

        out.push(NeighborRobustness {
            score,
            neighbor_mean: mean,
            neighbor_std: std,
        });
    }

    out
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BootstrapResult {
    pub pnl_distribution: Vec<f32>,
    pub max_drawdown_distribution: Vec<f32>,
    pub prob_dd_30: f32,
    pub prob_dd_50: f32,
}

pub fn bootstrap_trade_distribution(
    trade_rs: &[f32],
    paths: usize,
    trades_per_path: usize,
    seed: Option<u64>,
) -> BootstrapResult {
    let seed = seed.unwrap_or(42);
    let mut rng = StdRng::seed_from_u64(seed);
    if trade_rs.is_empty() || paths == 0 || trades_per_path == 0 {
        return BootstrapResult {
            pnl_distribution: Vec::new(),
            max_drawdown_distribution: Vec::new(),
            prob_dd_30: 0.0,
            prob_dd_50: 0.0,
        };
    }

    let mut pnl_distribution = Vec::with_capacity(paths);
    let mut dd_distribution = Vec::with_capacity(paths);

    for _ in 0..paths {
        let mut equity = 1.0_f32;
        let mut peak = 1.0_f32;
        let mut worst_dd = 0.0_f32;
        for _ in 0..trades_per_path {
            if let Some(&r) = trade_rs.choose(&mut rng) {
                equity *= 1.0 + r;
                if equity > peak {
                    peak = equity;
                } else {
                    let dd = (peak - equity) / peak.max(1e-6);
                    if dd > worst_dd {
                        worst_dd = dd;
                    }
                }
            }
        }
        pnl_distribution.push(equity);
        dd_distribution.push(worst_dd);
    }

    let prob_dd_30 = dd_distribution
        .iter()
        .filter(|&&dd| dd >= 0.30)
        .count() as f32
        / paths as f32;
    let prob_dd_50 = dd_distribution
        .iter()
        .filter(|&&dd| dd >= 0.50)
        .count() as f32
        / paths as f32;

    BootstrapResult {
        pnl_distribution,
        max_drawdown_distribution: dd_distribution,
        prob_dd_30,
        prob_dd_50,
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeneralizationStats {
    pub mean_profit_factor: f32,
    pub mean_sharpe: f32,
    pub worst_sharpe: f32,
    pub generalization_score: f32,
}

pub fn compute_generalization(metrics_per_market: &[Vec<BacktestMetrics>]) -> Vec<GeneralizationStats> {
    if metrics_per_market.is_empty() {
        return Vec::new();
    }

    let combos = metrics_per_market[0].len();
    let mut out = Vec::with_capacity(combos);

    for idx in 0..combos {
        let mut pf_sum = 0.0;
        let mut sharpe_sum = 0.0;
        let mut worst_sharpe = f32::INFINITY;

        for market_metrics in metrics_per_market {
            if let Some(m) = market_metrics.get(idx) {
                pf_sum += m.profit_factor;
                sharpe_sum += m.sharpe;
                if m.sharpe < worst_sharpe {
                    worst_sharpe = m.sharpe;
                }
            }
        }

        let count = metrics_per_market.len() as f32;
        let mean_pf = pf_sum / count;
        let mean_sharpe = sharpe_sum / count;
        let generalization_score = mean_sharpe - (mean_sharpe - worst_sharpe).abs() * 0.5;

        out.push(GeneralizationStats {
            mean_profit_factor: mean_pf,
            mean_sharpe,
            worst_sharpe,
            generalization_score,
        });
    }

    out
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum VolatilityRegime {
    Low,
    Normal,
    High,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct RegimeTag {
    pub trend_strength: f32,
    pub volatility: VolatilityRegime,
    pub is_crash: bool,
}

pub fn tag_regimes(ohlc: &[Ohlc], vol_lookback: usize, crash_threshold: f32) -> Vec<RegimeTag> {
    if ohlc.len() < 2 {
        return Vec::new();
    }
    let lookback = vol_lookback.max(2);
    let mut ema: f32 = ohlc[0].close;
    let alpha = 2.0 / (lookback as f32 + 1.0);
    let mut returns: Vec<f32> = Vec::with_capacity(ohlc.len());
    returns.push(0.0);

    for i in 1..ohlc.len() {
        let r = (ohlc[i].close / ohlc[i - 1].close) - 1.0;
        returns.push(r);
    }

    let mut out = Vec::with_capacity(ohlc.len());
    for i in 0..ohlc.len() {
        ema = alpha * ohlc[i].close + (1.0 - alpha) * ema;
        let trend_strength = ohlc[i].close - ema;

        let start = i.saturating_sub(lookback - 1);
        let mut mean = 0.0_f32;
        let mut count = 0.0_f32;
        for r in returns.iter().skip(start).take(lookback) {
            mean += *r;
            count += 1.0;
        }
        mean /= count.max(1.0);

        let mut var = 0.0_f32;
        for r in returns.iter().skip(start).take(lookback) {
            let d = *r - mean;
            var += d * d;
        }
        var /= count.max(1.0);
        let std = var.sqrt();

        let volatility = if std < 0.005 {
            VolatilityRegime::Low
        } else if std > 0.02 {
            VolatilityRegime::High
        } else {
            VolatilityRegime::Normal
        };

        let is_crash = returns[i] <= -crash_threshold;

        out.push(RegimeTag {
            trend_strength,
            volatility,
            is_crash,
        });
    }

    out
}

pub fn save_experiment_result(path: impl AsRef<Path>, result: &ExperimentResult) -> Result<()> {
    let file = fs::File::create(path)?;
    serde_json::to_writer_pretty(file, result)?;
    Ok(())
}

pub fn load_experiment_result(path: impl AsRef<Path>) -> Result<ExperimentResult> {
    let file = fs::File::open(path)?;
    let result = serde_json::from_reader(file)?;
    Ok(result)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm-bench"))]
    pub async fn wasm_perf_harness() -> Result<()> {
    fn log(msg: &str) {
        web_sys::console::log_1(&msg.into());
    }

    let ohlc = GpuBacktester::generate_dummy_ohlc(2_000);
    let grids = [
        GridConfig {
            ema: ParamRange { min: 10.0, max: 50.0, step: 5.0 },
            band: ParamRange { min: 1.0, max: 3.0, step: 0.5 },
            atr_stop: ParamRange { min: 1.0, max: 3.0, step: 0.5 },
            atr_target: ParamRange { min: 1.0, max: 5.0, step: 1.0 },
            risk: ParamRange { min: 0.0025, max: 0.01, step: 0.0025 },
        },
        GridConfig {
            ema: ParamRange { min: 5.0, max: 30.0, step: 5.0 },
            band: ParamRange { min: 0.5, max: 2.5, step: 0.5 },
            atr_stop: ParamRange { min: 0.5, max: 2.0, step: 0.5 },
            atr_target: ParamRange { min: 0.5, max: 3.0, step: 0.5 },
            risk: ParamRange { min: 0.001, max: 0.01, step: 0.001 },
        },
    ];

    let mut backtester = GpuBacktester::new().await?;
    let cost = CostModel::default();

    for (i, grid) in grids.iter().enumerate() {
        let start = js_sys::Date::now();
        let res = backtester.run_param_sweep(&ohlc, grid, cost).await?;
        let elapsed_ms = js_sys::Date::now() - start;
        let combos = res.len() as f64;
        let combos_per_s = combos / (elapsed_ms / 1000.0).max(1e-3);
        log(&format!(
            "Bench {}: combos={} elapsed_ms={:.2} combos/s={:.0}",
            i, combos, elapsed_ms, combos_per_s
        ));
    }

    Ok(())
}
