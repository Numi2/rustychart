use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::str::FromStr;

use data_feed::{DataEvent, DataSink, FeedStore};
use futures_util::StreamExt;
use gloo_net::http::Request;
use gloo_net::websocket::{futures::WebSocket, Message as WsMessage};
use js_sys::{Array, Date, Function, Reflect};
use serde::{Deserialize, Serialize};
use ta_engine::{
    default_output_for, IndicatorConfig, IndicatorId, IndicatorKind, IndicatorManager,
    IndicatorParams, LineStyle, OutputKind, SourceField,
};
use ts_core::{Candle, TimeFrame, Timestamp};
use wasm_bindgen::closure::Closure;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::spawn_local;
use web_sys::{
    CanvasRenderingContext2d, HtmlCanvasElement, MouseEvent, WebGl2RenderingContext, WebGlBuffer,
    WebGlProgram, WebGlShader, WebGlVertexArrayObject, WheelEvent,
};

trait RendererBackend {
    fn begin_frame(&mut self, width: f64, height: f64, clear_color: &str);
    fn draw_candles(&mut self, candles: &[PlotCandle], color_up: &str, color_down: &str);
    fn draw_polyline(&mut self, points: &[(f64, f64)], color: &str, width: f32);
    fn draw_segments(&mut self, segments: &[(f64, f64, f64, f64)], color: &str, width: f32);
    fn canvas_element(&self) -> HtmlCanvasElement;
}

struct CanvasBackend {
    canvas: HtmlCanvasElement,
    ctx: CanvasRenderingContext2d,
}

impl CanvasBackend {
    fn new(canvas: HtmlCanvasElement, ctx: CanvasRenderingContext2d) -> Self {
        Self { canvas, ctx }
    }
}

impl RendererBackend for CanvasBackend {
    fn begin_frame(&mut self, width: f64, height: f64, clear_color: &str) {
        self.canvas.set_width(width as u32);
        self.canvas.set_height(height as u32);
        self.ctx.set_fill_style_str(clear_color);
        self.ctx.fill_rect(0.0, 0.0, width, height);
    }

    fn draw_candles(&mut self, candles: &[PlotCandle], color_up: &str, color_down: &str) {
        let ctx = &self.ctx;
        for c in candles {
            let up = c.close >= c.open;
            let color = if up { color_up } else { color_down };
            ctx.set_stroke_style_str(color);
            ctx.set_fill_style_str(color);
            ctx.begin_path();
            ctx.move_to(c.x, c.y_high);
            ctx.line_to(c.x, c.y_low);
            ctx.stroke();

            let body_top = c.y_open.min(c.y_close);
            let body_bottom = c.y_open.max(c.y_close);
            let body_h = (body_bottom - body_top).max(1.0);
            ctx.fill_rect(c.x - c.half_w, body_top, c.half_w * 2.0, body_h);
        }
    }

    fn draw_polyline(&mut self, points: &[(f64, f64)], color: &str, width: f32) {
        if points.len() < 2 {
            return;
        }
        let ctx = &self.ctx;
        ctx.set_stroke_style_str(color);
        ctx.set_line_width(width as f64);
        ctx.begin_path();
        ctx.move_to(points[0].0, points[0].1);
        for p in points.iter().skip(1) {
            ctx.line_to(p.0, p.1);
        }
        ctx.stroke();
    }

    fn draw_segments(&mut self, segments: &[(f64, f64, f64, f64)], color: &str, width: f32) {
        if segments.is_empty() {
            return;
        }
        let ctx = &self.ctx;
        ctx.set_stroke_style_str(color);
        ctx.set_line_width(width as f64);
        for (x1, y1, x2, y2) in segments {
            ctx.begin_path();
            ctx.move_to(*x1, *y1);
            ctx.line_to(*x2, *y2);
            ctx.stroke();
        }
    }

    fn canvas_element(&self) -> HtmlCanvasElement {
        self.canvas.clone()
    }
}

struct WebGpuRendererBackend {
    inner: CanvasBackend,
}

impl WebGpuRendererBackend {
    fn new(canvas: HtmlCanvasElement, ctx: CanvasRenderingContext2d) -> Self {
        Self {
            inner: CanvasBackend::new(canvas, ctx),
        }
    }
}

impl RendererBackend for WebGpuRendererBackend {
    fn begin_frame(&mut self, width: f64, height: f64, clear_color: &str) {
        self.inner.begin_frame(width, height, clear_color);
    }

    fn draw_candles(&mut self, candles: &[PlotCandle], color_up: &str, color_down: &str) {
        self.inner.draw_candles(candles, color_up, color_down);
    }

    fn draw_polyline(&mut self, points: &[(f64, f64)], color: &str, width: f32) {
        self.inner.draw_polyline(points, color, width);
    }

    fn draw_segments(&mut self, segments: &[(f64, f64, f64, f64)], color: &str, width: f32) {
        self.inner.draw_segments(segments, color, width);
    }

    fn canvas_element(&self) -> HtmlCanvasElement {
        self.inner.canvas_element()
    }
}

fn webgpu_supported() -> bool {
    if let Some(win) = web_sys::window() {
        if let Ok(nav) = Reflect::get(&win, &JsValue::from_str("navigator")) {
            return Reflect::has(&nav, &JsValue::from_str("gpu")).unwrap_or(false);
        }
    }
    false
}

#[derive(Debug, Clone)]
struct PlotCandle {
    x: f64,
    half_w: f64,
    y_open: f64,
    y_close: f64,
    y_high: f64,
    y_low: f64,
    open: f64,
    close: f64,
}

struct WebGlBackend {
    canvas: HtmlCanvasElement,
    gl: WebGl2RenderingContext,
    program: WebGlProgram,
    vbo: WebGlBuffer,
    vao: WebGlVertexArrayObject,
    line_program: WebGlProgram,
    line_vbo: WebGlBuffer,
    line_vao: WebGlVertexArrayObject,
    width: f64,
    height: f64,
}

impl WebGlBackend {
    fn new(canvas: HtmlCanvasElement, gl: WebGl2RenderingContext) -> Result<Self, JsValue> {
        let program = Self::build_program(
            &gl,
            include_str!("shaders/candle.vert"),
            include_str!("shaders/candle.frag"),
        )?;
        let line_program = Self::build_program(
            &gl,
            include_str!("shaders/line.vert"),
            include_str!("shaders/line.frag"),
        )?;

        let vbo = gl
            .create_buffer()
            .ok_or_else(|| JsValue::from_str("failed to create candle buffer"))?;
        let vao = gl
            .create_vertex_array()
            .ok_or_else(|| JsValue::from_str("failed to create candle vao"))?;

        let line_vbo = gl
            .create_buffer()
            .ok_or_else(|| JsValue::from_str("failed to create line buffer"))?;
        let line_vao = gl
            .create_vertex_array()
            .ok_or_else(|| JsValue::from_str("failed to create line vao"))?;

        Ok(Self {
            canvas,
            gl,
            program,
            vbo,
            vao,
            line_program,
            line_vbo,
            line_vao,
            width: 1.0,
            height: 1.0,
        })
    }

    fn build_program(
        gl: &WebGl2RenderingContext,
        vs_src: &str,
        fs_src: &str,
    ) -> Result<WebGlProgram, JsValue> {
        let vs = Self::compile_shader(gl, WebGl2RenderingContext::VERTEX_SHADER, vs_src)?;
        let fs = Self::compile_shader(gl, WebGl2RenderingContext::FRAGMENT_SHADER, fs_src)?;
        let program = gl
            .create_program()
            .ok_or_else(|| JsValue::from_str("unable to create program"))?;
        gl.attach_shader(&program, &vs);
        gl.attach_shader(&program, &fs);
        gl.link_program(&program);
        if !gl
            .get_program_parameter(&program, WebGl2RenderingContext::LINK_STATUS)
            .as_bool()
            .unwrap_or(false)
        {
            let log = gl
                .get_program_info_log(&program)
                .unwrap_or_else(|| "unknown link error".to_string());
            return Err(JsValue::from_str(&log));
        }
        Ok(program)
    }

    fn compile_shader(
        gl: &WebGl2RenderingContext,
        kind: u32,
        src: &str,
    ) -> Result<WebGlShader, JsValue> {
        let shader = gl
            .create_shader(kind)
            .ok_or_else(|| JsValue::from_str("unable to create shader"))?;
        gl.shader_source(&shader, src);
        gl.compile_shader(&shader);
        if !gl
            .get_shader_parameter(&shader, WebGl2RenderingContext::COMPILE_STATUS)
            .as_bool()
            .unwrap_or(false)
        {
            let log = gl
                .get_shader_info_log(&shader)
                .unwrap_or_else(|| "unknown shader error".to_string());
            return Err(JsValue::from_str(&log));
        }
        Ok(shader)
    }

    fn to_ndc(&self, x: f64, y: f64) -> (f32, f32) {
        let nx = (x / self.width) as f32 * 2.0 - 1.0;
        let ny = 1.0 - (y / self.height) as f32 * 2.0;
        (nx, ny)
    }

    fn hex_to_rgba(color: &str) -> [f32; 4] {
        // Accept #RRGGBB only; fallback gray.
        if let Some(stripped) = color.strip_prefix('#') {
            if stripped.len() == 6 {
                if let (Ok(r), Ok(g), Ok(b)) = (
                    u8::from_str_radix(&stripped[0..2], 16),
                    u8::from_str_radix(&stripped[2..4], 16),
                    u8::from_str_radix(&stripped[4..6], 16),
                ) {
                    return [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0];
                }
            }
        }
        [0.7, 0.7, 0.7, 1.0]
    }
}

impl RendererBackend for WebGlBackend {
    fn begin_frame(&mut self, width: f64, height: f64, clear_color: &str) {
        self.width = width.max(1.0);
        self.height = height.max(1.0);
        self.canvas.set_width(self.width as u32);
        self.canvas.set_height(self.height as u32);
        self.gl
            .viewport(0, 0, self.width as i32, self.height as i32);
        let c = Self::hex_to_rgba(clear_color);
        self.gl.clear_color(c[0], c[1], c[2], c[3]);
        self.gl.clear(WebGl2RenderingContext::COLOR_BUFFER_BIT);
    }

    fn draw_candles(&mut self, candles: &[PlotCandle], color_up: &str, color_down: &str) {
        if candles.is_empty() {
            return;
        }
        let mut verts: Vec<f32> = Vec::with_capacity(candles.len() * 30);
        let color_up = Self::hex_to_rgba(color_up);
        let color_down = Self::hex_to_rgba(color_down);
        for c in candles {
            let up = c.close >= c.open;
            let color = if up { color_up } else { color_down };
            let (wx, wy1) = self.to_ndc(c.x, c.y_high);
            let (_, wy2) = self.to_ndc(c.x, c.y_low);
            // wick as line (2 verts)
            verts.extend_from_slice(&[wx, wy1, color[0], color[1], color[2], color[3]]);
            verts.extend_from_slice(&[wx, wy2, color[0], color[1], color[2], color[3]]);

            // body rectangle as two triangles (6 verts)
            let half_w = (c.half_w).max(0.5);
            let (x_left, y_top) = self.to_ndc(c.x - half_w, c.y_open.min(c.y_close));
            let (x_right, y_bottom) = self.to_ndc(c.x + half_w, c.y_open.max(c.y_close));

            let quad = [
                (x_left, y_top),
                (x_right, y_top),
                (x_right, y_bottom),
                (x_left, y_bottom),
            ];
            let idx = [0, 1, 2, 0, 2, 3];
            for i in idx {
                verts.extend_from_slice(&[
                    quad[i].0, quad[i].1, color[0], color[1], color[2], color[3],
                ]);
            }
        }

        self.gl.bind_vertex_array(Some(&self.vao));
        self.gl
            .bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&self.vbo));
        unsafe {
            let vert_array = js_sys::Float32Array::view(&verts);
            self.gl.buffer_data_with_array_buffer_view(
                WebGl2RenderingContext::ARRAY_BUFFER,
                &vert_array,
                WebGl2RenderingContext::STREAM_DRAW,
            );
        }
        self.gl.use_program(Some(&self.program));

        let stride = (6 * std::mem::size_of::<f32>()) as i32;
        self.gl.enable_vertex_attrib_array(0);
        self.gl.vertex_attrib_pointer_with_i32(
            0,
            2,
            WebGl2RenderingContext::FLOAT,
            false,
            stride,
            0,
        );
        self.gl.enable_vertex_attrib_array(1);
        self.gl.vertex_attrib_pointer_with_i32(
            1,
            4,
            WebGl2RenderingContext::FLOAT,
            false,
            stride,
            2 * 4,
        );
        let count = (verts.len() / 6) as i32;
        self.gl
            .draw_arrays(WebGl2RenderingContext::LINES, 0, (candles.len() * 2) as i32);
        let body_offset = (candles.len() * 2) as i32;
        let body_count = count - body_offset;
        if body_count > 0 {
            self.gl
                .draw_arrays(WebGl2RenderingContext::TRIANGLES, body_offset, body_count);
        }
        self.gl.bind_vertex_array(None);
    }

    fn draw_polyline(&mut self, points: &[(f64, f64)], color: &str, width: f32) {
        if points.len() < 2 {
            return;
        }
        let rgba = Self::hex_to_rgba(color);
        let mut verts: Vec<f32> = Vec::with_capacity(points.len() * 6);
        for (x, y) in points {
            let (nx, ny) = self.to_ndc(*x, *y);
            verts.extend_from_slice(&[nx, ny, rgba[0], rgba[1], rgba[2], rgba[3]]);
        }
        self.gl.bind_vertex_array(Some(&self.line_vao));
        self.gl
            .bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&self.line_vbo));
        unsafe {
            let vert_array = js_sys::Float32Array::view(&verts);
            self.gl.buffer_data_with_array_buffer_view(
                WebGl2RenderingContext::ARRAY_BUFFER,
                &vert_array,
                WebGl2RenderingContext::STREAM_DRAW,
            );
        }
        self.gl.use_program(Some(&self.line_program));
        let stride = (6 * std::mem::size_of::<f32>()) as i32;
        self.gl.enable_vertex_attrib_array(0);
        self.gl.vertex_attrib_pointer_with_i32(
            0,
            2,
            WebGl2RenderingContext::FLOAT,
            false,
            stride,
            0,
        );
        self.gl.enable_vertex_attrib_array(1);
        self.gl.vertex_attrib_pointer_with_i32(
            1,
            4,
            WebGl2RenderingContext::FLOAT,
            false,
            stride,
            2 * 4,
        );
        self.gl.line_width(width.clamp(1.0, 10.0)); // clamp for cross-browser support
        self.gl
            .draw_arrays(WebGl2RenderingContext::LINE_STRIP, 0, points.len() as i32);
        self.gl.bind_vertex_array(None);
    }

    fn draw_segments(&mut self, segments: &[(f64, f64, f64, f64)], color: &str, width: f32) {
        if segments.is_empty() {
            return;
        }
        let rgba = Self::hex_to_rgba(color);
        let mut verts: Vec<f32> = Vec::with_capacity(segments.len() * 12);
        for (x1, y1, x2, y2) in segments {
            let (nx1, ny1) = self.to_ndc(*x1, *y1);
            let (nx2, ny2) = self.to_ndc(*x2, *y2);
            verts.extend_from_slice(&[nx1, ny1, rgba[0], rgba[1], rgba[2], rgba[3]]);
            verts.extend_from_slice(&[nx2, ny2, rgba[0], rgba[1], rgba[2], rgba[3]]);
        }
        self.gl.bind_vertex_array(Some(&self.line_vao));
        self.gl
            .bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&self.line_vbo));
        unsafe {
            let vert_array = js_sys::Float32Array::view(&verts);
            self.gl.buffer_data_with_array_buffer_view(
                WebGl2RenderingContext::ARRAY_BUFFER,
                &vert_array,
                WebGl2RenderingContext::STREAM_DRAW,
            );
        }
        self.gl.use_program(Some(&self.line_program));
        let stride = (6 * std::mem::size_of::<f32>()) as i32;
        self.gl.enable_vertex_attrib_array(0);
        self.gl.vertex_attrib_pointer_with_i32(
            0,
            2,
            WebGl2RenderingContext::FLOAT,
            false,
            stride,
            0,
        );
        self.gl.enable_vertex_attrib_array(1);
        self.gl.vertex_attrib_pointer_with_i32(
            1,
            4,
            WebGl2RenderingContext::FLOAT,
            false,
            stride,
            2 * 4,
        );
        self.gl.line_width(width.clamp(1.0, 10.0)); // clamp for cross-browser support
        self.gl.draw_arrays(
            WebGl2RenderingContext::LINES,
            0,
            (segments.len() * 2) as i32,
        );
        self.gl.bind_vertex_array(None);
    }

    fn canvas_element(&self) -> HtmlCanvasElement {
        self.canvas.clone()
    }
}

/// Rendering style for the price series.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum SeriesStyle {
    #[default]
    Candle,
    Ohlc,
    Line,
    Area,
    Bar,
}

// --- Drawing types ----------------------------------------------------------

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DrawingKind {
    HorizontalLine,
    VerticalLine,
    TrendLine,
    Rectangle,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Drawing {
    pub id: u64,
    pub kind: DrawingKind,
    pub ts1: Timestamp,
    pub price1: f64,
    pub ts2: Option<Timestamp>,
    pub price2: Option<f64>,
    pub color: String,
    pub width: f64,
}

/// Order/position/alert overlays rendered on top of price pane.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderVisual {
    pub id: String,
    pub side: String,
    pub price: f64,
    pub qty: f64,
    pub label: String,
    pub stop_price: Option<f64>,
    pub take_profit_price: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertVisual {
    pub id: String,
    pub ts: Timestamp,
    pub price: Option<f64>,
    pub label: String,
    pub fired: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScriptSample {
    pub ts: Timestamp,
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScriptSeries {
    pub id: String,
    pub pane_id: Option<u32>,
    pub color: String,
    pub width: f64,
    pub samples: Vec<ScriptSample>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChartSnapshot {
    indicators: Vec<IndicatorConfig>,
    drawings: Vec<Drawing>,
}

struct Chart {
    canvas: HtmlCanvasElement,
    ctx: CanvasRenderingContext2d,
    background: Box<dyn RendererBackend>,
    feed_store: Rc<RefCell<FeedStore>>,
    timeframe: TimeFrame,
    style: SeriesStyle,
    log_scale: bool,

    width: f64,
    height: f64,

    // Visible time window on X axis.
    visible_start: Timestamp,
    visible_end: Timestamp,

    // Price panel Y scale.
    y_min: f64,
    y_max: f64,

    // Layout: price + indicator panes.
    price_panel_top: f64,
    price_panel_height: f64,
    price_weight: f64,
    pane_weights: HashMap<u32, f64>,

    auto_scroll: bool,

    // Interaction state.
    is_dragging: bool,
    last_pointer_x: f64,
    last_pointer_y: f64,
    last_pointer_time_ms: f64,
    kinetic_velocity_px_per_ms: f64,
    last_frame_time_ms: f64,
    dragging_drawing: Option<u64>,

    dirty: bool,

    // TA indicators.
    indicator_mgr: IndicatorManager,

    // Drawings.
    drawings: Vec<Drawing>,
    next_drawing_id: u64,

    // Undo/redo for indicator + drawing config.
    undo_stack: Vec<ChartSnapshot>,
    redo_stack: Vec<ChartSnapshot>,

    // Panel-level dirty flags.
    dirty_price_panel: bool,
    dirty_indicator_panes: bool,
    dirty_background: bool,
    overlay_dirty: bool,

    crosshair: Option<(f64, f64, Timestamp, f64)>,

    orders: Vec<OrderVisual>,
    positions: Vec<OrderVisual>,
    alerts: Vec<AlertVisual>,
    script_series: Vec<ScriptSeries>,

    // Reusable buffers to avoid per-frame allocations.
    downsample_buf: Vec<Candle>,

    // Lifecycle.
    destroyed: bool,
}

impl Chart {
    fn new(
        canvas: HtmlCanvasElement,
        ctx: CanvasRenderingContext2d,
        feed_store: Rc<RefCell<FeedStore>>,
        timeframe: TimeFrame,
    ) -> Result<Self, JsValue> {
        let width = canvas.width() as f64;
        let height = canvas.height() as f64;
        let base_tf_ms = Self::tf_ms(timeframe);

        let doc = canvas
            .owner_document()
            .ok_or_else(|| JsValue::from_str("no document"))?;
        let bg_canvas: HtmlCanvasElement = doc
            .create_element("canvas")?
            .dyn_into::<HtmlCanvasElement>()
            .map_err(|_| JsValue::from_str("bg canvas cast failed"))?;
        bg_canvas.set_width(width as u32);
        bg_canvas.set_height(height as u32);
        let background: Box<dyn RendererBackend> = if webgpu_supported() {
            let bg_ctx = bg_canvas
                .get_context("2d")?
                .ok_or_else(|| JsValue::from_str("no 2d context"))?
                .dyn_into::<CanvasRenderingContext2d>()?;
            Box::new(WebGpuRendererBackend::new(bg_canvas.clone(), bg_ctx))
        } else {
            match bg_canvas
                .get_context("webgl2")
                .and_then(|c| c.ok_or_else(|| JsValue::from_str("no webgl2")))
                .and_then(|c| {
                    c.dyn_into::<WebGl2RenderingContext>()
                        .map_err(|_| JsValue::from_str("webgl2 cast failed"))
                }) {
                Ok(gl_ctx) => {
                    if let Ok(gl_backend) = WebGlBackend::new(bg_canvas.clone(), gl_ctx) {
                        Box::new(gl_backend)
                    } else {
                        let bg_ctx = bg_canvas
                            .get_context("2d")?
                            .ok_or_else(|| JsValue::from_str("no 2d context"))?
                            .dyn_into::<CanvasRenderingContext2d>()?;
                        Box::new(CanvasBackend::new(bg_canvas.clone(), bg_ctx))
                    }
                }
                Err(_) => {
                    let bg_ctx = bg_canvas
                        .get_context("2d")?
                        .ok_or_else(|| JsValue::from_str("no 2d context"))?
                        .dyn_into::<CanvasRenderingContext2d>()?;
                    Box::new(CanvasBackend::new(bg_canvas.clone(), bg_ctx))
                }
            }
        };

        let now_ms = Date::now() as i64;
        let default_span_ms = (base_tf_ms * 400).max(10_000);

        Ok(Self {
            canvas,
            ctx,
            background,
            feed_store,
            timeframe,
            style: SeriesStyle::Candle,
            log_scale: false,
            width,
            height,
            visible_start: now_ms - default_span_ms,
            visible_end: now_ms,
            y_min: 0.0,
            y_max: 1.0,
            price_panel_top: 0.0,
            price_panel_height: height,
            price_weight: 1.0,
            pane_weights: HashMap::new(),
            auto_scroll: true,
            is_dragging: false,
            last_pointer_x: 0.0,
            last_pointer_y: 0.0,
            last_pointer_time_ms: 0.0,
            kinetic_velocity_px_per_ms: 0.0,
            last_frame_time_ms: 0.0,
            dragging_drawing: None,
            dirty: true,
            indicator_mgr: IndicatorManager::new(timeframe),
            drawings: Vec::new(),
            next_drawing_id: 1,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            dirty_price_panel: true,
            dirty_indicator_panes: true,
            dirty_background: true,
            overlay_dirty: true,
            crosshair: None,
            orders: Vec::new(),
            positions: Vec::new(),
            alerts: Vec::new(),
            script_series: Vec::new(),
            downsample_buf: Vec::new(),
            destroyed: false,
        })
    }

    #[inline]
    fn tf_ms(tf: TimeFrame) -> i64 {
        tf.duration_ms().max(1)
    }

    fn resize(&mut self, width: f64, height: f64) {
        self.width = width.max(1.0);
        self.height = height.max(1.0);
        self.canvas.set_width(self.width as u32);
        self.canvas.set_height(self.height as u32);
        self.background
            .begin_frame(self.width, self.height, "#0c111a");
        self.price_panel_top = 0.0;
        self.price_panel_height = self.height;
        self.dirty = true;
        self.dirty_price_panel = true;
        self.dirty_indicator_panes = true;
        self.dirty_background = true;
        self.overlay_dirty = true;
    }

    fn set_style(&mut self, style: &str) {
        self.style = match style {
            "candle" | "candles" => SeriesStyle::Candle,
            "ohlc" => SeriesStyle::Ohlc,
            "line" => SeriesStyle::Line,
            "area" => SeriesStyle::Area,
            "bar" | "bars" => SeriesStyle::Bar,
            _ => self.style,
        };
        self.dirty = true;
        self.dirty_price_panel = true;
        self.dirty_background = true;
    }

    fn set_pane_layout(&mut self, price_weight: f64, panes: &[(u32, f64)]) {
        self.price_weight = price_weight.max(0.1);
        self.pane_weights.clear();
        for (id, weight) in panes {
            self.pane_weights.insert(*id, weight.max(0.1));
        }
        self.dirty = true;
        self.dirty_price_panel = true;
        self.dirty_indicator_panes = true;
        self.dirty_background = true;
    }

    // --- indicator management ------------------------------------------------

    fn add_indicator(&mut self, config: IndicatorConfig) -> IndicatorId {
        let history: Vec<Candle> = {
            let fs = self.feed_store.borrow();
            let store = fs.store();
            store.series(self.timeframe).as_slice().to_vec()
        };
        self.push_undo_snapshot();
        let id = self.indicator_mgr.add_indicator(config, history.as_slice());
        self.dirty = true;
        self.dirty_price_panel = true;
        self.dirty_indicator_panes = true;
        id
    }

    fn remove_indicator(&mut self, id: IndicatorId) {
        self.push_undo_snapshot();
        self.indicator_mgr.remove_indicator(id);
        self.dirty = true;
        self.dirty_price_panel = true;
        self.dirty_indicator_panes = true;
    }

    fn set_indicator_styles(&mut self, id: IndicatorId, styles: Vec<LineStyle>) {
        self.push_undo_snapshot();
        self.indicator_mgr.set_line_styles(id, styles);
        self.dirty = true;
        self.dirty_price_panel = true;
        self.dirty_indicator_panes = true;
    }

    fn rebuild_indicators_from_store(&mut self) {
        let fs = self.feed_store.borrow();
        let store = fs.store();
        let series = store.series(self.timeframe);
        self.indicator_mgr.rebuild_all(series.as_slice());
        self.dirty = true;
        self.dirty_price_panel = true;
        self.dirty_indicator_panes = true;
    }

    fn snapshot(&self) -> ChartSnapshot {
        let indicators = self
            .indicator_mgr
            .indicators()
            .iter()
            .map(|inst| inst.config.clone())
            .collect();
        ChartSnapshot {
            indicators,
            drawings: self.drawings.clone(),
        }
    }

    fn restore_snapshot(&mut self, snap: ChartSnapshot) {
        let fs = self.feed_store.borrow();
        let store = fs.store();
        let history = store.series(self.timeframe).as_slice();

        self.indicator_mgr = IndicatorManager::new(self.timeframe);
        for cfg in snap.indicators.iter().cloned() {
            self.indicator_mgr.add_indicator(cfg, history);
        }

        self.drawings = snap.drawings;
        self.dirty = true;
        self.dirty_price_panel = true;
        self.dirty_indicator_panes = true;
    }

    fn push_undo_snapshot(&mut self) {
        self.undo_stack.push(self.snapshot());
        self.redo_stack.clear();
    }

    fn undo(&mut self) {
        if let Some(prev) = self.undo_stack.pop() {
            let cur = self.snapshot();
            self.restore_snapshot(prev);
            self.redo_stack.push(cur);
        }
    }

    fn redo(&mut self) {
        if let Some(next) = self.redo_stack.pop() {
            let cur = self.snapshot();
            self.restore_snapshot(next);
            self.undo_stack.push(cur);
        }
    }

    fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    // --- event application ---------------------------------------------------

    fn apply_data_event(&mut self, event: DataEvent) {
        let ev_clone = event.clone();

        {
            let mut fs = self.feed_store.borrow_mut();
            fs.on_event(event);
        }

        match ev_clone {
            DataEvent::HistoryBatch { timeframe, .. } => {
                if timeframe == self.timeframe {
                    self.rebuild_indicators_from_store();
                }
            }
            DataEvent::LiveCandle(c) => {
                self.indicator_mgr.on_new_candle(&c);
            }
            DataEvent::LiveTick(_) => {
                // Tick-based indicators could be added here if needed.
            }
            DataEvent::Reset => {
                self.indicator_mgr.clear();
            }
            DataEvent::Gap { .. } => {}
        }

        self.on_data_updated();
    }

    fn on_data_updated(&mut self) {
        let base_tf = self.timeframe;
        let base_tf_ms = Self::tf_ms(base_tf);
        let (first_ts, last_ts) = {
            let fs = self.feed_store.borrow();
            let store = fs.store();
            let series = store.series(base_tf);
            if let (Some(first), Some(last)) = (series.first(), series.last()) {
                (first.ts, last.ts)
            } else {
                return;
            }
        };

        if self.visible_end <= self.visible_start {
            let span = (last_ts - first_ts).abs().max(base_tf_ms * 4);
            self.visible_start = first_ts;
            self.visible_end = first_ts + span;
        } else if self.auto_scroll {
            let span = (self.visible_end - self.visible_start).max(base_tf_ms * 4);
            self.visible_end = last_ts + base_tf_ms;
            self.visible_start = self.visible_end - span;
        }

        self.dirty = true;
        self.dirty_price_panel = true;
        self.dirty_indicator_panes = true;
    }

    // --- main frame loop -----------------------------------------------------

    fn frame(&mut self, timestamp_ms: f64) {
        if self.destroyed {
            return;
        }
        if self.last_frame_time_ms == 0.0 {
            self.last_frame_time_ms = timestamp_ms;
        }
        let dt_ms = (timestamp_ms - self.last_frame_time_ms).max(0.0);
        self.last_frame_time_ms = timestamp_ms;

        if self.kinetic_velocity_px_per_ms.abs() > 0.01 {
            let dx = self.kinetic_velocity_px_per_ms * dt_ms;
            self.pan_pixels(dx);
            let friction = 0.95_f64.powf(dt_ms / 16.0);
            self.kinetic_velocity_px_per_ms *= friction;
            if self.kinetic_velocity_px_per_ms.abs() < 0.01 {
                self.kinetic_velocity_px_per_ms = 0.0;
            }
            self.dirty = true;
        }

        if self.dirty || self.overlay_dirty || self.dirty_background {
            self.render();
        }
    }

    // --- coordinate transforms ----------------------------------------------

    fn price_to_y(&self, price: f64) -> f64 {
        let p = if self.log_scale {
            if price > 0.0 {
                price.ln()
            } else {
                self.y_min
            }
        } else {
            price
        };
        let range = (self.y_max - self.y_min).max(1e-9);
        let norm = (p - self.y_min) / range;
        self.price_panel_top + self.price_panel_height - norm * self.price_panel_height
    }

    fn y_to_price(&self, y: f64) -> f64 {
        let range = (self.y_max - self.y_min).max(1e-9);
        let rel = (self.price_panel_top + self.price_panel_height - y) / self.price_panel_height;
        let rel_clamped = rel.clamp(0.0, 1.0);
        let v = self.y_min + rel_clamped * range;
        if self.log_scale {
            v.exp()
        } else {
            v
        }
    }

    fn time_to_x(&self, ts: Timestamp) -> f64 {
        let span = (self.visible_end - self.visible_start) as f64;
        if span <= 0.0 || self.width <= 0.0 {
            return 0.0;
        }
        let t = (ts - self.visible_start) as f64 / span;
        t * self.width
    }

    fn x_to_time(&self, x: f64) -> Timestamp {
        let span = (self.visible_end - self.visible_start) as f64;
        if span <= 0.0 || self.width <= 0.0 {
            return self.visible_start;
        }
        let ratio = (x / self.width).clamp(0.0, 1.0);
        self.visible_start + (ratio * span) as i64
    }

    fn visible_range(&self) -> (Timestamp, Timestamp) {
        (self.visible_start, self.visible_end)
    }

    fn map_point(&self, x: f64, y: f64) -> (Timestamp, f64) {
        (self.x_to_time(x), self.y_to_price(y))
    }

    fn nearest_candle(&self, ts: Timestamp) -> Option<Candle> {
        let fs = self.feed_store.borrow();
        let store = fs.store();
        let slice = store
            .series(self.timeframe)
            .range(self.visible_start, self.visible_end);
        slice.iter().min_by_key(|c| (c.ts - ts).abs()).cloned()
    }

    fn set_crosshair(&mut self, ts: Timestamp, price: f64) {
        let x = self.time_to_x(ts);
        let y = self.price_to_y(price);
        self.crosshair = Some((x, y, ts, price));
        self.overlay_dirty = true;
    }

    fn set_visible_range(&mut self, start: Timestamp, end: Timestamp) {
        if start >= end {
            return;
        }
        self.visible_start = start;
        self.visible_end = end;
        self.auto_scroll = false;
        self.dirty = true;
        self.dirty_price_panel = true;
        self.dirty_indicator_panes = true;
        self.dirty_background = true;
    }

    fn indicator_value_to_y(
        &self,
        value: f64,
        min_v: f64,
        max_v: f64,
        pane_top: f64,
        pane_height: f64,
    ) -> f64 {
        let range = (max_v - min_v).max(1e-9);
        let norm = (value - min_v) / range;
        pane_top + pane_height - norm * pane_height
    }

    // --- interaction ---------------------------------------------------------

    fn pan_pixels(&mut self, dx: f64) {
        let span = (self.visible_end - self.visible_start) as f64;
        if span <= 0.0 || self.width <= 0.0 {
            return;
        }
        let ts_per_px = span / self.width;
        let delta_ts = (dx * ts_per_px) as i64;
        self.visible_start -= delta_ts;
        self.visible_end -= delta_ts;
        self.auto_scroll = false;
        self.dirty = true;
        self.dirty_price_panel = true;
        self.dirty_indicator_panes = true;
    }

    fn zoom_at(&mut self, x: f64, factor: f64) {
        if factor <= 0.0 {
            return;
        }
        let center_ts = self.x_to_time(x);
        let span = (self.visible_end - self.visible_start) as f64;
        if span <= 0.0 {
            return;
        }

        let min_span = self.timeframe.duration_ms().max(1_000) as f64 * 10.0;
        let max_span = self.timeframe.duration_ms().max(1_000) as f64 * 10_000.0;

        let new_span = (span * factor).clamp(min_span, max_span);
        let center_offset = center_ts - self.visible_start;
        let center_ratio = (center_offset as f64 / span).clamp(0.0, 1.0);

        let new_start = center_ts - (new_span * center_ratio) as i64;
        let new_end = new_start + new_span as i64;

        self.visible_start = new_start;
        self.visible_end = new_end;
        self.auto_scroll = false;
        self.dirty = true;
        self.dirty_price_panel = true;
        self.dirty_indicator_panes = true;
    }

    fn on_mouse_down(&mut self, x: f64, y: f64) {
        self.last_pointer_x = x;
        self.last_pointer_y = y;
        self.last_pointer_time_ms = Date::now();
        self.kinetic_velocity_px_per_ms = 0.0;
        self.auto_scroll = false;
        self.crosshair = Some((x, y, self.x_to_time(x), self.y_to_price(y)));
        self.overlay_dirty = true;

        if let Some(id) = self.hit_test_drawing(x, y) {
            self.push_undo_snapshot();
            self.dragging_drawing = Some(id);
            return;
        }

        self.is_dragging = true;
    }

    fn on_mouse_move(&mut self, x: f64, y: f64) {
        self.crosshair = Some((x, y, self.x_to_time(x), self.y_to_price(y)));
        self.overlay_dirty = true;

        let now = Date::now();
        if let Some(drag_id) = self.dragging_drawing {
            let prev_ts = self.x_to_time(self.last_pointer_x);
            let new_ts = self.x_to_time(x);
            let prev_price = self.y_to_price(self.last_pointer_y);
            let new_price = self.y_to_price(y);
            let dt = new_ts - prev_ts;
            let dp = new_price - prev_price;
            self.move_drawing_by(drag_id, dt, dp);
            self.last_pointer_x = x;
            self.last_pointer_y = y;
            self.last_pointer_time_ms = now;
            return;
        }

        if !self.is_dragging {
            return;
        }
        let dt_ms = (now - self.last_pointer_time_ms).max(1.0);
        let dx = x - self.last_pointer_x;

        self.pan_pixels(dx);

        self.kinetic_velocity_px_per_ms = dx / dt_ms;

        self.last_pointer_x = x;
        self.last_pointer_y = y;
        self.last_pointer_time_ms = now;
    }

    fn on_mouse_up(&mut self) {
        self.is_dragging = false;
        self.dragging_drawing = None;
        if self.kinetic_velocity_px_per_ms.abs() < 0.01 {
            self.kinetic_velocity_px_per_ms = 0.0;
        }
    }

    fn on_wheel(&mut self, delta_y: f64, x: f64, _y: f64) {
        let factor = if delta_y < 0.0 { 0.9 } else { 1.1 };
        self.zoom_at(x, factor);
    }

    // --- drawings ------------------------------------------------------------

    fn add_horizontal_line(&mut self, price: f64, color: String, width: f64) -> u64 {
        self.push_undo_snapshot();
        let id = self.next_drawing_id;
        self.next_drawing_id += 1;
        self.drawings.push(Drawing {
            id,
            kind: DrawingKind::HorizontalLine,
            ts1: 0,
            price1: price,
            ts2: None,
            price2: None,
            color,
            width,
        });
        self.dirty = true;
        self.dirty_price_panel = true;
        id
    }

    fn add_vertical_line(&mut self, ts: Timestamp, color: String, width: f64) -> u64 {
        self.push_undo_snapshot();
        let id = self.next_drawing_id;
        self.next_drawing_id += 1;
        self.drawings.push(Drawing {
            id,
            kind: DrawingKind::VerticalLine,
            ts1: ts,
            price1: 0.0,
            ts2: None,
            price2: None,
            color,
            width,
        });
        self.dirty = true;
        self.dirty_price_panel = true;
        id
    }

    fn add_trend_line(
        &mut self,
        ts1: Timestamp,
        price1: f64,
        ts2: Timestamp,
        price2: f64,
        color: String,
        width: f64,
    ) -> u64 {
        self.push_undo_snapshot();
        let id = self.next_drawing_id;
        self.next_drawing_id += 1;
        self.drawings.push(Drawing {
            id,
            kind: DrawingKind::TrendLine,
            ts1,
            price1,
            ts2: Some(ts2),
            price2: Some(price2),
            color,
            width,
        });
        self.dirty = true;
        self.dirty_price_panel = true;
        id
    }

    fn add_rectangle(
        &mut self,
        ts1: Timestamp,
        price1: f64,
        ts2: Timestamp,
        price2: f64,
        color: String,
        width: f64,
    ) -> u64 {
        self.push_undo_snapshot();
        let id = self.next_drawing_id;
        self.next_drawing_id += 1;
        self.drawings.push(Drawing {
            id,
            kind: DrawingKind::Rectangle,
            ts1,
            price1,
            ts2: Some(ts2),
            price2: Some(price2),
            color,
            width,
        });
        self.dirty = true;
        self.dirty_price_panel = true;
        id
    }

    fn remove_drawing(&mut self, id: u64) {
        if let Some(idx) = self.drawings.iter().position(|d| d.id == id) {
            self.push_undo_snapshot();
            self.drawings.remove(idx);
            self.dirty = true;
            self.dirty_price_panel = true;
        }
    }

    fn clear_drawings(&mut self) {
        if !self.drawings.is_empty() {
            self.push_undo_snapshot();
            self.drawings.clear();
            self.dirty = true;
            self.dirty_price_panel = true;
        }
    }

    fn hit_test_drawing(&self, x: f64, y: f64) -> Option<u64> {
        let ts = self.x_to_time(x);
        let price = self.y_to_price(y);
        let tol_price = (self.y_max - self.y_min) * 0.01;
        for d in self.drawings.iter().rev() {
            match d.kind {
                DrawingKind::HorizontalLine => {
                    if (d.price1 - price).abs() <= tol_price {
                        return Some(d.id);
                    }
                }
                DrawingKind::VerticalLine => {
                    if (d.ts1 - ts).abs() as f64
                        <= (self.visible_end - self.visible_start) as f64 * 0.01
                    {
                        return Some(d.id);
                    }
                }
                DrawingKind::TrendLine => {
                    if let (Some(ts2), Some(price2)) = (d.ts2, d.price2) {
                        let t_min = d.ts1.min(ts2);
                        let t_max = d.ts1.max(ts2);
                        if ts < t_min || ts > t_max {
                            continue;
                        }
                        let p1 = d.price1;
                        let p2 = price2;
                        let ratio = if t_max == t_min {
                            0.0
                        } else {
                            (ts - t_min) as f64 / (t_max - t_min) as f64
                        };
                        let interp = p1 + (p2 - p1) * ratio;
                        if (interp - price).abs() <= tol_price {
                            return Some(d.id);
                        }
                    }
                }
                DrawingKind::Rectangle => {
                    if let (Some(ts2), Some(price2)) = (d.ts2, d.price2) {
                        let t_min = d.ts1.min(ts2);
                        let t_max = d.ts1.max(ts2);
                        let p_min = d.price1.min(price2);
                        let p_max = d.price1.max(price2);
                        if ts >= t_min && ts <= t_max && price >= p_min && price <= p_max {
                            return Some(d.id);
                        }
                    }
                }
            }
        }
        None
    }

    fn move_drawing_by(&mut self, id: u64, dt: Timestamp, dp: f64) {
        if let Some(d) = self.drawings.iter_mut().find(|d| d.id == id) {
            d.ts1 += dt;
            if let Some(ts2) = d.ts2.as_mut() {
                *ts2 += dt;
            }
            d.price1 += dp;
            if let Some(p2) = d.price2.as_mut() {
                *p2 += dp;
            }
            self.dirty = true;
            self.dirty_price_panel = true;
            self.dirty_background = true;
            self.overlay_dirty = true;
        }
    }

    fn clear_indicators(&mut self) {
        self.push_undo_snapshot();
        self.indicator_mgr.clear();
        self.dirty = true;
        self.dirty_price_panel = true;
        self.dirty_indicator_panes = true;
        self.dirty_background = true;
    }

    fn set_script_series(&mut self, series: Vec<ScriptSeries>) {
        self.script_series = series;
        self.dirty = true;
        self.dirty_background = true;
    }

    fn set_orders(&mut self, orders: Vec<OrderVisual>) {
        self.orders = orders;
        self.overlay_dirty = true;
    }

    fn set_positions(&mut self, positions: Vec<OrderVisual>) {
        self.positions = positions;
        self.overlay_dirty = true;
    }

    fn set_alerts(&mut self, alerts: Vec<AlertVisual>) {
        self.alerts = alerts;
        self.overlay_dirty = true;
    }

    fn render_layers(&mut self) -> (Vec<f64>, Vec<Timestamp>) {
        // Determine how many separate panes we need.
        let pane_ids: Vec<u32> = {
            let mut ids = Vec::new();
            for inst in self.indicator_mgr.indicators() {
                if inst.config.output == OutputKind::SeparatePane {
                    if let Some(pid) = inst.config.pane_id {
                        if !ids.contains(&pid) {
                            ids.push(pid);
                        }
                    }
                }
            }
            ids.sort();
            ids
        };

        let mut pane_layout: Vec<(u32, f64, f64)> = Vec::new();
        if pane_ids.is_empty() {
            self.price_panel_top = 0.0;
            self.price_panel_height = self.height;
        } else {
            let mut total_weight = self.price_weight.max(0.1);
            for pid in pane_ids.iter() {
                total_weight += self.pane_weights.get(pid).copied().unwrap_or(1.0).max(0.1);
            }
            self.price_panel_top = 0.0;
            self.price_panel_height =
                self.height * (self.price_weight.max(0.1) / total_weight.max(0.1));
            let mut cursor = self.price_panel_height;
            for pid in pane_ids.iter() {
                let w = self.pane_weights.get(pid).copied().unwrap_or(1.0).max(0.1);
                let h = self.height * (w / total_weight.max(0.1));
                pane_layout.push((*pid, cursor, h));
                cursor += h;
            }
        }

        // Downsample visible main candles (~ one per pixel).
        self.downsample_buf.clear();
        {
            let fs = self.feed_store.borrow();
            let store = fs.store();
            let max_points = self.width.max(1.0) as usize;
            let tmp = store.downsample(
                self.timeframe,
                self.visible_start,
                self.visible_end,
                max_points,
            );
            self.downsample_buf.extend_from_slice(&tmp);
        }
        let candles = &self.downsample_buf;

        if candles.is_empty() {
            return (Vec::new(), Vec::new());
        }

        // Dynamic price Y range.
        let mut min_p = f64::MAX;
        let mut max_p = f64::MIN;
        for c in candles {
            min_p = min_p.min(c.low);
            max_p = max_p.max(c.high);
        }
        if !min_p.is_finite() || !max_p.is_finite() || max_p <= min_p {
            return (Vec::new(), Vec::new());
        }
        if self.log_scale && min_p <= 0.0 {
            // Fallback to linear if prices are non-positive.
            self.log_scale = false;
        }
        if self.log_scale {
            let min_t = min_p.ln();
            let max_t = max_p.ln();
            let pad = (max_t - min_t).max(1e-6) * 0.05;
            self.y_min = min_t - pad;
            self.y_max = max_t + pad;
        } else {
            let pad = (max_p - min_p) * 0.035;
            self.y_min = min_p - pad;
            self.y_max = max_p + pad;
        }

        let price_ticks = self.price_ticks(5);
        let time_ticks = self.time_ticks(6);

        let mut grid_segments: Vec<(f64, f64, f64, f64)> = Vec::new();
        for p in &price_ticks {
            let y = self.price_to_y(*p);
            grid_segments.push((0.0, y, self.width, y));
        }
        for t in &time_ticks {
            let x = self.time_to_x(*t);
            grid_segments.push((x, self.price_panel_top, x, self.height));
        }

        // Slightly brighter canvas background and higher-contrast grid.
        self.background
            .begin_frame(self.width, self.height, "#0c111a");
        self.background
            .draw_segments(&grid_segments, "#1b2836", 1.0);

        // Main price series.
        let bar_width = (self.width / candles.len().max(1) as f64 * 0.7).clamp(2.0, 14.0);
        let half_w = bar_width * 0.5;
        let mut plot_candles = Vec::with_capacity(candles.len());
        for c in candles {
            plot_candles.push(PlotCandle {
                x: self.time_to_x(c.ts),
                half_w,
                y_open: self.price_to_y(c.open),
                y_close: self.price_to_y(c.close),
                y_high: self.price_to_y(c.high),
                y_low: self.price_to_y(c.low),
                open: c.open,
                close: c.close,
            });
        }
        self.background
            .draw_candles(&plot_candles, "#4ade80", "#ef4444");

        // Overlays on price panel.
        for inst in self.indicator_mgr.indicators() {
            if inst.config.output != OutputKind::Overlay {
                continue;
            }
            let series = inst.series();
            let slice = series.range(self.visible_start, self.visible_end);
            if slice.len() < 2 {
                continue;
            }
            let dim = inst.output_dimension();
            for line_idx in 0..dim {
                let style = &inst.config.line_styles[line_idx];
                let mut pts = Vec::with_capacity(slice.len());
                for sample in slice.iter() {
                    if let Some(v) = sample.values.get(line_idx) {
                        if v.is_finite() {
                            pts.push((self.time_to_x(sample.ts), self.price_to_y(*v)));
                        }
                    }
                }
                self.background
                    .draw_polyline(&pts, &style.color, style.width as f32);
            }
        }

        // Separate panes.
        if !pane_layout.is_empty() {
            for (pane_id, pane_top, pane_height) in pane_layout.iter() {
                // y range
                let mut min_v = f64::MAX;
                let mut max_v = f64::MIN;
                for inst in self.indicator_mgr.indicators() {
                    if inst.config.output != OutputKind::SeparatePane {
                        continue;
                    }
                    if inst.config.pane_id != Some(*pane_id) {
                        continue;
                    }
                    let series = inst.series();
                    let slice = series.range(self.visible_start, self.visible_end);
                    for sample in slice.iter() {
                        for v in &sample.values {
                            if v.is_finite() {
                                min_v = min_v.min(*v);
                                max_v = max_v.max(*v);
                            }
                        }
                    }
                }
                if !min_v.is_finite() || !max_v.is_finite() || max_v <= min_v {
                    continue;
                }

                for inst in self.indicator_mgr.indicators() {
                    if inst.config.output != OutputKind::SeparatePane {
                        continue;
                    }
                    if inst.config.pane_id != Some(*pane_id) {
                        continue;
                    }
                    let series = inst.series();
                    let slice = series.range(self.visible_start, self.visible_end);
                    if slice.len() < 2 {
                        continue;
                    }
                    let dim = inst.output_dimension();
                    for line_idx in 0..dim {
                        let style = &inst.config.line_styles[line_idx];
                        let mut pts = Vec::with_capacity(slice.len());
                        for sample in slice.iter() {
                            if let Some(v) = sample.values.get(line_idx) {
                                if v.is_finite() {
                                    let y = self.indicator_value_to_y(
                                        *v,
                                        min_v,
                                        max_v,
                                        *pane_top,
                                        *pane_height,
                                    );
                                    pts.push((self.time_to_x(sample.ts), y));
                                }
                            }
                        }
                        self.background
                            .draw_polyline(&pts, &style.color, style.width as f32);
                    }
                }
            }
        }

        // Drawings on price panel.
        let mut drawing_segments: Vec<(f64, f64, f64, f64)> = Vec::new();
        for d in &self.drawings {
            match d.kind {
                DrawingKind::HorizontalLine => {
                    let y = self.price_to_y(d.price1);
                    drawing_segments.push((0.0, y, self.width, y));
                }
                DrawingKind::VerticalLine => {
                    let x = self.time_to_x(d.ts1);
                    drawing_segments.push((x, self.price_panel_top, x, self.height));
                }
                DrawingKind::TrendLine => {
                    if let (Some(ts2), Some(price2)) = (d.ts2, d.price2) {
                        drawing_segments.push((
                            self.time_to_x(d.ts1),
                            self.price_to_y(d.price1),
                            self.time_to_x(ts2),
                            self.price_to_y(price2),
                        ));
                    }
                }
                DrawingKind::Rectangle => {
                    if let (Some(ts2), Some(price2)) = (d.ts2, d.price2) {
                        let x1 = self.time_to_x(d.ts1);
                        let y1 = self.price_to_y(d.price1);
                        let x2 = self.time_to_x(ts2);
                        let y2 = self.price_to_y(price2);
                        let (xmin, xmax) = if x1 < x2 { (x1, x2) } else { (x2, x1) };
                        let (ymin, ymax) = if y1 < y2 { (y1, y2) } else { (y2, y1) };
                        drawing_segments.push((xmin, ymin, xmax, ymin));
                        drawing_segments.push((xmax, ymin, xmax, ymax));
                        drawing_segments.push((xmax, ymax, xmin, ymax));
                        drawing_segments.push((xmin, ymax, xmin, ymin));
                    }
                }
            }
        }
        self.background
            .draw_segments(&drawing_segments, "#9ea7b3", 1.5);

        self.dirty_price_panel = false;
        self.dirty_indicator_panes = false;
        self.dirty_background = false;
        (price_ticks, time_ticks)
    }

    fn render_overlay(&self) {
        let ctx = &self.ctx;

        // Orders and positions.
        for order in self.orders.iter().chain(self.positions.iter()) {
            let y = self.price_to_y(order.price);
            ctx.set_line_width(1.0);
            let color = if order.side.to_lowercase() == "buy" {
                "#26a69a"
            } else {
                "#ef5350"
            };
            ctx.set_stroke_style_str(color);
            ctx.begin_path();
            ctx.move_to(0.0, y);
            ctx.line_to(self.width, y);
            ctx.stroke();

            // Label box.
            ctx.set_fill_style_str(color);
            let label = format!("{} @ {}", order.label, order.price);
            ctx.fill_rect(0.0, y - 10.0, (label.len() as f64 * 6.0) + 8.0, 18.0);
            ctx.set_fill_style_str("#0f1724");
            ctx.set_font("12px 'Inter', sans-serif");
            ctx.fill_text(&label, 4.0, y + 3.0).unwrap_or_default();

            if let Some(stop) = order.stop_price {
                let y_stop = self.price_to_y(stop);
                ctx.set_stroke_style_str("#f6c343");
                ctx.begin_path();
                ctx.move_to(0.0, y_stop);
                ctx.line_to(self.width, y_stop);
                ctx.stroke();
            }
            if let Some(tp) = order.take_profit_price {
                let y_tp = self.price_to_y(tp);
                ctx.set_stroke_style_str("#64b5f6");
                ctx.begin_path();
                ctx.move_to(0.0, y_tp);
                ctx.line_to(self.width, y_tp);
                ctx.stroke();
            }
        }

        // Alerts
        for alert in &self.alerts {
            let x = self.time_to_x(alert.ts);
            let y = if let Some(price) = alert.price {
                self.price_to_y(price)
            } else {
                self.price_panel_top + 12.0
            };
            let color = if alert.fired { "#f6c343" } else { "#64b5f6" };
            ctx.set_fill_style_str(color);
            ctx.begin_path();
            ctx.arc(x, y, 5.0, 0.0, std::f64::consts::PI * 2.0).ok();
            ctx.fill();
        }

        // Script overlays.
        for series in &self.script_series {
            ctx.set_line_width(series.width);
            ctx.set_stroke_style_str(&series.color);
            ctx.begin_path();
            let mut started = false;
            for sample in &series.samples {
                if sample.ts < self.visible_start || sample.ts > self.visible_end {
                    continue;
                }
                if let Some(val) = sample.values.first() {
                    if !val.is_finite() {
                        continue;
                    }
                    let x = self.time_to_x(sample.ts);
                    let y = self.price_to_y(*val);
                    if !started {
                        ctx.move_to(x, y);
                        started = true;
                    } else {
                        ctx.line_to(x, y);
                    }
                }
            }
            if started {
                ctx.stroke();
            }
        }

        // Crosshair overlay.
        if let Some((x, y, ts, price)) = self.crosshair {
            ctx.set_stroke_style_str("#8ab4ff");
            ctx.set_line_width(1.0);
            ctx.set_line_dash(&Array::of2(
                &JsValue::from_f64(4.0),
                &JsValue::from_f64(4.0),
            ))
            .ok();
            ctx.begin_path();
            ctx.move_to(x, self.price_panel_top);
            ctx.line_to(x, self.price_panel_top + self.price_panel_height);
            ctx.move_to(0.0, y);
            ctx.line_to(self.width, y);
            ctx.stroke();
            let _ = ctx.set_line_dash(&Array::new());

            // Price label
            let label = format!("{price:.2}");
            ctx.set_fill_style_str("#0f1724");
            let box_width = (label.len() as f64 * 7.0) + 10.0;
            ctx.fill_rect(self.width - box_width, y - 10.0, box_width, 20.0);
            ctx.set_fill_style_str("#ffffff");
            ctx.set_font("12px 'Inter', sans-serif");
            ctx.fill_text(&label, self.width - box_width + 4.0, y + 4.0)
                .ok();

            // Time label
            let dt = js_sys::Date::new(&JsValue::from_f64(ts as f64));
            let time_label = dt
                .to_locale_string("en-GB", &JsValue::UNDEFINED)
                .as_string()
                .unwrap_or_else(|| dt.to_iso_string().as_string().unwrap_or_default());
            let time_box = (time_label.len() as f64 * 6.0) + 10.0;
            ctx.set_fill_style_str("#0f1724");
            ctx.fill_rect(x - time_box * 0.5, self.height - 24.0, time_box, 20.0);
            ctx.set_fill_style_str("#ffffff");
            ctx.set_font("12px 'Inter', sans-serif");
            ctx.fill_text(&time_label, x - time_box * 0.5 + 4.0, self.height - 8.0)
                .ok();

            if let Some(candle) = self.nearest_candle(ts) {
                let info = format!(
                    "O:{:.2} H:{:.2} L:{:.2} C:{:.2} V:{:.0}",
                    candle.open, candle.high, candle.low, candle.close, candle.volume
                );
                let info_w = (info.len() as f64 * 6.5) + 12.0;
                ctx.set_fill_style_str("#0f1724");
                ctx.fill_rect(8.0, self.price_panel_top + 8.0, info_w, 20.0);
                ctx.set_fill_style_str("#ffffff");
                ctx.set_font("12px 'Inter', sans-serif");
                ctx.fill_text(&info, 12.0, self.price_panel_top + 22.0).ok();
            }
        }
    }

    fn render(&mut self) {
        if self.destroyed {
            return;
        }

        if self.dirty_price_panel || self.dirty_indicator_panes {
            self.dirty_background = true;
        }

        let (price_ticks, time_ticks) = if self.dirty_background || self.dirty {
            self.render_layers()
        } else {
            (self.price_ticks(5), self.time_ticks(6))
        };

        self.ctx.clear_rect(0.0, 0.0, self.width, self.height);
        let bg_canvas = self.background.canvas_element();
        let _ = self
            .ctx
            .draw_image_with_html_canvas_element(&bg_canvas, 0.0, 0.0);
        self.draw_axis_labels(&price_ticks, &time_ticks);
        self.render_overlay();
        self.dirty = false;
        self.overlay_dirty = false;
    }

    fn price_ticks(&self, count: usize) -> Vec<f64> {
        if count < 2 {
            return Vec::new();
        }
        let span = (self.y_max - self.y_min).max(1e-6);
        let step = span / (count as f64 - 1.0);
        if self.log_scale {
            (0..count)
                .map(|i| (self.y_min + step * i as f64).exp())
                .collect()
        } else {
            (0..count).map(|i| self.y_min + step * i as f64).collect()
        }
    }

    fn time_ticks(&self, count: usize) -> Vec<Timestamp> {
        if count < 1 {
            return Vec::new();
        }
        let span = self.visible_end - self.visible_start;
        if span <= 0 {
            return Vec::new();
        }
        let step = span / (count as i64 + 1);
        (0..=count)
            .map(|i| self.visible_start + step * i as i64)
            .collect()
    }

    fn draw_axis_labels(&self, price_ticks: &[f64], time_ticks: &[Timestamp]) {
        let ctx = &self.ctx;
        ctx.set_fill_style_str("#d5e0ef");
        ctx.set_font("12px 'Inter', sans-serif");
        for p in price_ticks {
            let y = self.price_to_y(*p);
            let label = format!("{p:.2}");
            ctx.fill_text(&label, self.width - 64.0, y - 1.0).ok();
        }

        for t in time_ticks {
            let x = self.time_to_x(*t);
            let dt = js_sys::Date::new(&JsValue::from_f64(*t as f64));
            let label = dt
                .to_locale_time_string("en-GB")
                .as_string()
                .unwrap_or_else(|| dt.to_time_string().as_string().unwrap_or_default());
            ctx.fill_text(&label, x - 28.0, self.height - 4.0).ok();
        }
    }
}

// ---------- ChartHandle / JS API -------------------------------------------

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum ChartEvent {
    Click {
        x: f64,
        y: f64,
        ts: Timestamp,
        price: f64,
        button: i16,
    },
    CrosshairMove {
        x: f64,
        y: f64,
        ts: Timestamp,
        price: f64,
    },
    ViewChanged {
        start: Timestamp,
        end: Timestamp,
    },
}

struct EventSubscription {
    id: u32,
    callback: Function,
}

/// Ensure the canvas matches the layouted size of its container.
fn resize_canvas_to_parent(inner: &Rc<RefCell<ChartHandleInner>>) {
    let canvas = { inner.borrow().chart.canvas.clone() };
    let rect = canvas.get_bounding_client_rect();
    let width = rect.width().max(1.0);
    let height = rect.height().max(1.0);
    {
        let mut inner_mut = inner.borrow_mut();
        inner_mut.chart.resize(width, height);
    }
}

struct ChartHandleInner {
    symbol: String,
    base_timeframe: TimeFrame,
    http_base: String,
    ws_base: String,
    provider_override: Option<String>,
    chart: Chart,

    feeds_live: bool,

    next_event_id: u32,
    subscribers: Vec<EventSubscription>,
}

impl ChartHandleInner {
    fn dispatch_event(&self, event: &ChartEvent) {
        if self.subscribers.is_empty() {
            return;
        }
        if let Ok(json) = serde_json::to_string(event) {
            let val = JsValue::from_str(&json);
            for sub in &self.subscribers {
                let _ = sub.callback.call1(&JsValue::NULL, &val);
            }
        }
    }

    fn add_subscription(&mut self, cb: Function) -> u32 {
        let id = self.next_event_id;
        self.next_event_id = self.next_event_id.wrapping_add(1);
        self.subscribers
            .push(EventSubscription { id, callback: cb });
        id
    }

    fn remove_subscription(&mut self, id: u32) {
        if let Some(idx) = self.subscribers.iter().position(|s| s.id == id) {
            self.subscribers.remove(idx);
        }
    }
}

fn setup_mouse_events(inner_rc: &Rc<RefCell<ChartHandleInner>>) -> Result<(), JsValue> {
    let canvas = inner_rc.borrow().chart.canvas.clone();

    // mousedown
    {
        let inner_rc = inner_rc.clone();
        let canvas_clone = canvas.clone();
        let closure = Closure::<dyn FnMut(MouseEvent)>::wrap(Box::new(move |event: MouseEvent| {
            event.prevent_default();
            let rect = canvas_clone.get_bounding_client_rect();
            let x = event.client_x() as f64 - rect.left();
            let y = event.client_y() as f64 - rect.top();
            let button = event.button();

            {
                let mut inner = inner_rc.borrow_mut();
                inner.chart.on_mouse_down(x, y);
            }

            {
                let inner = inner_rc.borrow();
                let (ts, price) = inner.chart.map_point(x, y);
                let ev = ChartEvent::Click {
                    x,
                    y,
                    ts,
                    price,
                    button,
                };
                inner.dispatch_event(&ev);
            }
        }));
        canvas.add_event_listener_with_callback("mousedown", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }

    // mousemove
    {
        let inner_rc = inner_rc.clone();
        let canvas_clone = canvas.clone();
        let closure = Closure::<dyn FnMut(MouseEvent)>::wrap(Box::new(move |event: MouseEvent| {
            event.prevent_default();
            let rect = canvas_clone.get_bounding_client_rect();
            let x = event.client_x() as f64 - rect.left();
            let y = event.client_y() as f64 - rect.top();

            {
                let mut inner = inner_rc.borrow_mut();
                inner.chart.on_mouse_move(x, y);
            }

            {
                let inner = inner_rc.borrow();
                let (ts, price) = inner.chart.map_point(x, y);
                let ev = ChartEvent::CrosshairMove { x, y, ts, price };
                inner.dispatch_event(&ev);
                if inner.chart.is_dragging || inner.chart.dragging_drawing.is_some() {
                    let (start, end) = inner.chart.visible_range();
                    inner.dispatch_event(&ChartEvent::ViewChanged { start, end });
                }
            }
        }));
        canvas.add_event_listener_with_callback("mousemove", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }

    // mouseup
    {
        let inner = inner_rc.clone();
        let window = web_sys::window().unwrap();
        let closure = Closure::<dyn FnMut(MouseEvent)>::wrap(Box::new(move |event: MouseEvent| {
            event.prevent_default();
            inner.borrow_mut().chart.on_mouse_up();
        }));
        window.add_event_listener_with_callback("mouseup", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }

    // wheel
    {
        let inner = inner_rc.clone();
        let canvas_clone = canvas.clone();
        let opts = web_sys::AddEventListenerOptions::new();
        opts.set_passive(true);
        let closure = Closure::<dyn FnMut(WheelEvent)>::wrap(Box::new(move |event: WheelEvent| {
            let rect = canvas_clone.get_bounding_client_rect();
            let x = event.client_x() as f64 - rect.left();
            let y = event.client_y() as f64 - rect.top();
            {
                let mut inner_mut = inner.borrow_mut();
                inner_mut.chart.on_wheel(event.delta_y(), x, y);
                let (start, end) = inner_mut.chart.visible_range();
                inner_mut.dispatch_event(&ChartEvent::ViewChanged { start, end });
            }
        }));
        canvas.add_event_listener_with_callback_and_add_event_listener_options(
            "wheel",
            closure.as_ref().unchecked_ref(),
            &opts,
        )?;
        closure.forget();
    }

    Ok(())
}

fn start_render_loop(inner_rc: Rc<RefCell<ChartHandleInner>>) {
    let f = Rc::new(RefCell::new(None::<Closure<dyn FnMut(f64)>>));
    let g = f.clone();

    *g.borrow_mut() = Some(Closure::wrap(Box::new(move |timestamp: f64| {
        let mut should_continue = true;
        {
            let mut inner = inner_rc.borrow_mut();
            if inner.chart.destroyed {
                should_continue = false;
            } else {
                inner.chart.frame(timestamp);
            }
        }

        if should_continue {
            let window = web_sys::window().unwrap();
            window
                .request_animation_frame(f.borrow().as_ref().unwrap().as_ref().unchecked_ref())
                .unwrap();
        }
    }) as Box<dyn FnMut(f64)>));

    let window = web_sys::window().unwrap();
    window
        .request_animation_frame(g.borrow().as_ref().unwrap().as_ref().unchecked_ref())
        .unwrap();
}

#[derive(Deserialize)]
struct MarketDataResponse {
    data: Vec<MarketDataBar>,
}

#[derive(Deserialize)]
struct MarketDataBar {
    ts: Option<i64>,
    date: String,
    time: String,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: Option<f64>,
}

fn read_global(key: &str) -> Option<String> {
    Reflect::get(&js_sys::global(), &JsValue::from_str(key))
        .ok()
        .and_then(|v| v.as_string())
}

fn normalize_provider(value: &str) -> Option<String> {
    match value.to_ascii_lowercase().as_str() {
        "alpha" | "alphavantage" | "av" => Some("alpha".to_string()),
        "coinbase" | "cb" => Some("coinbase".to_string()),
        "yahoo" => Some("yahoo".to_string()),
        _ => None,
    }
}

fn provider_for_symbol(symbol: &str, override_provider: Option<&str>) -> String {
    if let Some(p) = override_provider {
        if let Some(norm) = normalize_provider(p) {
            return norm;
        }
    }
    let s = symbol.to_ascii_uppercase();
    if s.contains('-') {
        return "coinbase".to_string();
    }
    if s.ends_with("USD") || s.ends_with("USDT") || s.ends_with("USDC") {
        return "coinbase".to_string();
    }
    "alpha".to_string()
}

fn seed_price_for(symbol: &str) -> f64 {
    let upper = symbol.to_ascii_uppercase();
    if upper.contains("BTC") {
        30_000.0
    } else if upper.contains("ETH") {
        2_000.0
    } else if upper.contains("ES=") || upper.contains("SPY") {
        4_500.0
    } else {
        100.0
    }
}

fn fallback_history(tf: TimeFrame, days: u64, seed: f64) -> Vec<Candle> {
    let step_ms = tf.duration_ms().max(1);
    let day_ms = 86_400_000_i64;
    let approx = ((days as i64 * day_ms) / step_ms).clamp(120, 1200) as usize;
    let count = approx.max(120);
    let now = Date::now() as i64;
    let mut candles = Vec::with_capacity(count);
    let mut price = seed.max(1.0);

    for idx in (0..count).rev() {
        let ts = now - step_ms * (count as i64 - idx as i64);
        let drift = ((idx as f64) * 0.07).sin() * 0.9 + ((idx as f64) * 0.013).cos() * 0.6;
        let trend = (idx as f64 / count as f64 - 0.5) * 0.4;
        let open = price;
        price = (price + drift + trend).max(0.5);
        let close = price;
        let high = open.max(close) + 0.6;
        let low = open.min(close) - 0.6;
        candles.push(Candle {
            ts,
            timeframe: tf,
            open,
            high,
            low,
            close,
            volume: 1.0,
        });
    }

    candles
}

fn init_feeds(inner_rc: Rc<RefCell<ChartHandleInner>>) {
    {
        let mut inner = inner_rc.borrow_mut();
        inner.feeds_live = true;
    }
    let rc = inner_rc.clone();
    spawn_local(async move {
        let (symbol, base_tf, http_base, ws_base) = {
            let inner = rc.borrow();
            (
                inner.symbol.clone(),
                inner.base_timeframe,
                inner.http_base.clone(),
                inner.ws_base.clone(),
            )
        };

        if !rc.borrow().feeds_live {
            return;
        }

        let mut history_loaded = false;
        let push_fallback_history = |rc: &Rc<RefCell<ChartHandleInner>>| {
            if !rc.borrow().feeds_live {
                return;
            }
            let mut inner_mut = rc.borrow_mut();
            if !inner_mut.feeds_live {
                return;
            }
            let seed = seed_price_for(&symbol);
            let candles = fallback_history(base_tf, 5, seed);
            let ev = DataEvent::HistoryBatch {
                timeframe: base_tf,
                candles,
                prepend: false,
            };
            inner_mut.chart.apply_data_event(ev);
        };

        let provider_override = { rc.borrow().provider_override.clone() };
        let provider = provider_for_symbol(&symbol, provider_override.as_deref());
        let base_http = http_base.trim_end_matches('/');
        let base_ws = ws_base.trim_end_matches('/');

        if provider == "alpha" {
            let history_url = format!(
                "{}/history?symbol={}&tf={}&provider=alpha&outputsize=compact",
                base_http,
                symbol,
                base_tf.name()
            );
            if let Ok(resp) = Request::get(&history_url).send().await {
                if !rc.borrow().feeds_live {
                    return;
                }
                if let Ok(candles) = resp.json::<Vec<Candle>>().await {
                    let mut candles = candles;
                    candles.sort_by_key(|c| c.ts);
                    let has_candles = !candles.is_empty();
                    let ev = DataEvent::HistoryBatch {
                        timeframe: base_tf,
                        candles,
                        prepend: false,
                    };
                    let mut inner_mut = rc.borrow_mut();
                    if !inner_mut.feeds_live {
                        return;
                    }
                    if has_candles {
                        history_loaded = true;
                    }
                    inner_mut.chart.apply_data_event(ev);
                }
            }
            if !history_loaded {
                push_fallback_history(&rc);
            }
        } else {
            // History via existing market-data endpoint.
            let history_url = format!(
                "{}/market-data?symbol={}&interval={}&provider={}",
                base_http,
                symbol,
                base_tf.name(),
                provider
            );
            if let Ok(resp) = Request::get(&history_url).send().await {
                if !rc.borrow().feeds_live {
                    return;
                }
                if let Ok(body) = resp.json::<MarketDataResponse>().await {
                    let mut candles: Vec<Candle> = body
                        .data
                        .into_iter()
                        .filter_map(|bar| {
                            let ts = bar.ts.or_else(|| {
                                let iso = format!("{}T{}:00Z", bar.date, bar.time);
                                let ts_ms = Date::new(&JsValue::from_str(&iso)).get_time();
                                if ts_ms.is_nan() {
                                    None
                                } else {
                                    Some(ts_ms as i64)
                                }
                            })?;
                            Some(Candle {
                                ts,
                                timeframe: base_tf,
                                open: bar.open,
                                high: bar.high,
                                low: bar.low,
                                close: bar.close,
                                volume: bar.volume.unwrap_or(0.0),
                            })
                        })
                        .collect();

                    candles.sort_by_key(|c| c.ts);

                    let has_candles = !candles.is_empty();
                    let ev = DataEvent::HistoryBatch {
                        timeframe: base_tf,
                        candles,
                        prepend: false,
                    };
                    let mut inner_mut = rc.borrow_mut();
                    if !inner_mut.feeds_live {
                        return;
                    }
                    if has_candles {
                        history_loaded = true;
                    }
                    inner_mut.chart.apply_data_event(ev);
                }
            }

            if !history_loaded {
                push_fallback_history(&rc);
            }

            // WebSocket for supported providers.
            let ws_url = format!(
                "{}?symbol={}&tf={}&provider={}",
                base_ws,
                symbol,
                base_tf.name(),
                provider
            );
            if let Ok(mut ws) = WebSocket::open(&ws_url) {
                while let Some(msg) = ws.next().await {
                    if !rc.borrow().feeds_live {
                        break;
                    }
                    if let Ok(WsMessage::Text(txt)) = msg {
                        if let Ok(ev) = serde_json::from_str::<DataEvent>(&txt) {
                            let mut inner_mut = rc.borrow_mut();
                            if !inner_mut.feeds_live {
                                break;
                            }
                            inner_mut.chart.apply_data_event(ev);
                        }
                    }
                }
            }
        }
    });
}

/// Public chart handle for JS.
#[wasm_bindgen]
pub struct ChartHandle {
    inner: Rc<RefCell<ChartHandleInner>>,
}

#[wasm_bindgen]
impl ChartHandle {
    #[wasm_bindgen(constructor)]
    pub fn new(
        canvas_id: &str,
        symbol: &str,
        base_timeframe: &str,
        http_base: &str,
        ws_base: &str,
    ) -> Result<ChartHandle, JsValue> {
        let tf = TimeFrame::from_str(base_timeframe)
            .ok_or_else(|| JsValue::from_str("invalid timeframe"))?;

        let window = web_sys::window().ok_or_else(|| JsValue::from_str("no window"))?;
        let document = window
            .document()
            .ok_or_else(|| JsValue::from_str("no document"))?;
        let element = document
            .get_element_by_id(canvas_id)
            .ok_or_else(|| JsValue::from_str("canvas not found"))?;

        let canvas: HtmlCanvasElement = element
            .dyn_into::<HtmlCanvasElement>()
            .map_err(|_| JsValue::from_str("element is not a canvas"))?;

        let context = canvas
            .get_context("2d")?
            .ok_or_else(|| JsValue::from_str("no 2d context"))?
            .dyn_into::<CanvasRenderingContext2d>()?;

        let feed_store_rc = Rc::new(RefCell::new(FeedStore::new(symbol.to_string(), tf)));
        let chart = Chart::new(canvas.clone(), context, feed_store_rc, tf)?;

        let provider_override =
            read_global("RUSTYCHART_DEFAULT_PROVIDER").and_then(|p| normalize_provider(&p));

        let inner = Rc::new(RefCell::new(ChartHandleInner {
            symbol: symbol.to_string(),
            base_timeframe: tf,
            http_base: http_base.to_string(),
            ws_base: ws_base.to_string(),
            provider_override,
            chart,
            feeds_live: true,
            next_event_id: 1,
            subscribers: Vec::new(),
        }));

        // Fit the canvas to the rendered cell size (including after layout shifts).
        resize_canvas_to_parent(&inner);
        {
            let inner_clone = inner.clone();
            let window = web_sys::window().unwrap();
            let resize_cb = Closure::<dyn FnMut()>::wrap(Box::new(move || {
                resize_canvas_to_parent(&inner_clone);
            }));
            window
                .add_event_listener_with_callback("resize", resize_cb.as_ref().unchecked_ref())?;
            resize_cb.forget();
        }
        {
            // One extra pass on the next frame to catch initial layout.
            let inner_clone = inner.clone();
            let window = web_sys::window().unwrap();
            let raf = Closure::<dyn FnMut(f64)>::wrap(Box::new(move |_| {
                resize_canvas_to_parent(&inner_clone);
            }));
            window
                .request_animation_frame(raf.as_ref().unchecked_ref())
                .map_err(|_| JsValue::from_str("failed to schedule resize"))?;
            raf.forget();
        }

        setup_mouse_events(&inner)?;
        start_render_loop(inner.clone());
        init_feeds(inner.clone());

        Ok(ChartHandle { inner })
    }

    /// Resize chart (call from JS on window resize).
    pub fn resize(&self, width: f64, height: f64) {
        let mut inner = self.inner.borrow_mut();
        inner.chart.resize(width, height);
    }

    /// Change main series style: "candles", "ohlc", "line", "area", "bars".
    pub fn set_style(&self, style: &str) {
        let mut inner = self.inner.borrow_mut();
        inner.chart.set_style(style);
    }

    /// Set price scale mode: "linear" or "log".
    pub fn set_scale(&self, scale: &str) {
        let mut inner = self.inner.borrow_mut();
        let log = matches!(scale.to_ascii_lowercase().as_str(), "log" | "logarithmic");
        if inner.chart.log_scale != log {
            inner.chart.log_scale = log;
            inner.chart.dirty = true;
            inner.chart.dirty_price_panel = true;
            inner.chart.dirty_indicator_panes = true;
        }
    }

    /// Update pane weights for price + indicator panes.
    /// Accepts parallel arrays of pane IDs and weights for JS-friendly FFI.
    pub fn set_pane_layout(&self, price_weight: f64, pane_ids: Vec<u32>, pane_weights: Vec<f64>) {
        let mut inner = self.inner.borrow_mut();
        let panes: Vec<(u32, f64)> = pane_ids.into_iter().zip(pane_weights).collect();
        inner.chart.set_pane_layout(price_weight, panes.as_slice());
    }

    /// Add an indicator.
    ///
    /// `kind`: "sma", "ema", "rsi", "macd", "bbands", "atr".
    /// `params_json`: kind-specific params (see below).
    /// `output`: "overlay" or "pane" (empty string => default per indicator).
    /// `pane_id`: for separate panes; indicators with same pane_id share a pane.
    ///
    /// Example params:
    ///  SMA:   {"period":20,"source":"close"}
    ///  EMA:   {"period":20,"source":"close"}
    ///  RSI:   {"period":14,"source":"close"}
    ///  MACD:  {"fast":12,"slow":26,"signal":9,"source":"close"}
    ///  BBands:{"period":20,"stddev":2.0,"source":"close"}
    ///  ATR:   {"period":14}
    pub fn add_indicator(
        &self,
        kind: &str,
        params_json: &str,
        output: &str,
        pane_id: Option<u32>,
    ) -> Result<u32, JsValue> {
        let kind = IndicatorKind::from_str(kind)
            .map_err(|_| JsValue::from_str("unknown indicator kind"))?;
        let params = parse_indicator_params(kind, params_json)?;
        let output_kind = if output.is_empty() {
            default_output_for(kind)
        } else {
            OutputKind::from_str(output).unwrap_or_else(|_| default_output_for(kind))
        };

        let config = IndicatorConfig::with_default_styles(kind, params, output_kind, pane_id);

        let mut inner = self.inner.borrow_mut();
        let id = inner.chart.add_indicator(config);
        Ok(id as u32)
    }

    /// Add indicator using a serialized IndicatorConfig (used by UI shell restore).
    pub fn add_indicator_from_config(&self, config_json: &str) -> Result<u32, JsValue> {
        let cfg: IndicatorConfig =
            serde_json::from_str(config_json).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let mut inner = self.inner.borrow_mut();
        let id = inner.chart.add_indicator(cfg);
        Ok(id as u32)
    }

    pub fn remove_indicator(&self, id: u32) {
        let mut inner = self.inner.borrow_mut();
        inner.chart.remove_indicator(id as IndicatorId);
    }

    pub fn clear_indicators(&self) {
        let mut inner = self.inner.borrow_mut();
        inner.chart.clear_indicators();
    }

    /// Override indicator styles from JS.
    ///
    /// styles_json: e.g.
    ///   [
    ///      {"color":"#ff0000","width":1.5,"pattern":"solid"},
    ///      {"color":"#00ff00","width":1.0,"pattern":"dashed"}
    ///   ]
    pub fn set_indicator_styles(&self, id: u32, styles_json: &str) -> Result<(), JsValue> {
        let styles: Vec<LineStyle> =
            serde_json::from_str(styles_json).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let mut inner = self.inner.borrow_mut();
        inner.chart.set_indicator_styles(id as IndicatorId, styles);
        Ok(())
    }

    /// Undo last indicator/drawing configuration change.
    pub fn undo(&self) {
        let mut inner = self.inner.borrow_mut();
        inner.chart.undo();
    }

    /// Redo last undone indicator/drawing configuration change.
    pub fn redo(&self) {
        let mut inner = self.inner.borrow_mut();
        inner.chart.redo();
    }

    pub fn can_undo(&self) -> bool {
        let inner = self.inner.borrow();
        inner.chart.can_undo()
    }

    pub fn can_redo(&self) -> bool {
        let inner = self.inner.borrow();
        inner.chart.can_redo()
    }

    /// Add a horizontal line drawing on price panel.
    pub fn add_horizontal_line(&self, price: f64, color: &str, width: f64) -> u32 {
        let mut inner = self.inner.borrow_mut();
        let id = inner
            .chart
            .add_horizontal_line(price, color.to_string(), width);
        id as u32
    }

    pub fn add_vertical_line(&self, ts: i64, color: &str, width: f64) -> u32 {
        let mut inner = self.inner.borrow_mut();
        let id = inner.chart.add_vertical_line(ts, color.to_string(), width);
        id as u32
    }

    pub fn add_trend_line(
        &self,
        ts1: i64,
        price1: f64,
        ts2: i64,
        price2: f64,
        color: &str,
        width: f64,
    ) -> u32 {
        let mut inner = self.inner.borrow_mut();
        let id = inner
            .chart
            .add_trend_line(ts1, price1, ts2, price2, color.to_string(), width);
        id as u32
    }

    pub fn add_rectangle(
        &self,
        ts1: i64,
        price1: f64,
        ts2: i64,
        price2: f64,
        color: &str,
        width: f64,
    ) -> u32 {
        let mut inner = self.inner.borrow_mut();
        let id = inner
            .chart
            .add_rectangle(ts1, price1, ts2, price2, color.to_string(), width);
        id as u32
    }

    pub fn remove_drawing(&self, id: u32) {
        let mut inner = self.inner.borrow_mut();
        inner.chart.remove_drawing(id as u64);
    }

    pub fn clear_drawings(&self) {
        let mut inner = self.inner.borrow_mut();
        inner.chart.clear_drawings();
    }

    /// Replace script overlay series from JSON.
    pub fn set_script_series(&self, json: &str) -> Result<(), JsValue> {
        let series: Vec<ScriptSeries> =
            serde_json::from_str(json).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let mut inner = self.inner.borrow_mut();
        inner.chart.set_script_series(series);
        Ok(())
    }

    pub fn set_orders(&self, json: &str) -> Result<(), JsValue> {
        let orders: Vec<OrderVisual> =
            serde_json::from_str(json).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let mut inner = self.inner.borrow_mut();
        inner.chart.set_orders(orders);
        Ok(())
    }

    pub fn set_positions(&self, json: &str) -> Result<(), JsValue> {
        let positions: Vec<OrderVisual> =
            serde_json::from_str(json).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let mut inner = self.inner.borrow_mut();
        inner.chart.set_positions(positions);
        Ok(())
    }

    pub fn set_alerts(&self, json: &str) -> Result<(), JsValue> {
        let alerts: Vec<AlertVisual> =
            serde_json::from_str(json).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let mut inner = self.inner.borrow_mut();
        inner.chart.set_alerts(alerts);
        Ok(())
    }

    /// Subscribe to chart events (click, crosshair move, etc.).
    ///
    /// The callback receives a JSON string payload.
    pub fn subscribe_events(&self, callback: &Function) -> u32 {
        let mut inner = self.inner.borrow_mut();
        inner.add_subscription(callback.clone())
    }

    pub fn unsubscribe_events(&self, id: u32) {
        let mut inner = self.inner.borrow_mut();
        inner.remove_subscription(id);
    }

    /// Attach default HTTP+WebSocket feed (if detached).
    pub fn attach_feeds(&self) {
        let mut inner = self.inner.borrow_mut();
        if inner.feeds_live {
            return;
        }
        inner.feeds_live = true;
        drop(inner);
        init_feeds(self.inner.clone());
    }

    /// Detach feeds (stop processing history/live updates).
    pub fn detach_feeds(&self) {
        let mut inner = self.inner.borrow_mut();
        inner.feeds_live = false;
    }

    /// Change symbol and reset feeds.
    pub fn set_symbol(&self, symbol: &str) {
        {
            let mut inner = self.inner.borrow_mut();
            if inner.symbol == symbol {
                return;
            }
            inner.symbol = symbol.to_string();
            inner.feeds_live = false;

            let tf = inner.base_timeframe;
            inner.chart.feed_store = Rc::new(RefCell::new(FeedStore::new(symbol.to_string(), tf)));
            inner.chart.indicator_mgr = IndicatorManager::new(tf);
            inner.chart.drawings.clear();
            inner.chart.undo_stack.clear();
            inner.chart.redo_stack.clear();
            inner.chart.dirty = true;
            inner.chart.dirty_price_panel = true;
            inner.chart.dirty_indicator_panes = true;
        }
        init_feeds(self.inner.clone());
    }

    /// Change data provider ("alpha", "yahoo", "coinbase") and restart feeds.
    pub fn set_provider(&self, provider: &str) {
        let Some(norm) = normalize_provider(provider) else {
            return;
        };
        {
            let mut inner = self.inner.borrow_mut();
            if inner.provider_override.as_deref() == Some(norm.as_str()) {
                return;
            }
            inner.provider_override = Some(norm);
            inner.feeds_live = false;

            let tf = inner.base_timeframe;
            let symbol = inner.symbol.clone();
            inner.chart.feed_store = Rc::new(RefCell::new(FeedStore::new(symbol.clone(), tf)));
            inner.chart.indicator_mgr = IndicatorManager::new(tf);
            inner.chart.dirty = true;
            inner.chart.dirty_price_panel = true;
            inner.chart.dirty_indicator_panes = true;
        }
        init_feeds(self.inner.clone());
    }

    /// Apply a linked view window from another chart.
    pub fn sync_view(&self, start: i64, end: i64) -> Result<(), JsValue> {
        if start >= end {
            return Err(JsValue::from_str("invalid view range"));
        }
        let mut inner = self.inner.borrow_mut();
        inner.chart.set_visible_range(start, end);
        Ok(())
    }

    /// Display crosshair at a given timestamp/price without recomputing data.
    pub fn show_crosshair(&self, ts: i64, price: f64) {
        let mut inner = self.inner.borrow_mut();
        inner.chart.set_crosshair(ts, price);
    }

    /// Change timeframe and reset feeds + indicator store.
    pub fn set_timeframe(&self, timeframe: &str) -> Result<(), JsValue> {
        let tf =
            TimeFrame::from_str(timeframe).ok_or_else(|| JsValue::from_str("invalid timeframe"))?;

        {
            let mut inner = self.inner.borrow_mut();
            if inner.base_timeframe == tf {
                return Ok(());
            }
            inner.base_timeframe = tf;
            inner.feeds_live = false;

            let symbol = inner.symbol.clone();
            inner.chart.timeframe = tf;
            inner.chart.feed_store = Rc::new(RefCell::new(FeedStore::new(symbol.clone(), tf)));
            inner.chart.indicator_mgr = IndicatorManager::new(tf);
            inner.chart.drawings.clear();
            inner.chart.undo_stack.clear();
            inner.chart.redo_stack.clear();
            inner.chart.dirty = true;
            inner.chart.dirty_price_panel = true;
            inner.chart.dirty_indicator_panes = true;
        }
        init_feeds(self.inner.clone());
        Ok(())
    }

    pub fn destroy(&self) {
        let mut inner = self.inner.borrow_mut();
        inner.feeds_live = false;
        inner.chart.destroyed = true;
        inner.subscribers.clear();
    }
}

// ---------- parameter parsing from JSON -------------------------------------

#[derive(Deserialize)]
struct SmaParamsJson {
    period: Option<usize>,
    source: Option<String>,
}

#[derive(Deserialize)]
struct EmaParamsJson {
    period: Option<usize>,
    source: Option<String>,
}

#[derive(Deserialize)]
struct RsiParamsJson {
    period: Option<usize>,
    source: Option<String>,
}

#[derive(Deserialize)]
struct MacdParamsJson {
    fast: Option<usize>,
    slow: Option<usize>,
    signal: Option<usize>,
    source: Option<String>,
}

#[derive(Deserialize)]
struct BbandsParamsJson {
    period: Option<usize>,
    stddev: Option<f64>,
    source: Option<String>,
}

#[derive(Deserialize)]
struct AtrParamsJson {
    period: Option<usize>,
}

#[derive(Deserialize)]
struct StochParamsJson {
    k_period: Option<usize>,
    d_period: Option<usize>,
}

#[derive(Deserialize)]
struct VwapParamsJson {
    reset_each_day: Option<bool>,
}

#[derive(Deserialize)]
struct CciParamsJson {
    period: Option<usize>,
    source: Option<String>,
    constant: Option<f64>,
}

#[derive(Deserialize)]
struct VwmoParamsJson {
    period: Option<usize>,
    source: Option<String>,
}
fn parse_source_field(from: Option<String>) -> SourceField {
    from.as_deref()
        .and_then(|s| SourceField::from_str(s).ok())
        .unwrap_or(SourceField::Close)
}

fn parse_indicator_params(
    kind: IndicatorKind,
    params_json: &str,
) -> Result<IndicatorParams, JsValue> {
    let params = match kind {
        IndicatorKind::Sma => {
            let p: SmaParamsJson = if params_json.is_empty() {
                SmaParamsJson {
                    period: None,
                    source: None,
                }
            } else {
                serde_json::from_str(params_json).map_err(|e| JsValue::from_str(&e.to_string()))?
            };
            let period = p.period.unwrap_or(20);
            let source = parse_source_field(p.source);
            IndicatorParams::Sma { period, source }
        }
        IndicatorKind::Ema => {
            let p: EmaParamsJson = if params_json.is_empty() {
                EmaParamsJson {
                    period: None,
                    source: None,
                }
            } else {
                serde_json::from_str(params_json).map_err(|e| JsValue::from_str(&e.to_string()))?
            };
            let period = p.period.unwrap_or(20);
            let source = parse_source_field(p.source);
            IndicatorParams::Ema { period, source }
        }
        IndicatorKind::Rsi => {
            let p: RsiParamsJson = if params_json.is_empty() {
                RsiParamsJson {
                    period: None,
                    source: None,
                }
            } else {
                serde_json::from_str(params_json).map_err(|e| JsValue::from_str(&e.to_string()))?
            };
            let period = p.period.unwrap_or(14);
            let source = parse_source_field(p.source);
            IndicatorParams::Rsi { period, source }
        }
        IndicatorKind::Macd => {
            let p: MacdParamsJson = if params_json.is_empty() {
                MacdParamsJson {
                    fast: None,
                    slow: None,
                    signal: None,
                    source: None,
                }
            } else {
                serde_json::from_str(params_json).map_err(|e| JsValue::from_str(&e.to_string()))?
            };
            let fast = p.fast.unwrap_or(12);
            let slow = p.slow.unwrap_or(26);
            let signal = p.signal.unwrap_or(9);
            let source = parse_source_field(p.source);
            IndicatorParams::Macd {
                fast,
                slow,
                signal,
                source,
            }
        }
        IndicatorKind::Bbands => {
            let p: BbandsParamsJson = if params_json.is_empty() {
                BbandsParamsJson {
                    period: None,
                    stddev: None,
                    source: None,
                }
            } else {
                serde_json::from_str(params_json).map_err(|e| JsValue::from_str(&e.to_string()))?
            };
            let period = p.period.unwrap_or(20);
            let stddev = p.stddev.unwrap_or(2.0);
            let source = parse_source_field(p.source);
            IndicatorParams::Bbands {
                period,
                stddev,
                source,
            }
        }
        IndicatorKind::Atr => {
            let p: AtrParamsJson = if params_json.is_empty() {
                AtrParamsJson { period: None }
            } else {
                serde_json::from_str(params_json).map_err(|e| JsValue::from_str(&e.to_string()))?
            };
            let period = p.period.unwrap_or(14);
            IndicatorParams::Atr { period }
        }
        IndicatorKind::Stoch => {
            let p: StochParamsJson = if params_json.is_empty() {
                StochParamsJson {
                    k_period: None,
                    d_period: None,
                }
            } else {
                serde_json::from_str(params_json).map_err(|e| JsValue::from_str(&e.to_string()))?
            };
            IndicatorParams::Stoch {
                k_period: p.k_period.unwrap_or(14),
                d_period: p.d_period.unwrap_or(3),
            }
        }
        IndicatorKind::Vwap => {
            let p: VwapParamsJson = if params_json.is_empty() {
                VwapParamsJson {
                    reset_each_day: None,
                }
            } else {
                serde_json::from_str(params_json).map_err(|e| JsValue::from_str(&e.to_string()))?
            };
            IndicatorParams::Vwap {
                reset_each_day: p.reset_each_day.unwrap_or(true),
            }
        }
        IndicatorKind::Cci => {
            let p: CciParamsJson = if params_json.is_empty() {
                CciParamsJson {
                    period: None,
                    source: None,
                    constant: None,
                }
            } else {
                serde_json::from_str(params_json).map_err(|e| JsValue::from_str(&e.to_string()))?
            };
            IndicatorParams::Cci {
                period: p.period.unwrap_or(20),
                source: parse_source_field(p.source),
                constant: p.constant.unwrap_or(0.015),
            }
        }
        IndicatorKind::Vwmo => {
            let p: VwmoParamsJson = if params_json.is_empty() {
                VwmoParamsJson {
                    period: None,
                    source: None,
                }
            } else {
                serde_json::from_str(params_json).map_err(|e| JsValue::from_str(&e.to_string()))?
            };
            IndicatorParams::Vwmo {
                period: p.period.unwrap_or(20),
                source: parse_source_field(p.source),
            }
        }
    };
    Ok(params)
}
