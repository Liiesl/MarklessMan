use std::path::{Path, PathBuf};

use clap::{Parser, ValueEnum};
use image::{Rgb, RgbImage, imageops::FilterType};
use ndarray::Array4;
use ort::{ep, session::Session, value::Tensor};

/// MarklessMan CLI: det / clean / det+clean via ORT DirectML.
#[derive(ValueEnum, Clone, Copy, Debug, Default)]
enum Mode {
    Det,
    #[default]
    Clean,
    Full,
}

#[derive(Parser, Debug)]
struct Args {
    /// Cleaner ONNX (e.g. weights/marklessman-clean-slbr_256_int8.onnx)
    #[arg(long)]
    model: PathBuf,
    /// Input image file or directory (dir mode mirrors py --images folder runs)
    #[arg(long)]
    input: PathBuf,
    /// Output image file or directory
    #[arg(long)]
    output: PathBuf,
    /// det | clean | full
    #[arg(long, value_enum, default_value_t = Mode::Clean)]
    mode: Mode,
    /// Detector ONNX (e.g. weights/marklessman-det.onnx); required for det/full
    #[arg(long)]
    det_model: Option<PathBuf>,
    /// Detector confidence threshold (mirrors detect_onnx CONF_DEFAULT)
    #[arg(long, default_value_t = 0.25)]
    conf: f32,
    /// Detector NMS IoU (mirrors detect_onnx IOU_DEFAULT)
    #[arg(long, default_value_t = 0.5)]
    iou: f32,
    /// DirectML device id (default adapter when omitted)
    #[arg(long)]
    device_id: Option<i32>,
    /// Force CPU-only (skip DirectML registration)
    #[arg(long, default_value_t = false)]
    cpu: bool,
    /// Run cleaner on DirectML (10x faster than CPU, bit-exact with graph fusion disabled)
    #[arg(long, default_value_t = false)]
    clean_dml: bool,
    /// Force deterministic compute (DML accuracy knob; slower but may recover quality)
    #[arg(long, default_value_t = false)]
    deterministic: bool,
    /// CPU intra-op threads (default: all cores). Tunes CPU cleaner speed.
    #[arg(long)]
    intra_threads: Option<usize>,
    /// Benchmark cleaner N iters on a real 256 crop and exit (0 = disabled)
    #[arg(long, default_value_t = 0)]
    bench_clean: usize,
    /// Graph optimization level for cleaner session: disable|l1|all (default l1)
    #[arg(long, default_value = "l1")]
    opt_level: String,
    /// Enable ORT profiling for cleaner session, write profile JSON with this prefix
    #[arg(long)]
    profile: Option<PathBuf>,
    /// Context padding around each det box; the padded rect is what gets tiled (mirrors test/tile_vs_resize --pad)
    #[arg(long, default_value_t = 64)]
    pad: i32,
    /// Tile stride for native-256 tiling; 192 = 64px overlap (mirrors --stride)
    #[arg(long, default_value_t = 192)]
    stride: i32,
    /// Feather ramp px used inside the tile mosaic (mirrors --feather)
    #[arg(long, default_value_t = 32)]
    tile_feather: usize,
    /// Verbose DML runtime inspection: ORT logs + session/EP dump + per-infer timing
    #[arg(long, default_value_t = false)]
    verbose: bool,
}

static VERBOSE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

fn verbose_on() -> bool {
    *VERBOSE.get().unwrap_or(&false)
}

/// eprintln session I/O + EP info after commit. Shows what DML actually loaded.
fn log_session(tag: &str, session: &Session, ep: &str) {
    if !verbose_on() {
        return;
    }
    eprintln!("[verbose] {tag}: ep={ep}");
    for i in session.inputs() {
        eprintln!("[verbose] {tag} input: name={} dtype={:?}", i.name(), i.dtype());
    }
    for o in session.outputs() {
        eprintln!("[verbose] {tag} output: name={} dtype={:?}", o.name(), o.dtype());
    }
}

fn build_session(model: &Path, cpu: bool, device_id: Option<i32>) -> ort::Result<Session> {
    build_session_full(
        model,
        cpu,
        device_id,
        false,
        None,
        ort::session::builder::GraphOptimizationLevel::Level1,
        None,
        verbose_on(),
    )
}

fn parse_opt_level(s: &str) -> ort::session::builder::GraphOptimizationLevel {
    use ort::session::builder::GraphOptimizationLevel as L;
    match s.to_ascii_lowercase().as_str() {
        "disable" | "off" | "0" => L::Disable,
        "all" | "3" | "l3" => L::All,
        _ => L::Level1,
    }
}

fn build_session_full(
    model: &Path,
    cpu: bool,
    device_id: Option<i32>,
    deterministic: bool,
    intra_threads: Option<usize>,
    opt: ort::session::builder::GraphOptimizationLevel,
    profile: Option<PathBuf>,
    verbose: bool,
) -> ort::Result<Session> {
    // DML-first; ORT falls back to CPU per-op automatically (DequantizeLinear etc).
    // NOTE: BuilderResult carries SessionBuilder inside the error so it is not
    // Send/Sync; use recover() instead of `?` into anyhow here.
    let builder = Session::builder()?;
    // Verbose: surface ORT's own load/execution logs (EP fallback errors etc).
    let builder = if verbose {
        use std::sync::Arc;
        let sink: ort::logging::LoggerFunction = Arc::new(|lvl, _cat, id, loc, msg| {
            eprintln!("[ort:{lvl:?}] {id} {loc}: {msg}");
        });
        builder
            .with_logger(sink)
            .unwrap_or_else(|e| {
                eprintln!("logger unsupported ({e:?}), continuing");
                e.recover()
            })
            .with_log_level(ort::logging::LogLevel::Verbose)
            .unwrap_or_else(|e| {
                eprintln!("log-level unsupported ({e:?}), continuing");
                e.recover()
            })
            .with_log_verbosity(1)
            .unwrap_or_else(|e| {
                eprintln!("log-verbosity unsupported ({e:?}), continuing");
                e.recover()
            })
    } else {
        builder
    };
    // Deterministic compute is a DML accuracy knob (slower, potentially cleaner output).
    let builder = if deterministic {
        builder
            .with_deterministic_compute(true)
            .unwrap_or_else(|e| {
                eprintln!("deterministic flag unsupported ({e:?}), continuing");
                e.recover()
            })
    } else {
        builder
    };
    let builder = builder
        .with_optimization_level(opt)
        .unwrap_or_else(|e| {
            eprintln!("opt-level unsupported ({e:?}), continuing");
            e.recover()
        });
    let builder = match intra_threads {
        Some(n) => builder.with_intra_threads(n).unwrap_or_else(|e| {
            eprintln!("intra-threads unsupported ({e:?}), continuing");
            e.recover()
        }),
        None => builder,
    };
    let builder = match profile {
        Some(p) => builder.with_profiling(p).unwrap_or_else(|e| {
            eprintln!("profiling unsupported ({e:?}), continuing");
            e.recover()
        }),
        None => builder,
    };
    let mut builder = if cpu {
        if verbose {
            eprintln!("[verbose] {}: CPU session (no EP registration)", model.display());
        }
        builder
    } else {
        if verbose {
            eprintln!(
                "[verbose] {}: registering DML (device_id={:?}, deterministic={deterministic})",
                model.display(),
                device_id
            );
        }
        // Root-cause fix for the DML mask drift: the DML graph fusion transformer
        // mis-compiles the mask branch (mask logits attenuated ~1.56x with ~+1.5
        // bias, identically in fp32 and int8). With fusion disabled, DML output
        // is bit-exact vs CPU (mask meanabs ~1e-7 over 18 boxes) at ~10% slower
        // pace. Metacommands, opt-level and deterministic compute were all
        // ruled out as causes (test/knob_dml.py).
        let builder = builder
            .with_config_entry("ep.dml.disable_graph_fusion", "1")
            .unwrap_or_else(|e| {
                eprintln!("dml no-fusion config unsupported ({e:?}), continuing");
                e.recover()
            });
        if verbose {
            eprintln!("[verbose] DML graph fusion disabled (mask-parity fix)");
        }
        let dml = match device_id {
            Some(id) => ep::DirectML::default().with_device_id(id),
            None => ep::DirectML::default(),
        };
        let builder = builder
            .with_execution_providers([dml.build()])
            .unwrap_or_else(|e| {
                eprintln!("DML register failed ({e:?}), falling back to CPU");
                e.recover()
            });
        if verbose {
            use ort::ep::directml::DMLSessionBuilderExt as _;
            match builder.dml_device() {
                Some(Ok(ptr)) => eprintln!("[verbose] DML device ptr={ptr:?}"),
                Some(Err(e)) => eprintln!("[verbose] DML device query failed: {e:?}"),
                None => eprintln!("[verbose] DML API unavailable in this build"),
            }
            match builder.d3d_command_queue() {
                Some(Ok(ptr)) => eprintln!("[verbose] D3D12 queue ptr={ptr:?}"),
                Some(Err(e)) => eprintln!("[verbose] D3D12 queue query failed: {e:?}"),
                None => eprintln!("[verbose] D3D12 queue unavailable"),
            }
        }
        builder
    };
    let session = builder.commit_from_file(model)?;
    log_session(
        &format!("load {}", model.file_name().and_then(|s| s.to_str()).unwrap_or("?")),
        &session,
        if cpu { "cpu" } else { "dml" },
    );
    Ok(session)
}

fn cpu_threads(default: Option<usize>) -> Option<usize> {
    match default {
        Some(n) => Some(n),
        None => std::thread::available_parallelism().ok().map(|n| n.get()),
    }
}

fn is_image(path: &Path) -> bool {
    match path.extension().and_then(|e| e.to_str()) {
        Some(e) => matches!(
            e.to_ascii_lowercase().as_str(),
            "jpg" | "jpeg" | "png" | "webp" | "bmp"
        ),
        None => false,
    }
}

fn ensure_parent(path: &Path) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

/// Quality-matched save: Python uses cv2 quality 90 for cleaned pages,
/// 88 for det boxes, cv2-default (~95) for before/after crops.
/// `image` default JPEG quality (75) looks visibly worse, so set it explicitly.
fn save_image(path: &Path, img: &RgbImage, jpg_quality: u8) -> anyhow::Result<()> {
    ensure_parent(path)?;
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if ext == "jpg" || ext == "jpeg" {
        let f = std::fs::File::create(path)?;
        let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(f, jpg_quality);
        enc.encode_image(img)?;
    } else {
        img.save(path)?;
    }
    Ok(())
}

// ---- cleaner (s256: 256 in, cubic back up) ----

fn run_clean(session: &mut Session, small_256: &RgbImage) -> anyhow::Result<RgbImage> {
    let mut input_array = Array4::<f32>::zeros((1, 3, 256, 256));
    for y in 0..256 {
        for x in 0..256 {
            let p = small_256.get_pixel(x, y);
            input_array[[0, 0, y as usize, x as usize]] = p[0] as f32 / 255.0;
            input_array[[0, 1, y as usize, x as usize]] = p[1] as f32 / 255.0;
            input_array[[0, 2, y as usize, x as usize]] = p[2] as f32 / 255.0;
        }
    }
    let tensor = Tensor::from_array(input_array)?;
    let t0 = verbose_on().then(std::time::Instant::now);
    let outputs = session.run(ort::inputs![tensor])?;
    if let Some(t0) = t0 {
        eprintln!("[verbose] run_clean session.run: {:.1}ms", t0.elapsed().as_secs_f64() * 1e3);
    }
    let out_view = outputs[0].try_extract_array::<f32>()?;
    let data = out_view
        .as_slice()
        .ok_or_else(|| anyhow::anyhow!("non-contiguous model output"))?;
    anyhow::ensure!(
        data.len() == 1 * 3 * 256 * 256,
        "unexpected output len {}, want 196608",
        data.len()
    );
    let mut out = RgbImage::new(256, 256);
    for y in 0..256 {
        for x in 0..256 {
            let r = (data[(0 * 3 + 0) * 256 * 256 + y as usize * 256 + x as usize].clamp(0.0, 1.0)
                * 255.0)
                .round() as u8;
            let g = (data[(0 * 3 + 1) * 256 * 256 + y as usize * 256 + x as usize].clamp(0.0, 1.0)
                * 255.0)
                .round() as u8;
            let b = (data[(0 * 3 + 2) * 256 * 256 + y as usize * 256 + x as usize].clamp(0.0, 1.0)
                * 255.0)
                .round() as u8;
            out.put_pixel(x, y, Rgb([r, g, b]));
        }
    }
    Ok(out)
}

fn clean_whole(
    session: &mut Session,
    input: &Path,
    output: &Path,
) -> anyhow::Result<()> {
    let src = image::open(input)?.to_rgb8();
    let (ow, oh) = src.dimensions();
    let small = resize_bilinear_cv2(&src, 256, 256);
    let fixed = run_clean(session, &small)?;
    let out_full = image::imageops::resize(&fixed, ow, oh, FilterType::CatmullRom);
    save_image(output, &out_full, 90)?;
    eprintln!("cleaned {ow}x{oh} {} -> {}", input.display(), output.display());
    Ok(())
}

// ---- detector (bare-ONNX YOLO, mirrors detect_onnx.py) ----

const DET_SIZE: u32 = 640;
const DET_PAD: u8 = 114;

/// Bilinear resize matching OpenCV INTER_LINEAR (cv2.resize).
/// `image::imageops::Triangle` aliases tiny strokes away on strong
/// downscales (e.g. 800x2600 -> 197x640); cv2 preserves them, which is
/// what the detector was trained/evaluated against (detect_onnx.py).
fn resize_bilinear_cv2(img: &RgbImage, nw: u32, nh: u32) -> RgbImage {
    let (w, h) = img.dimensions();
    if w == nw && h == nh {
        return img.clone();
    }
    let sx = w as f32 / nw as f32;
    let sy = h as f32 / nh as f32;
    let mut out = RgbImage::new(nw, nh);
    for y in 0..nh {
        let fy = (y as f32 + 0.5) * sy - 0.5;
        let y0 = fy.floor() as i32;
        let wy = fy - y0 as f32;
        let ya = y0.clamp(0, h as i32 - 1) as u32;
        let yb = (y0 + 1).clamp(0, h as i32 - 1) as u32;
        for x in 0..nw {
            let fx = (x as f32 + 0.5) * sx - 0.5;
            let x0 = fx.floor() as i32;
            let wx = fx - x0 as f32;
            let xa = x0.clamp(0, w as i32 - 1) as u32;
            let xb = (x0 + 1).clamp(0, w as i32 - 1) as u32;
            let p00 = img.get_pixel(xa, ya);
            let p10 = img.get_pixel(xb, ya);
            let p01 = img.get_pixel(xa, yb);
            let p11 = img.get_pixel(xb, yb);
            let mut px = [0u8; 3];
            for c in 0..3 {
                let v = p00[c] as f32 * (1.0 - wx) * (1.0 - wy)
                    + p10[c] as f32 * wx * (1.0 - wy)
                    + p01[c] as f32 * (1.0 - wx) * wy
                    + p11[c] as f32 * wx * wy;
                px[c] = v.round().clamp(0.0, 255.0) as u8;
            }
            out.put_pixel(x, y, Rgb(px));
        }
    }
    out
}

fn letterbox(img: &RgbImage) -> (Array4<f32>, f32, f32, f32) {
    let (w, h) = img.dimensions();
    let s = (DET_SIZE as f32 / h as f32).min(DET_SIZE as f32 / w as f32);
    let nw = (w as f32 * s).round() as u32;
    let nh = (h as f32 * s).round() as u32;
    let resized = resize_bilinear_cv2(img, nw, nh);
    let pad_h = DET_SIZE - nh;
    let pad_w = DET_SIZE - nw;
    let top = pad_h / 2;
    let left = pad_w / 2;
    let mut canvas = RgbImage::from_pixel(DET_SIZE, DET_SIZE, Rgb([DET_PAD, DET_PAD, DET_PAD]));
    image::imageops::replace(&mut canvas, &resized, left as i64, top as i64);
    let mut arr = Array4::<f32>::zeros((1, 3, DET_SIZE as usize, DET_SIZE as usize));
    for y in 0..DET_SIZE {
        for x in 0..DET_SIZE {
            let p = canvas.get_pixel(x, y);
            arr[[0, 0, y as usize, x as usize]] = p[0] as f32 / 255.0;
            arr[[0, 1, y as usize, x as usize]] = p[1] as f32 / 255.0;
            arr[[0, 2, y as usize, x as usize]] = p[2] as f32 / 255.0;
        }
    }
    (arr, s, left as f32, top as f32)
}

fn nms_greedy(boxes: &[(f32, f32, f32, f32)], scores: &[f32], iou_thr: f32) -> Vec<usize> {
    if boxes.is_empty() {
        return Vec::new();
    }
    let areas: Vec<f32> = boxes
        .iter()
        .map(|(x0, y0, x1, y1)| (x1 - x0).max(0.0) * (y1 - y0).max(0.0))
        .collect();
    let mut order: Vec<usize> = (0..boxes.len()).collect();
    order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
    let mut keep = Vec::new();
    while let Some(&i) = order.first() {
        keep.push(i);
        let (ax0, ay0, ax1, ay1) = boxes[i];
        let mut rest = Vec::with_capacity(order.len().saturating_sub(1));
        for &j in &order[1..] {
            let (bx0, by0, bx1, by1) = boxes[j];
            let inter = (ax1.min(bx1) - ax0.max(bx0)).max(0.0)
                * (ay1.min(by1) - ay0.max(by0)).max(0.0);
            let ovr = inter / (areas[i] + areas[j] - inter).max(1e-9);
            if ovr <= iou_thr {
                rest.push(j);
            }
        }
        order = rest;
    }
    keep
}

/// Raw detect: xyxy float pixels + scores (mirrors OnnxDetector.raw).
fn detect_raw(
    session: &mut Session,
    img: &RgbImage,
    conf: f32,
    iou: f32,
) -> anyhow::Result<(Vec<(f32, f32, f32, f32)>, Vec<f32>)> {
    let (w, h) = img.dimensions();
    let (arr, s, px, py) = letterbox(img);
    let tensor = Tensor::from_array(arr)?;
    let t0 = verbose_on().then(std::time::Instant::now);
    let outputs = session.run(ort::inputs![tensor])?;
    if let Some(t0) = t0 {
        eprintln!("[verbose] detect_raw session.run: {:.1}ms", t0.elapsed().as_secs_f64() * 1e3);
    }
    let view = outputs[0].try_extract_array::<f32>()?;
    let shape = view.shape().to_vec();
    // Expect [1, C, 8400] (or [C, 8400]): row0..3 = cxcywh in 640 space, row4.. = scores.
    let (c, n, stride_batch) = match shape.as_slice() {
        [_, c, n] => (*c, *n, c * n),
        [c, n] => (*c, *n, 0),
        _ => anyhow::bail!("unexpected det output shape {shape:?}"),
    };
    let _ = stride_batch;
    let data = view
        .as_slice()
        .ok_or_else(|| anyhow::anyhow!("non-contiguous det output"))?;
    let at = |ch: usize, i: usize| -> f32 {
        if shape.len() == 3 {
            data[(0 * c + ch) * n + i]
        } else {
            data[ch * n + i]
        }
    };
    // Per-anchor (cx, cy, w, h, score); multi-class fallback = best class wins.
    let mut boxes = Vec::new();
    let mut scores = Vec::new();
    for i in 0..n {
        let (score, cx, cy, bw, bh) = if c > 5 {
            let mut best = 0.0f32;
            for ch in 4..c {
                best = best.max(at(ch, i));
            }
            (best, at(0, i), at(1, i), at(2, i), at(3, i))
        } else {
            (at(4, i), at(0, i), at(1, i), at(2, i), at(3, i))
        };
        if score < conf {
            continue;
        }
        let x0 = ((cx - bw / 2.0 - px) / s).clamp(0.0, w as f32);
        let y0 = ((cy - bh / 2.0 - py) / s).clamp(0.0, h as f32);
        let x1 = ((cx + bw / 2.0 - px) / s).clamp(0.0, w as f32);
        let y1 = ((cy + bh / 2.0 - py) / s).clamp(0.0, h as f32);
        boxes.push((x0, y0, x1, y1));
        scores.push(score);
    }
    let keep = nms_greedy(&boxes, &scores, iou);
    Ok((
        keep.iter().map(|&k| boxes[k]).collect(),
        keep.iter().map(|&k| scores[k]).collect(),
    ))
}

/// Int (x, y, w, h) boxes (mirrors OnnxDetector.detect).
fn detect_boxes(
    session: &mut Session,
    img: &RgbImage,
    conf: f32,
    iou: f32,
) -> anyhow::Result<Vec<(i32, i32, i32, i32)>> {
    let (boxes, _) = detect_raw(session, img, conf, iou)?;
    Ok(boxes
        .into_iter()
        .map(|(x0, y0, x1, y1)| {
            (
                x0 as i32,
                y0 as i32,
                0.max((x1 - x0) as i32),
                0.max((y1 - y0) as i32),
            )
        })
        .collect())
}

fn draw_box(img: &mut RgbImage, x0: i32, y0: i32, x1: i32, y1: i32, color: Rgb<u8>, thick: u32) {
    let (w, h) = (img.width() as i32, img.height() as i32);
    for t in 0..thick as i32 {
        let (xa, ya, xb, yb) = (x0 - t, y0 - t, x1 + t, y1 + t);
        for x in xa..=xb {
            for y in [ya, yb] {
                if x >= 0 && y >= 0 && x < w && y < h {
                    img.put_pixel(x as u32, y as u32, color);
                }
            }
        }
        for y in ya..=yb {
            for x in [xa, xb] {
                if x >= 0 && y >= 0 && x < w && y < h {
                    img.put_pixel(x as u32, y as u32, color);
                }
            }
        }
    }
}

fn det_draw(
    session: &mut Session,
    input: &Path,
    output: &Path,
    conf: f32,
    iou: f32,
) -> anyhow::Result<usize> {
    let src = image::open(input)?.to_rgb8();
    let (boxes, _) = detect_raw(session, &src, conf, iou)?;
    let mut vis = src;
    for (x0, y0, x1, y1) in &boxes {
        draw_box(
            &mut vis,
            *x0 as i32,
            *y0 as i32,
            *x1 as i32,
            *y1 as i32,
            Rgb([0, 255, 0]),
            3,
        );
    }
    save_image(output, &vis, 88)?;
    eprintln!("det {}: {} boxes -> {}", input.display(), boxes.len(), output.display());
    Ok(boxes.len())
}

// ---- full pipeline: padded-bbox + native-256 tiling (mirrors test/tile_vs_resize method_B) ----

const TILE: i32 = 256;

 /// wm bbox + surrounding non-wm context, clamped to the image -> (x, y, w, h).
fn padded_bbox(x0: i32, y0: i32, x1: i32, y1: i32, img_w: i32, img_h: i32, pad: i32) -> (i32, i32, u32, u32) {
    let x0 = (x0 - pad).max(0);
    let y0 = (y0 - pad).max(0);
    let x1 = (x1 + pad).min(img_w);
    let y1 = (y1 + pad).min(img_h);
    (x0, y0, (x1 - x0).max(1) as u32, (y1 - y0).max(1) as u32)
}

/// Tile origins (1-D) covering [reg0, reg0+reg_len), tiles kept inside the image.
fn tile_origins(reg0: i32, reg_len: i32, img_len: i32, size: i32, stride: i32) -> Vec<i32> {
    if reg_len <= size {
        let o = reg0 - (size - reg_len) / 2;
        return vec![o.clamp(0, 0.max(img_len - size))];
    }
    let first = 0.max(reg0.min(img_len - size));
    let last = first.max((reg0 + reg_len - size).min(img_len - size));
    let mut origins: Vec<i32> = Vec::new();
    let mut o = first;
    while o < last {
        origins.push(o);
        o += stride;
    }
    if origins.is_empty() || *origins.last().unwrap() != last {
        origins.push(last);
    }
    origins
}

/// Feather weight table (mirrors tile_vs_resize.feather_alpha: ramp = linspace(0,1,r), v = i/(r-1)).
fn feather_alpha_table(w: usize, h: usize, feather: usize) -> Vec<f32> {
    let mut alpha = vec![1.0f32; w * h];
    let r = feather.min(h / 2).min(w / 2).max(1);
    for i in 0..r {
        let v = if r > 1 { i as f32 / (r - 1) as f32 } else { 0.0 };
        for x in 0..w {
            alpha[i * w + x] = alpha[i * w + x].min(v);
            alpha[(h - 1 - i) * w + x] = alpha[(h - 1 - i) * w + x].min(v);
        }
        for y in 0..h {
            alpha[y * w + i] = alpha[y * w + i].min(v);
            alpha[y * w + (w - 1 - i)] = alpha[y * w + (w - 1 - i)].min(v);
        }
    }
    alpha
}

/// Square-expand resize clean for rects just over 256 (side 257-380):
/// expand short side to a square centered on the rect, run once at 256,
/// then crop the original rw x rh center back. Extra background is context
/// only, never pasted to the page (caller pastes the returned rw x rh).
const SQUARE_RESIZE_MIN: i32 = 257;
const SQUARE_RESIZE_MAX: i32 = 380;

fn square_for_rect(
    rx: i32,
    ry: i32,
    rw: u32,
    rh: u32,
    img_w: i32,
    img_h: i32,
) -> (i32, i32, u32) {
    let rw_i = rw as i32;
    let rh_i = rh as i32;
    let mut side = rw_i.max(rh_i);
    side = side.min(img_w).min(img_h).max(1);
    // Center on the rect, then clamp so the square stays inside the image.
    // Square always contains the rect: side >= rw/rh and both live in-image.
    let cx = rx + rw_i / 2;
    let cy = ry + rh_i / 2;
    let sx = (cx - side / 2).clamp(0, 0.max(img_w - side));
    let sy = (cy - side / 2).clamp(0, 0.max(img_h - side));
    (sx, sy, side as u32)
}

fn clean_rect_square_resize(
    session: &mut Session,
    src: &RgbImage,
    rx: i32,
    ry: i32,
    rw: u32,
    rh: u32,
    sx: i32,
    sy: i32,
    side: u32,
) -> anyhow::Result<RgbImage> {
    let square =
        image::imageops::crop_imm(src, sx as u32, sy as u32, side, side).to_image();
    let small = resize_bilinear_cv2(&square, 256, 256);
    let fixed = run_clean(session, &small)?;
    let full = image::imageops::resize(&fixed, side, side, FilterType::CatmullRom);
    // Original rect offset inside the square; square contains rect by construction.
    let ox = (rx - sx).max(0) as u32;
    let oy = (ry - sy).max(0) as u32;
    let ox = ox.min(side.saturating_sub(rw));
    let oy = oy.min(side.saturating_sub(rh));
    Ok(image::imageops::crop_imm(&full, ox, oy, rw, rh).to_image())
}

/// Native-256 tiled clean of a padded-bbox rect: each 256 tile runs at true
/// scale (no resize), outputs are feather-mosaicked back to rw x rh.
fn clean_rect_tiled(
    session: &mut Session,
    src: &RgbImage,
    rx: i32,
    ry: i32,
    rw: u32,
    rh: u32,
    stride: i32,
    feather: usize,
) -> anyhow::Result<RgbImage> {
    let (img_w, img_h) = (src.width() as i32, src.height() as i32);
    let rw_i = rw as i32;
    let rh_i = rh as i32;
    let xs = tile_origins(rx, rw_i, img_w, TILE, stride);
    let ys = tile_origins(ry, rh_i, img_h, TILE, stride);
    let alpha = feather_alpha_table(TILE as usize, TILE as usize, feather);
    let mut accum = vec![0.0f32; rw as usize * rh as usize * 3];
    let mut wsum = vec![0.0f32; rw as usize * rh as usize];
    for &oy in &ys {
        for &ox in &xs {
            if ox < 0 || oy < 0 || ox + TILE > img_w || oy + TILE > img_h {
                continue;
            }
            let tile =
                image::imageops::crop_imm(src, ox as u32, oy as u32, TILE as u32, TILE as u32)
                    .to_image();
            let cleaned = run_clean(session, &tile)?;
            // tile rect -> region-relative coords, clipped to region
            let x0 = ox - rx;
            let y0 = oy - ry;
            let tx0 = 0.max(-x0);
            let ty0 = 0.max(-y0);
            let tx1 = TILE.min(rw_i - x0);
            let ty1 = TILE.min(rh_i - y0);
            if ty1 <= ty0 || tx1 <= tx0 {
                continue;
            }
            let rx0 = x0 + tx0;
            let ry0 = y0 + ty0;
            let cw = (tx1 - tx0) as usize;
            let ch = (ty1 - ty0) as usize;
            for dy in 0..ch {
                for dx in 0..cw {
                    let a = alpha[(ty0 as usize + dy) * TILE as usize + (tx0 as usize + dx)];
                    let p = cleaned.get_pixel((tx0 + dx as i32) as u32, (ty0 + dy as i32) as u32);
                    let idx = ((ry0 as usize + dy) * rw as usize + (rx0 as usize + dx)) * 3;
                    accum[idx] += a * p[0] as f32;
                    accum[idx + 1] += a * p[1] as f32;
                    accum[idx + 2] += a * p[2] as f32;
                    wsum[(ry0 as usize + dy) * rw as usize + (rx0 as usize + dx)] += a;
                }
            }
        }
    }
    let mut out = RgbImage::new(rw, rh);
    for y in 0..rh {
        for x in 0..rw {
            let w = wsum[y as usize * rw as usize + x as usize];
            if w > 1e-6 {
                let idx = (y as usize * rw as usize + x as usize) * 3;
                let r = (accum[idx] / w).round().clamp(0.0, 255.0) as u8;
                let g = (accum[idx + 1] / w).round().clamp(0.0, 255.0) as u8;
                let b = (accum[idx + 2] / w).round().clamp(0.0, 255.0) as u8;
                out.put_pixel(x, y, Rgb([r, g, b]));
            } else {
                out.put_pixel(x, y, *src.get_pixel((rx + x as i32) as u32, (ry + y as i32) as u32));
            }
        }
    }
    Ok(out)
}

/// Feathered paste (mirrors inference.feather_blend, feather=7, truncating u8 cast).
/// Python ramp is np.linspace(0, 1, r): v = i / (r-1), not i / r.
fn feather_blend(dst: &mut RgbImage, patch: &RgbImage, x0: i32, y0: i32) {
    let (w, h) = (patch.width() as usize, patch.height() as usize);
    let feather = 7usize.min(h / 2).min(w / 2);
    let mut alpha = vec![1.0f32; w * h];
    for i in 0..feather {
        let v = if feather > 1 {
            i as f32 / (feather - 1) as f32
        } else {
            0.0
        };
        for x in 0..w {
            alpha[i * w + x] = alpha[i * w + x].min(v);
            alpha[(h - 1 - i) * w + x] = alpha[(h - 1 - i) * w + x].min(v);
        }
        for y in 0..h {
            alpha[y * w + i] = alpha[y * w + i].min(v);
            alpha[y * w + (w - 1 - i)] = alpha[y * w + (w - 1 - i)].min(v);
        }
    }
    for y in 0..h {
        for x in 0..w {
            let a = alpha[y * w + x];
            let d = dst.get_pixel((x0 + x as i32) as u32, (y0 + y as i32) as u32);
            let p = patch.get_pixel(x as u32, y as u32);
            let mix = |dd: u8, pp: u8| (a * pp as f32 + (1.0 - a) * dd as f32) as u8;
            dst.put_pixel(
                (x0 + x as i32) as u32,
                (y0 + y as i32) as u32,
                Rgb([mix(d[0], p[0]), mix(d[1], p[1]), mix(d[2], p[2])]),
            );
        }
    }
}

/// End-to-end page clean (mirrors test/tile_vs_resize method_B tiled path).
fn full_clean(
    det: &mut Session,
    clean: &mut Session,
    input: &Path,
    output: &Path,
    crops_dir: &Path,
    stem: &str,
    conf: f32,
    iou: f32,
    pad: i32,
    stride: i32,
    tile_feather: usize,
) -> anyhow::Result<usize> {
    let src = image::open(input)?.to_rgb8();
    let (img_w, img_h) = (src.width() as i32, src.height() as i32);
    let boxes = detect_boxes(det, &src, conf, iou)?;
    let mut out_img = src.clone();
    std::fs::create_dir_all(crops_dir)?;
    for (i, (x, y, w, h)) in boxes.iter().enumerate() {
        let (rx, ry, rw, rh) = padded_bbox(*x, *y, x + w, y + h, img_w, img_h, pad);
        let before =
            image::imageops::crop_imm(&src, rx as u32, ry as u32, rw, rh).to_image();
        let side = (rw as i32).max(rh as i32);
        // Just-over-256 rects: square-expand + single resize pass, paste back
        // the original rw x rh only. Avoids the 257-320 tiling gap entirely.
        let fixed = if side >= SQUARE_RESIZE_MIN && side <= SQUARE_RESIZE_MAX {
            let (sx, sy, sq) = square_for_rect(rx, ry, rw, rh, img_w, img_h);
            clean_rect_square_resize(clean, &src, rx, ry, rw, rh, sx, sy, sq)?
        } else {
            clean_rect_tiled(clean, &src, rx, ry, rw, rh, stride, tile_feather)?
        };
        save_image(&crops_dir.join(format!("{stem}_box{i}_before.jpg")), &before, 95)?;
        save_image(&crops_dir.join(format!("{stem}_box{i}_after.jpg")), &fixed, 95)?;
        feather_blend(&mut out_img, &fixed, rx, ry);
    }
    save_image(output, &out_img, 90)?;
    eprintln!("full {}: {} boxes -> {}", input.display(), boxes.len(), output.display());
    Ok(boxes.len())
}

fn collect_inputs(input: &Path) -> anyhow::Result<Vec<PathBuf>> {
    if input.is_dir() {
        let mut files: Vec<PathBuf> = std::fs::read_dir(input)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_file() && is_image(p))
            .collect();
        files.sort();
        anyhow::ensure!(!files.is_empty(), "no images in {}", input.display());
        Ok(files)
    } else {
        Ok(vec![input.to_path_buf()])
    }
}

fn out_for(dir_mode: bool, out_base: &Path, stem: &str, suffix: &str) -> PathBuf {
    if dir_mode {
        out_base.join(format!("{stem}{suffix}"))
    } else {
        out_base.to_path_buf()
    }
}

fn bench_cleaner(
    model: &Path,
    cpu: bool,
    device_id: Option<i32>,
    deterministic: bool,
    intra_threads: Option<usize>,
    opt: ort::session::builder::GraphOptimizationLevel,
    iters: usize,
) -> anyhow::Result<()> {
    use std::time::Instant;
    let mut session = build_session_full(model, cpu, device_id, deterministic, intra_threads, opt, None, false)
        .map_err(|e| anyhow::anyhow!("bench session: {e:?}"))?;
    // Real 256 input: reuse tmp_ghost crop if present, else flat gray.
    let small: RgbImage = if Path::new("test/tmp_ghost0.jpg").exists() {
        let src = image::open("test/tmp_ghost0.jpg")?.to_rgb8();
        let (w, h) = (src.width().min(512), src.height().min(512));
        let crop = image::imageops::crop_imm(&src, 0, 0, w.min(h), w.min(h)).to_image();
        resize_bilinear_cv2(&crop, 256, 256)
    } else {
        RgbImage::from_pixel(256, 256, Rgb([200, 200, 200]))
    };
    for _ in 0..5 {
        let _ = run_clean(&mut session, &small)?;
    }
    let mut ms: Vec<f64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        let _ = run_clean(&mut session, &small)?;
        ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = ms.iter().sum::<f64>() / ms.len() as f64;
    let p50 = ms[ms.len() / 2];
    let p95 = ms[(ms.len() as f64 * 0.95) as usize];
    eprintln!(
        "bench_clean cpu={cpu} det={deterministic} threads={intra_threads:?} opt={opt:?}: n={} mean={:.1}ms p50={:.1}ms p95={:.1}ms",
        ms.len(),
        mean,
        p50,
        p95
    );
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let _ = VERBOSE.set(args.verbose);
    if args.verbose {
        eprintln!("[verbose] args: {args:?}");
    }
    if args.bench_clean > 0 {
        let threads = cpu_threads(args.intra_threads);
        if !args.clean_dml {
            // CPU baseline + tuned (Level3) back-to-back for the 5x question.
            bench_cleaner(
                &args.model,
                true,
                None,
                false,
                threads,
                ort::session::builder::GraphOptimizationLevel::Level1,
                args.bench_clean,
            )?;
            bench_cleaner(
                &args.model,
                true,
                None,
                false,
                threads,
                ort::session::builder::GraphOptimizationLevel::Level3,
                args.bench_clean,
            )?;
        } else {
            // DML bench runs the same single-model exact path as --clean-dml.
            bench_cleaner(
                &args.model,
                false,
                args.device_id,
                false,
                threads,
                ort::session::builder::GraphOptimizationLevel::Level1,
                args.bench_clean,
            )?;
            // Deterministic DML: accuracy knob, expect slower but maybe cleaner.
            bench_cleaner(
                &args.model,
                false,
                args.device_id,
                true,
                threads,
                ort::session::builder::GraphOptimizationLevel::Level1,
                args.bench_clean,
            )?;
        }
        return Ok(());
    }
    let dir_mode = args.input.is_dir();
    if dir_mode {
        std::fs::create_dir_all(&args.output)?;
    }
    let files = collect_inputs(&args.input)?;

    match args.mode {
        Mode::Clean => {
            // One model for every backend: --clean-dml runs the same --model on
            // DML with graph fusion disabled (bit-exact vs CPU, ~10x faster).
            let use_cpu = !args.clean_dml;
            if !use_cpu {
                eprintln!("cleaner on DML no-fusion (fast, quality-matched)");
            }
            let threads = cpu_threads(args.intra_threads);
            let mut session = build_session_full(
                &args.model,
                use_cpu,
                args.device_id,
                args.deterministic,
                threads,
                parse_opt_level(&args.opt_level),
                args.profile.clone(),
                args.verbose,
            )
            .map_err(|e| anyhow::anyhow!("clean session: {e:?}"))?;
            for src in &files {
                let stem = src.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
                let dst = out_for(dir_mode, &args.output, stem, "_cleaned.jpg");
                if let Err(e) = clean_whole(&mut session, src, &dst) {
                    eprintln!("[warn] skip {}: {e:#}", src.display());
                }
            }
        }
        Mode::Det => {
            let det_path = args
                .det_model
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("--det-model required for det mode"))?;
            let mut session = build_session(det_path, args.cpu, args.device_id)
                .map_err(|e| anyhow::anyhow!("det session: {e:?}"))?;
            let mut total = 0;
            for src in &files {
                let stem = src.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
                let dst = out_for(dir_mode, &args.output, stem, "_boxed.jpg");
                match det_draw(&mut session, src, &dst, args.conf, args.iou) {
                    Ok(n) => total += n,
                    Err(e) => eprintln!("[warn] skip {}: {e:#}", src.display()),
                }
            }
            eprintln!("done: {} images, {total} boxes (cpu={})", files.len(), args.cpu);
        }
        Mode::Full => {
            let det_path = args
                .det_model
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("--det-model required for full mode"))?;
            let mut det = build_session(det_path, args.cpu, args.device_id)
                .map_err(|e| anyhow::anyhow!("det session: {e:?}"))?;
            // See Mode::Clean: one model on every backend.
            let use_cpu = !args.clean_dml;
            if !use_cpu {
                eprintln!("cleaner on DML no-fusion (fast, quality-matched)");
            }
            let threads = cpu_threads(args.intra_threads);
            let mut clean = build_session_full(
                &args.model,
                use_cpu,
                args.device_id,
                args.deterministic,
                threads,
                parse_opt_level(&args.opt_level),
                args.profile.clone(),
                args.verbose,
            )
            .map_err(|e| anyhow::anyhow!("clean session: {e:?}"))?;
            let crops_base = if dir_mode {
                args.output.join("crops")
            } else {
                args.output
                    .parent()
                    .map(|p| p.join("crops"))
                    .unwrap_or_else(|| PathBuf::from("crops"))
            };
            let mut total = 0;
            for src in &files {
                let stem = src.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
                let dst = out_for(dir_mode, &args.output, stem, "_cleaned.jpg");
                match full_clean(&mut det, &mut clean, src, &dst, &crops_base, stem, args.conf, args.iou, args.pad, args.stride, args.tile_feather) {
                    Ok(n) => total += n,
                    Err(e) => eprintln!("[warn] skip {}: {e:#}", src.display()),
                }
            }
            eprintln!("done: {} images, {total} boxes (cpu={})", files.len(), args.cpu);
        }
    }
    Ok(())
}
