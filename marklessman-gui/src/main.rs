#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod icons;
mod models;
mod theme;
mod updater;
mod widgets;

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
    mpsc::{self, Receiver},
};
use std::time::{Instant, SystemTime};

use eframe::egui;
use marklessman_core::{Backend, Cleaner, Detector, Pipeline, SessionOptions, is_image_file};

use theme::M3Colors;

#[cfg(all(windows, feature = "updates"))]
use velopack::VelopackApp;

fn main() -> eframe::Result<()> {
    // ---- Velopack lifecycle (handles install/update/uninstall and exits) ----
    #[cfg(all(windows, feature = "updates"))]
    VelopackApp::build().run();

    // Silent auto-update: check once shortly after startup and download
    // immediately when available; the staged update is applied on close
    // via `on_exit` (no UI, no prompts).
    let pending_update: Arc<Mutex<Option<updater::UpdateInfo>>> =
        Arc::new(Mutex::new(None));
    #[cfg(feature = "updates")]
    {
        let pending_w = pending_update.clone();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(3));
            if let Some(info) = updater::check_and_download() {
                *pending_w.lock().expect("update lock") = Some(info);
            }
        });
    }

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([960.0, 680.0])
            .with_min_inner_size([680.0, 500.0]),
        ..Default::default()
    };
    eframe::run_native(
        "MarklessMan",
        options,
        Box::new(move |cc| {
            theme::apply(&cc.egui_ctx);
            theme::install_cjk_fallback(&cc.egui_ctx);
            egui_extras::install_image_loaders(&cc.egui_ctx);
            Ok(Box::new({
                let mut app = MarklessApp::default();
                app.pending_update = pending_update.clone();
                app
            }))
        }),
    )
}

#[derive(Debug, Clone, PartialEq)]
enum FileStatus {
    Queued,
    Running,
    Done { boxes: usize },
    Error(String),
}

#[derive(Debug, Clone)]
struct QueueItem {
    path: PathBuf,
    status: FileStatus,
    output: Option<PathBuf>,
    size: u64,
    modified: Option<SystemTime>,
}

impl QueueItem {
    fn new(path: PathBuf) -> Self {
        let (size, modified) = std::fs::metadata(&path)
            .map(|m| (m.len(), m.modified().ok()))
            .unwrap_or((0, None));
        Self {
            path,
            status: FileStatus::Queued,
            output: None,
            size,
            modified,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SortKey {
    Added,
    NameAsc,
    NameDesc,
    SizeLarge,
    SizeSmall,
    DateNew,
    DateOld,
    Status,
}

impl SortKey {
    const ALL: [SortKey; 8] = [
        SortKey::Added,
        SortKey::NameAsc,
        SortKey::NameDesc,
        SortKey::SizeLarge,
        SortKey::SizeSmall,
        SortKey::DateNew,
        SortKey::DateOld,
        SortKey::Status,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::Added => "Added order",
            Self::NameAsc => "Name A–Z",
            Self::NameDesc => "Name Z–A",
            Self::SizeLarge => "Largest first",
            Self::SizeSmall => "Smallest first",
            Self::DateNew => "Newest first",
            Self::DateOld => "Oldest first",
            Self::Status => "Status",
        }
    }
}

fn status_rank(s: &FileStatus) -> u8 {
    match s {
        FileStatus::Running => 0,
        FileStatus::Error(_) => 1,
        FileStatus::Queued => 2,
        FileStatus::Done { .. } => 3,
    }
}

fn fmt_secs(secs: f32) -> String {
    let s = secs.max(0.0).round() as u64;
    if s < 60 {
        format!("{s}s")
    } else if s < 3600 {
        format!("{}m {:02}s", s / 60, s % 60)
    } else {
        format!("{}h {:02}m", s / 3600, (s % 3600) / 60)
    }
}

fn fmt_size(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    let b = bytes as f64;
    if b < KB {
        format!("{bytes} B")
    } else if b < MB {
        format!("{:.1} KB", b / KB)
    } else {
        format!("{:.1} MB", b / MB)
    }
}

fn fmt_age(modified: Option<SystemTime>) -> String {
    let Some(t) = modified else {
        return "date unknown".to_string();
    };
    let Ok(d) = SystemTime::now().duration_since(t) else {
        return "just now".to_string();
    };
    let s = d.as_secs();
    if s < 60 {
        "just now".to_string()
    } else if s < 3600 {
        format!("{}m ago", s / 60)
    } else if s < 86400 {
        format!("{}h ago", s / 3600)
    } else {
        format!("{}d ago", s / 86400)
    }
}

#[derive(Debug)]
enum GuiEvent {
    Started {
        total: usize,
    },
    FileStarted {
        index: usize,
    },
    FileDone {
        index: usize,
        boxes: usize,
        output: PathBuf,
    },
    FileError {
        index: usize,
        message: String,
    },
    Finished,
    SetupError(String),
}

/// One in-flight `fast-down-api` model download (lazy on Run).
struct ActiveDownload {
    spec: &'static models::ModelSpec,
    percent: f32,
    downloaded: u64,
    total: u64,
    progress_rx: Receiver<(f32, u64, u64)>,
    result_rx: Receiver<Result<PathBuf, String>>,
    cancel: tokio_util::sync::CancellationToken,
}

#[derive(PartialEq, Clone, Copy)]
enum BackendChoice {
    Cpu,
    DirectMl,
}

impl BackendChoice {
    fn to_core(self) -> Backend {
        match self {
            Self::Cpu => Backend::Cpu,
            Self::DirectMl => Backend::DirectMl,
        }
    }
}

struct MarklessApp {
    items: Vec<QueueItem>,
    selected: Option<usize>,
    output_dir: PathBuf,
    backend: BackendChoice,
    det_model: PathBuf,
    clean_model: PathBuf,
    running: bool,
    done: usize,
    total: usize,
    current: String,
    log: String,
    rx: Option<Receiver<GuiEvent>>,
    cancel: Option<Arc<AtomicBool>>,
    preview_for: Option<(usize, bool)>,
    before_tex: Option<egui::TextureHandle>,
    after_tex: Option<egui::TextureHandle>,
    preview_open: bool,
    output_chosen: bool,
    hover_row: Option<usize>,
    img_scroll: egui::Vec2,
    dark_mode: bool,
    show_advanced: bool,
    thumbs: HashMap<PathBuf, egui::TextureHandle>,
    sort: SortKey,
    run_started: Option<Instant>,
    file_started: Option<Instant>,
    time_sum: f64,
    time_n: usize,
    recent: VecDeque<f32>,
    last_run_secs: Option<f32>,
    /// Spec ids queued for download (each `&'static ModelSpec` looked up by id).
    dl_queue: VecDeque<&'static str>,
    dl_current: Option<ActiveDownload>,
    dl_error: Option<String>,
    dl_pending_run: bool,
    /// Update staged by the silent background check; applied on close.
    pending_update: Arc<Mutex<Option<updater::UpdateInfo>>>,
}

impl Default for MarklessApp {
    fn default() -> Self {
        Self {
            items: Vec::new(),
            selected: None,
            output_dir: PathBuf::from("output"),
            backend: BackendChoice::DirectMl,
            det_model: models::resolve_model_path(&models::DET),
            clean_model: models::resolve_model_path(&models::CLEAN),
            running: false,
            done: 0,
            total: 0,
            current: String::new(),
            log: String::from("Drop images onto the window, then press Run."),
            rx: None,
            cancel: None,
            preview_for: None,
            before_tex: None,
            after_tex: None,
            preview_open: false,
            output_chosen: false,
            hover_row: None,
            img_scroll: egui::Vec2::ZERO,
            dark_mode: true,
            show_advanced: false,
            thumbs: HashMap::new(),
            sort: SortKey::NameAsc,
            run_started: None,
            file_started: None,
            time_sum: 0.0,
            time_n: 0,
            recent: VecDeque::new(),
            last_run_secs: None,
            dl_queue: VecDeque::new(),
            dl_current: None,
            dl_error: None,
            dl_pending_run: false,
            pending_update: Arc::new(Mutex::new(None)),
        }
    }
}

fn output_path_for(input: &Path, out_dir: &Path) -> PathBuf {
    let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
    out_dir.join(format!("{stem}_cleaned.jpg"))
}

fn reveal_in_folder(path: &Path) {
    let dir = path.parent().unwrap_or(path);
    #[cfg(target_os = "windows")]
    {
        let _ = std::process::Command::new("explorer").arg(dir).spawn();
    }
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open").arg(dir).spawn();
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        let _ = std::process::Command::new("xdg-open").arg(dir).spawn();
    }
}

fn file_name_of(path: &Path) -> String {
    path.file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("?")
        .to_string()
}

fn load_preview_texture(
    ctx: &egui::Context,
    path: &Path,
    name: String,
) -> Option<egui::TextureHandle> {
    let img = image::open(path).ok()?.to_rgb8();
    let (w, h) = img.dimensions();
    let max_side = 700u32;
    let scale = (w.max(h) as f32 / max_side as f32).max(1.0);
    let small = if scale > 1.0 {
        let (pw, ph) = ((w as f32 / scale) as u32, (h as f32 / scale) as u32);
        image::imageops::resize(
            &img,
            pw.max(1),
            ph.max(1),
            image::imageops::FilterType::Triangle,
        )
    } else {
        img
    };
    let (sw, sh) = small.dimensions();
    let pixels: Vec<u8> = small.into_raw();
    let color = egui::ColorImage::from_rgb([sw as usize, sh as usize], &pixels);
    Some(ctx.load_texture(name, color, egui::TextureOptions::LINEAR))
}

/// Small 112px thumbnail for M3 list leading (shown at 56dp, 2x for sharpness).
fn load_thumb_texture(
    ctx: &egui::Context,
    path: &Path,
    name: String,
) -> Option<egui::TextureHandle> {
    let img = image::open(path).ok()?.to_rgb8();
    let (w, h) = img.dimensions();
    let max_side = 112u32;
    let scale = (w.max(h) as f32 / max_side as f32).max(1.0);
    let small = if scale > 1.0 {
        let (pw, ph) = ((w as f32 / scale) as u32, (h as f32 / scale) as u32);
        image::imageops::resize(
            &img,
            pw.max(1),
            ph.max(1),
            image::imageops::FilterType::Triangle,
        )
    } else {
        img
    };
    let (sw, sh) = small.dimensions();
    let pixels: Vec<u8> = small.into_raw();
    let color = egui::ColorImage::from_rgb([sw as usize, sh as usize], &pixels);
    Some(ctx.load_texture(name, color, egui::TextureOptions::LINEAR))
}

impl MarklessApp {
    fn avg_secs(&self) -> Option<f32> {
        if self.recent.is_empty() {
            return None;
        }
        Some(self.recent.iter().sum::<f32>() / self.recent.len() as f32)
    }

    fn set_sort(&mut self, key: SortKey) {
        if self.running || self.sort == key {
            return;
        }
        self.sort = key;
        self.apply_sort();
    }

    fn apply_sort(&mut self) {
        if self.sort == SortKey::Added {
            return;
        }
        let sel_path = self
            .selected
            .and_then(|i| self.items.get(i).map(|it| it.path.clone()));
        match self.sort {
            SortKey::Added => {}
            SortKey::NameAsc => self.items.sort_by(|a, b| {
                file_name_of(&a.path)
                    .to_lowercase()
                    .cmp(&file_name_of(&b.path).to_lowercase())
            }),
            SortKey::NameDesc => self.items.sort_by(|a, b| {
                file_name_of(&b.path)
                    .to_lowercase()
                    .cmp(&file_name_of(&a.path).to_lowercase())
            }),
            SortKey::SizeLarge => self.items.sort_by_key(|b| std::cmp::Reverse(b.size)),
            SortKey::SizeSmall => self.items.sort_by_key(|a| a.size),
            SortKey::DateNew => self.items.sort_by_key(|b| std::cmp::Reverse(b.modified)),
            SortKey::DateOld => self.items.sort_by_key(|a| a.modified),
            SortKey::Status => self.items.sort_by(|a, b| {
                status_rank(&a.status)
                    .cmp(&status_rank(&b.status))
                    .then_with(|| {
                        file_name_of(&a.path)
                            .to_lowercase()
                            .cmp(&file_name_of(&b.path).to_lowercase())
                    })
            }),
        }
        if let Some(p) = sel_path {
            self.selected = self.items.iter().position(|it| it.path == p);
        }
        self.preview_for = None;
    }

    /// Bottom-bar ETA string. Live-ticking while running by subtracting the
    /// in-progress file's elapsed time from the estimate.
    fn eta_text(&self) -> String {
        if self.running {
            let remaining = self.total.saturating_sub(self.done);
            if remaining == 0 {
                return "finishing…".to_string();
            }
            if let Some(avg) = self.avg_secs() {
                let elapsed_cur = self
                    .file_started
                    .map(|t| t.elapsed().as_secs_f32())
                    .unwrap_or(0.0);
                let eta = (avg * remaining as f32 - elapsed_cur).max(0.0);
                return format!("~{} left • {} avg", fmt_secs(eta), fmt_secs(avg));
            }
            if let Some(t) = self.file_started {
                return format!("working… {} elapsed", fmt_secs(t.elapsed().as_secs_f32()));
            }
            return "calculating…".to_string();
        }
        if let Some(total) = self.last_run_secs {
            let avg = if self.time_n > 0 {
                self.time_sum / self.time_n as f64
            } else {
                0.0
            };
            if avg > 0.0 {
                return format!("done in {} • {} avg", fmt_secs(total), fmt_secs(avg as f32));
            }
            return format!("done in {}", fmt_secs(total));
        }
        String::new()
    }

    fn record_file_time(&mut self) {
        if let Some(t) = self.file_started.take() {
            let secs = t.elapsed().as_secs_f32();
            self.time_sum += secs as f64;
            self.time_n += 1;
            self.recent.push_back(secs);
            while self.recent.len() > 10 {
                self.recent.pop_front();
            }
        }
    }

    fn add_paths(&mut self, paths: Vec<PathBuf>) {
        let mut added = 0;
        for p in paths {
            if p.is_dir() {
                match std::fs::read_dir(&p) {
                    Ok(entries) => {
                        let mut files: Vec<PathBuf> = entries
                            .filter_map(|e| e.ok().map(|e| e.path()))
                            .filter(|f| f.is_file() && is_image_file(f))
                            .collect();
                        files.sort();
                        for f in files {
                            if !self.items.iter().any(|it| it.path == f) {
                                self.items.push(QueueItem::new(f));
                                added += 1;
                            }
                        }
                    }
                    Err(e) => self.log = format!("cannot read dir {}: {e}", p.display()),
                }
                continue;
            }
            if !is_image_file(&p) {
                continue;
            }
            if !self.items.iter().any(|it| it.path == p) {
                self.items.push(QueueItem::new(p));
                added += 1;
            }
        }
        if self.selected.is_none() && !self.items.is_empty() {
            self.selected = Some(0);
        }
        if added > 0 {
            self.apply_sort();
            // apply_sort preserves selection by path, but a fresh add with no
            // prior selection should still select the first row.
            if self.selected.is_none() && !self.items.is_empty() {
                self.selected = Some(0);
            }
            self.log = format!("added {added} file(s), {} total", self.items.len());
            self.preview_for = None;
        }
    }

    fn remove_item(&mut self, index: usize) {
        if self.running || index >= self.items.len() {
            return;
        }
        let removed = self.items.remove(index);
        self.thumbs.remove(&removed.path);
        self.preview_for = None;
        self.before_tex = None;
        self.after_tex = None;
        self.selected = match self.selected {
            None => None,
            Some(_) if self.items.is_empty() => None,
            Some(sel) if sel == index => Some(index.min(self.items.len().saturating_sub(1))),
            Some(sel) if sel > index => Some(sel - 1),
            Some(sel) => Some(sel),
        };
        if self.items.is_empty() {
            self.preview_open = false;
        }
    }

    fn clear_all(&mut self) {
        if self.running {
            return;
        }
        self.items.clear();
        self.thumbs.clear();
        self.selected = None;
        self.done = 0;
        self.total = 0;
        self.preview_for = None;
        self.before_tex = None;
        self.after_tex = None;
        self.preview_open = false;
        self.run_started = None;
        self.file_started = None;
        self.time_sum = 0.0;
        self.time_n = 0;
        self.recent.clear();
        self.last_run_secs = None;
        self.log = "queue cleared.".to_string();
    }

    fn ingest_dropped(&mut self, ctx: &egui::Context) {
        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        if dropped.is_empty() {
            return;
        }
        let paths: Vec<PathBuf> = dropped.iter().map(|f| f.path().to_path_buf()).collect();
        self.add_paths(paths);
    }

    fn poll_events(&mut self) {
        let events: Vec<GuiEvent> = {
            let Some(rx) = self.rx.as_ref() else {
                return;
            };
            let mut out = Vec::new();
            while let Ok(ev) = rx.try_recv() {
                out.push(ev);
            }
            out
        };
        for ev in events {
            match ev {
                GuiEvent::Started { total } => {
                    self.total = total;
                    self.done = 0;
                    self.run_started = Some(Instant::now());
                    self.file_started = None;
                    self.time_sum = 0.0;
                    self.time_n = 0;
                    self.recent.clear();
                    self.last_run_secs = None;
                }
                GuiEvent::FileStarted { index } => {
                    if let Some(it) = self.items.get_mut(index) {
                        it.status = FileStatus::Running;
                    }
                    self.file_started = Some(Instant::now());
                    self.current = self
                        .items
                        .get(index)
                        .map(|it| it.path.display().to_string())
                        .unwrap_or_default();
                }
                GuiEvent::FileDone {
                    index,
                    boxes,
                    output,
                } => {
                    if let Some(it) = self.items.get_mut(index) {
                        it.status = FileStatus::Done { boxes };
                        it.output = Some(output);
                    }
                    self.record_file_time();
                    self.done += 1;
                }
                GuiEvent::FileError { index, message } => {
                    if let Some(it) = self.items.get_mut(index) {
                        it.status = FileStatus::Error(message.clone());
                    }
                    self.record_file_time();
                    self.done += 1;
                    self.log = format!("error: {message}");
                }
                GuiEvent::SetupError(msg) => {
                    self.log = format!("setup failed: {msg}");
                    self.running = false;
                    self.file_started = None;
                }
                GuiEvent::Finished => {
                    self.running = false;
                    self.current.clear();
                    self.file_started = None;
                    if let Some(t) = self.run_started.take() {
                        self.last_run_secs = Some(t.elapsed().as_secs_f32());
                    }
                    self.log = format!("done: {}/{} files", self.done, self.total);
                }
            }
        }
    }

    fn is_downloading(&self) -> bool {
        self.dl_current.is_some() || !self.dl_queue.is_empty()
    }

    fn apply_canonical_if_present(&mut self) {
        // Adopt AppData files that appeared (e.g. downloaded externally).
        let det_canon = models::model_path(&models::DET);
        if !self.det_model.exists() && det_canon.exists() {
            self.det_model = det_canon;
        }
        let clean_canon = models::model_path(&models::CLEAN);
        if !self.clean_model.exists() && clean_canon.exists() {
            self.clean_model = clean_canon;
        }
    }

    /// Specs that still need downloading for a run (lazy on Run).
    /// A user-picked override that exists on disk counts as satisfied.
    fn needed_specs(&mut self) -> Vec<&'static models::ModelSpec> {
        self.apply_canonical_if_present();
        let mut out = Vec::new();
        if !self.det_model.exists() && !models::is_downloaded(&models::DET) {
            out.push(&models::DET);
        }
        if !self.clean_model.exists() && !models::is_downloaded(&models::CLEAN) {
            out.push(&models::CLEAN);
        }
        out
    }

    fn begin_downloads(&mut self, specs: Vec<&'static models::ModelSpec>, pending_run: bool) {
        self.dl_error = None;
        if pending_run {
            self.dl_pending_run = true;
        }
        for s in specs {
            if self.dl_queue.iter().any(|id| *id == s.id)
                || self.dl_current.as_ref().is_some_and(|d| d.spec.id == s.id)
            {
                continue;
            }
            self.dl_queue.push_back(s.id);
        }
        if self.dl_current.is_none() {
            self.pump_download_queue();
        }
    }

    fn download_spec(&mut self, spec: &'static models::ModelSpec) {
        self.begin_downloads(vec![spec], false);
    }

    fn pump_download_queue(&mut self) {
        if self.dl_current.is_some() {
            return;
        }
        let Some(id) = self.dl_queue.pop_front() else {
            return;
        };
        let Some(spec) = models::get_model(id) else {
            return;
        };
        let (progress_tx, progress_rx) = mpsc::channel::<(f32, u64, u64)>();
        let (result_tx, result_rx) = mpsc::channel::<Result<PathBuf, String>>();
        let cancel = models::spawn_download(spec, progress_tx, result_tx, false);
        self.dl_current = Some(ActiveDownload {
            spec,
            percent: 0.0,
            downloaded: 0,
            total: 0,
            progress_rx,
            result_rx,
            cancel,
        });
        self.log = format!("downloading {} …", spec.filename);
    }

    fn poll_downloads(&mut self) {
        // Drain progress for the active download.
        if let Some(dl) = self.dl_current.as_mut() {
            while let Ok((pct, done, total)) = dl.progress_rx.try_recv() {
                dl.percent = pct;
                dl.downloaded = done;
                dl.total = total;
            }
        }
        // Check completion without holding the borrow across queue pumping.
        let finished: Option<Result<PathBuf, String>> = self
            .dl_current
            .as_ref()
            .and_then(|dl| dl.result_rx.try_recv().ok());
        let Some(result) = finished else {
            return;
        };
        let dl = self.dl_current.take().expect("polled current");
        match result {
            Ok(path) => {
                if dl.spec.id == models::DET.id {
                    self.det_model = path.clone();
                } else if dl.spec.id == models::CLEAN.id {
                    self.clean_model = path.clone();
                }
                self.dl_error = None;
                self.log = format!("downloaded {}", dl.spec.filename);
            }
            Err(e) => {
                self.dl_error = Some(e.clone());
                self.log = format!("model download failed: {e}");
                // Drop the rest of the queue so Retry is explicit.
                self.dl_queue.clear();
                self.dl_pending_run = false;
                return;
            }
        }
        if self.dl_queue.is_empty() {
            // Re-resolve (legacy + canonical) then continue a pending run.
            self.apply_canonical_if_present();
            if self.dl_pending_run {
                self.dl_pending_run = false;
                self.start_run();
            }
        } else {
            self.pump_download_queue();
        }
    }

    fn cancel_download(&mut self) {
        if let Some(dl) = self.dl_current.take() {
            dl.cancel.cancel();
        }
        self.dl_queue.clear();
        self.dl_pending_run = false;
        self.log = "model download cancelled (resumes next time).".to_string();
    }

    /// Snapshot of download state for a spec (cloned to avoid borrow fights in UI).
    fn dl_snapshot(&self, id: &str) -> Option<(f32, u64, u64)> {
        self.dl_current
            .as_ref()
            .filter(|d| d.spec.id == id)
            .map(|d| (d.percent, d.downloaded, d.total))
    }

    fn dl_queued(&self, id: &str) -> bool {
        self.dl_queue.iter().any(|q| *q == id)
    }

    fn show_model_row(
        &mut self,
        ui: &mut egui::Ui,
        m3: &M3Colors,
        spec: &'static models::ModelSpec,
        title: &str,
        tip: &str,
    ) {
        let (model_ok, model_path) = match spec.id {
            "det" => (self.det_model.exists(), self.det_model.clone()),
            _ => (self.clean_model.exists(), self.clean_model.clone()),
        };
        let active = self.dl_snapshot(spec.id);
        let queued = self.dl_queued(spec.id);
        let pick_salt = match spec.id {
            "det" => "det-pick",
            _ => "clean-pick",
        };
        let icon_salt = match spec.id {
            "det" => "det-ok",
            _ => "clean-ok",
        };
        ui.horizontal(|ui| {
            if active.is_some() {
                ui.spinner();
            } else {
                ui.add(icons::tinted(
                    icon_salt,
                    if model_ok { icons::CHECK } else { icons::ALERT },
                    16.0,
                    if model_ok { m3.success } else { m3.error },
                ));
            }
            ui.label(
                egui::RichText::new(title)
                    .size(13.0)
                    .color(m3.on_surface_variant),
            )
            .on_hover_text(tip);
            ui.add(
                egui::Label::new(
                    egui::RichText::new(widgets::trunc_middle(
                        &model_path.display().to_string(),
                        48,
                    ))
                    .monospace()
                    .size(12.0)
                    .color(m3.on_surface_variant),
                )
                .truncate(),
            )
            .on_hover_text(model_path.display().to_string());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if widgets::icon_button(
                    ui,
                    icons::tinted(pick_salt, icons::FOLDER, 18.0, m3.primary),
                    &format!("Choose {title} model (.onnx)"),
                )
                .clicked()
                    && let Some(p) = rfd::FileDialog::new()
                        .add_filter("onnx", &["onnx"])
                        .pick_file()
                {
                    match spec.id {
                        "det" => self.det_model = p,
                        _ => self.clean_model = p,
                    }
                    self.dl_error = None;
                }
                if active.is_some() {
                    if widgets::icon_button(
                        ui,
                        icons::tinted("dl-cancel", icons::X, 18.0, m3.error),
                        "Cancel download (resumes next time)",
                    )
                    .clicked()
                    {
                        self.cancel_download();
                    }
                } else if !model_ok && !queued {
                    if widgets::icon_button(
                        ui,
                        icons::tinted("dl-start", icons::DOWNLOAD, 18.0, m3.primary),
                        &format!("Download {} from Hugging Face", spec.filename),
                    )
                    .clicked()
                    {
                        self.download_spec(spec);
                    }
                }
            });
        });
        if let Some((pct, done, total)) = active {
            let frac = (pct / 100.0).clamp(0.0, 1.0);
            ui.horizontal(|ui| {
                ui.add_space(24.0);
                let bar = egui::ProgressBar::new(frac)
                    .show_percentage()
                    .text(format!(
                        "downloading {} • {} / {}",
                        spec.filename,
                        fmt_size(done),
                        if total > 0 {
                            fmt_size(total)
                        } else {
                            "…".to_string()
                        }
                    ))
                    .corner_radius(egui::CornerRadius::same(4))
                    .desired_width(ui.available_width().max(80.0));
                ui.add(bar);
            });
        } else if queued {
            ui.horizontal(|ui| {
                ui.add_space(24.0);
                ui.label(
                    egui::RichText::new("queued for download…")
                        .size(12.0)
                        .color(m3.on_surface_variant),
                );
            });
        } else if !model_ok {
            ui.horizontal(|ui| {
                ui.add_space(24.0);
                ui.label(
                    egui::RichText::new(format!(
                        "missing — auto-downloads on Run from Hugging Face into {}",
                        models::models_dir().display()
                    ))
                    .size(11.5)
                    .color(m3.error),
                );
            });
        }
    }

    fn start_run(&mut self) {
        if self.running || self.is_downloading() || self.items.is_empty() {
            return;
        }
        if !self.output_chosen {
            if let Some(d) = rfd::FileDialog::new().pick_folder() {
                self.output_dir = d;
                self.output_chosen = true;
            } else {
                self.log = "run cancelled: no output folder selected.".to_string();
                return;
            }
        }
        // Lazy model resolving: download missing AppData models first.
        let needed = self.needed_specs();
        if !needed.is_empty() {
            let names: Vec<&str> = needed.iter().map(|s| s.filename).collect();
            self.log = format!("models missing, downloading: {}", names.join(", "));
            self.show_advanced = true;
            self.begin_downloads(needed, true);
            return;
        }
        if !self.det_model.exists() {
            self.log = format!("detector not found: {}", self.det_model.display());
            return;
        }
        if !self.clean_model.exists() {
            self.log = format!("cleaner not found: {}", self.clean_model.display());
            return;
        }
        self.start_pipeline();
    }

    fn start_pipeline(&mut self) {
        if let Err(e) = std::fs::create_dir_all(&self.output_dir) {
            self.log = format!("cannot create output dir: {e}");
            return;
        }
        let queued: Vec<(usize, PathBuf)> = self
            .items
            .iter()
            .enumerate()
            .filter(|(_, it)| !matches!(it.status, FileStatus::Done { .. }))
            .map(|(i, it)| (i, it.path.clone()))
            .collect();
        if queued.is_empty() {
            self.log = "nothing to do (all done). Clear or add files.".to_string();
            return;
        }
        for (i, _) in &queued {
            if let Some(it) = self.items.get_mut(*i) {
                it.status = FileStatus::Queued;
            }
        }
        let (tx, rx) = mpsc::channel::<GuiEvent>();
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_w = cancel.clone();
        let out_dir = self.output_dir.clone();
        let det = self.det_model.clone();
        let clean = self.clean_model.clone();
        let backend = self.backend.to_core();
        let total = queued.len();

        self.rx = Some(rx);
        self.cancel = Some(cancel);
        self.running = true;
        self.done = 0;
        self.total = total;
        self.run_started = Some(Instant::now());
        self.file_started = None;
        self.time_sum = 0.0;
        self.time_n = 0;
        self.recent.clear();
        self.last_run_secs = None;

        std::thread::spawn(move || {
            let opts = match backend {
                Backend::Cpu => SessionOptions::cpu(),
                Backend::DirectMl => SessionOptions::directml(),
            };
            let detector = Detector::open(&det, &opts);
            let cleaner = Cleaner::open(&clean, &opts);
            let (detector, cleaner) = match (detector, cleaner) {
                (Ok(d), Ok(c)) => (d, c),
                (Err(e), _) | (_, Err(e)) => {
                    let _ = tx.send(GuiEvent::SetupError(e.to_string()));
                    let _ = tx.send(GuiEvent::Finished);
                    return;
                }
            };
            let mut pipeline = Pipeline::new(detector, cleaner);
            let _ = tx.send(GuiEvent::Started { total });
            for (idx, path) in queued {
                if cancel_w.load(Ordering::Relaxed) {
                    break;
                }
                let _ = tx.send(GuiEvent::FileStarted { index: idx });
                let out = output_path_for(&path, &out_dir);
                let res: Result<usize, String> = (|| {
                    let src = image::open(&path).map_err(|e| e.to_string())?.to_rgb8();
                    let processed = pipeline.process(&src).map_err(|e| e.to_string())?;
                    let n = processed.detections.len();
                    marklessman_core::save_image(&out, &processed.image, 90)
                        .map_err(|e| e.to_string())?;
                    Ok(n)
                })();
                match res {
                    Ok(n) => {
                        let _ = tx.send(GuiEvent::FileDone {
                            index: idx,
                            boxes: n,
                            output: out,
                        });
                    }
                    Err(msg) => {
                        let _ = tx.send(GuiEvent::FileError {
                            index: idx,
                            message: msg,
                        });
                    }
                }
            }
            let _ = tx.send(GuiEvent::Finished);
        });
    }

    fn cancel_run(&mut self) {
        if let Some(c) = self.cancel.as_ref() {
            c.store(true, Ordering::Relaxed);
            self.log = "cancelling…".to_string();
        }
    }

    fn update_preview(&mut self, ctx: &egui::Context) {
        let Some(sel) = self.selected else {
            self.before_tex = None;
            self.after_tex = None;
            return;
        };
        let Some(item) = self.items.get(sel) else {
            return;
        };
        let has_output = item.output.is_some() && matches!(item.status, FileStatus::Done { .. });
        match self.preview_for {
            Some((s, h)) if s == sel && h == has_output => return,
            Some((s, _)) if s == sel => {
                self.after_tex = item
                    .output
                    .as_ref()
                    .filter(|_| has_output)
                    .and_then(|p| load_preview_texture(ctx, p, format!("after-{sel}")));
                self.preview_for = Some((sel, has_output));
            }
            _ => {
                self.before_tex = load_preview_texture(ctx, &item.path, format!("before-{sel}"));
                self.after_tex = item
                    .output
                    .as_ref()
                    .filter(|_| has_output)
                    .and_then(|p| load_preview_texture(ctx, p, format!("after-{sel}")));
                self.img_scroll = egui::Vec2::ZERO;
                self.preview_for = Some((sel, has_output));
            }
        }
    }

    fn handle_shortcuts(&mut self, ctx: &egui::Context) {
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            if self.is_downloading() {
                self.cancel_download();
            } else if self.running {
                self.cancel_run();
            } else if self.preview_open {
                self.preview_open = false;
            }
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Enter))
            && !self.running
            && !self.is_downloading()
            && !self.items.is_empty()
        {
            self.start_run();
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Delete)) && !self.running {
            if let Some(sel) = self.selected {
                self.remove_item(sel);
            }
        }
    }
}

fn status_meta(status: &FileStatus, m3: &M3Colors) -> (&'static str, &'static [u8], egui::Color32) {
    match status {
        FileStatus::Queued => ("queued", icons::CLOCK, m3.on_surface_variant),
        FileStatus::Running => ("running", icons::LOADER, m3.primary),
        FileStatus::Done { .. } => ("done", icons::CHECK, m3.success),
        FileStatus::Error(_) => ("error", icons::ALERT, m3.error),
    }
}

/// Leading tonal circle colors per status (container, icon fg).
#[allow(dead_code)]
fn status_container(status: &FileStatus, m3: &M3Colors) -> (egui::Color32, egui::Color32) {
    match status {
        FileStatus::Queued => (m3.secondary_container, m3.on_secondary_container),
        FileStatus::Running => (m3.primary_container, m3.on_primary_container),
        FileStatus::Done { .. } => (m3.success_container, m3.success),
        FileStatus::Error(_) => (m3.error_container, m3.error),
    }
}

/// One half of the Before/After split: M3 media card + synced scroll.
///
/// Header is a proper M3 list-style row (12px icon-title gap, 14sp title,
/// 12sp supporting dims). Media is shrunk to fit the card width so it never
/// bleeds edge-to-edge; empty state uses a tonal circle like the rest of
/// the UI. `scroll` is shared between both cards so the views stay synced.
fn image_card(
    ui: &mut egui::Ui,
    m3: &M3Colors,
    title: &str,
    salt: &'static str,
    icon_bytes: &'static [u8],
    tint: egui::Color32,
    container: egui::Color32,
    tex: Option<egui::TextureHandle>,
    dims: Option<[usize; 2]>,
    empty_hint: &str,
    scroll: &mut egui::Vec2,
) {
    widgets::card_frame(m3).show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing = egui::vec2(12.0, 0.0);
            widgets::tonal_icon(ui, salt, icon_bytes, 16.0, tint, container, 32.0);
            ui.label(egui::RichText::new(title).size(14.0).color(m3.on_surface));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if let Some(d) = dims {
                    ui.label(widgets::on_surface_variant(
                        &format!("{}×{}", d[0], d[1]),
                        m3,
                        12.0,
                    ));
                }
            });
        });
        ui.add_space(8.0);
        // Fit-width media with synced vertical scroll: each image is scaled to
        // the card width (up or down), so BEFORE/AFTER stay comparable and pan
        // together vertically. No horizontal scroll.
        let out = egui::ScrollArea::vertical()
            .id_salt(salt)
            .vertical_scroll_offset(scroll.y)
            .auto_shrink([false; 2])
            .show(ui, |ui| {
                if let Some(tex) = tex.as_ref() {
                    let [tw, th] = tex.size();
                    let avail_w = ui.available_width().max(1.0);
                    let scale = avail_w / (tw as f32).max(1.0);
                    let target = egui::vec2(avail_w, th as f32 * scale);
                    let img = egui::Image::from_texture((tex.id(), tex.size_vec2()))
                        .fit_to_exact_size(target)
                        .corner_radius(egui::CornerRadius::same(8));
                    ui.add(img);
                } else {
                    ui.centered_and_justified(|ui| {
                        ui.vertical_centered(|ui| {
                            ui.add_space(24.0);
                            widgets::tonal_icon(ui, salt, icon_bytes, 22.0, tint, container, 52.0);
                            ui.add_space(12.0);
                            ui.label(widgets::on_surface_variant(empty_hint, m3, 14.0));
                            ui.add_space(24.0);
                        });
                    });
                }
            });
        if out.content_size.y > out.inner_rect.height() + 1.0 {
            scroll.y = out.state.offset.y;
        }
        scroll.x = 0.0;
    });
}

impl eframe::App for MarklessApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let m3 = theme::colors(self.dark_mode);
        self.ingest_dropped(&ctx);
        self.poll_events();
        self.poll_downloads();
        if self.running || self.is_downloading() {
            ctx.request_repaint();
        }
        self.handle_shortcuts(&ctx);
        self.update_preview(&ctx);

        let hovered = ctx.input(|i| !i.raw.hovered_files.is_empty());

        // ── M3 Top App Bar ──────────────────────────────────────
        egui::Panel::top("header").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.add_space(4.0);
                ui.add(icons::svg("logo", icons::LOGO, 34.0));
                ui.vertical(|ui| {
                    ui.add_space(2.0);
                    ui.label(
                        egui::RichText::new("MarklessMan")
                            .size(20.0)
                            .strong()
                            .color(m3.on_surface),
                    );
                    ui.label(widgets::on_surface_variant("Watermark remover", &m3, 12.0));
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if self.running {
                        if widgets::filled_button(
                            ui,
                            "stop",
                            icons::X,
                            "Cancel",
                            m3.error,
                            m3.on_error,
                            "Stop after current file (Esc)",
                            egui::vec2(112.0, 40.0),
                        )
                        .clicked()
                        {
                            self.cancel_run();
                        }
                    } else if self.is_downloading() {
                        if widgets::filled_button(
                            ui,
                            "stop-dl",
                            icons::X,
                            "Cancel",
                            m3.error,
                            m3.on_error,
                            "Cancel model download (Esc, resumes next time)",
                            egui::vec2(112.0, 40.0),
                        )
                        .clicked()
                        {
                            self.cancel_download();
                        }
                    } else {
                        let run_enabled = !self.items.is_empty();
                        let run_resp = widgets::filled_button(
                            ui,
                            "play",
                            icons::PLAY,
                            "Run",
                            m3.primary,
                            m3.on_primary,
                            "Process queued files (Enter)",
                            egui::vec2(112.0, 40.0),
                        )
                        .on_hover_cursor(widgets::click_cursor(run_enabled));
                        if run_resp.clicked() && run_enabled {
                            self.start_run();
                        }
                    }
                    {
                        let clear_enabled = !self.running && !self.items.is_empty();
                        let clear_resp = widgets::outlined_button(
                            ui,
                            "clear",
                            icons::TRASH,
                            "Clear",
                            &m3,
                            "Remove all files from the queue",
                            egui::vec2(96.0, 40.0),
                        )
                        .on_hover_cursor(widgets::click_cursor(clear_enabled));
                        if clear_resp.clicked() && clear_enabled {
                            self.clear_all();
                        }
                    }
                    let theme_icon = if self.dark_mode {
                        icons::tinted("theme", icons::LIGHT_MODE, 20.0, m3.on_surface_variant)
                    } else {
                        icons::tinted("theme", icons::DARK_MODE, 20.0, m3.on_surface_variant)
                    };
                    if widgets::icon_button(
                        ui,
                        theme_icon,
                        if self.dark_mode {
                            "Switch to light theme"
                        } else {
                            "Switch to dark theme"
                        },
                    )
                    .clicked()
                    {
                        self.dark_mode = !self.dark_mode;
                        theme::set_dark(&ctx, self.dark_mode);
                    }
                });
            });
            ui.add_space(4.0);
        });

        // ── M3 Settings: output + segmented backend ─────────────
        egui::Panel::top("settings").show(ui, |ui| {
            ui.add_space(4.0);
            widgets::card_frame(&m3).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(16.0, 0.0);
                    widgets::tonal_icon(
                        ui,
                        "out-lead",
                        icons::FOLDER_OPEN,
                        18.0,
                        m3.primary,
                        m3.primary_container,
                        40.0,
                    );
                    ui.vertical(|ui| {
                        ui.spacing_mut().item_spacing = egui::vec2(0.0, 2.0);
                        ui.add_space(2.0);
                        ui.label(
                            egui::RichText::new("Output folder")
                                .size(14.0)
                                .color(m3.on_surface),
                        );
                        ui.add(
                            egui::Label::new(
                                egui::RichText::new(widgets::trunc_middle(
                                    &self.output_dir.display().to_string(),
                                    70,
                                ))
                                .monospace()
                                .size(13.0)
                                .color(m3.on_surface_variant),
                            )
                            .truncate(),
                        )
                        .on_hover_text(self.output_dir.display().to_string());
                        ui.add_space(2.0);
                    });
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if widgets::icon_button(
                            ui,
                            icons::tinted("out-pick", icons::FOLDER, 18.0, m3.primary),
                            "Choose output folder",
                        )
                        .clicked()
                            && let Some(d) = rfd::FileDialog::new().pick_folder()
                        {
                            self.output_dir = d;
                            self.output_chosen = true;
                        }
                    });
                });
            });
            ui.add_space(4.0);
            // Advanced collapsible with M3 expand chevron (SVG).
            let adv_open = self.show_advanced;
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing = egui::vec2(4.0, 0.0);
                let chevron = if adv_open {
                    icons::tinted("adv-less", icons::EXPAND_LESS, 20.0, m3.primary)
                } else {
                    icons::tinted("adv-more", icons::EXPAND_MORE, 20.0, m3.primary)
                };
                if ui
                    .add(
                        egui::Button::image_and_text(
                            chevron,
                            egui::RichText::new("Advanced")
                                .size(13.0)
                                .strong()
                                .color(m3.primary),
                        )
                        .frame(false),
                    )
                    .on_hover_text("Models and inference provider")
                    .on_hover_cursor(egui::CursorIcon::PointingHand)
                    .clicked()
                {
                    self.show_advanced = !self.show_advanced;
                }
            });
            if self.show_advanced {
                widgets::card_frame(&m3).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("Backend")
                                .size(13.0)
                                .strong()
                                .color(m3.on_surface),
                        )
                        .on_hover_text("Inference provider");
                        ui.add_space(8.0);
                        // M3 segmented buttons.
                        egui::Frame::NONE
                            .fill(m3.surface_highest)
                            .stroke(egui::Stroke::new(1.0, m3.outline_variant))
                            .corner_radius(egui::CornerRadius::same(20))
                            .inner_margin(egui::Margin::same(4))
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.spacing_mut().item_spacing = egui::vec2(4.0, 4.0);
                                    let cpu_sel = self.backend == BackendChoice::Cpu;
                                    let cpu_btn = egui::Button::image_and_text(
                                        icons::tinted(
                                            "cpu",
                                            icons::CPU,
                                            16.0,
                                            if cpu_sel {
                                                m3.on_secondary_container
                                            } else {
                                                m3.on_surface_variant
                                            },
                                        ),
                                        egui::RichText::new("CPU").size(13.0).strong().color(
                                            if cpu_sel {
                                                m3.on_secondary_container
                                            } else {
                                                m3.on_surface_variant
                                            },
                                        ),
                                    )
                                    .fill(if cpu_sel {
                                        m3.secondary_container
                                    } else {
                                        egui::Color32::TRANSPARENT
                                    })
                                    .stroke(egui::Stroke::NONE)
                                    .corner_radius(egui::CornerRadius::same(16))
                                    .min_size(egui::vec2(118.0, 36.0));
                                    if ui
                                        .add(cpu_btn)
                                        .on_hover_cursor(egui::CursorIcon::PointingHand)
                                        .clicked()
                                    {
                                        self.backend = BackendChoice::Cpu;
                                    }
                                    let gpu_sel = self.backend == BackendChoice::DirectMl;
                                    let gpu_btn = egui::Button::image_and_text(
                                        icons::tinted(
                                            "gpu",
                                            icons::GPU,
                                            16.0,
                                            if gpu_sel {
                                                m3.on_secondary_container
                                            } else {
                                                m3.on_surface_variant
                                            },
                                        ),
                                        egui::RichText::new("DirectML").size(13.0).strong().color(
                                            if gpu_sel {
                                                m3.on_secondary_container
                                            } else {
                                                m3.on_surface_variant
                                            },
                                        ),
                                    )
                                    .fill(if gpu_sel {
                                        m3.secondary_container
                                    } else {
                                        egui::Color32::TRANSPARENT
                                    })
                                    .stroke(egui::Stroke::NONE)
                                    .corner_radius(egui::CornerRadius::same(16))
                                    .min_size(egui::vec2(128.0, 36.0));
                                    if ui
                                        .add(gpu_btn)
                                        .on_hover_cursor(egui::CursorIcon::PointingHand)
                                        .clicked()
                                    {
                                        self.backend = BackendChoice::DirectMl;
                                    }
                                });
                            });
                    });
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("Models")
                                .size(13.0)
                                .strong()
                                .color(m3.on_surface),
                        )
                        .on_hover_text(format!(
                            "Hugging Face models cached in {}",
                            models::models_dir().display()
                        ));
                        ui.with_layout(
                            egui::Layout::right_to_left(egui::Align::Center),
                            |ui| {
                                if widgets::icon_button(
                                    ui,
                                    icons::tinted(
                                        "models-folder",
                                        icons::FOLDER_OPEN,
                                        18.0,
                                        m3.primary,
                                    ),
                                    &format!("Open {}", models::models_dir().display()),
                                )
                                .clicked()
                                {
                                    let dir = models::ensure_models_dir()
                                        .unwrap_or_else(|_| models::models_dir());
                                    reveal_in_folder(&dir.join("_.tmp"));
                                }
                            },
                        );
                    });
                    self.show_model_row(
                        ui,
                        &m3,
                        &models::DET,
                        "Detector",
                        "Detector ONNX model",
                    );
                    self.show_model_row(
                        ui,
                        &m3,
                        &models::CLEAN,
                        "Cleaner",
                        "Cleaner ONNX model",
                    );
                    if let Some(err) = self.dl_error.clone() {
                        ui.horizontal(|ui| {
                            ui.add_space(24.0);
                            ui.label(
                                egui::RichText::new(format!("download failed: {err}"))
                                    .size(12.0)
                                    .color(m3.error),
                            );
                            if ui
                                .small_button("Retry")
                                .on_hover_text("Retry failed download")
                                .clicked()
                            {
                                self.dl_error = None;
                                let needed = self.needed_specs();
                                if !needed.is_empty() {
                                    self.begin_downloads(needed, self.dl_pending_run);
                                }
                            }
                        });
                    }
                });
            }
            ui.add_space(4.0);
        });

        // ── M3 bottom bar: linear progress + log ─────────────────
        egui::Panel::bottom("status").show(ui, |ui| {
            ui.add_space(4.0);
            // Model download takes over the progress slot while active (lazy on Run).
            if let Some(dl) = self.dl_current.as_ref() {
                let frac = (dl.percent / 100.0).clamp(0.0, 1.0);
                ui.horizontal(|ui| {
                    ui.spinner();
                    let text = format!(
                        "downloading {} • {} / {}",
                        dl.spec.filename,
                        fmt_size(dl.downloaded),
                        if dl.total > 0 {
                            fmt_size(dl.total)
                        } else {
                            "…".to_string()
                        }
                    );
                    ui.add(
                        egui::ProgressBar::new(frac)
                            .show_percentage()
                            .text(text)
                            .corner_radius(egui::CornerRadius::same(4)),
                    );
                });
            } else {
                let frac = if self.total > 0 {
                    (self.done as f32 / self.total as f32).clamp(0.0, 1.0)
                } else {
                    0.0
                };
            ui.horizontal(|ui| {
                let denom = self.total.max(self.items.len());
                widgets::pill(
                    ui,
                    &format!("{}/{}", self.done, denom),
                    m3.surface_highest,
                    m3.on_surface,
                );
                let text = if self.running && !self.current.is_empty() {
                    widgets::trunc_middle(&self.current, 70)
                } else if self.total > 0 {
                    format!("{}/{} done", self.done, self.total)
                } else {
                    "idle".to_string()
                };
                let eta = self.eta_text();
                // ProgressBar defaults to all remaining row width, which would
                // starve the ETA label — cap it to reserve room when ETA shows.
                let mut bar = egui::ProgressBar::new(frac)
                    .show_percentage()
                    .text(text)
                    .corner_radius(egui::CornerRadius::same(4));
                if !eta.is_empty() {
                    let reserve = 220.0;
                    let bar_w =
                        (ui.available_width() - reserve - ui.spacing().item_spacing.x).max(80.0);
                    bar = bar.desired_width(bar_w);
                }
                ui.add(bar);
                if !eta.is_empty() {
                    ui.label(
                        egui::RichText::new(eta)
                            .size(12.0)
                            .color(m3.on_surface_variant),
                    );
                }
            });
            }
            ui.horizontal(|ui| {
                let downloading = self.dl_current.is_some();
                let (icon_bytes, icon_name, color) =
                    if self.log.starts_with("error")
                        || self.log.starts_with("setup failed")
                        || self.log.starts_with("model download failed")
                    {
                        (icons::ALERT, "log-err", m3.error)
                    } else if self.log.starts_with("done:") {
                        (icons::CHECK, "log-ok", m3.success)
                    } else if self.running || downloading {
                        (icons::LOADER, "log-run", m3.primary)
                    } else {
                        (icons::CLOCK, "log-idle", m3.outline)
                    };
                if self.running || downloading {
                    ui.spinner();
                } else {
                    ui.add(icons::tinted(icon_name, icon_bytes, 16.0, color));
                }
                ui.label(
                    egui::RichText::new(&self.log)
                        .size(12.5)
                        .color(m3.on_surface_variant),
                );
            });
            ui.add_space(4.0);
        });

        // ── Preview side sheet (toggleable) ──────────────────────
        if self.preview_open && self.selected.is_some_and(|s| s < self.items.len()) {
            egui::Panel::right("preview")
                .resizable(true)
                .default_size(480.0)
                .min_size(340.0)
                .max_size(680.0)
                .show(ui, |ui| {
                    self.show_preview_panel(ui, &m3);
                });
        }

        // ── Queue ────────────────────────────────────────────────
        egui::CentralPanel::default().show(ui, |ui| {
            if hovered {
                let rect = ui.available_rect_before_wrap();
                ui.painter().rect_filled(
                    rect,
                    egui::CornerRadius::same(28),
                    m3.primary_container,
                );
                ui.painter().rect_stroke(
                    rect,
                    egui::CornerRadius::same(28),
                    egui::Stroke::new(2.0, m3.primary),
                    egui::StrokeKind::Inside,
                );
                ui.centered_and_justified(|ui| {
                    ui.vertical_centered(|ui| {
                        widgets::tonal_icon(
                            ui,
                            "drop",
                            icons::UPLOAD,
                            40.0,
                            m3.on_primary_container,
                            m3.primary,
                            80.0,
                        );
                        ui.add_space(12.0);
                        ui.label(
                            egui::RichText::new("Drop to add images")
                                .size(20.0)
                                .strong()
                                .color(m3.on_primary_container),
                        );
                    });
                });
                return;
            }
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("Queue")
                        .size(18.0)
                        .strong()
                        .color(m3.on_surface),
                );
                widgets::pill(
                    ui,
                    &format!("{}", self.items.len()),
                    m3.surface_highest,
                    m3.on_surface,
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    {
                        let add_enabled = !self.running;
                        let add_resp = widgets::tonal_button(
                            ui,
                            "add",
                            icons::PLUS,
                            "Add",
                            &m3,
                            "Add images (files or folders via drag & drop too)",
                            egui::vec2(96.0, 40.0),
                        )
                        .on_hover_cursor(widgets::click_cursor(add_enabled));
                        if add_resp.clicked()
                            && add_enabled
                            && let Some(files) = rfd::FileDialog::new()
                                .add_filter("images", &["jpg", "jpeg", "png", "webp", "bmp"])
                                .pick_files()
                        {
                            self.add_paths(files);
                        }
                    }
                    ui.add_space(8.0);
                    {
                        // M3 filter chip: same 40dp stadium height as Add so the
                        // row stays aligned. Replaces ComboBox (shorter box +
                        // no icon slot) with a popup menu.
                        let cur = self.sort;
                        let mut next_sort = cur;
                        ui.add_enabled_ui(!self.running, |ui| {
                            let btn = egui::Button::image_and_text(
                                icons::tinted(
                                    "sort-lead",
                                    icons::SORT,
                                    16.0,
                                    m3.on_surface_variant,
                                ),
                                egui::RichText::new(cur.label())
                                    .size(13.0)
                                    .color(m3.on_surface),
                            )
                            .fill(m3.surface_highest)
                            .stroke(egui::Stroke::new(1.0, m3.outline_variant))
                            .corner_radius(egui::CornerRadius::same(20))
                            .min_size(egui::vec2(168.0, 40.0));
                            let resp = ui
                                .add(btn)
                                .on_hover_text("Sort queue (disabled while running)")
                                .on_hover_cursor(egui::CursorIcon::PointingHand);
                            // Trailing chevron painted (not a widget) so the whole
                            // chip stays one click target. A "▾" text glyph would
                            // be tofu — it is missing from egui's bundled font.
                            let chev_tint = if self.running {
                                m3.outline
                            } else {
                                m3.on_surface_variant
                            };
                            icons::tinted("sort-chev", icons::EXPAND_MORE, 16.0, chev_tint)
                                .paint_at(
                                    ui,
                                    egui::Rect::from_center_size(
                                        resp.rect.right_center() - egui::vec2(18.0, 0.0),
                                        egui::vec2(16.0, 16.0),
                                    ),
                                );
                            egui::Popup::menu(&resp).show(|ui| {
                                for key in SortKey::ALL {
                                    ui.selectable_value(&mut next_sort, key, key.label());
                                }
                            });
                        });
                        if next_sort != self.sort {
                            self.set_sort(next_sort);
                        }
                    }
                });
            });
            ui.label(widgets::on_surface_variant(
                "Drag & drop files or folders anywhere. Eye opens preview, Del removes, Enter runs.",
                &m3,
                12.0,
            ));
            ui.add_space(4.0);
            ui.separator();
            ui.add_space(4.0);
            // Lazily fill the 56dp thumbnail cache (max 8 per frame to avoid jank).
            {
                let mut loaded = 0;
                for it in &self.items {
                    if loaded >= 8 {
                        break;
                    }
                    if !self.thumbs.contains_key(&it.path) {
                        let name = format!("thumb-{}", it.path.display());
                        if let Some(tex) = load_thumb_texture(&ctx, &it.path, name) {
                            self.thumbs.insert(it.path.clone(), tex);
                        }
                        loaded += 1;
                    }
                }
            }
            egui::ScrollArea::vertical()
                .auto_shrink([false; 2])
                .show(ui, |ui| {
                    if self.items.is_empty() {
                        ui.centered_and_justified(|ui| {
                            widgets::card_frame(&m3).show(ui, |ui| {
                                ui.vertical_centered(|ui| {
                                    ui.add_space(12.0);
                                    widgets::tonal_icon(
                                        ui,
                                        "empty",
                                        icons::UPLOAD,
                                        32.0,
                                        m3.primary,
                                        m3.primary_container,
                                        64.0,
                                    );
                                    ui.add_space(12.0);
                                    ui.label(
                                        egui::RichText::new("Queue is empty")
                                            .size(16.0)
                                            .strong()
                                            .color(m3.on_surface),
                                    );
                                    ui.label(widgets::on_surface_variant(
                                        "Drop images to start",
                                        &m3,
                                        13.0,
                                    ));
                                    ui.add_space(12.0);
                                });
                            });
                        });
                        return;
                    }
                    let mut select: Option<usize> = None;
                    let mut remove: Option<usize> = None;
                    let mut preview: Option<usize> = None;
                    let mut hover_row: Option<usize> = None;
                    // M3 list: flat full-bleed rows on the panel surface with
                    // dividers (no nested cards). Selected = secondary-container.
                    ui.spacing_mut().item_spacing = egui::vec2(0.0, 0.0);
                    {
                        let n = self.items.len();
                        for (i, it) in self.items.iter().enumerate() {
                            let selected = self.selected == Some(i);
                            let name = file_name_of(&it.path);
                            let preview_active =
                                self.preview_open && self.selected == Some(i);
                            let row_hovered = self.hover_row == Some(i);
                            let (support_text, support_color) = match &it.status {
                                FileStatus::Queued => {
                                    ("Queued".to_string(), m3.on_surface_variant)
                                }
                                FileStatus::Running => {
                                    ("Running…".to_string(), m3.primary)
                                }
                                FileStatus::Done { boxes } => {
                                    let s = if *boxes == 1 {
                                        "Done • 1 region".to_string()
                                    } else {
                                        format!("Done • {boxes} regions")
                                    };
                                    (s, m3.on_surface_variant)
                                }
                                FileStatus::Error(msg) => (
                                    format!(
                                        "Error • {}",
                                        widgets::trunc_middle(msg, 40)
                                    ),
                                    m3.error,
                                ),
                            };
                            let hover_text = {
                                let meta =
                                    format!("{} • {}", fmt_size(it.size), fmt_age(it.modified));
                                match &it.status {
                                    FileStatus::Error(msg) => {
                                        format!("{}\n{meta}\nError: {msg}", it.path.display())
                                    }
                                    _ => format!("{}\n{meta}", it.path.display()),
                                }
                            };
                            let thumb = self.thumbs.get(&it.path).cloned();
                            let row = egui::Frame::NONE
                                .fill(if selected {
                                    m3.secondary_container
                                } else if row_hovered {
                                    m3.surface_high
                                } else {
                                    egui::Color32::TRANSPARENT
                                })
                                .corner_radius(egui::CornerRadius::same(0))
                                .inner_margin(egui::Margin {
                                    left: 16,
                                    right: 8,
                                    top: 8,
                                    bottom: 8,
                                })
                                .show(ui, |ui| {
                                    let mut label_clicked = false;
                                    let mut remove_clicked = false;
                                    let mut preview_clicked = false;
                                    ui.horizontal(|ui| {
                                        if matches!(it.status, FileStatus::Running) {
                                            widgets::thumbnail_spinner(ui, 44.0, &m3);
                                        } else {
                                            widgets::thumbnail(ui, thumb.as_ref(), 44.0, &m3);
                                        }
                                        ui.add_space(4.0);
                                        ui.vertical(|ui| {
                                            ui.set_min_size(egui::vec2(80.0, 44.0));
                                            ui.add(
                                                egui::Label::new(
                                                    egui::RichText::new(
                                                        widgets::trunc_middle(&name, 40),
                                                    )
                                                    .size(14.0)
                                                    .color(if selected {
                                                        m3.on_secondary_container
                                                    } else {
                                                        m3.on_surface
                                                    }),
                                                )
                                                .truncate()
                                                .sense(egui::Sense::click()),
                                            )
                                            .on_hover_text(hover_text.clone());
                                            ui.add(
                                                egui::Label::new(
                                                    egui::RichText::new(support_text.clone())
                                                        .size(12.5)
                                                        .color(if selected {
                                                            m3.on_secondary_container
                                                        } else {
                                                            support_color
                                                        }),
                                                )
                                                .truncate(),
                                            );
                                            // Clicking text selects the row.
                                            if ui.ctx().input(|inp| {
                                                inp.pointer.primary_clicked()
                                            }) && ui
                                                .rect_contains_pointer(
                                                    ui.min_rect(),
                                                )
                                            {
                                                label_clicked = true;
                                            }
                                        });
                                        ui.with_layout(
                                            egui::Layout::right_to_left(egui::Align::Center),
                                            |ui| {
                                                // Always reserve the delete slot so the eye
                                                // never shifts under the cursor on hover.
                                                // `add_visible(false, …)` keeps layout space.
                                                let show_x = !self.running;
                                                let x_resp = ui.add_visible(
                                                    show_x,
                                                    egui::Button::image(
                                                        icons::tinted(
                                                            "row-x",
                                                            icons::X,
                                                            20.0,
                                                            m3.on_surface_variant,
                                                        ),
                                                    )
                                                    .frame(false)
                                                    .min_size(egui::vec2(40.0, 40.0)),
                                                );
                                                let x_resp = x_resp
                                                    .on_hover_text("Remove from queue (Del)")
                                                    .on_hover_cursor(
                                                        egui::CursorIcon::PointingHand,
                                                    );
                                                if show_x && x_resp.clicked() {
                                                    remove_clicked = true;
                                                }
                                                if widgets::icon_button(
                                                    ui,
                                                    icons::tinted(
                                                        "row-eye",
                                                        icons::EYE,
                                                        20.0,
                                                        if preview_active {
                                                            m3.primary
                                                        } else {
                                                            m3.on_surface_variant
                                                        },
                                                    ),
                                                    "Open preview panel",
                                                )
                                                .clicked()
                                                {
                                                    preview_clicked = true;
                                                }
                                            },
                                        );
                                    });
                                    (label_clicked, remove_clicked, preview_clicked)
                                });
                            let row_resp =
                                row.response.on_hover_cursor(egui::CursorIcon::PointingHand);
                            let (label_clicked, remove_clicked, preview_clicked) = row.inner;
                            if row_resp.hovered() {
                                hover_row = Some(i);
                            }
                            // Whole row selects (M3 list behavior).
                            // NOTE: Frame responses only sense hover, so use
                            // hovered()+primary_clicked() instead of clicked_by().
                            if remove_clicked {
                                remove = Some(i);
                            } else if preview_clicked {
                                preview = Some(i);
                            } else if label_clicked
                                || (row_resp.hovered()
                                    && ui.ctx().input(|inp| inp.pointer.primary_clicked()))
                            {
                                select = Some(i);
                            }
                            if i + 1 < n {
                                ui.separator();
                            }
                        }
                    }
                    if let Some(i) = preview {
                        self.selected = Some(i);
                        self.preview_for = None;
                        self.preview_open = true;
                    } else if let Some(i) = select {
                        self.selected = Some(i);
                        self.preview_for = None;
                    }
                    if let Some(i) = remove {
                        self.remove_item(i);
                    }
                    self.hover_row = hover_row;
                });
        });
    }

    fn on_exit(&mut self) {
        // Auto-apply a staged update on close: hands off to Velopack's
        // Update.exe (silent, no restart); the next launch runs the new version.
        let staged = self
            .pending_update
            .lock()
            .map(|g| g.clone())
            .unwrap_or(None);
        if let Some(info) = staged.as_ref() {
            updater::apply_on_exit(info);
        }
    }
}

impl MarklessApp {
    /// Before/After preview in an M3 side sheet.
    fn show_preview_panel(&mut self, ui: &mut egui::Ui, m3: &M3Colors) {
        let Some(sel) = self.selected else {
            ui.vertical_centered(|ui| {
                ui.add_space(60.0);
                widgets::tonal_icon(
                    ui,
                    "no-sel",
                    icons::IMAGE,
                    36.0,
                    m3.primary,
                    m3.primary_container,
                    72.0,
                );
                ui.add_space(12.0);
                ui.label(
                    egui::RichText::new("Nothing selected")
                        .size(18.0)
                        .strong()
                        .color(m3.on_surface),
                );
                ui.label(widgets::on_surface_variant(
                    "Add files or pick an item in the queue",
                    m3,
                    13.0,
                ));
            });
            return;
        };
        let Some(item) = self.items.get(sel) else {
            return;
        };
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing = egui::vec2(16.0, 0.0);
            widgets::tonal_icon(
                ui,
                "preview",
                icons::COMPARE,
                18.0,
                m3.on_primary_container,
                m3.primary_container,
                36.0,
            );
            ui.label(
                egui::RichText::new("Preview")
                    .size(18.0)
                    .strong()
                    .color(m3.on_surface),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if widgets::icon_button(
                    ui,
                    icons::tinted("preview-close", icons::X, 18.0, m3.on_surface_variant),
                    "Close preview panel (Esc)",
                )
                .clicked()
                {
                    self.preview_open = false;
                }
                if let Some(out) = item.output.as_ref() {
                    let tip = format!("Open output folder\n{}", out.display());
                    if widgets::icon_button(
                        ui,
                        icons::tinted("reveal", icons::FOLDER_OPEN, 18.0, m3.on_surface_variant),
                        &tip,
                    )
                    .clicked()
                    {
                        reveal_in_folder(out);
                    }
                }
                let (pill_text, _, color) = status_meta(&item.status, m3);
                widgets::status_pill(ui, pill_text, color);
            });
        });
        ui.add(
            egui::Label::new(widgets::on_surface_variant(
                &widgets::trunc_middle(&item.path.display().to_string(), 80),
                m3,
                12.0,
            ))
            .truncate(),
        )
        .on_hover_text(item.path.display().to_string());
        ui.add_space(4.0);
        ui.separator();
        ui.add_space(4.0);

        let before = self.before_tex.clone();
        let after = self.after_tex.clone();
        let before_dims = before.as_ref().map(|t| t.size());
        let after_dims = after.as_ref().map(|t| t.size());
        let after_hint = match &item.status {
            FileStatus::Done { .. } => "output missing".to_string(),
            FileStatus::Running => "processing…".to_string(),
            FileStatus::Error(e) => e.clone(),
            FileStatus::Queued => "press Run".to_string(),
        };
        let gap = ui.spacing().item_spacing.x;
        if ui.available_width() >= 480.0 {
            ui.columns(2, |cols| {
                image_card(
                    &mut cols[0],
                    m3,
                    "BEFORE",
                    "mm-img-before",
                    icons::IMAGE,
                    m3.on_secondary_container,
                    m3.secondary_container,
                    before,
                    before_dims,
                    "no preview",
                    &mut self.img_scroll,
                );
                image_card(
                    &mut cols[1],
                    m3,
                    "AFTER",
                    "mm-img-after",
                    icons::CHECK,
                    m3.success,
                    m3.success_container,
                    after,
                    after_dims,
                    &after_hint,
                    &mut self.img_scroll,
                );
            });
        } else {
            let half_w = ui.available_width();
            let half_h = ((ui.available_height() - gap) / 2.0).max(140.0);
            ui.allocate_ui(egui::vec2(half_w, half_h), |ui| {
                image_card(
                    ui,
                    m3,
                    "BEFORE",
                    "mm-img-before",
                    icons::IMAGE,
                    m3.on_secondary_container,
                    m3.secondary_container,
                    before,
                    before_dims,
                    "no preview",
                    &mut self.img_scroll,
                );
            });
            ui.allocate_ui(egui::vec2(half_w, half_h), |ui| {
                image_card(
                    ui,
                    m3,
                    "AFTER",
                    "mm-img-after",
                    icons::CHECK,
                    m3.success,
                    m3.success_container,
                    after,
                    after_dims,
                    &after_hint,
                    &mut self.img_scroll,
                );
            });
        }
    }
}
