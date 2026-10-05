//! Model resolving for the MarklessMan GUI.
//!
//! Mirrors `ManhwaOCR/engine-pool/model-store` but GUI-owned (no new crate):
//! a tiny registry of Hugging Face assets plus `fast-down-api` resumable
//! downloads persisted under `%APPDATA%\marklessman\models`.
//!
//! The default is always the canonical AppData path. The Advanced panel only
//! lets the user swap in a custom file via the picker (manual override).

use std::path::PathBuf;
use std::sync::{OnceLock, mpsc};

/// Describes a single downloadable model asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelSpec {
    /// Stable identifier, e.g. `det`.
    pub id: &'static str,
    /// Filename as stored on disk inside [`models_dir`].
    pub filename: &'static str,
    /// Download URL (kept verbatim including `?download=true` for Hugging Face).
    pub url: &'static str,
    /// Human-readable description.
    pub description: &'static str,
}

pub const DET: ModelSpec = ModelSpec {
    id: "det",
    filename: "marklessman-detv1.onnx",
    url: "https://huggingface.co/Liiesl/marklessman-detv1/resolve/main/marklessman-detv1.onnx?download=true",
    description: "Watermark detector (YOLO)",
};

pub const CLEAN: ModelSpec = ModelSpec {
    id: "clean",
    filename: "marklessman-cleanv1_256_fp32.onnx",
    url: "https://huggingface.co/Liiesl/marklessman-cleanv1/resolve/main/marklessman-cleanv1_256_fp32.onnx?download=true",
    description: "Watermark cleaner (SLBR 256px fp32)",
};

/// All registered models in display order.
pub const MODELS: &[ModelSpec] = &[DET, CLEAN];

/// Find a model by its stable `id`.
pub fn get_model(id: &str) -> Option<&'static ModelSpec> {
    MODELS.iter().find(|m| m.id == id)
}

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

/// `%APPDATA%\marklessman\models` on Windows,
/// `~/.config/marklessman/models` on Linux/macOS.
pub fn models_dir() -> PathBuf {
    if let Some(dir) = dirs::config_dir() {
        return dir.join("marklessman").join("models");
    }
    // Fallback: raw APPDATA env (Windows) or cwd-relative.
    if let Ok(appdata) = std::env::var("APPDATA") {
        return PathBuf::from(appdata).join("marklessman").join("models");
    }
    PathBuf::from("models")
}

/// Ensure [`models_dir`] exists, creating it recursively if needed.
pub fn ensure_models_dir() -> std::io::Result<PathBuf> {
    let dir = models_dir();
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Expected on-disk (canonical AppData) path for a registered model.
pub fn model_path(spec: &ModelSpec) -> PathBuf {
    models_dir().join(spec.filename)
}

/// Resolve the file to load: always the canonical AppData path.
/// Anything else (e.g. a file-picker override) is the user's explicit choice
/// stored in the GUI state, never auto-resolved here.
pub fn resolve_model_path(spec: &ModelSpec) -> PathBuf {
    model_path(spec)
}

/// Returns true if the canonical AppData file exists.
pub fn is_downloaded(spec: &ModelSpec) -> bool {
    model_path(spec).exists()
}

/// List all models missing from the canonical AppData dir.
#[allow(dead_code)]
pub fn missing_models() -> Vec<&'static ModelSpec> {
    MODELS.iter().filter(|m| !is_downloaded(m)).collect()
}

// ---------------------------------------------------------------------------
// Downloads (fast-down-api)
// ---------------------------------------------------------------------------

fn rt() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_name("marklessman-dl")
            .worker_threads(2)
            .build()
            .expect("download runtime")
    })
}

fn config_for(spec: &ModelSpec, overwrite: bool) -> fast_down_api::PartialConfig {
    fast_down_api::PartialConfig {
        save_dir: Some(models_dir()),
        filename: Some(spec.filename.to_string()),
        overwrite: Some(overwrite),
        // Same defaults as ManhwaOCR: 16 threads, resume, 20 redirects for HF.
        threads: Some(16),
        resume: Some(true),
        max_redirects: Some(20),
        ..Default::default()
    }
}

fn prepare_url(spec: &ModelSpec) -> Result<url::Url, String> {
    if spec.url.is_empty() {
        return Err(format!("model '{}' has no URL", spec.id));
    }
    url::Url::parse(spec.url).map_err(|e| format!("invalid URL for {}: {e}", spec.id))
}

/// Strip Windows `\\?\` verbatim prefixes. `fast-down-api` hands back
/// `Event::Renamed` paths in that form (`\\?\C:\...`); keeping them would
/// leak the prefix into displayed and stored model paths.
pub fn normalize(path: PathBuf) -> PathBuf {
    #[cfg(windows)]
    {
        if let Some(s) = path.to_str() {
            if let Some(rest) = s.strip_prefix(r"\\?\UNC\") {
                return PathBuf::from(format!(r"\\{rest}"));
            }
            if let Some(rest) = s.strip_prefix(r"\\?\") {
                return PathBuf::from(rest);
            }
        }
        path
    }
    #[cfg(not(windows))]
    {
        path
    }
}

/// Drive a `fast-down-api` download to completion, forwarding
/// `(percent 0..100, downloaded, total)` over `sender`.
#[allow(dead_code)]
pub async fn download_model_with_sender(
    spec: &ModelSpec,
    sender: mpsc::Sender<(f32, u64, u64)>,
    overwrite: bool,
) -> Result<PathBuf, String> {
    use fast_down_api::{Event, create_channel, create_cancellation_token, download};

    let url = prepare_url(spec)?;
    ensure_models_dir().map_err(|e| format!("failed to create models dir: {e}"))?;
    if !overwrite && model_path(spec).exists() {
        return Ok(model_path(spec));
    }
    let (tx, rx) = create_channel();
    let token = create_cancellation_token();
    download(url, config_for(spec, overwrite), tx, token);
    loop {
        let event = rx
            .recv()
            .await
            .map_err(|_| "download channel closed without completion".to_string())?;
        match event {
            Event::Progress(sample) => {
                let _ = sender.send((sample.percent as f32, sample.downloaded, sample.total));
            }
            Event::Renamed(p) => {
                if let Ok(meta) = std::fs::metadata(&p) {
                    let total = meta.len();
                    let _ = sender.send((100.0, total, total));
                } else {
                    let _ = sender.send((100.0, 0, 0));
                }
                return Ok(normalize(p));
            }
            Event::RenameFailed(e) => return Err(format!("rename failed for {}: {e}", spec.id)),
            Event::PrefetchError(e) => return Err(format!("prefetch failed for {}: {e}", spec.id)),
            Event::GenPathError(e) => return Err(format!("gen path failed for {}: {e}", spec.id)),
            Event::BuildClientError(e) => {
                return Err(format!("client build failed for {}: {e}", spec.id));
            }
            Event::BuildPusherError(e) => {
                return Err(format!("pusher build failed for {}: {e}", spec.id));
            }
            Event::ResumeError(e) => return Err(format!("resume failed for {}: {e:?}", spec.id)),
            _ => continue,
        }
    }
}

/// Spawn a download onto the shared runtime. Progress goes to `progress_tx`,
/// the final result to `result_tx`. Returns a token for cooperative cancel
/// (leaves `.part`/`.fd` sidecars for resume).
pub fn spawn_download(
    spec: &'static ModelSpec,
    progress_tx: mpsc::Sender<(f32, u64, u64)>,
    result_tx: mpsc::Sender<Result<PathBuf, String>>,
    overwrite: bool,
) -> tokio_util::sync::CancellationToken {
    use fast_down_api::{Event, create_cancellation_token, create_channel, download};

    let token = create_cancellation_token();
    let token_w = token.clone();
    rt().spawn(async move {
        let res: Result<PathBuf, String> = (|| async {
            let url = prepare_url(spec)?;
            ensure_models_dir().map_err(|e| format!("failed to create models dir: {e}"))?;
            if !overwrite && model_path(spec).exists() {
                return Ok(model_path(spec));
            }
            let (tx, rx) = create_channel();
            download(url, config_for(spec, overwrite), tx, token_w.clone());
            loop {
                let event = rx
                    .recv()
                    .await
                    .map_err(|_| "download channel closed without completion".to_string())?;
                match event {
                    Event::Progress(sample) => {
                        let _ = progress_tx.send((
                            sample.percent as f32,
                            sample.downloaded,
                            sample.total,
                        ));
                    }
                    Event::Renamed(p) => {
                        if let Ok(meta) = std::fs::metadata(&p) {
                            let total = meta.len();
                            let _ = progress_tx.send((100.0, total, total));
                        } else {
                            let _ = progress_tx.send((100.0, 0, 0));
                        }
                        return Ok(normalize(p));
                    }
                    Event::RenameFailed(e) => {
                        return Err(format!("rename failed for {}: {e}", spec.id));
                    }
                    Event::PrefetchError(e) => {
                        return Err(format!("prefetch failed for {}: {e}", spec.id));
                    }
                    Event::GenPathError(e) => {
                        return Err(format!("gen path failed for {}: {e}", spec.id));
                    }
                    Event::BuildClientError(e) => {
                        return Err(format!("client build failed for {}: {e}", spec.id));
                    }
                    Event::BuildPusherError(e) => {
                        return Err(format!("pusher build failed for {}: {e}", spec.id));
                    }
                    Event::ResumeError(e) => {
                        return Err(format!("resume failed for {}: {e:?}", spec.id));
                    }
                    _ => continue,
                }
            }
        })()
        .await;
        let _ = result_tx.send(res);
    });
    token
}
