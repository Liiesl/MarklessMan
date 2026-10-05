//! Silent Velopack updater: auto-download when available, apply on close.
//!
//! A background thread calls [`check_and_download`] once shortly after
//! startup. When a release is found on GitHub it is downloaded immediately
//! (no prompt, no progress UI) and staged; [`apply_on_exit`] hands the
//! staged update to Velopack's Update.exe when the app closes.
//!
//! Gated behind the `updates` feature so `--no-default-features` skips
//! velopack entirely. The stubs below mirror the real API so `main.rs`
//! compiles unchanged in both configs. Outside a Velopack install
//! (e.g. `cargo run`) `create_manager` fails and every call is a silent
//! no-op.

#[cfg(feature = "updates")]
use velopack::{sources, UpdateCheck, UpdateManager};

/// Update descriptor. Real type when `updates` is on, empty stub otherwise
/// (never constructed — `check_and_download` always returns `None`).
#[cfg(feature = "updates")]
pub use velopack::UpdateInfo;
#[cfg(not(feature = "updates"))]
#[derive(Debug, Clone, Default)]
pub struct UpdateInfo;

#[cfg(feature = "updates")]
const GITHUB_REPO: &str = "https://github.com/Liiesl/MarklessMan";

#[cfg(feature = "updates")]
fn create_manager() -> Option<UpdateManager> {
    let source = sources::GithubSource::new(GITHUB_REPO, None, false);
    UpdateManager::new(source, None, None).ok()
}

/// Check for updates and download immediately when one is available.
/// Returns the staged update so the app can apply it on exit.
#[cfg(feature = "updates")]
pub fn check_and_download() -> Option<UpdateInfo> {
    let um = create_manager()?;
    let info = match um.check_for_updates().ok()? {
        UpdateCheck::UpdateAvailable(info) => *info,
        _ => return None,
    };
    um.download_updates(&info, None).ok()?;
    Some(info)
}

/// Hand a staged update to the Velopack updater and return immediately.
/// Update.exe waits for this process to exit, applies silently, and does
/// not restart — the next launch runs the new version.
#[cfg(feature = "updates")]
pub fn apply_on_exit(info: &UpdateInfo) {
    if let Some(um) = create_manager() {
        let _ = um.wait_exit_then_apply_updates(info, true, false, Vec::<String>::new());
    }
}

#[cfg(not(feature = "updates"))]
#[allow(dead_code)]
pub fn check_and_download() -> Option<UpdateInfo> {
    None
}

#[cfg(not(feature = "updates"))]
#[allow(dead_code)]
pub fn apply_on_exit(_info: &UpdateInfo) {}
