//! Material 3 (Material You) theme for MarklessMan.
//!
//! M3 design tokens derived from seed `#5AA0FF` so brand stays blue.
//! Dark scheme is default; light scheme is styled too so the app bar
//! theme toggle just flips `ThemePreference`.

use eframe::egui;

/// M3 color set. `Copy` so call sites can grab `theme::colors(dark)`.
#[derive(Clone, Copy)]
pub struct M3Colors {
    pub primary: egui::Color32,
    pub on_primary: egui::Color32,
    pub primary_container: egui::Color32,
    pub on_primary_container: egui::Color32,
    pub secondary_container: egui::Color32,
    pub on_secondary_container: egui::Color32,
    pub surface: egui::Color32,
    pub surface_low: egui::Color32,
    pub surface_container: egui::Color32,
    pub surface_high: egui::Color32,
    pub surface_highest: egui::Color32,
    pub on_surface: egui::Color32,
    pub on_surface_variant: egui::Color32,
    pub outline: egui::Color32,
    pub outline_variant: egui::Color32,
    pub error: egui::Color32,
    pub on_error: egui::Color32,
    pub error_container: egui::Color32,
    pub success: egui::Color32,
    pub success_container: egui::Color32,
    pub warn: egui::Color32,
}

pub fn colors(dark: bool) -> M3Colors {
    if dark {
        M3Colors {
            primary: egui::Color32::from_rgb(0xA8, 0xC7, 0xFA),
            on_primary: egui::Color32::from_rgb(0x0A, 0x30, 0x5F),
            primary_container: egui::Color32::from_rgb(0x28, 0x47, 0x77),
            on_primary_container: egui::Color32::from_rgb(0xD6, 0xE3, 0xFF),
            secondary_container: egui::Color32::from_rgb(0x3E, 0x47, 0x59),
            on_secondary_container: egui::Color32::from_rgb(0xD6, 0xE3, 0xFF),
            surface: egui::Color32::from_rgb(0x13, 0x13, 0x18),
            surface_low: egui::Color32::from_rgb(0x1B, 0x1B, 0x1F),
            surface_container: egui::Color32::from_rgb(0x1E, 0x1F, 0x25),
            surface_high: egui::Color32::from_rgb(0x28, 0x2A, 0x31),
            surface_highest: egui::Color32::from_rgb(0x33, 0x35, 0x3D),
            on_surface: egui::Color32::from_rgb(0xE3, 0xE2, 0xE9),
            on_surface_variant: egui::Color32::from_rgb(0xC4, 0xC6, 0xD0),
            outline: egui::Color32::from_rgb(0x44, 0x47, 0x4F),
            outline_variant: egui::Color32::from_rgb(0x2A, 0x2C, 0x33),
            error: egui::Color32::from_rgb(0xFF, 0xB4, 0xAB),
            on_error: egui::Color32::from_rgb(0x69, 0x00, 0x05),
            error_container: egui::Color32::from_rgb(0x93, 0x00, 0x0A),
            success: egui::Color32::from_rgb(0x6D, 0xD5, 0x8C),
            success_container: egui::Color32::from_rgb(0x14, 0x53, 0x2D),
            warn: egui::Color32::from_rgb(0xFF, 0xB2, 0x24),
        }
    } else {
        M3Colors {
            primary: egui::Color32::from_rgb(0x0B, 0x57, 0xD0),
            on_primary: egui::Color32::WHITE,
            primary_container: egui::Color32::from_rgb(0xD6, 0xE3, 0xFF),
            on_primary_container: egui::Color32::from_rgb(0x0A, 0x30, 0x5F),
            secondary_container: egui::Color32::from_rgb(0xDC, 0xE2, 0xF0),
            on_secondary_container: egui::Color32::from_rgb(0x1A, 0x2C, 0x4E),
            surface: egui::Color32::from_rgb(0xFD, 0xF7, 0xFF),
            surface_low: egui::Color32::from_rgb(0xF7, 0xF2, 0xFA),
            surface_container: egui::Color32::from_rgb(0xF3, 0xED, 0xF7),
            surface_high: egui::Color32::from_rgb(0xEC, 0xE6, 0xF0),
            surface_highest: egui::Color32::from_rgb(0xE6, 0xE0, 0xE9),
            on_surface: egui::Color32::from_rgb(0x1D, 0x1B, 0x20),
            on_surface_variant: egui::Color32::from_rgb(0x49, 0x45, 0x4F),
            outline: egui::Color32::from_rgb(0x79, 0x74, 0x7E),
            outline_variant: egui::Color32::from_rgb(0xCA, 0xC4, 0xD0),
            error: egui::Color32::from_rgb(0xBA, 0x1A, 0x1A),
            on_error: egui::Color32::WHITE,
            error_container: egui::Color32::from_rgb(0xFF, 0xDA, 0xD6),
            success: egui::Color32::from_rgb(0x14, 0x6C, 0x2E),
            success_container: egui::Color32::from_rgb(0xC4, 0xEE, 0xD0),
            warn: egui::Color32::from_rgb(0x7A, 0x4D, 0x00),
        }
    }
}

/// Legacy palette kept so older call sites still compile.
/// Values are the M3 dark scheme.
#[allow(dead_code)]
pub mod palette {
    use eframe::egui::Color32;

    pub const BG: Color32 = Color32::from_rgb(0x13, 0x13, 0x18);
    pub const PANEL: Color32 = Color32::from_rgb(0x1B, 0x1B, 0x1F);
    pub const CARD: Color32 = Color32::from_rgb(0x1E, 0x1F, 0x25);
    pub const CARD_HOVER: Color32 = Color32::from_rgb(0x28, 0x2A, 0x31);
    pub const BORDER: Color32 = Color32::from_rgb(0x2A, 0x2C, 0x33);
    pub const TEXT: Color32 = Color32::from_rgb(0xE3, 0xE2, 0xE9);
    pub const MUTED: Color32 = Color32::from_rgb(0xC4, 0xC6, 0xD0);
    pub const FAINT: Color32 = Color32::from_rgb(0x8E, 0x90, 0x99);
    pub const ACCENT: Color32 = Color32::from_rgb(0xA8, 0xC7, 0xFA);
    pub const ACCENT_DIM: Color32 = Color32::from_rgb(0x28, 0x47, 0x77);
    pub const SUCCESS: Color32 = Color32::from_rgb(0x6D, 0xD5, 0x8C);
    pub const WARN: Color32 = Color32::from_rgb(0xFF, 0xB2, 0x24);
    pub const ERROR: Color32 = Color32::from_rgb(0xFF, 0xB4, 0xAB);

    // M3 aliases.
    pub const PRIMARY: Color32 = ACCENT;
    pub const ON_PRIMARY: Color32 = Color32::from_rgb(0x0A, 0x30, 0x5F);
    pub const PRIMARY_CONTAINER: Color32 = ACCENT_DIM;
    pub const ON_PRIMARY_CONTAINER: Color32 = Color32::from_rgb(0xD6, 0xE3, 0xFF);
    pub const SECONDARY_CONTAINER: Color32 = Color32::from_rgb(0x3E, 0x47, 0x59);
    pub const SURFACE: Color32 = BG;
    pub const SURFACE_LOW: Color32 = PANEL;
    pub const SURFACE_CONTAINER: Color32 = CARD;
    pub const SURFACE_HIGH: Color32 = CARD_HOVER;
    pub const SURFACE_HIGHEST: Color32 = Color32::from_rgb(0x33, 0x35, 0x3D);
    pub const ON_SURFACE: Color32 = TEXT;
    pub const ON_SURFACE_VARIANT: Color32 = MUTED;
    pub const OUTLINE: Color32 = Color32::from_rgb(0x44, 0x47, 0x4F);
    pub const OUTLINE_VARIANT: Color32 = BORDER;
    pub const ERROR_CONTAINER: Color32 = Color32::from_rgb(0x93, 0x00, 0x0A);
    pub const SUCCESS_CONTAINER: Color32 = Color32::from_rgb(0x14, 0x53, 0x2D);
}

/// Light palette mirror (for direct use if needed).
#[allow(dead_code)]
pub mod light_palette {
    use eframe::egui::Color32;

    pub const BG: Color32 = Color32::from_rgb(0xFD, 0xF7, 0xFF);
    pub const PANEL: Color32 = Color32::from_rgb(0xF7, 0xF2, 0xFA);
    pub const CARD: Color32 = Color32::from_rgb(0xF3, 0xED, 0xF7);
    pub const CARD_HOVER: Color32 = Color32::from_rgb(0xEC, 0xE6, 0xF0);
    pub const BORDER: Color32 = Color32::from_rgb(0xCA, 0xC4, 0xD0);
    pub const TEXT: Color32 = Color32::from_rgb(0x1D, 0x1B, 0x20);
    pub const MUTED: Color32 = Color32::from_rgb(0x49, 0x45, 0x4F);
    pub const FAINT: Color32 = Color32::from_rgb(0x79, 0x74, 0x7E);
    pub const ACCENT: Color32 = Color32::from_rgb(0x0B, 0x57, 0xD0);
    pub const ACCENT_DIM: Color32 = Color32::from_rgb(0xD6, 0xE3, 0xFF);
    pub const SUCCESS: Color32 = Color32::from_rgb(0x14, 0x6C, 0x2E);
    pub const WARN: Color32 = Color32::from_rgb(0x7A, 0x4D, 0x00);
    pub const ERROR: Color32 = Color32::from_rgb(0xBA, 0x1A, 0x1A);
}

fn style_m3(style: &mut egui::Style, c: M3Colors, dark: bool) {
    style.visuals.dark_mode = dark;
    style.visuals.panel_fill = c.surface_low;
    style.visuals.window_fill = c.surface_low;
    style.visuals.extreme_bg_color = c.surface;
    style.visuals.code_bg_color = c.surface_container;
    style.visuals.hyperlink_color = c.primary;
    style.visuals.warn_fg_color = c.warn;
    style.visuals.error_fg_color = c.error;
    style.visuals.selection.bg_fill = c.primary_container;
    style.visuals.selection.stroke = egui::Stroke::new(1.0, c.primary);

    // M3 shapes: dialogs extra-large, menus small, buttons stadium.
    style.visuals.window_corner_radius = egui::CornerRadius::same(28);
    style.visuals.menu_corner_radius = egui::CornerRadius::same(8);
    style.visuals.window_shadow = egui::Shadow {
        offset: [0, 4],
        blur: 16,
        spread: 0,
        color: egui::Color32::from_black_alpha(if dark { 120 } else { 40 }),
    };
    style.visuals.popup_shadow = egui::Shadow {
        offset: [0, 2],
        blur: 8,
        spread: 0,
        color: egui::Color32::from_black_alpha(if dark { 100 } else { 32 }),
    };

    for widgets in [
        &mut style.visuals.widgets.noninteractive,
        &mut style.visuals.widgets.inactive,
        &mut style.visuals.widgets.hovered,
        &mut style.visuals.widgets.active,
        &mut style.visuals.widgets.open,
    ] {
        // Stadium buttons / chips; cards set their own 12-16dp radius.
        widgets.corner_radius = egui::CornerRadius::same(20);
    }
    style.visuals.widgets.noninteractive.bg_fill = c.surface_container;
    style.visuals.widgets.noninteractive.weak_bg_fill = c.surface_container;
    style.visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, c.outline_variant);
    style.visuals.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0, c.on_surface_variant);

    style.visuals.widgets.inactive.bg_fill = c.surface_highest;
    style.visuals.widgets.inactive.weak_bg_fill = c.surface_highest;
    style.visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, c.outline_variant);
    style.visuals.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, c.on_surface);

    // State layers: 8% hover, 12% press approximated with lighter fill.
    style.visuals.widgets.hovered.bg_fill = c.surface_highest;
    style.visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, c.primary);
    style.visuals.widgets.hovered.fg_stroke = egui::Stroke::new(1.0, c.on_surface);
    style.visuals.widgets.hovered.expansion = 1.0;

    style.visuals.widgets.active.bg_fill = c.primary_container;
    style.visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0, c.primary);
    style.visuals.widgets.active.fg_stroke = egui::Stroke::new(1.0, c.on_primary_container);

    // M3 type scale (egui default font, M3 sizes).
    use egui::{FontFamily, FontId, TextStyle};
    style.text_styles.insert(
        TextStyle::Heading,
        FontId::new(22.0, FontFamily::Proportional),
    );
    style
        .text_styles
        .insert(TextStyle::Body, FontId::new(14.0, FontFamily::Proportional));
    style.text_styles.insert(
        TextStyle::Button,
        FontId::new(14.0, FontFamily::Proportional),
    );
    style.text_styles.insert(
        TextStyle::Small,
        FontId::new(12.0, FontFamily::Proportional),
    );
    style.text_styles.insert(
        TextStyle::Monospace,
        FontId::new(12.0, FontFamily::Monospace),
    );

    // 4dp grid-ish spacing.
    style.spacing.item_spacing = egui::vec2(8.0, 8.0);
    style.spacing.button_padding = egui::vec2(16.0, 10.0);
    style.spacing.window_margin = egui::Margin::same(16);
    style.spacing.menu_margin = egui::Margin::same(8);
    style.spacing.indent = 16.0;
    style.spacing.scroll = egui::style::ScrollStyle {
        floating: true,
        bar_width: 8.0,
        ..Default::default()
    };
}

pub fn apply(ctx: &egui::Context) {
    ctx.set_theme(egui::ThemePreference::Dark);
    let dark = colors(true);
    let light = colors(false);
    ctx.style_mut_of(egui::Theme::Dark, |style| {
        *style = egui::Style::default();
        style_m3(style, dark, true);
    });
    ctx.style_mut_of(egui::Theme::Light, |style| {
        *style = egui::Style::default();
        style_m3(style, light, false);
    });
}

/// Flip between pre-styled M3 dark / light schemes.
pub fn set_dark(ctx: &egui::Context, dark: bool) {
    ctx.set_theme(if dark {
        egui::ThemePreference::Dark
    } else {
        egui::ThemePreference::Light
    });
}

/// Install a system CJK fallback font so CJK file paths render instead of tofu.
///
/// `egui`'s bundled fonts (Hack / Ubuntu-Light) have no CJK glyphs. This
/// probes well-known OS CJK fonts, loads the first few found, and appends
/// them as lowest-priority fallbacks to both Proportional and Monospace
/// families (queue paths use `.monospace()`). No new deps, no binary bloat.
///
/// Returns the number of fallback fonts registered (0 when none found).
pub fn install_cjk_fallback(ctx: &egui::Context) -> usize {
    use std::sync::Arc;

    let candidates = cjk_candidates();
    let mut loaded: Vec<(String, egui::FontData)> = Vec::new();

    for path in candidates {
        if loaded.len() >= 2 {
            break;
        }
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        if bytes.is_empty() {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("cjk")
            .to_string();
        // Note: for `.ttc` collections index 0 is the regular weight, which is
        // what we want for UI fallback. Invalid indices yield no glyphs, so
        // stick to 0 rather than registering every face.
        loaded.push((
            format!("cjk-{stem}"),
            egui::FontData::from_owned(bytes),
        ));
    }

    if loaded.is_empty() {
        eprintln!("marklessman: no system CJK font found, CJK paths may show tofu");
        return 0;
    }

    let count = loaded.len();
    let mut defs = egui::FontDefinitions::default();
    for (name, data) in loaded {
        defs.font_data.insert(name.clone(), Arc::new(data));
        if let Some(fam) = defs.families.get_mut(&egui::FontFamily::Proportional) {
            fam.push(name.clone());
        }
        if let Some(fam) = defs.families.get_mut(&egui::FontFamily::Monospace) {
            fam.push(name);
        }
    }
    ctx.set_fonts(defs);
    count
}

/// Well-known system CJK font locations, highest-priority first.
fn cjk_candidates() -> Vec<std::path::PathBuf> {
    use std::path::PathBuf;

    let mut out: Vec<PathBuf> = Vec::new();

    #[cfg(target_os = "windows")]
    {
        let windir = std::env::var("WINDIR").unwrap_or_else(|_| r"C:\Windows".to_string());
        let fonts = PathBuf::from(windir).join("Fonts");
        for f in [
            // Simplified-first: YaHei regular covers CJK Unified Ideographs.
            "msyh.ttc",
            "simhei.ttf",
            "simsun.ttc",
            // Traditional / fallback.
            "msjh.ttc",
            "mingliu.ttc",
        ] {
            out.push(fonts.join(f));
        }
    }

    #[cfg(target_os = "macos")]
    {
        for f in [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/Library/Fonts/Noto Sans CJK SC.ttc",
        ] {
            out.push(PathBuf::from(f));
        }
    }

    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    {
        for f in [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansSC-Regular.ttf",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        ] {
            out.push(PathBuf::from(f));
        }
        // Also probe Windows fonts dir under Wine/dual-boot, harmless elsewhere.
        if let Ok(windir) = std::env::var("WINDIR") {
            let fonts = PathBuf::from(windir).join("Fonts");
            for f in ["msyh.ttc", "simhei.ttf", "simsun.ttc"] {
                out.push(fonts.join(f));
            }
        }
    }

    // User override: drop any TTF/OTF/TTC at this path to force it first.
    if let Some(dir) = dirs::config_dir() {
        out.insert(0, dir.join("marklessman").join("cjk-fallback.ttf"));
    }

    out
}
