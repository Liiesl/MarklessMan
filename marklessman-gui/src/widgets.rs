//! Material 3 reusable widgets on top of the SVG icon set.

use eframe::egui;

use crate::theme::{M3Colors, palette};

/// M3 elevated card: surface-container, 1dp outline-variant, 16dp radius.
pub fn card_frame(c: &M3Colors) -> egui::Frame {
    egui::Frame::NONE
        .fill(c.surface_container)
        .stroke(egui::Stroke::new(1.0, c.outline_variant))
        .corner_radius(egui::CornerRadius::same(16))
        .inner_margin(egui::Margin::same(12))
        .shadow(egui::Shadow {
            offset: [0, 1],
            blur: 6,
            spread: 0,
            color: egui::Color32::from_black_alpha(60),
        })
}

/// M3 filled tonal icon container (fixed-size circle) with centered icon.
///
/// Uses `allocate_exact_size` + manual paint so it never stretches to fill
/// the parent (a `Frame` + `centered_and_justified` would expand full-width
/// inside `horizontal` / `vertical_centered` layouts).
pub fn tonal_icon(
    ui: &mut egui::Ui,
    name: &'static str,
    bytes: &'static [u8],
    icon_size: f32,
    fg: egui::Color32,
    container: egui::Color32,
    diam: f32,
) {
    let diam = diam.clamp(24.0, 96.0);
    let side = icon_size.clamp(12.0, diam);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(diam, diam), egui::Sense::hover());
    ui.painter().rect_filled(
        rect,
        egui::CornerRadius::same((diam / 2.0) as u8),
        container,
    );
    let min = rect.center() - egui::vec2(side / 2.0, side / 2.0);
    ui.put(
        egui::Rect::from_min_size(min, egui::vec2(side, side)),
        crate::icons::tinted(name, bytes, side, fg),
    );
}

/// M3 icon button: 40dp touch target, circular hover state layer.
pub fn icon_button(ui: &mut egui::Ui, icon: egui::Image<'static>, tooltip: &str) -> egui::Response {
    ui.add(
        egui::Button::image(icon)
            .frame(false)
            .min_size(egui::vec2(40.0, 40.0)),
    )
    .on_hover_text(tooltip)
    .on_hover_cursor(egui::CursorIcon::PointingHand)
}

/// Hover cursor for clickable controls: hand when enabled, forbidden when disabled.
pub fn click_cursor(enabled: bool) -> egui::CursorIcon {
    if enabled {
        egui::CursorIcon::PointingHand
    } else {
        egui::CursorIcon::NotAllowed
    }
}

/// M3 Filled button (stadium, 40dp). Caller picks M3 fill/fg.
pub fn filled_button(
    ui: &mut egui::Ui,
    icon_name: &'static str,
    icon_bytes: &'static [u8],
    label: &str,
    fill: egui::Color32,
    fg: egui::Color32,
    tooltip: &str,
    min_size: egui::Vec2,
) -> egui::Response {
    let btn = egui::Button::image_and_text(
        crate::icons::tinted(icon_name, icon_bytes, 16.0, fg),
        egui::RichText::new(label).size(14.0).strong().color(fg),
    )
    .fill(fill)
    .stroke(egui::Stroke::NONE)
    .corner_radius(egui::CornerRadius::same(20))
    .min_size(min_size);
    ui.add(btn)
        .on_hover_text(tooltip)
        .on_hover_cursor(egui::CursorIcon::PointingHand)
}

/// M3 Tonal button (secondary-container).
pub fn tonal_button(
    ui: &mut egui::Ui,
    icon_name: &'static str,
    icon_bytes: &'static [u8],
    label: &str,
    c: &M3Colors,
    tooltip: &str,
    min_size: egui::Vec2,
) -> egui::Response {
    filled_button(
        ui,
        icon_name,
        icon_bytes,
        label,
        c.secondary_container,
        c.on_secondary_container,
        tooltip,
        min_size,
    )
}

/// M3 Outlined button.
pub fn outlined_button(
    ui: &mut egui::Ui,
    icon_name: &'static str,
    icon_bytes: &'static [u8],
    label: &str,
    c: &M3Colors,
    tooltip: &str,
    min_size: egui::Vec2,
) -> egui::Response {
    let btn = egui::Button::image_and_text(
        crate::icons::tinted(icon_name, icon_bytes, 16.0, c.primary),
        egui::RichText::new(label)
            .size(14.0)
            .strong()
            .color(c.primary),
    )
    .fill(egui::Color32::TRANSPARENT)
    .stroke(egui::Stroke::new(1.0, c.outline))
    .corner_radius(egui::CornerRadius::same(20))
    .min_size(min_size);
    ui.add(btn)
        .on_hover_text(tooltip)
        .on_hover_cursor(egui::CursorIcon::PointingHand)
}

/// Small count chip, e.g. `3/12` or `5`. M3 assist-chip shape.
pub fn pill(ui: &mut egui::Ui, text: &str, fill: egui::Color32, fg: egui::Color32) {
    egui::Frame::NONE
        .fill(fill)
        .corner_radius(egui::CornerRadius::same(20))
        .inner_margin(egui::Margin::symmetric(12, 5))
        .show(ui, |ui| {
            ui.label(egui::RichText::new(text).size(12.0).color(fg).strong());
        });
}

/// Status chip for queue rows / preview cards. Tonal fill + outline.
pub fn status_pill(ui: &mut egui::Ui, text: &str, color: egui::Color32) {
    egui::Frame::NONE
        .fill(egui::Color32::from_rgba_unmultiplied(
            color.r(),
            color.g(),
            color.b(),
            32,
        ))
        .stroke(egui::Stroke::new(
            1.0,
            egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 110),
        ))
        .corner_radius(egui::CornerRadius::same(20))
        .inner_margin(egui::Margin::symmetric(12, 5))
        .show(ui, |ui| {
            ui.label(egui::RichText::new(text).size(12.0).color(color).strong());
        });
}

/// Shorten a long path for display, keeping head + tail.
pub fn trunc_middle(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    if max_chars <= 4 {
        return "…".to_string();
    }
    let keep = (max_chars - 1) / 2;
    let head: String = s.chars().take(keep).collect();
    let tail: String = s
        .chars()
        .rev()
        .take(max_chars - keep - 1)
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    format!("{head}…{tail}")
}

#[allow(dead_code)]
pub fn muted(s: &str) -> egui::RichText {
    egui::RichText::new(s).color(palette::MUTED)
}

#[allow(dead_code)]
pub fn faint(s: &str) -> egui::RichText {
    egui::RichText::new(s).color(palette::FAINT).size(12.0)
}

/// Theme-aware variants (prefer these in new M3 layout code).
pub fn on_surface_variant(s: &str, c: &M3Colors, size: f32) -> egui::RichText {
    egui::RichText::new(s)
        .color(c.on_surface_variant)
        .size(size)
}

/// M3 list thumbnail: fixed-size rounded image (8dp), or placeholder.
///
/// Never stretches — uses `allocate_exact_size` like `tonal_icon`.
pub fn thumbnail(ui: &mut egui::Ui, tex: Option<&egui::TextureHandle>, size: f32, c: &M3Colors) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(size, size), egui::Sense::hover());
    ui.painter()
        .rect_filled(rect, egui::CornerRadius::same(8), c.surface_highest);
    if let Some(tex) = tex {
        ui.put(
            rect,
            egui::Image::from_texture((tex.id(), tex.size_vec2()))
                .max_size(egui::vec2(size, size))
                .corner_radius(egui::CornerRadius::same(8)),
        );
    } else {
        let side = 24.0;
        let min = rect.center() - egui::vec2(side / 2.0, side / 2.0);
        ui.put(
            egui::Rect::from_min_size(min, egui::vec2(side, side)),
            crate::icons::tinted("thumb-ph", crate::icons::IMAGE, side, c.outline),
        );
    }
}

/// Small spinner box the same size as a thumbnail (for Running rows).
pub fn thumbnail_spinner(ui: &mut egui::Ui, size: f32, c: &M3Colors) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(size, size), egui::Sense::hover());
    ui.painter()
        .rect_filled(rect, egui::CornerRadius::same(8), c.surface_highest);
    let side = 20.0;
    let min = rect.center() - egui::vec2(side / 2.0, side / 2.0);
    ui.put(
        egui::Rect::from_min_size(min, egui::vec2(side, side)),
        egui::Spinner::new().size(side),
    );
}
