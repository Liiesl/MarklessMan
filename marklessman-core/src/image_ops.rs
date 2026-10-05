use std::path::{Path, PathBuf};

use image::{RgbImage, codecs::jpeg::JpegEncoder};
use ndarray::Array4;
use rayon::prelude::*;
use wide::f32x8;

use crate::Result;

/// Bilinear resize with OpenCV's `INTER_LINEAR` sampling convention.
///
/// `image`'s `Triangle` filter aliases fine strokes away on hard downscales
/// (an 800x2600 page shrunk to 197x640 loses them), while OpenCV keeps them
/// — and the detector was trained and evaluated on OpenCV resizes.
pub fn resize_bilinear(image: &RgbImage, width: u32, height: u32) -> RgbImage {
    let (source_width, source_height) = image.dimensions();
    if source_width == width && source_height == height {
        return image.clone();
    }

    let scale_x = source_width as f32 / width as f32;
    let scale_y = source_height as f32 / height as f32;
    let (sw_i32, sh_i32) = (source_width as i32, source_height as i32);
    let (sw_usize, w_usize) = (source_width as usize, width as usize);
    let src_buf = image.as_raw();
    let mut resized = RgbImage::new(width, height);

    {
        let flat = resized.as_flat_samples_mut();
        let dst_buf = flat.samples;
        // Each output row is independent, so rayon row parallelism is
        // bit-identical to the scalar loop. Per-pixel math is unchanged
        // (same op order, round-then-clamp) to avoid model drift.
        dst_buf
            .par_chunks_mut(w_usize * 3)
            .enumerate()
            .for_each(|(y, dst_row)| {
                let yf = y as u32;
                let source_y = (yf as f32 + 0.5) * scale_y - 0.5;
                let y0 = source_y.floor() as i32;
                let weight_y = source_y - y0 as f32;
                let inv_wy = 1.0 - weight_y;
                let y_top = y0.clamp(0, sh_i32 - 1) as usize;
                let y_bottom = (y0 + 1).clamp(0, sh_i32 - 1) as usize;
                let top_base = y_top * sw_usize * 3;
                let bottom_base = y_bottom * sw_usize * 3;

                for (x, dst_px) in dst_row.chunks_exact_mut(3).enumerate() {
                    let source_x = (x as f32 + 0.5) * scale_x - 0.5;
                    let x0 = source_x.floor() as i32;
                    let weight_x = source_x - x0 as f32;
                    let inv_wx = 1.0 - weight_x;
                    let x_left = x0.clamp(0, sw_i32 - 1) as usize * 3;
                    let x_right = (x0 + 1).clamp(0, sw_i32 - 1) as usize * 3;

                    let tl = top_base + x_left;
                    let tr = top_base + x_right;
                    let bl = bottom_base + x_left;
                    let br = bottom_base + x_right;
                    for channel in 0..3 {
                        // Same order as the scalar reference:
                        // tl*inv_wx*inv_wy + tr*wx*inv_wy + bl*inv_wx*wy + br*wx*wy
                        let value = src_buf[tl + channel] as f32 * inv_wx * inv_wy
                            + src_buf[tr + channel] as f32 * weight_x * inv_wy
                            + src_buf[bl + channel] as f32 * inv_wx * weight_y
                            + src_buf[br + channel] as f32 * weight_x * weight_y;
                        dst_px[channel] = value.round().clamp(0.0, 255.0) as u8;
                    }
                }
            });
    }

    resized
}

/// Write `image` to `path`, creating missing parent directories.
///
/// JPEG output uses `jpeg_quality` explicitly: the crate default (75) is
/// visibly softer than the quality 90 used elsewhere in the project.
pub fn save_image(path: impl AsRef<Path>, image: &RgbImage, jpeg_quality: u8) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }

    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if extension == "jpg" || extension == "jpeg" {
        let file = std::fs::File::create(path)?;
        let mut encoder = JpegEncoder::new_with_quality(file, jpeg_quality);
        encoder.encode_image(image)?;
    } else {
        image.save(path)?;
    }
    Ok(())
}

/// True when the path's extension names a format the crate handles.
pub fn is_image_file(path: &Path) -> bool {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some(extension) => matches!(
            extension.to_ascii_lowercase().as_str(),
            "jpg" | "jpeg" | "png" | "webp" | "bmp"
        ),
        None => false,
    }
}

/// The image files directly inside `directory`, sorted by path.
pub fn collect_images(directory: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
    let mut images: Vec<PathBuf> = std::fs::read_dir(directory)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_file() && is_image_file(path))
        .collect();
    images.sort();
    Ok(images)
}

/// Scale `image` to `[0, 1]` floats in NCHW layout, ready for a model input.
///
/// rayon splits the 3 planes into row chunks (each output element is
/// independent); `wide` vectorizes the `/ 255.0` with true IEEE division,
/// the same op as the scalar path, so tensors are bit-identical.
pub(crate) fn image_to_nchw(image: &RgbImage) -> Array4<f32> {
    let (width, height) = image.dimensions();
    let (w, h) = (width as usize, height as usize);
    if w == 0 || h == 0 {
        return Array4::<f32>::zeros((1, 3, h, w));
    }
    let plane = w * h;
    let mut flat = vec![0.0f32; 3 * plane];
    {
        let src = image.as_raw();
        let divisor = f32x8::splat(255.0);
        // 3*h row chunks of length w: chunk i holds channel (i / h), row (i % h).
        flat.par_chunks_mut(w)
            .enumerate()
            .for_each(|(chunk, dst_row)| {
                let channel = chunk / h;
                let y = chunk % h;
                let src_row_base = y * w * 3;
                let mut x = 0;
                // SIMD fast path: 8 pixels at a time, gather strided bytes,
                // vector divide — same values as scalar `b as f32 / 255.0`.
                while x + 8 <= w {
                    let mut tmp = [0.0f32; 8];
                    for k in 0..8 {
                        tmp[k] = src[src_row_base + (x + k) * 3 + channel] as f32;
                    }
                    let v = f32x8::from(tmp) / divisor;
                    let out: [f32; 8] = v.into();
                    dst_row[x..x + 8].copy_from_slice(&out);
                    x += 8;
                }
                for dx in x..w {
                    dst_row[dx] = src[src_row_base + dx * 3 + channel] as f32 / 255.0;
                }
            });
    }
    Array4::from_shape_vec((1, 3, h, w), flat).expect("nchw shape")
}

/// Per-pixel cross-fade weights: 0 at the border, 1 past `radius` pixels.
///
/// The ramp follows the project's `linspace(0, 1, radius)` convention, i.e.
/// step `i` weighs `i / (radius - 1)` rather than `i / radius`.
pub(crate) fn feather_alpha(width: usize, height: usize, radius: usize) -> Vec<f32> {
    let mut alpha = vec![1.0f32; width * height];
    let radius = radius.min(height / 2).min(width / 2).max(1);
    for i in 0..radius {
        let weight = if radius > 1 {
            i as f32 / (radius - 1) as f32
        } else {
            0.0
        };
        for x in 0..width {
            alpha[i * width + x] = alpha[i * width + x].min(weight);
            alpha[(height - 1 - i) * width + x] = alpha[(height - 1 - i) * width + x].min(weight);
        }
        for y in 0..height {
            alpha[y * width + i] = alpha[y * width + i].min(weight);
            alpha[y * width + (width - 1 - i)] = alpha[y * width + (width - 1 - i)].min(weight);
        }
    }
    alpha
}

/// Blend `patch` over `destination` with a `radius`-pixel feathered border,
/// starting at `(x, y)`.
///
/// rayon parallelizes over destination rows (disjoint writes); `wide`
/// vectorizes `w*over + (1-w)*base` per channel with the same op order as
/// scalar. Narrowing uses scalar `as u8` truncation to match the reference
/// exactly — no `round`, no FMA.
pub fn feather_paste(destination: &mut RgbImage, patch: &RgbImage, x: i32, y: i32, radius: usize) {
    let (width, height) = (patch.width() as usize, patch.height() as usize);
    if width == 0 || height == 0 {
        return;
    }
    let alpha = feather_alpha(width, height, radius);
    let dw = destination.width() as usize;
    let patch_buf = patch.as_raw();
    let one = f32x8::splat(1.0);

    {
        let flat = destination.as_flat_samples_mut();
        let dst_buf = flat.samples;
        dst_buf
            .par_chunks_mut(dw * 3)
            .enumerate()
            .for_each(|(dy_global, dst_row)| {
                let py = dy_global as i32 - y;
                if py < 0 || py >= height as i32 {
                    return;
                }
                let py_usize = py as usize;
                let alpha_row_base = py_usize * width;
                let patch_row_base = py_usize * width * 3;
                let dst_col_base = x;
                let mut px = 0;
                while px + 8 <= width {
                    let mut w_tmp = [0.0f32; 8];
                    for k in 0..8 {
                        w_tmp[k] = alpha[alpha_row_base + px + k];
                    }
                    let w = f32x8::from(w_tmp);
                    let inv = one - w;
                    for channel in 0..3 {
                        let mut o_tmp = [0.0f32; 8];
                        let mut b_tmp = [0.0f32; 8];
                        for k in 0..8 {
                            o_tmp[k] =
                                patch_buf[patch_row_base + (px + k) * 3 + channel] as f32;
                            let dx = dst_col_base + px as i32 + k as i32;
                            b_tmp[k] = dst_row[(dx as usize) * 3 + channel] as f32;
                        }
                        // Same order as scalar: w*o + (1-w)*b.
                        let blended = w * f32x8::from(o_tmp) + inv * f32x8::from(b_tmp);
                        let out: [f32; 8] = blended.into();
                        for (k, v) in out.iter().enumerate() {
                            let dx = dst_col_base + px as i32 + k as i32;
                            dst_row[(dx as usize) * 3 + channel] = *v as u8;
                        }
                    }
                    px += 8;
                }
                for dx_local in px..width {
                    let weight = alpha[alpha_row_base + dx_local];
                    let inv = 1.0 - weight;
                    let dx = dst_col_base + dx_local as i32;
                    let dst_base = (dx as usize) * 3;
                    let src_base = patch_row_base + dx_local * 3;
                    for channel in 0..3 {
                        let blended = weight * patch_buf[src_base + channel] as f32
                            + inv * dst_row[dst_base + channel] as f32;
                        dst_row[dst_base + channel] = blended as u8;
                    }
                }
            });
    }
}
