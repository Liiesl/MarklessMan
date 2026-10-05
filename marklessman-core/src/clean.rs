use std::path::Path;

use image::{RgbImage, imageops::FilterType};
use ort::{session::Session, value::Tensor};
use rayon::prelude::*;
use wide::f32x8;

use crate::detect::Rect;
use crate::error::Error;
use crate::image_ops::{feather_alpha, image_to_nchw, resize_bilinear};
use crate::session::open_session;
use crate::{Result, SessionOptions};

/// Max tiles per single ONNX run. Bounds DirectML memory on large regions;
/// larger jobs are chunked. Fixed-batch-1 weights fall back to sequential
/// (bit-identical) automatically — see [`Cleaner::clean_patches`].
const BATCH_CHUNK: usize = 8;

/// The cleaner model always operates on square patches of this edge length.
pub const PATCH_SIZE: u32 = 256;

/// Regions in this size band are cleaned with one square-resized pass instead
/// of tiling: the 257..320 range is where tiled output picks up seam artifacts.
const SQUARE_RESIZE_MIN: u32 = 257;
const SQUARE_RESIZE_MAX: u32 = 380;

/// Watermark inpainting model (SLBR-style, 256x256 native resolution).
pub struct Cleaner {
    session: Session,
}

impl Cleaner {
    /// Load an inpainting model.
    pub fn open(model: impl AsRef<Path>, options: &SessionOptions) -> Result<Self> {
        Ok(Self {
            session: open_session(model.as_ref(), options)?,
        })
    }

    /// Run the model on a single [`PATCH_SIZE`] square patch.
    pub fn clean_patch(&mut self, patch: &RgbImage) -> Result<RgbImage> {
        let (width, height) = patch.dimensions();
        if width != PATCH_SIZE || height != PATCH_SIZE {
            return Err(Error::UnexpectedPatchSize {
                expected: PATCH_SIZE,
                width,
                height,
            });
        }

        let tensor = Tensor::from_array(image_to_nchw(patch))?;
        let outputs = self.session.run(ort::inputs![tensor])?;
        let view = outputs[0].try_extract_array::<f32>()?;
        let data = view.as_slice().ok_or(Error::NonContiguousOutput)?;

        let expected = (PATCH_SIZE * PATCH_SIZE * 3) as usize;
        if data.len() != expected {
            return Err(Error::UnexpectedOutputLength {
                expected,
                actual: data.len(),
            });
        }
        Ok(rgb_from_nchw(data, PATCH_SIZE, PATCH_SIZE))
    }

    /// Run the model on several [`PATCH_SIZE`] patches, returning one cleaned
    /// image per input in order.
    ///
    /// Tiles are pre/post-processed in parallel (rayon + SIMD, bit-identical)
    /// and inferred in chunks of [`BATCH_CHUNK`] with a single `session.run`
    /// per chunk. Shipped weights are fixed batch-1 (`[1,3,256,256]`), so a
    /// chunk that the model rejects falls back to sequential `clean_patch`
    /// calls — pixel-perfect with the old loop. Dynamic-batch exports take
    /// the single-run fast path.
    pub fn clean_patches(&mut self, patches: &[RgbImage]) -> Result<Vec<RgbImage>> {
        for patch in patches {
            let (width, height) = patch.dimensions();
            if width != PATCH_SIZE || height != PATCH_SIZE {
                return Err(Error::UnexpectedPatchSize {
                    expected: PATCH_SIZE,
                    width,
                    height,
                });
            }
        }
        if patches.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = Vec::with_capacity(patches.len());
        for chunk in patches.chunks(BATCH_CHUNK) {
            if chunk.len() == 1 {
                out.push(self.clean_patch(&chunk[0])?);
                continue;
            }
            match self.run_batch_chunk(chunk) {
                Ok(mut images) => out.append(&mut images),
                Err(_) => {
                    // Fixed-batch model (or any batch-shape rejection):
                    // sequential path is exactly the old behaviour.
                    for patch in chunk {
                        out.push(self.clean_patch(patch)?);
                    }
                }
            }
        }
        Ok(out)
    }

    /// Single `session.run` for `chunk.len()` tiles. Errors when the model
    /// expects a different batch size, letting the caller fall back.
    fn run_batch_chunk(&mut self, chunk: &[RgbImage]) -> Result<Vec<RgbImage>> {
        use ndarray::Array4;

        let n = chunk.len();
        let plane = PATCH_SIZE as usize * PATCH_SIZE as usize;
        // Sequential over tiles here; each image_to_nchw already fans out
        // over all cores, so nesting another rayon level would oversubscribe.
        let mut flat = vec![0.0f32; n * 3 * plane];
        for (i, patch) in chunk.iter().enumerate() {
            let t = image_to_nchw(patch);
            flat[i * 3 * plane..(i + 1) * 3 * plane]
                .copy_from_slice(t.as_slice().expect("nchw contiguous"));
        }
        let input =
            Array4::from_shape_vec((n, 3, PATCH_SIZE as usize, PATCH_SIZE as usize), flat)
                .expect("batch shape");
        let tensor = Tensor::from_array(input)?;
        let outputs = self.session.run(ort::inputs![tensor])?;
        let view = outputs[0].try_extract_array::<f32>()?;
        let shape: Vec<usize> = view.shape().to_vec();
        if shape != vec![n, 3, PATCH_SIZE as usize, PATCH_SIZE as usize] {
            return Err(Error::UnexpectedOutputLength {
                expected: n * 3 * plane,
                actual: view.len(),
            });
        }
        let data = view.as_slice().ok_or(Error::NonContiguousOutput)?;
        Ok(rgb_from_nchw_batch(data, n, PATCH_SIZE, PATCH_SIZE))
    }

    /// Clean a whole image by scaling it down to [`PATCH_SIZE`], inpainting
    /// once, and scaling the result back to the original size.
    ///
    /// Best suited to images close to the patch size; large pages should go
    /// through [`Pipeline`](crate::Pipeline), which cleans per detection.
    pub fn clean_image(&mut self, image: &RgbImage) -> Result<RgbImage> {
        let (width, height) = image.dimensions();
        let scaled = resize_bilinear(image, PATCH_SIZE, PATCH_SIZE);
        let cleaned = self.clean_patch(&scaled)?;
        Ok(image::imageops::resize(
            &cleaned,
            width,
            height,
            FilterType::CatmullRom,
        ))
    }

    /// Clean a rectangular region out of `image`, returning just the region.
    ///
    /// Small regions are cleaned in one square-resized pass; larger ones are
    /// covered by overlapping [`PATCH_SIZE`] tiles whose outputs are
    /// cross-faded together.
    pub fn clean_region(
        &mut self,
        image: &RgbImage,
        region: Rect,
        tiling: &TilingConfig,
    ) -> Result<RgbImage> {
        let side = region.width.max(region.height);
        if (SQUARE_RESIZE_MIN..=SQUARE_RESIZE_MAX).contains(&side) {
            self.clean_region_square(image, region)
        } else {
            self.clean_region_tiled(image, region, tiling)
        }
    }

    /// Clean `region` through a single square window centred on it: the
    /// window's short side is stretched to match the long one, inpainted at
    /// [`PATCH_SIZE`], scaled back, then cropped down to the region. The extra
    /// window area is context only and never leaves this function.
    fn clean_region_square(&mut self, image: &RgbImage, region: Rect) -> Result<RgbImage> {
        let (image_width, image_height) = (image.width() as i32, image.height() as i32);
        let (origin_x, origin_y, side) = square_window(region, image_width, image_height);

        let square =
            image::imageops::crop_imm(image, origin_x as u32, origin_y as u32, side, side)
                .to_image();
        let cleaned = self.clean_image(&square)?;

        // The square contains the region by construction; clamp anyway so a
        // rounding surprise can never crop outside the cleaned area.
        let offset_x = (region.x - origin_x)
            .max(0)
            .min(side.saturating_sub(region.width) as i32) as u32;
        let offset_y = (region.y - origin_y)
            .max(0)
            .min(side.saturating_sub(region.height) as i32) as u32;
        Ok(image::imageops::crop_imm(
            &cleaned,
            offset_x,
            offset_y,
            region.width,
            region.height,
        )
        .to_image())
    }

    /// Cover `region` with overlapping [`PATCH_SIZE`] tiles, each inpainted at
    /// true scale, and cross-fade the results back into one region image.
    ///
    /// Inference is batched ([`Cleaner::clean_patches`]); accumulation keeps
    /// the original tile order and `weight * pixel` op order so blended
    /// floats — and the final `round`-after-divide — are bit-identical.
    /// Only the final per-pixel normalize fans out over rayon (independent
    /// pixels).
    fn clean_region_tiled(
        &mut self,
        image: &RgbImage,
        region: Rect,
        tiling: &TilingConfig,
    ) -> Result<RgbImage> {
        let (image_width, image_height) = (image.width() as i32, image.height() as i32);
        let (region_width, region_height) = (region.width as i32, region.height as i32);
        let patch = PATCH_SIZE as i32;

        let offsets_x = tile_origins(region.x, region_width, image_width, patch, tiling.stride);
        let offsets_y = tile_origins(region.y, region_height, image_height, patch, tiling.stride);
        let alpha = feather_alpha(PATCH_SIZE as usize, PATCH_SIZE as usize, tiling.feather);

        // Collect valid tiles first so inference can run batched.
        let mut jobs: Vec<(i32, i32, RgbImage)> = Vec::new();
        for &offset_y in &offsets_y {
            for &offset_x in &offsets_x {
                if offset_x < 0
                    || offset_y < 0
                    || offset_x + patch > image_width
                    || offset_y + patch > image_height
                {
                    continue;
                }
                let tile = image::imageops::crop_imm(
                    image,
                    offset_x as u32,
                    offset_y as u32,
                    PATCH_SIZE,
                    PATCH_SIZE,
                )
                .to_image();
                jobs.push((offset_x, offset_y, tile));
            }
        }

        let tiles: Vec<RgbImage> = jobs.iter().map(|(_, _, t)| t.clone()).collect();
        // Raw buffers once: avoids per-pixel get_pixel overhead in the hot loop.
        let cleaned_bufs: Vec<Vec<u8>> = self
            .clean_patches(&tiles)?
            .iter()
            .map(|img| img.as_raw().clone())
            .collect();

        let mut accum = vec![0.0f32; region.width as usize * region.height as usize * 3];
        let mut weights = vec![0.0f32; region.width as usize * region.height as usize];

        // Sequential, original order: float summation order affects the
        // 1-LSB after rounding, so this must not run in parallel.
        for ((offset_x, offset_y, _), cleaned_raw) in jobs.iter().zip(cleaned_bufs.iter()) {
            let (offset_x, offset_y) = (*offset_x, *offset_y);
            // Where the tile lands inside the region; it may overhang any edge.
            let relative_x = offset_x - region.x;
            let relative_y = offset_y - region.y;
            let start_x = (-relative_x).max(0);
            let start_y = (-relative_y).max(0);
            let end_x = patch.min(region_width - relative_x);
            let end_y = patch.min(region_height - relative_y);
            if end_y <= start_y || end_x <= start_x {
                continue;
            }
            let paste_x = relative_x + start_x;
            let paste_y = relative_y + start_y;
            let copy_width = (end_x - start_x) as usize;
            let copy_height = (end_y - start_y) as usize;
            let patch_usize = PATCH_SIZE as usize;

            for dy in 0..copy_height {
                let alpha_base = (start_y as usize + dy) * patch_usize + start_x as usize;
                let clean_base = ((start_y + dy as i32) as usize) * patch_usize * 3
                    + (start_x as usize) * 3;
                let region_row = paste_y as usize + dy;
                for dx in 0..copy_width {
                    let weight = alpha[alpha_base + dx];
                    let src_off = clean_base + dx * 3;
                    let col = paste_x as usize + dx;
                    let index = (region_row * region.width as usize + col) * 3;
                    accum[index] += weight * cleaned_raw[src_off] as f32;
                    accum[index + 1] += weight * cleaned_raw[src_off + 1] as f32;
                    accum[index + 2] += weight * cleaned_raw[src_off + 2] as f32;
                    weights[region_row * region.width as usize + col] += weight;
                }
            }
        }

        let (rw, rh) = (region.width as usize, region.height as usize);
        let mut out_buf = vec![0u8; rw * rh * 3];
        {
            let src_raw = image.as_raw();
            let img_w = image.width() as usize;
            out_buf
                .par_chunks_mut(rw * 3)
                .enumerate()
                .for_each(|(y, dst_row)| {
                    for x in 0..rw {
                        let flat = y * rw + x;
                        let weight = weights[flat];
                        if weight > 1e-6 {
                            let base = flat * 3;
                            for c in 0..3 {
                                // Same as scalar: divide, then round, then clamp.
                                dst_row[x * 3 + c] =
                                    (accum[base + c] / weight).round().clamp(0.0, 255.0) as u8;
                            }
                        } else {
                            // Untouched corner: keep the original pixels.
                            let sx = (region.x + x as i32) as usize;
                            let sy = (region.y + y as i32) as usize;
                            let src_off = (sy * img_w + sx) * 3;
                            dst_row[x * 3..x * 3 + 3]
                                .copy_from_slice(&src_raw[src_off..src_off + 3]);
                        }
                    }
                });
        }
        Ok(RgbImage::from_raw(region.width, region.height, out_buf).expect("region buffer"))
    }
}

/// How regions larger than a single patch are cut up and stitched back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TilingConfig {
    /// Distance between consecutive tile origins. The default of 192 leaves
    /// 64 pixels of overlap between neighbouring 256-pixel tiles.
    pub stride: i32,
    /// Width of the cross-fade ramp at tile borders, in pixels.
    pub feather: usize,
}

impl Default for TilingConfig {
    fn default() -> Self {
        Self {
            stride: 192,
            feather: 32,
        }
    }
}

/// A square window that contains `region` and stays inside the image,
/// returned as `(x, y, side)`.
fn square_window(region: Rect, image_width: i32, image_height: i32) -> (i32, i32, u32) {
    let side = (region.width.max(region.height) as i32)
        .min(image_width)
        .min(image_height)
        .max(1);
    let center_x = region.x + region.width as i32 / 2;
    let center_y = region.y + region.height as i32 / 2;
    let x = (center_x - side / 2).clamp(0, (image_width - side).max(0));
    let y = (center_y - side / 2).clamp(0, (image_height - side).max(0));
    (x, y, side as u32)
}

/// Tile origins covering `region_start .. region_start + region_len`, with
/// every tile kept fully inside `0 .. image_len`.
fn tile_origins(region_start: i32, region_len: i32, image_len: i32, size: i32, stride: i32) -> Vec<i32> {
    if region_len <= size {
        // A single tile centred on the region, nudged back into the image.
        let origin = region_start - (size - region_len) / 2;
        return vec![origin.clamp(0, (image_len - size).max(0))];
    }

    let first = region_start.min(image_len - size).max(0);
    let last = first.max((region_start + region_len - size).min(image_len - size));
    let mut origins = Vec::new();
    let mut origin = first;
    while origin < last {
        origins.push(origin);
        origin += stride;
    }
    if origins.last() != Some(&last) {
        origins.push(last);
    }
    origins
}

/// Read a `[1, 3, height, width]` float tensor back into an 8-bit image.
///
/// rayon parallelizes over rows (each pixel independent); `wide`
/// vectorizes the `* 255.0` with true IEEE multiply, the same op as scalar.
/// Clamp/round/narrow stay scalar (`round` away-from-zero then `as u8`) so
/// output bytes are bit-identical — `wide::round` uses banker's rounding and
/// would drift on x.5 values.
fn rgb_from_nchw(data: &[f32], width: u32, height: u32) -> RgbImage {
    let (w, h) = (width as usize, height as usize);
    if w == 0 || h == 0 {
        return RgbImage::new(width, height);
    }
    let plane = w * h;
    debug_assert_eq!(data.len(), plane * 3);
    let mut out = vec![0u8; w * h * 3];
    {
        let scale = f32x8::splat(255.0);
        out.par_chunks_mut(w * 3)
            .enumerate()
            .for_each(|(y, dst_row)| {
                let mut x = 0;
                while x + 8 <= w {
                    let base = y * w + x;
                    for channel in 0..3 {
                        let mut tmp = [0.0f32; 8];
                        for k in 0..8 {
                            tmp[k] = data[channel * plane + base + k];
                        }
                        // Same op as scalar: clamp, then * 255.0.
                        let mut clamped = [0.0f32; 8];
                        for k in 0..8 {
                            clamped[k] = tmp[k].clamp(0.0, 1.0);
                        }
                        let scaled: [f32; 8] = (f32x8::from(clamped) * scale).into();
                        for k in 0..8 {
                            dst_row[(x + k) * 3 + channel] = scaled[k].round() as u8;
                        }
                    }
                    x += 8;
                }
                for dx in x..w {
                    let offset = y * w + dx;
                    for channel in 0..3 {
                        dst_row[dx * 3 + channel] =
                            (data[channel * plane + offset].clamp(0.0, 1.0) * 255.0).round()
                                as u8;
                    }
                }
            });
    }
    RgbImage::from_raw(width, height, out).expect("rgb buffer")
}

/// Convert one image of a batched `[N, 3, H, W]` output back to 8-bit.
/// Same math as [`rgb_from_nchw`]; rows across the batch run in parallel.
fn rgb_from_nchw_batch(data: &[f32], batch: usize, width: u32, height: u32) -> Vec<RgbImage> {
    let (w, h) = (width as usize, height as usize);
    let plane = w * h;
    let img_floats = plane * 3;
    debug_assert_eq!(data.len(), batch * img_floats);
    (0..batch)
        .into_par_iter()
        .map(|n| rgb_from_nchw(&data[n * img_floats..(n + 1) * img_floats], width, height))
        .collect()
}
