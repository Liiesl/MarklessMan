use image::{Rgb, RgbImage};
use marklessman_core::{feather_paste, resize_bilinear};

fn test_image(w: u32, h: u32, seed: u32) -> RgbImage {
    let mut img = RgbImage::new(w, h);
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        (s >> 16) as u8
    };
    for p in img.pixels_mut() {
        *p = Rgb([next(), next(), next()]);
    }
    img
}

fn reference_resize(image: &RgbImage, width: u32, height: u32) -> RgbImage {
    let (sw, sh) = image.dimensions();
    if sw == width && sh == height {
        return image.clone();
    }
    let sx = sw as f32 / width as f32;
    let sy = sh as f32 / height as f32;
    let mut out = RgbImage::new(width, height);
    for y in 0..height {
        let soy = (y as f32 + 0.5) * sy - 0.5;
        let y0 = soy.floor() as i32;
        let wy = soy - y0 as f32;
        let yt = y0.clamp(0, sh as i32 - 1) as u32;
        let yb = (y0 + 1).clamp(0, sh as i32 - 1) as u32;
        for x in 0..width {
            let sox = (x as f32 + 0.5) * sx - 0.5;
            let x0 = sox.floor() as i32;
            let wx = sox - x0 as f32;
            let xl = x0.clamp(0, sw as i32 - 1) as u32;
            let xr = (x0 + 1).clamp(0, sw as i32 - 1) as u32;
            let tl = image.get_pixel(xl, yt);
            let tr = image.get_pixel(xr, yt);
            let bl = image.get_pixel(xl, yb);
            let br = image.get_pixel(xr, yb);
            let mut px = [0u8; 3];
            for c in 0..3 {
                let v = tl[c] as f32 * (1.0 - wx) * (1.0 - wy)
                    + tr[c] as f32 * wx * (1.0 - wy)
                    + bl[c] as f32 * (1.0 - wx) * wy
                    + br[c] as f32 * wx * wy;
                px[c] = v.round().clamp(0.0, 255.0) as u8;
            }
            out.put_pixel(x, y, Rgb(px));
        }
    }
    out
}

fn reference_feather(dst: &mut RgbImage, patch: &RgbImage, x: i32, y: i32, radius: usize) {
    let (w, h) = (patch.width() as usize, patch.height() as usize);
    let mut alpha = vec![1.0f32; w * h];
    let r = radius.min(h / 2).min(w / 2).max(1);
    for i in 0..r {
        let wt = if r > 1 { i as f32 / (r - 1) as f32 } else { 0.0 };
        for xx in 0..w {
            alpha[i * w + xx] = alpha[i * w + xx].min(wt);
            alpha[(h - 1 - i) * w + xx] = alpha[(h - 1 - i) * w + xx].min(wt);
        }
        for yy in 0..h {
            alpha[yy * w + i] = alpha[yy * w + i].min(wt);
            alpha[yy * w + (w - 1 - i)] = alpha[yy * w + (w - 1 - i)].min(wt);
        }
    }
    for py in 0..h {
        for px in 0..w {
            let wt = alpha[py * w + px];
            let t = dst.get_pixel((x + px as i32) as u32, (y + py as i32) as u32);
            let s = patch.get_pixel(px as u32, py as u32);
            let b = |base: u8, over: u8| (wt * over as f32 + (1.0 - wt) * base as f32) as u8;
            dst.put_pixel(
                (x + px as i32) as u32,
                (y + py as i32) as u32,
                Rgb([b(t[0], s[0]), b(t[1], s[1]), b(t[2], s[2])]),
            );
        }
    }
}

#[test]
fn resize_is_pixel_perfect() {
    for (w, h, nw, nh) in [(37, 53, 64, 64), (800, 61, 197, 64), (256, 256, 256, 256), (640, 480, 256, 256), (17, 91, 53, 29)] {
        let img = test_image(w, h, 12345);
        let fast = resize_bilinear(&img, nw, nh);
        let slow = reference_resize(&img, nw, nh);
        assert_eq!(fast.as_raw(), slow.as_raw(), "resize {w}x{h} -> {nw}x{nh}");
    }
}

#[test]
fn feather_is_pixel_perfect() {
    let mut dst_fast = test_image(300, 300, 777);
    let mut dst_slow = dst_fast.clone();
    let patch = test_image(256, 256, 999);
    feather_paste(&mut dst_fast, &patch, 22, 11, 7);
    reference_feather(&mut dst_slow, &patch, 22, 11, 7);
    assert_eq!(dst_fast.as_raw(), dst_slow.as_raw(), "feather paste drift");
}

#[test]
fn resize_and_feather_are_deterministic() {
    let img = test_image(123, 77, 42);
    let a = resize_bilinear(&img, 64, 64);
    let b = resize_bilinear(&img, 64, 64);
    assert_eq!(a.as_raw(), b.as_raw());
}
