use std::path::Path;

use image::RgbImage;
use ndarray::Array4;
use ort::{session::Session, value::Tensor};

use crate::image_ops::{image_to_nchw, resize_bilinear};
use crate::session::open_session;
use crate::{Result, SessionOptions};

/// The detector always reads a letterboxed 640x640 canvas.
const INPUT_SIZE: u32 = 640;
/// Letterbox fill, matching the 114-gray the dataset pipeline used.
const LETTERBOX_FILL: u8 = 114;

/// Confidence threshold applied when a [`Detector`] is opened.
pub const DEFAULT_CONFIDENCE: f32 = 0.25;
/// Non-maximum-suppression IoU threshold applied when a [`Detector`] is opened.
pub const DEFAULT_NMS_IOU: f32 = 0.5;

/// An axis-aligned rectangle in image coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    /// Left edge.
    pub x: i32,
    /// Top edge.
    pub y: i32,
    /// Edge length in pixels; never negative.
    pub width: u32,
    /// Edge length in pixels; never negative.
    pub height: u32,
}

impl Rect {
    /// Right edge (exclusive).
    pub fn right(&self) -> i32 {
        self.x + self.width as i32
    }

    /// Bottom edge (exclusive).
    pub fn bottom(&self) -> i32 {
        self.y + self.height as i32
    }

    /// Grow by `padding` on every side and clamp to an image of
    /// `image_width` x `image_height`. The result is always at least 1x1.
    pub fn padded(&self, padding: i32, image_width: i32, image_height: i32) -> Rect {
        let x0 = (self.x - padding).max(0);
        let y0 = (self.y - padding).max(0);
        let x1 = (self.right() + padding).min(image_width);
        let y1 = (self.bottom() + padding).min(image_height);
        Rect {
            x: x0,
            y: y0,
            width: (x1 - x0).max(1) as u32,
            height: (y1 - y0).max(1) as u32,
        }
    }
}

/// A single watermark detection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    /// Where the watermark sits in the source image.
    pub bounds: Rect,
    /// Model confidence, in `[0, 1]`.
    pub score: f32,
}

/// Watermark detector backed by a bare ONNX YOLO export.
pub struct Detector {
    /// Boxes scoring below this are dropped before suppression.
    pub confidence: f32,
    /// A kept box suppresses overlapping boxes above this IoU.
    pub nms_iou: f32,
    session: Session,
}

impl Detector {
    /// Load a detector model.
    pub fn open(model: impl AsRef<Path>, options: &SessionOptions) -> Result<Self> {
        Ok(Self {
            confidence: DEFAULT_CONFIDENCE,
            nms_iou: DEFAULT_NMS_IOU,
            session: open_session(model.as_ref(), options)?,
        })
    }

    /// Run detection over an image, strongest boxes first after suppression.
    pub fn detect(&mut self, image: &RgbImage) -> Result<Vec<Detection>> {
        let (width, height) = image.dimensions();
        let letterboxed = letterbox(image);

        let tensor = Tensor::from_array(letterboxed.tensor)?;
        let outputs = self.session.run(ort::inputs![tensor])?;
        let view = outputs[0].try_extract_array::<f32>()?;

        // Layout is [1, C, N] (or [C, N]): rows 0..4 hold cxcywh in 640-space,
        // remaining rows hold per-class scores.
        let shape = view.shape().to_vec();
        let (channels, anchors) = match shape.as_slice() {
            [_, channels, anchors] => (*channels, *anchors),
            [channels, anchors] => (*channels, *anchors),
            shape => return Err(crate::Error::UnexpectedOutputShape { shape: shape.to_vec() }),
        };
        let data = view.as_slice().ok_or(crate::Error::NonContiguousOutput)?;
        let at = |channel: usize, anchor: usize| data[channel * anchors + anchor];

        let mut boxes = Vec::new();
        let mut scores = Vec::new();
        for anchor in 0..anchors {
            // Multi-class exports carry one score row per class; the best one wins.
            let score = if channels > 5 {
                (4..channels)
                    .map(|channel| at(channel, anchor))
                    .fold(0.0f32, f32::max)
            } else {
                at(4, anchor)
            };
            if score < self.confidence {
                continue;
            }

            let (center_x, center_y, box_width, box_height) =
                (at(0, anchor), at(1, anchor), at(2, anchor), at(3, anchor));
            let x0 = ((center_x - box_width / 2.0 - letterboxed.pad_x) / letterboxed.scale)
                .clamp(0.0, width as f32);
            let y0 = ((center_y - box_height / 2.0 - letterboxed.pad_y) / letterboxed.scale)
                .clamp(0.0, height as f32);
            let x1 = ((center_x + box_width / 2.0 - letterboxed.pad_x) / letterboxed.scale)
                .clamp(0.0, width as f32);
            let y1 = ((center_y + box_height / 2.0 - letterboxed.pad_y) / letterboxed.scale)
                .clamp(0.0, height as f32);

            boxes.push((x0, y0, x1, y1));
            scores.push(score);
        }

        let keep = non_max_suppression(&boxes, &scores, self.nms_iou);
        Ok(keep
            .into_iter()
            .map(|index| Detection {
                bounds: bounds_from_corners(boxes[index]),
                score: scores[index],
            })
            .collect())
    }
}

/// Input image scaled and padded onto the detector's square canvas.
struct Letterboxed {
    tensor: Array4<f32>,
    scale: f32,
    pad_x: f32,
    pad_y: f32,
}

fn letterbox(image: &RgbImage) -> Letterboxed {
    let (width, height) = image.dimensions();
    let scale = (INPUT_SIZE as f32 / height as f32).min(INPUT_SIZE as f32 / width as f32);
    let scaled_width = (width as f32 * scale).round() as u32;
    let scaled_height = (height as f32 * scale).round() as u32;

    let resized = resize_bilinear(image, scaled_width, scaled_height);
    let pad_x = ((INPUT_SIZE - scaled_width) / 2) as f32;
    let pad_y = ((INPUT_SIZE - scaled_height) / 2) as f32;

    let mut canvas =
        RgbImage::from_pixel(INPUT_SIZE, INPUT_SIZE, image::Rgb([LETTERBOX_FILL; 3]));
    image::imageops::replace(&mut canvas, &resized, pad_x as i64, pad_y as i64);

    Letterboxed {
        tensor: image_to_nchw(&canvas),
        scale,
        pad_x,
        pad_y,
    }
}

/// Greedy non-maximum suppression: keep the strongest box, drop every box
/// overlapping it above `iou_threshold`, repeat.
fn non_max_suppression(
    boxes: &[(f32, f32, f32, f32)],
    scores: &[f32],
    iou_threshold: f32,
) -> Vec<usize> {
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
    while let Some(&best) = order.first() {
        keep.push(best);
        let (ax0, ay0, ax1, ay1) = boxes[best];
        let mut survivors = Vec::with_capacity(order.len().saturating_sub(1));
        for &candidate in &order[1..] {
            let (bx0, by0, bx1, by1) = boxes[candidate];
            let intersection = (ax1.min(bx1) - ax0.max(bx0)).max(0.0)
                * (ay1.min(by1) - ay0.max(by0)).max(0.0);
            let overlap = intersection / (areas[best] + areas[candidate] - intersection).max(1e-9);
            if overlap <= iou_threshold {
                survivors.push(candidate);
            }
        }
        order = survivors;
    }
    keep
}

fn bounds_from_corners((x0, y0, x1, y1): (f32, f32, f32, f32)) -> Rect {
    Rect {
        x: x0 as i32,
        y: y0 as i32,
        width: (x1 - x0).max(0.0) as i32 as u32,
        height: (y1 - y0).max(0.0) as i32 as u32,
    }
}
