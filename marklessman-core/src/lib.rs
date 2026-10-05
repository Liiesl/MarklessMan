//! Detection and removal of visible watermarks from images.
//!
//! The pipeline pairs a YOLO-style detector with an SLBR-style inpainting
//! model, both executed through ONNX Runtime (DirectML or CPU). Every
//! detection is padded with surrounding context, cleaned at the model's
//! native 256x256 patch size — tiled or square-resized depending on the
//! region's size — and feathered back into the page.
//!
//! ```no_run
//! # fn main() -> marklessman_core::Result<()> {
//! use marklessman_core::{Cleaner, Detector, Pipeline, SessionOptions};
//!
//! let backend = SessionOptions::directml();
//! let detector = Detector::open("weights/marklessman-det.onnx", &backend)?;
//! let cleaner = Cleaner::open("weights/marklessman-clean_256_fp32.onnx", &backend)?;
//! let mut pipeline = Pipeline::new(detector, cleaner);
//!
//! let source = image::open("page.jpg")?.to_rgb8();
//! let processed = pipeline.process(&source)?;
//! processed.image.save("page_clean.jpg")?;
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]

mod clean;
mod detect;
mod error;
mod image_ops;
mod pipeline;
mod session;

pub use clean::{Cleaner, TilingConfig, PATCH_SIZE};
pub use detect::{Detection, Detector, Rect, DEFAULT_CONFIDENCE, DEFAULT_NMS_IOU};
pub use error::{Error, Result};
pub use image_ops::{collect_images, feather_paste, is_image_file, resize_bilinear, save_image};
pub use pipeline::{Pipeline, PipelineOptions, Processed};
pub use session::{Backend, GraphOptimization, SessionOptions};
