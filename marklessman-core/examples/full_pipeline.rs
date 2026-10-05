//! Clean a single image through the full pipeline.
//!
//! ```text
//! cargo run -p marklessman-core --example full_pipeline -- <input.jpg> [output.jpg]
//! ```

use std::path::Path;

use marklessman_core::{Cleaner, Detector, Pipeline, SessionOptions};

fn main() -> marklessman_core::Result<()> {
    let backend = SessionOptions::directml();
    let detector = Detector::open("weights/marklessman-det.onnx", &backend)?;
    let cleaner = Cleaner::open("weights/marklessman-clean_256_fp32.onnx", &backend)?;
    let mut pipeline = Pipeline::new(detector, cleaner);

    let input = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "test/new/1_t09.jpg".to_owned());
    let output = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "out_cleaned.jpg".to_owned());

    let processed = pipeline.process_path(Path::new(&input), Path::new(&output))?;
    eprintln!(
        "{} detection(s) written to {}",
        processed.detections.len(),
        output
    );
    Ok(())
}
