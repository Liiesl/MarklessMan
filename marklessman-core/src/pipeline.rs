use std::path::Path;

use image::RgbImage;

use crate::clean::{Cleaner, TilingConfig};
use crate::detect::{Detection, Detector};
use crate::image_ops::{feather_paste, save_image};
use crate::Result;

/// Feather width used when cleaned regions are blended back into the page.
const BLEND_RADIUS: usize = 7;

/// Knobs for [`Pipeline::process`] and [`Pipeline::process_path`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PipelineOptions {
    /// Context added around every detection before it is cleaned; the padded
    /// rectangle is what gets inpainted and blended back.
    pub context_padding: i32,
    /// How regions larger than a patch are tiled.
    pub tiling: TilingConfig,
    /// JPEG quality used by [`Pipeline::process_path`].
    pub jpeg_quality: u8,
}

impl Default for PipelineOptions {
    fn default() -> Self {
        Self {
            context_padding: 64,
            tiling: TilingConfig::default(),
            jpeg_quality: 90,
        }
    }
}

/// What [`Pipeline::process`] produced for one image.
pub struct Processed {
    /// The source with every detection cleaned and blended back in.
    pub image: RgbImage,
    /// The detections that were cleaned, before context padding.
    pub detections: Vec<Detection>,
}

/// The full watermark-removal pipeline: detect, inpaint each detection,
/// feather the results into the page.
pub struct Pipeline {
    detector: Detector,
    cleaner: Cleaner,
    options: PipelineOptions,
}

impl Pipeline {
    /// Combine a detector and a cleaner. Detector thresholds come from the
    /// [`Detector`] itself; pipeline behaviour is configured through
    /// [`PipelineOptions`].
    pub fn new(detector: Detector, cleaner: Cleaner) -> Self {
        Self {
            detector,
            cleaner,
            options: PipelineOptions::default(),
        }
    }

    /// Replace the pipeline options.
    pub fn with_options(mut self, options: PipelineOptions) -> Self {
        self.options = options;
        self
    }

    /// The current pipeline options.
    pub fn options(&self) -> &PipelineOptions {
        &self.options
    }

    /// The current pipeline options, for in-place edits.
    pub fn options_mut(&mut self) -> &mut PipelineOptions {
        &mut self.options
    }

    /// The detector, e.g. to adjust `confidence` between calls.
    pub fn detector(&self) -> &Detector {
        &self.detector
    }

    /// The detector, for in-place edits.
    pub fn detector_mut(&mut self) -> &mut Detector {
        &mut self.detector
    }

    /// The cleaner.
    pub fn cleaner(&self) -> &Cleaner {
        &self.cleaner
    }

    /// The cleaner, for in-place edits.
    pub fn cleaner_mut(&mut self) -> &mut Cleaner {
        &mut self.cleaner
    }

    /// Detect every watermark in `image`, inpaint it, and blend the result
    /// back into a copy of the source.
    pub fn process(&mut self, image: &RgbImage) -> Result<Processed> {
        let (image_width, image_height) = (image.width() as i32, image.height() as i32);
        let options = self.options;

        let detections = self.detector.detect(image)?;
        let mut output = image.clone();

        for detection in &detections {
            let region = detection.bounds.padded(options.context_padding, image_width, image_height);
            let cleaned = self.cleaner.clean_region(image, region, &options.tiling)?;
            feather_paste(&mut output, &cleaned, region.x, region.y, BLEND_RADIUS);
        }

        Ok(Processed {
            image: output,
            detections,
        })
    }

    /// Read `input`, run [`Pipeline::process`], and write the cleaned page to
    /// `output` at [`PipelineOptions::jpeg_quality`] for JPEG targets.
    pub fn process_path(
        &mut self,
        input: impl AsRef<Path>,
        output: impl AsRef<Path>,
    ) -> Result<Processed> {
        let source = image::open(input.as_ref())?.to_rgb8();
        let processed = self.process(&source)?;
        save_image(
            output.as_ref(),
            &processed.image,
            self.options.jpeg_quality,
        )?;
        Ok(processed)
    }
}
