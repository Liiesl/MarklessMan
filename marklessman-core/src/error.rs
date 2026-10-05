/// Errors surfaced by the pipeline.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Reading or writing a file failed.
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),

    /// An image could not be decoded or encoded.
    #[error("image error: {0}")]
    Image(#[from] image::ImageError),

    /// ONNX Runtime rejected a call: session build, tensor creation or inference.
    #[error("onnx runtime error: {0}")]
    Runtime(#[from] ort::Error),

    /// The model returned a tensor layout the pipeline cannot interpret.
    #[error("unexpected model output shape {shape:?}")]
    UnexpectedOutputShape {
        /// Shape reported by the runtime.
        shape: Vec<usize>,
    },

    /// The model returned the wrong number of values for the expected layout.
    #[error("model returned {actual} values, expected {expected}")]
    UnexpectedOutputLength {
        /// Number of values the pipeline expects.
        expected: usize,
        /// Number of values the model produced.
        actual: usize,
    },

    /// Only contiguous tensors can be read back into an image.
    #[error("model output tensor is not contiguous")]
    NonContiguousOutput,

    /// The cleaner only accepts square patches of [`crate::PATCH_SIZE`].
    #[error("cleaner expects a {expected}x{expected} patch, got {width}x{height}")]
    UnexpectedPatchSize {
        /// Required edge length in pixels.
        expected: u32,
        /// Actual width.
        width: u32,
        /// Actual height.
        height: u32,
    },
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;
