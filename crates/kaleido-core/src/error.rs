//! Unified error type for kaleido-core public APIs.

use std::io;

/// Errors returned by kaleido-core public functions.
#[derive(Debug, thiserror::Error)]
pub enum KaleidoError {
    /// Filesystem I/O failure (missing file, permission, ...).
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    /// Image decoding failure (unsupported or corrupted data).
    #[error("image decode error: {0}")]
    Image(#[from] image::ImageError),
    /// Structural problem with caller-supplied input.
    #[error("invalid input: {0}")]
    InvalidInput(String),
    /// Network failure during web search or download (DNS, timeout, bad status).
    #[error("network error: {0}")]
    Network(#[from] reqwest::Error),
    /// JSON serialization/deserialization failure.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    /// Downloaded resource did not declare an image content type.
    #[error("unexpected content type: {0}")]
    ContentType(String),
    /// Downloaded resource exceeded the configured size limit.
    #[error("download too large: {size} bytes exceeds the {limit} byte limit")]
    TooLarge { size: u64, limit: u64 },
    /// Downloaded image is a perceptual duplicate of one already on disk.
    #[error("duplicate image (dhash {0})")]
    Duplicate(String),
    /// Model files missing, corrupted, or failing checksum verification.
    #[error("model error: {0}")]
    Model(String),
    /// OCR engine failure (unknown engine, runtime/inference error).
    #[error("engine error: {0}")]
    Engine(String),
}

/// Convenience alias for results yielding [`KaleidoError`].
pub type KaleidoResult<T> = Result<T, KaleidoError>;
