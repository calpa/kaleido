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
    /// Model files missing, corrupted, or failing checksum verification.
    #[error("model error: {0}")]
    Model(String),
    /// OCR engine failure (unknown engine, runtime/inference error).
    #[error("engine error: {0}")]
    Engine(String),
}

/// Convenience alias for results yielding [`KaleidoError`].
pub type KaleidoResult<T> = Result<T, KaleidoError>;
