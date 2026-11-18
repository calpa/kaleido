//! kaleido-core: image analysis primitives for the kaleido macOS gallery.
//!
//! Pure-Rust, no I/O side effects beyond reading image files. Designed to be
//! embeddable from both the CLI and the future GUI crate.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

pub mod cluster;
pub mod error;
pub mod palette;
pub mod phash;
pub mod scan;

pub use cluster::cluster_similar;
pub use error::{KaleidoError, KaleidoResult};
pub use palette::{palette, palette_file, Palette, Swatch};
pub use phash::{dhash, dhash_file, phash, phash_file, Hash64};
pub use scan::{dms_to_decimal, exif_from_file, scan, Exif, Gps, ImageEntry, ScanStats};
