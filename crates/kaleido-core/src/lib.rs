//! kaleido-core: image analysis primitives for the kaleido macOS gallery.
//!
//! Pure-Rust, no I/O side effects beyond reading image files. Designed to be
//! embeddable from both the CLI and the future GUI crate.

pub mod palette;
pub mod phash;
pub mod scan;

pub use palette::{palette, palette_file, Palette, Swatch};
pub use phash::{dhash, dhash_file, Hash64};
pub use scan::{scan, ImageEntry, ScanStats};
