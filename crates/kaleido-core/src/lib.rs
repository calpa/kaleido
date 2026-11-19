//! kaleido-core: image analysis primitives for the kaleido macOS gallery.
//!
//! Pure-Rust, no I/O side effects beyond reading image files. Designed to be
//! embeddable from both the CLI and the future GUI crate.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

pub mod cluster;
pub mod engine;
pub mod error;
pub mod model_store;
pub mod models;
pub mod ocr;
pub mod palette;
pub mod phash;
pub mod ppocr;
pub mod scan;
mod unionfind;
pub mod web;

pub use cluster::cluster_similar;
pub use engine::{run_ensemble, FakeEngine, OcrEngine};
pub use error::{KaleidoError, KaleidoResult};
pub use model_store::{FileState, Manifest, ModelStatus, ModelStore};
pub use models::{readiness_all, MachineProfile, ModelSpec, Verdict};
pub use ocr::{ensemble, EngineBlock, EngineProfile, Evidence, OcrBlock, ScriptSet};
pub use palette::{palette, palette_file, Palette, Swatch};
pub use phash::{dhash, dhash_file, phash, phash_file, Hash64};
#[cfg(feature = "engine-ort")]
pub use ppocr::PaddleOcrV5;
pub use scan::{dms_to_decimal, exif_from_file, scan, Exif, Gps, ImageEntry, ScanStats};
pub use web::{
    DownloadConfig, Downloader, FetchManifest, FetchReport, ImageProvider, ItemOutcome,
    ManifestEntry, OpenverseProvider, SearchResult, MAX_PAGE_SIZE, OPENVERSE_API,
};
