//! Folder scanning: enumerate supported images with metadata.

use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use walkdir::WalkDir;

/// Supported raster formats (matches the `image` crate features enabled).
pub const SUPPORTED_EXTENSIONS: &[&str] =
    &["jpg", "jpeg", "png", "gif", "webp", "tif", "tiff", "bmp"];

#[derive(Debug, Clone, Serialize)]
pub struct ImageEntry {
    pub path: PathBuf,
    pub width: u32,
    pub height: u32,
    pub bytes: u64,
    pub modified: Option<SystemTime>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ScanStats {
    pub scanned: u64,
    pub skipped: u64,
    pub entries: Vec<ImageEntry>,
}

/// Recursively collect image entries under `root`, in parallel.
pub fn scan(root: &Path) -> anyhow::Result<ScanStats> {
    use rayon::prelude::*;

    let paths: Vec<PathBuf> = WalkDir::new(root)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .filter(|p| is_supported(p))
        .collect();

    let scanned_total = paths.len() as u64;
    let entries: Vec<ImageEntry> = paths
        .into_par_iter()
        .filter_map(|p| read_entry(&p).ok())
        .collect();

    let read = entries.len() as u64;
    Ok(ScanStats {
        scanned: scanned_total,
        skipped: scanned_total - read,
        entries,
    })
}

pub fn is_supported(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| SUPPORTED_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

fn read_entry(path: &Path) -> anyhow::Result<ImageEntry> {
    let meta = std::fs::metadata(path)?;
    let (width, height) = image::image_dimensions(path)?;
    Ok(ImageEntry {
        path: path.to_path_buf(),
        width,
        height,
        bytes: meta.len(),
        modified: meta.modified().ok(),
    })
}
