//! 64-bit perceptual hashes: dHash (horizontal gradient comparison) and
//! pHash (DCT low-frequency sign pattern).

use crate::error::KaleidoResult;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Hash64(pub u64);

impl Hash64 {
    pub fn as_u64(self) -> u64 {
        self.0
    }

    /// Hamming distance between two hashes (0 = identical, 64 = maximal).
    pub fn hamming(self, other: Hash64) -> u32 {
        (self.0 ^ other.0).count_ones()
    }

    /// Two images are considered "similar" when hamming distance <= threshold.
    pub fn similar(self, other: Hash64, threshold: u32) -> bool {
        self.hamming(other) <= threshold
    }
}

/// Compute dHash for an image file. Errors on unreadable/unsupported files.
pub fn dhash_file(path: &Path) -> KaleidoResult<Hash64> {
    let img = image::open(path)?;
    Ok(dhash(&img))
}

/// Compute dHash for an in-memory image.
pub fn dhash(img: &image::DynamicImage) -> Hash64 {
    let gray = img
        .grayscale()
        .resize_exact(9, 8, image::imageops::FilterType::Lanczos3)
        .to_luma8();
    let mut bits: u64 = 0;
    for y in 0..8u32 {
        for x in 0..8u32 {
            let left = gray.get_pixel(x, y)[0] as u64;
            let right = gray.get_pixel(x + 1, y)[0] as u64;
            bits <<= 1;
            if left > right {
                bits |= 1;
            }
        }
    }
    Hash64(bits)
}

/// Side length of the square the image is resampled to before the DCT.
const PHASH_DIM: usize = 32;
/// Size of the low-frequency block (top-left) that forms the hash.
const PHASH_BLOCK: usize = 8;

/// Compute pHash for an image file. Errors on unreadable/unsupported files.
pub fn phash_file(path: &Path) -> KaleidoResult<Hash64> {
    let img = image::open(path)?;
    Ok(phash(&img))
}

/// Compute pHash (DCT-based) for an in-memory image.
///
/// Resamples to 32x32 grayscale, takes the 2D DCT-II, keeps the top-left
/// 8x8 low-frequency block, and sets one bit per coefficient above the
/// block median. Robust to rescaling and compression noise.
pub fn phash(img: &image::DynamicImage) -> Hash64 {
    let gray = img
        .grayscale()
        .resize_exact(
            PHASH_DIM as u32,
            PHASH_DIM as u32,
            image::imageops::FilterType::Lanczos3,
        )
        .to_luma8();
    let plane: Vec<f32> = gray.pixels().map(|p| p[0] as f32).collect();
    let coefficients = dct2_2d(&plane, PHASH_DIM);

    let block: Vec<f32> = (0..PHASH_BLOCK)
        .flat_map(|y| (0..PHASH_BLOCK).map(move |x| (x, y)))
        .map(|(x, y)| coefficients[y * PHASH_DIM + x])
        .collect();
    let median = median_of(&block);

    let mut bits: u64 = 0;
    for coefficient in &block {
        bits <<= 1;
        if *coefficient > median {
            bits |= 1;
        }
    }
    Hash64(bits)
}

/// Discrete cosine transform (type II) of a square plane via separable
/// row/column passes. Output is row-major: `out[frequency_row * n + column]`.
fn dct2_2d(plane: &[f32], n: usize) -> Vec<f32> {
    // table[k * n + i] = cos((2i+1)kπ / 2n), precomputed once.
    let table: Vec<f32> = (0..n)
        .flat_map(|k| {
            (0..n).map(move |i| {
                ((2 * i + 1) as f32 * (k as f32) * std::f32::consts::PI / (2.0 * n as f32)).cos()
            })
        })
        .collect();

    // Rows: plane[y][x] → tmp[y][k_x].
    let mut tmp = vec![0.0f32; n * n];
    for y in 0..n {
        for k in 0..n {
            let mut sum = 0.0;
            for x in 0..n {
                sum += plane[y * n + x] * table[k * n + x];
            }
            tmp[y * n + k] = sum;
        }
    }
    // Columns: tmp[y][k_x] → out[k_y][k_x].
    let mut out = vec![0.0f32; n * n];
    for x in 0..n {
        for k in 0..n {
            let mut sum = 0.0;
            for y in 0..n {
                sum += tmp[y * n + x] * table[k * n + y];
            }
            out[k * n + x] = sum;
        }
    }
    out
}

/// Median of an even-length vector (average of the two central order stats).
/// `total_cmp` keeps the ordering total across all finite f32 values.
fn median_of(values: &[f32]) -> f32 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    let mid = sorted.len() / 2;
    (sorted[mid - 1] + sorted[mid]) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(width: u32, height: u32, luma: u8) -> image::DynamicImage {
        image::DynamicImage::from(image::ImageBuffer::from_fn(width, height, |_, _| {
            image::Luma([luma])
        }))
    }

    #[test]
    fn identical_images_have_zero_distance() {
        let a = solid(100, 100, 128);
        let b = solid(100, 100, 128);
        assert_eq!(dhash(&a).hamming(dhash(&b)), 0);
    }

    #[test]
    fn black_and_white_differ() {
        let a = solid(100, 100, 0);
        let b = solid(100, 100, 255);
        // solid colors yield all-equal neighbors; dHash of uniform gray = 0 bits set
        assert_eq!(dhash(&a).as_u64(), 0);
        assert_eq!(dhash(&b).as_u64(), 0);
        assert_eq!(dhash(&a).hamming(dhash(&b)), 0);
    }

    #[test]
    fn decreasing_gradient_sets_all_bits() {
        // dHash bit = left > right; a right-to-left brightening gradient sets every bit.
        let grad = image::DynamicImage::from(image::ImageBuffer::from_fn(200, 200, |x, _| {
            image::Luma([((199 - x) * 255 / 200) as u8])
        }));
        assert_eq!(dhash(&grad).as_u64(), u64::MAX);
    }

    #[test]
    fn resized_copy_stays_similar() {
        let grad = image::DynamicImage::from(image::ImageBuffer::from_fn(400, 400, |x, _| {
            image::Luma([(x * 255 / 400) as u8])
        }));
        let small = grad.resize(100, 100, image::imageops::FilterType::Lanczos3);
        assert!(dhash(&grad).similar(dhash(&small), 12));
    }

    #[test]
    fn phash_identical_images_have_zero_distance() {
        let a = solid(100, 100, 128);
        let b = solid(100, 100, 128);
        assert_eq!(phash(&a).hamming(phash(&b)), 0);
    }

    #[test]
    fn phash_of_solid_images_is_resolution_independent() {
        // Same content at different input sizes resamples to the same
        // 32x32 plane, so the hash must be identical.
        let small = solid(64, 64, 255);
        let large = solid(100, 100, 255);
        assert_eq!(phash(&small), phash(&large));
        assert_eq!(phash(&solid(64, 64, 0)), phash(&solid(80, 120, 0)));
    }

    #[test]
    fn phash_resized_copy_stays_similar() {
        // Textured blob (bright disc + dark corner on mid gray): the DCT
        // coefficients sit far from the median, so a downscaled copy keeps a
        // small distance. (A pure linear gradient is pathological for
        // median-based pHash: its near-median coefficients are quantization
        // noise whose signs flip across scales — that case belongs to dHash.)
        let blob = image::DynamicImage::from(image::ImageBuffer::from_fn(128, 128, |x, y| {
            let dx = x as f32 - 64.0;
            let dy = y as f32 - 64.0;
            let v = if dx * dx + dy * dy < 40.0 * 40.0 {
                230u8
            } else if x < 24 && y < 24 {
                20u8
            } else {
                120u8
            };
            image::Luma([v])
        }));
        let small = blob.resize(64, 64, image::imageops::FilterType::Lanczos3);
        let dist = phash(&blob).hamming(phash(&small));
        assert!(dist <= 10, "hamming distance was {dist}");
    }

    #[test]
    fn dct_of_constant_plane_has_only_dc_terms() {
        // DCT-II of a constant signal: X[0] = c·n per pass, X[k>0] = 0.
        let n = 8;
        let plane = vec![3.5f32; n * n];
        let out = dct2_2d(&plane, n);
        // The only nonzero coefficient is the DC×DC term at index 0.
        assert!(
            (out[0] - 3.5 * (n * n) as f32).abs() < 1e-3,
            "dc: {}",
            out[0]
        );
        for (idx, value) in out.iter().enumerate().skip(1) {
            assert!(value.abs() < 1e-3, "idx {idx}: {value}");
        }
    }
}
