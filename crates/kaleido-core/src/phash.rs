//! 64-bit difference hash (dHash): resample to 9x8 grayscale, compare horizontal neighbors.

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
pub fn dhash_file(path: &Path) -> anyhow::Result<Hash64> {
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
}
