//! Color palette extraction: k-means over a downsampled RGB pixel sample.

use crate::error::KaleidoResult;
use image::GenericImageView;
use serde::Serialize;
use std::path::Path;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Swatch {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    /// Fraction of sampled pixels belonging to this cluster (0.0-1.0).
    pub share: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct Palette {
    pub swatches: Vec<Swatch>,
}

impl Swatch {
    /// Lowercase hex color string, e.g. "#1a2b3c".
    pub fn hex(&self) -> String {
        format!("#{:02x}{:02x}{:02x}", self.r, self.g, self.b)
    }
}

impl Palette {
    /// Hex string of the dominant color, e.g. `Some("#1a2b3c")`; `None`
    /// for an empty palette.
    pub fn dominant_hex(&self) -> Option<String> {
        let s = self.swatches.first()?;
        Some(format!("#{:02x}{:02x}{:02x}", s.r, s.g, s.b))
    }

    /// Export as CSS custom properties, one `--c<i>` per swatch:
    /// `:root {` / `  --c1: #1a2b3c;` / ... / `}`.
    pub fn to_css_vars(&self) -> String {
        let mut css = String::from(":root {\n");
        for (i, swatch) in self.swatches.iter().enumerate() {
            css.push_str(&format!("  --c{}: {};\n", i + 1, swatch.hex()));
        }
        css.push('}');
        css
    }
}

/// Extract a `k`-color palette from an image file.
pub fn palette_file(path: &Path, k: u8) -> KaleidoResult<Palette> {
    let img = image::open(path)?;
    Ok(palette(&img, k))
}

/// Extract a `k`-color palette from an in-memory image.
///
/// Samples at most 32x32 pixels (lanczos downsample) then runs k-means with
/// deterministic seeding (farthest-point init), 8 iterations.
pub fn palette(img: &image::DynamicImage, k: u8) -> Palette {
    let k = k.clamp(1, 16) as usize;
    let thumb = img.resize_exact(32, 32, image::imageops::FilterType::Lanczos3);
    let (w, h) = thumb.dimensions();
    let pixels: Vec<[f32; 3]> = (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .map(|(x, y)| {
            let p = thumb.get_pixel(x, y);
            [p[0] as f32, p[1] as f32, p[2] as f32]
        })
        .collect();

    let centroids = kmeans(&pixels, k, 8);
    let mut swatches: Vec<Swatch> = centroids
        .into_iter()
        .map(|(c, share)| Swatch {
            r: c[0].round().clamp(0.0, 255.0) as u8,
            g: c[1].round().clamp(0.0, 255.0) as u8,
            b: c[2].round().clamp(0.0, 255.0) as u8,
            share,
        })
        .collect();
    swatches.sort_by(|a, b| {
        b.share
            .partial_cmp(&a.share)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| (a.r, a.g, a.b).cmp(&(b.r, b.g, b.b)))
    });
    Palette { swatches }
}

fn kmeans(pixels: &[[f32; 3]], k: usize, iterations: u8) -> Vec<([f32; 3], f32)> {
    let n = pixels.len();
    if n == 0 {
        return vec![([0.0; 3], 1.0); k.max(1)];
    }
    // Farthest-point seeding for deterministic, spread-out initial centroids.
    let mut centroids: Vec<[f32; 3]> = vec![pixels[0]];
    while centroids.len() < k {
        let next = pixels
            .iter()
            .max_by_key(|p| {
                let d = centroids
                    .iter()
                    .map(|c| dist2(p, c))
                    .fold(f32::INFINITY, f32::min);
                (d * 1e6) as u64
            })
            .copied()
            .unwrap_or([0.0; 3]);
        centroids.push(next);
    }

    let mut assign = vec![0usize; n];
    for _ in 0..iterations {
        for (i, p) in pixels.iter().enumerate() {
            let (best, _) = centroids
                .iter()
                .enumerate()
                .map(|(j, c)| (j, dist2(p, c)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, 0.0));
            assign[i] = best;
        }
        let mut sums = vec![[0.0f64; 3]; k];
        let mut counts = vec![0f64; k];
        for (p, j) in pixels.iter().zip(&assign) {
            for ch in 0..3 {
                sums[*j][ch] += p[ch] as f64;
            }
            counts[*j] += 1.0;
        }
        for j in 0..k {
            if counts[j] > 0.0 {
                for ch in 0..3 {
                    centroids[j][ch] = (sums[j][ch] / counts[j]) as f32;
                }
            }
        }
    }

    let mut counts = vec![0f64; k];
    for j in &assign {
        counts[*j] += 1.0;
    }
    let total = n as f64;
    centroids
        .into_iter()
        .zip(counts)
        .map(|(c, count)| (c, (count / total) as f32))
        .collect()
}

fn dist2(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let d0 = a[0] - b[0];
    let d1 = a[1] - b[1];
    let d2 = a[2] - b[2];
    d0 * d0 + d1 * d1 + d2 * d2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn palette_returns_k_sorted_swatches() {
        let img = image::DynamicImage::from(image::ImageBuffer::from_fn(64, 64, |x, _| {
            if x < 32 {
                image::Rgb([255u8, 0, 0])
            } else {
                image::Rgb([0u8, 0, 255])
            }
        }));
        let p = palette(&img, 2);
        assert_eq!(p.swatches.len(), 2);
        assert!(p.swatches[0].share >= p.swatches[1].share);
        let total: f32 = p.swatches.iter().map(|s| s.share).sum();
        assert!(
            (total - 1.0).abs() < 1e-3,
            "shares sum to ~1.0, got {total}"
        );
    }

    #[test]
    fn dominant_hex_format() {
        let img = image::DynamicImage::from(image::ImageBuffer::from_fn(16, 16, |_, _| {
            image::Rgb([26u8, 43, 60])
        }));
        let p = palette(&img, 1);
        assert_eq!(p.dominant_hex(), Some("#1a2b3c".to_string()));
    }

    #[test]
    fn css_vars_cover_c1_through_ck() {
        let img = image::DynamicImage::from(image::ImageBuffer::from_fn(64, 64, |x, _| {
            let band = (x / 16) as u8;
            image::Rgb([band * 60, 255 - band * 40, 90 + band * 30])
        }));
        let css = palette(&img, 3).to_css_vars();
        assert!(css.starts_with(":root {\n"), "got: {css}");
        assert!(css.ends_with('}'));
        assert!(css.contains("  --c1: #"));
        assert!(css.contains("  --c2: #"));
        assert!(css.contains("  --c3: #"));
        assert!(!css.contains("--c4"), "got: {css}");
    }

    #[test]
    fn json_output_is_deterministic() {
        let img = image::DynamicImage::from(image::ImageBuffer::from_fn(64, 64, |x, y| {
            image::Rgb([(x * 4) as u8, (y * 3) as u8, ((x + y) * 2) as u8])
        }));
        let a = serde_json::to_string(&palette(&img, 5)).unwrap();
        let b = serde_json::to_string(&palette(&img, 5)).unwrap();
        assert_eq!(a, b);
        // Field order is fixed: swatches array first, then r/g/b per swatch.
        assert!(a.starts_with(r#"{"swatches":[{"r":"#), "got: {a}");
        // k clamps into 1..=16.
        assert_eq!(palette(&img, 0).swatches.len(), 1);
    }
}
