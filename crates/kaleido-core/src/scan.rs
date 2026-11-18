//! Folder scanning: enumerate supported images with EXIF metadata.
//!
//! Corrupted images are skipped gracefully: a file whose header cannot be
//! decoded never fails the scan, it only bumps [`ScanStats::skipped`].

use crate::error::{KaleidoError, KaleidoResult};
use exif::{Reader, Tag, Value};
use serde::Serialize;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use walkdir::WalkDir;

/// Supported raster formats (matches the `image` crate features enabled).
pub const SUPPORTED_EXTENSIONS: &[&str] =
    &["jpg", "jpeg", "png", "gif", "webp", "tif", "tiff", "bmp"];

/// EXIF summary embedded in an [`ImageEntry`].
///
/// Absent data is `None`. Extraction never fails the scan: a file without
/// EXIF (or with malformed metadata) yields the all-`None` default.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Exif {
    /// Camera make and model, deduplicated, e.g. "Canon EOS R5".
    pub camera: Option<String>,
    /// Lens model string, e.g. "RF24-70mm F2.8 L IS USM".
    pub lens: Option<String>,
    /// ISO sensitivity.
    pub iso: Option<u16>,
    /// Aperture as f-number string, e.g. "f/2.8".
    pub aperture: Option<String>,
    /// GPS position in signed decimal degrees (north/east positive).
    pub gps: Option<Gps>,
}

/// GPS position in signed decimal degrees.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Gps {
    pub latitude: f64,
    pub longitude: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImageEntry {
    pub path: PathBuf,
    pub width: u32,
    pub height: u32,
    pub bytes: u64,
    pub modified: Option<SystemTime>,
    pub exif: Exif,
}

#[derive(Debug, Clone, Serialize)]
pub struct ScanStats {
    pub scanned: u64,
    pub skipped: u64,
    pub entries: Vec<ImageEntry>,
}

/// Recursively collect image entries under `root`, in parallel.
pub fn scan(root: &Path) -> KaleidoResult<ScanStats> {
    use rayon::prelude::*;

    if !root.is_dir() {
        return Err(KaleidoError::InvalidInput(format!(
            "scan root is not a directory: {}",
            root.display()
        )));
    }

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

/// Extract an EXIF summary from an image file. Never fails: unreadable,
/// unsupported, or malformed metadata yields the all-`None` default.
pub fn exif_from_file(path: &Path) -> Exif {
    let file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(_) => return Exif::default(),
    };
    let mut reader = BufReader::new(file);
    let raw = match Reader::new().read_from_container(&mut reader) {
        Ok(raw) => raw,
        Err(_) => return Exif::default(),
    };
    Exif {
        camera: camera_name(&raw),
        lens: ascii_field(&raw, Tag::LensModel),
        iso: iso_value(&raw),
        aperture: aperture_value(&raw),
        gps: gps_position(&raw),
    }
}

/// Convert EXIF DMS rationals (degrees, minutes, seconds) to signed decimal
/// degrees. `hemisphere` is one of 'N'/'S'/'E'/'W'; south and west are
/// negative. Returns `None` for malformed input (wrong arity or a zero
/// denominator, which would divide by zero).
pub fn dms_to_decimal(dms: &[exif::Rational], hemisphere: char) -> Option<f64> {
    if dms.len() != 3 || dms.iter().any(|r| r.denom == 0) {
        return None;
    }
    let value = dms[0].to_f64() + dms[1].to_f64() / 60.0 + dms[2].to_f64() / 3600.0;
    let sign = match hemisphere.to_ascii_uppercase() {
        'N' | 'E' => 1.0,
        'S' | 'W' => -1.0,
        _ => return None,
    };
    Some(sign * value)
}

fn read_entry(path: &Path) -> KaleidoResult<ImageEntry> {
    let meta = std::fs::metadata(path)?;
    let (width, height) = image::image_dimensions(path)?;
    Ok(ImageEntry {
        path: path.to_path_buf(),
        width,
        height,
        bytes: meta.len(),
        modified: meta.modified().ok(),
        exif: exif_from_file(path),
    })
}

fn ascii_field(exif: &exif::Exif, tag: Tag) -> Option<String> {
    let field = exif.get_field(tag, exif::In::PRIMARY)?;
    match &field.value {
        Value::Ascii(parts) => parts
            .first()
            .map(|bytes| {
                String::from_utf8_lossy(bytes)
                    .trim_end_matches('\0')
                    .to_string()
            })
            .filter(|s| !s.is_empty()),
        _ => None,
    }
}

fn rationals(exif: &exif::Exif, tag: Tag) -> Vec<exif::Rational> {
    let field = match exif.get_field(tag, exif::In::PRIMARY) {
        Some(field) => field,
        None => return Vec::new(),
    };
    match &field.value {
        Value::Rational(rs) => rs.clone(),
        _ => Vec::new(),
    }
}

/// "Make Model", collapsing the common case where the model string already
/// starts with the make (e.g. Make="Canon", Model="Canon EOS R5" → "Canon EOS R5").
fn camera_name(exif: &exif::Exif) -> Option<String> {
    let make = ascii_field(exif, Tag::Make);
    let model = ascii_field(exif, Tag::Model);
    match (make, model) {
        (None, None) => None,
        (Some(single), None) | (None, Some(single)) => Some(single),
        (Some(make), Some(model)) => {
            let prefix = format!("{} ", make);
            if model
                .to_ascii_lowercase()
                .starts_with(&prefix.to_ascii_lowercase())
            {
                Some(model)
            } else {
                Some(format!("{prefix}{model}"))
            }
        }
    }
}

fn iso_value(exif: &exif::Exif) -> Option<u16> {
    let field = exif.get_field(Tag::PhotographicSensitivity, exif::In::PRIMARY)?;
    match &field.value {
        Value::Short(v) => v.first().copied(),
        _ => None,
    }
}

fn aperture_value(exif: &exif::Exif) -> Option<String> {
    let rs = rationals(exif, Tag::FNumber);
    let r = rs.first()?;
    if r.denom == 0 {
        return None;
    }
    let s = format!("{:.1}", r.to_f64());
    let trimmed = s.strip_suffix(".0").unwrap_or(&s);
    Some(format!("f/{trimmed}"))
}

fn gps_position(exif: &exif::Exif) -> Option<Gps> {
    let lat_ref = ascii_field(exif, Tag::GPSLatitudeRef)?;
    let lat = rationals(exif, Tag::GPSLatitude);
    let lon_ref = ascii_field(exif, Tag::GPSLongitudeRef)?;
    let lon = rationals(exif, Tag::GPSLongitude);
    Some(Gps {
        latitude: dms_to_decimal(&lat, lat_ref.chars().next()?)?,
        longitude: dms_to_decimal(&lon, lon_ref.chars().next()?)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("kaleido-core-scan-{}-{}", std::process::id(), tag));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_file(dir: &Path, name: &str, bytes: &[u8]) {
        std::fs::write(dir.join(name), bytes).unwrap();
    }

    fn solid_png() -> Vec<u8> {
        let img = image::DynamicImage::from(image::ImageBuffer::from_fn(4, 4, |_, _| {
            image::Rgb([10u8, 20, 30])
        }));
        let mut out = std::io::Cursor::new(Vec::new());
        img.write_to(&mut out, image::ImageFormat::Png).unwrap();
        out.into_inner()
    }

    /// A valid 8x8 JPEG with an APP1 EXIF segment carrying Make/Model/
    /// LensModel/ISO/FNumber and a GPS position (Tokio-ish 35°39'30" N,
    /// 139°47'30" E). Built by encoding a tiny JPEG with the `image` crate,
    /// then splicing an APP1 segment (payload authored with kamadak-exif's
    /// writer) right after the SOI marker.
    fn exif_jpeg_bytes() -> Vec<u8> {
        use exif::experimental::Writer;

        let jpeg = {
            let img = image::DynamicImage::from(image::ImageBuffer::from_fn(8, 8, |_, _| {
                image::Rgb([90u8, 140, 200])
            }));
            let mut out = std::io::Cursor::new(Vec::new());
            img.write_to(&mut out, image::ImageFormat::Jpeg).unwrap();
            out.into_inner()
        };

        let ascii = |s: &str| Value::Ascii(vec![format!("{s}\0").into_bytes()]);
        let rational = |num: u32, denom: u32| exif::Rational { num, denom };

        let make = exif::Field {
            tag: Tag::Make,
            ifd_num: exif::In::PRIMARY,
            value: ascii("TestMake"),
        };
        let model = exif::Field {
            tag: Tag::Model,
            ifd_num: exif::In::PRIMARY,
            value: ascii("TestCam 100"),
        };
        let lens = exif::Field {
            tag: Tag::LensModel,
            ifd_num: exif::In::PRIMARY,
            value: ascii("TestLens 50mm"),
        };
        let iso = exif::Field {
            tag: Tag::PhotographicSensitivity,
            ifd_num: exif::In::PRIMARY,
            value: Value::Short(vec![200]),
        };
        let fnumber = exif::Field {
            tag: Tag::FNumber,
            ifd_num: exif::In::PRIMARY,
            value: Value::Rational(vec![rational(28, 10)]),
        };
        let lat_ref = exif::Field {
            tag: Tag::GPSLatitudeRef,
            ifd_num: exif::In::PRIMARY,
            value: ascii("N"),
        };
        let lat = exif::Field {
            tag: Tag::GPSLatitude,
            ifd_num: exif::In::PRIMARY,
            value: Value::Rational(vec![rational(35, 1), rational(39, 1), rational(30, 1)]),
        };
        let lon_ref = exif::Field {
            tag: Tag::GPSLongitudeRef,
            ifd_num: exif::In::PRIMARY,
            value: ascii("E"),
        };
        let lon = exif::Field {
            tag: Tag::GPSLongitude,
            ifd_num: exif::In::PRIMARY,
            value: Value::Rational(vec![rational(139, 1), rational(47, 1), rational(30, 1)]),
        };

        let mut writer = Writer::new();
        for field in [
            &make, &model, &lens, &iso, &fnumber, &lat_ref, &lat, &lon_ref, &lon,
        ] {
            writer.push_field(field);
        }
        let mut tiff = std::io::Cursor::new(Vec::new());
        writer.write(&mut tiff, false).unwrap();

        let payload = [b"Exif\0\0".as_slice(), tiff.get_ref().as_slice()].concat();
        let mut out = jpeg[..2].to_vec();
        out.extend_from_slice(&[0xFF, 0xE1]);
        // Segment length includes its own two length bytes.
        out.extend_from_slice((payload.len() as u16 + 2).to_be_bytes().as_slice());
        out.extend_from_slice(&payload);
        out.extend_from_slice(&jpeg[2..]);
        out
    }

    #[test]
    fn corrupted_files_are_skipped_gracefully() {
        let dir = temp_dir("corrupt");
        write_file(&dir, "broken.jpg", b"this is definitely not an image");
        write_file(&dir, "ok.png", &solid_png());
        let stats = scan(&dir).unwrap();
        assert_eq!(stats.scanned, 2);
        assert_eq!(stats.entries.len(), 1);
        assert_eq!(stats.skipped, 1);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn nonexistent_root_is_invalid_input() {
        let dir =
            std::env::temp_dir().join(format!("kaleido-core-scan-missing-{}", std::process::id()));
        assert!(matches!(scan(&dir), Err(KaleidoError::InvalidInput(_))));
    }

    #[test]
    fn png_without_exif_yields_default() {
        let dir = temp_dir("noexif");
        write_file(&dir, "plain.png", &solid_png());
        let stats = scan(&dir).unwrap();
        assert_eq!(stats.entries.len(), 1);
        assert_eq!(stats.entries[0].exif, Exif::default());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn exif_fields_are_extracted() {
        let dir = temp_dir("exif");
        write_file(&dir, "photo.jpg", &exif_jpeg_bytes());
        let stats = scan(&dir).unwrap();
        assert_eq!(stats.entries.len(), 1);
        let entry = &stats.entries[0];
        assert_eq!(entry.width, 8);
        assert_eq!(entry.height, 8);
        assert_eq!(entry.exif.camera.as_deref(), Some("TestMake TestCam 100"));
        assert_eq!(entry.exif.lens.as_deref(), Some("TestLens 50mm"));
        assert_eq!(entry.exif.iso, Some(200));
        assert_eq!(entry.exif.aperture.as_deref(), Some("f/2.8"));
        let gps = entry.exif.gps.unwrap_or(Gps {
            latitude: 0.0,
            longitude: 0.0,
        });
        assert!(
            (gps.latitude - 35.658_333).abs() < 1e-4,
            "lat {}",
            gps.latitude
        );
        assert!(
            (gps.longitude - 139.791_666).abs() < 1e-4,
            "lon {}",
            gps.longitude
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn camera_make_prefix_is_collapsed() {
        let dir = temp_dir("camera");
        write_file(&dir, "photo.jpg", &exif_jpeg_bytes());
        let stats = scan(&dir).unwrap();
        // Make="TestMake", Model="TestCam 100": no prefix overlap → joined.
        assert_eq!(
            stats.entries[0].exif.camera,
            Some("TestMake TestCam 100".to_string())
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn dms_conversion_signs_and_error_paths() {
        let r = |num: u32, denom: u32| exif::Rational { num, denom };
        let dms = [r(35, 1), r(39, 1), r(30, 1)];
        assert!((dms_to_decimal(&dms, 'N').unwrap() - 35.658_333).abs() < 1e-5);
        assert!((dms_to_decimal(&dms, 'S').unwrap() + 35.658_333).abs() < 1e-5);
        assert!((dms_to_decimal(&dms, 'e').unwrap() - 35.658_333).abs() < 1e-5);
        assert!((dms_to_decimal(&dms, 'W').unwrap() + 35.658_333).abs() < 1e-5);
        assert!(dms_to_decimal(&dms, 'X').is_none());
        assert!(dms_to_decimal(&dms[..2], 'N').is_none());
        let zero_denom = [r(1, 1), r(1, 0), r(0, 1)];
        assert!(dms_to_decimal(&zero_denom, 'N').is_none());
    }
}
