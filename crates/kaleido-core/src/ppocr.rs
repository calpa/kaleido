//! PP-OCRv5 mobile pipeline (research §2 and §5): DB text detection followed
//! by CTC text recognition, both ONNX models run on ONNX Runtime via `ort`.
//!
//! The pre/post-processing below is pure Rust and always compiled, so it is
//! unit tested in the default build; only [`PaddleOcrV5`] needs the
//! `engine-ort` feature.

use crate::ocr::{EngineProfile, ScriptSet};
use image::{imageops::FilterType, RgbImage};

/// Engine and model id (matches `MODEL_SPECS` and `models/manifest.json`).
pub const ENGINE_ID: &str = "paddle-ocrv5-mobile";
/// Detection model file name.
pub const DET_FILE: &str = "pp-ocrv5_mobile_det.onnx";
/// Recognition model file name.
pub const REC_FILE: &str = "pp-ocrv5_mobile_rec.onnx";
/// Recognition dictionary file name.
pub const DICT_FILE: &str = "ppocrv5_dict.txt";

/// Longest detector input side (PaddleOCR `DetResizeForTest` limit).
const DET_LIMIT_SIDE: u32 = 960;
/// DB binarization threshold.
const DB_THRESH: f32 = 0.3;
/// Minimum mean probability inside a candidate box.
const DB_BOX_THRESH: f32 = 0.6;
/// DB unclip expansion ratio.
const DB_UNCLIP_RATIO: f32 = 1.5;
/// Boxes whose shorter side is below this (in map pixels) are dropped.
const DB_MIN_SIZE: f32 = 3.0;
/// Recognition input height.
pub const REC_HEIGHT: u32 = 48;
/// Minimum recognition input width (PaddleOCR `rec_image_shape` 3×48×320).
const REC_MIN_WIDTH: u32 = 320;
/// Width cap so very long lines do not explode memory.
const REC_MAX_WIDTH: u32 = 3200;
/// ImageNet mean/std used by the detector, in the BGR order PaddleOCR feeds.
const DET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const DET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Ensemble profile: detector-class, weight 1.0, reads Han/Kana/Latin
/// (research §2: one recognizer covers zh/zh-Hant/en/ja).
pub fn profile() -> EngineProfile {
    EngineProfile::new(
        ENGINE_ID,
        1.0,
        ScriptSet {
            han: true,
            kana: true,
            latin: true,
        },
        true,
    )
}

/// Detector input size: scale so the longer side is at most
/// [`DET_LIMIT_SIDE`], then round both sides to a multiple of 32 (≥ 32).
pub fn det_input_size(width: u32, height: u32) -> (u32, u32) {
    let longest = width.max(height).max(1) as f32;
    let ratio = if longest > DET_LIMIT_SIDE as f32 {
        DET_LIMIT_SIDE as f32 / longest
    } else {
        1.0
    };
    let round32 = |v: u32| -> u32 {
        let scaled = (v as f32 * ratio).round() as u32;
        (scaled.div_ceil(32) * 32).max(32)
    };
    (round32(width), round32(height))
}

/// Planar CHW tensor data in BGR channel order, `(px / 255 - mean) / std`.
pub fn to_chw_bgr(img: &RgbImage, mean: [f32; 3], std: [f32; 3]) -> Vec<f32> {
    let (w, h) = img.dimensions();
    let plane = (w * h) as usize;
    let mut out = vec![0.0f32; plane * 3];
    for (i, px) in img.pixels().enumerate() {
        // BGR: channel 0 is blue.
        for (c, value) in [px[2], px[1], px[0]].into_iter().enumerate() {
            out[c * plane + i] = (value as f32 / 255.0 - mean[c]) / std[c];
        }
    }
    out
}

/// Detector preprocessing: resize and normalize. Returns tensor data and the
/// (width, height) of the input.
pub fn det_preprocess(img: &RgbImage) -> (Vec<f32>, u32, u32) {
    let (w, h) = det_input_size(img.width(), img.height());
    let resized = image::imageops::resize(img, w, h, FilterType::Triangle);
    (to_chw_bgr(&resized, DET_MEAN, DET_STD), w, h)
}

/// Axis-aligned text box in probability-map pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetBox {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    /// Mean probability inside the box before unclipping.
    pub score: f32,
}

/// DB post-processing on a `width × height` probability map: binarize,
/// take 8-connected components, score each bounding box by its mean
/// probability, then unclip by `area · ratio / perimeter` on every side.
///
/// PaddleOCR fits rotated rectangles; this first version keeps axis-aligned
/// boxes, which is what [`crate::ocr::Rect`] carries anyway.
pub fn db_boxes(prob: &[f32], width: usize, height: usize) -> Vec<DetBox> {
    if prob.len() < width * height {
        return Vec::new();
    }
    let mut seen = vec![false; width * height];
    let mut boxes = Vec::new();
    let mut stack = Vec::new();
    for start in 0..width * height {
        if seen[start] || prob[start] <= DB_THRESH {
            continue;
        }
        seen[start] = true;
        stack.push(start);
        let (mut x0, mut y0, mut x1, mut y1) = (width, height, 0usize, 0usize);
        while let Some(idx) = stack.pop() {
            let (x, y) = (idx % width, idx / width);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                    if nx < 0 || ny < 0 || nx >= width as i64 || ny >= height as i64 {
                        continue;
                    }
                    let n = ny as usize * width + nx as usize;
                    if !seen[n] && prob[n] > DB_THRESH {
                        seen[n] = true;
                        stack.push(n);
                    }
                }
            }
        }
        let (bw, bh) = ((x1 - x0 + 1) as f32, (y1 - y0 + 1) as f32);
        if bw.min(bh) < DB_MIN_SIZE {
            continue;
        }
        let mut sum = 0.0f32;
        for y in y0..=y1 {
            sum += prob[y * width + x0..=y * width + x1].iter().sum::<f32>();
        }
        let score = sum / (bw * bh);
        if score < DB_BOX_THRESH {
            continue;
        }
        let offset = bw * bh * DB_UNCLIP_RATIO / (2.0 * (bw + bh));
        let b = DetBox {
            x0: (x0 as f32 - offset).max(0.0),
            y0: (y0 as f32 - offset).max(0.0),
            x1: (x1 as f32 + 1.0 + offset).min(width as f32),
            y1: (y1 as f32 + 1.0 + offset).min(height as f32),
            score,
        };
        if (b.x1 - b.x0).min(b.y1 - b.y0) >= DB_MIN_SIZE + 2.0 {
            boxes.push(b);
        }
    }
    boxes
}

/// Recognizer preprocessing: rotate tall crops (vertical text) by 90°, resize
/// to height [`REC_HEIGHT`] keeping aspect ratio, normalize to [-1, 1], and
/// zero-pad the width to at least [`REC_MIN_WIDTH`]. Returns tensor data and
/// the padded width.
pub fn rec_preprocess(crop: &RgbImage) -> (Vec<f32>, u32) {
    let rotated;
    let crop = if crop.height() as f32 >= 1.5 * crop.width() as f32 {
        rotated = image::imageops::rotate270(crop);
        &rotated
    } else {
        crop
    };
    let ratio = crop.width().max(1) as f32 / crop.height().max(1) as f32;
    let resized_w = ((REC_HEIGHT as f32 * ratio).ceil() as u32).clamp(1, REC_MAX_WIDTH);
    let padded_w = resized_w.max(REC_MIN_WIDTH);
    let resized = image::imageops::resize(crop, resized_w, REC_HEIGHT, FilterType::Triangle);
    let plane = (padded_w * REC_HEIGHT) as usize;
    let mut out = vec![0.0f32; plane * 3];
    for (x, y, px) in resized.enumerate_pixels() {
        let i = (y * padded_w + x) as usize;
        for (c, value) in [px[2], px[1], px[0]].into_iter().enumerate() {
            out[c * plane + i] = value as f32 / 127.5 - 1.0;
        }
    }
    (out, padded_w)
}

/// Parse a PaddleOCR dictionary: one symbol per line. CTC class 0 is the
/// blank, classes `1..=n` map to the lines, and class `n + 1` is a space
/// (`use_space_char`).
pub fn load_dict(text: &str) -> Vec<String> {
    let mut dict: Vec<String> = text
        .lines()
        .map(|l| l.trim_end_matches('\r').to_string())
        .collect();
    dict.push(" ".to_string());
    dict
}

/// Greedy CTC decoding of a `steps × classes` probability matrix: argmax per
/// step, collapse repeats, drop blanks. Confidence is the mean max
/// probability of the emitted characters (0 when nothing is emitted).
pub fn ctc_decode(probs: &[f32], steps: usize, classes: usize, dict: &[String]) -> (String, f32) {
    let mut text = String::new();
    let mut conf_sum = 0.0f32;
    let mut emitted = 0usize;
    let mut prev = 0usize;
    for row in probs.chunks_exact(classes.max(1)).take(steps) {
        let (best, p) = row
            .iter()
            .copied()
            .enumerate()
            .fold(
                (0usize, f32::MIN),
                |acc, (i, v)| if v > acc.1 { (i, v) } else { acc },
            );
        if best != 0 && best != prev {
            if let Some(symbol) = dict.get(best - 1) {
                text.push_str(symbol);
                conf_sum += p;
                emitted += 1;
            }
        }
        prev = best;
    }
    let conf = if emitted == 0 {
        0.0
    } else {
        conf_sum / emitted as f32
    };
    (text, conf)
}

#[cfg(feature = "engine-ort")]
pub use self::runtime::PaddleOcrV5;

#[cfg(feature = "engine-ort")]
mod runtime {
    use super::*;
    use crate::engine::OcrEngine;
    use crate::error::{KaleidoError, KaleidoResult};
    use crate::ocr::{EngineBlock, Rect};
    use image::DynamicImage;
    use ort::environment::Environment;
    use ort::session::Session;
    use ort::value::TensorRef;
    use std::path::Path;
    use std::sync::OnceLock;

    fn engine_err(e: impl std::fmt::Display) -> KaleidoError {
        KaleidoError::Engine(format!("{ENGINE_ID}: {e}"))
    }

    /// ONNX Runtime allows one environment per process; share it.
    fn environment() -> KaleidoResult<&'static Environment> {
        static ENV: OnceLock<Environment> = OnceLock::new();
        if let Some(env) = ENV.get() {
            return Ok(env);
        }
        let builder = ort::init().with_name("kaleido");
        #[cfg(feature = "engine-ort-coreml")]
        let builder = builder.with_execution_providers([ort::ep::CoreML::default().build()]);
        let env = builder.build().map_err(engine_err)?;
        Ok(ENV.get_or_init(|| env))
    }

    fn load_session(path: &Path) -> KaleidoResult<Session> {
        if !path.is_file() {
            return Err(KaleidoError::Model(format!(
                "{} missing; run scripts/fetch-models.sh",
                path.display()
            )));
        }
        Session::builder(environment()?)
            .map_err(engine_err)?
            .commit_from_file(path)
            .map_err(engine_err)
    }

    /// PP-OCRv5 mobile det + rec on ONNX Runtime (CPU, or CoreML with the
    /// `engine-ort-coreml` feature when the provider registers).
    pub struct PaddleOcrV5 {
        det: Session,
        rec: Session,
        dict: Vec<String>,
    }

    impl PaddleOcrV5 {
        fn run_f32(
            session: &mut Session,
            shape: [usize; 4],
            data: &[f32],
        ) -> KaleidoResult<(Vec<i64>, Vec<f32>)> {
            let input = TensorRef::from_array_view((shape, data)).map_err(engine_err)?;
            let outputs = session.run(ort::inputs![input]).map_err(engine_err)?;
            let (out_shape, values) = outputs[0].try_extract_tensor::<f32>().map_err(engine_err)?;
            Ok((out_shape.to_vec(), values.to_vec()))
        }

        fn recognize(&mut self, crop: &RgbImage) -> KaleidoResult<(String, f32)> {
            let (data, width) = rec_preprocess(crop);
            let (shape, probs) = Self::run_f32(
                &mut self.rec,
                [1, 3, REC_HEIGHT as usize, width as usize],
                &data,
            )?;
            let [_, steps, classes] = shape[..] else {
                return Err(engine_err(format!("unexpected rec output shape {shape:?}")));
            };
            let (steps, classes) = (steps.max(0) as usize, classes.max(0) as usize);
            if classes != self.dict.len() + 1 {
                return Err(engine_err(format!(
                    "rec has {classes} classes but dictionary implies {}",
                    self.dict.len() + 1
                )));
            }
            Ok(ctc_decode(&probs, steps, classes, &self.dict))
        }
    }

    impl OcrEngine for PaddleOcrV5 {
        fn init(model_path: &Path) -> KaleidoResult<Self> {
            let dict_path = model_path.join(DICT_FILE);
            let dict_text = std::fs::read_to_string(&dict_path).map_err(|e| {
                KaleidoError::Model(format!("cannot read {}: {e}", dict_path.display()))
            })?;
            Ok(Self {
                det: load_session(&model_path.join(DET_FILE))?,
                rec: load_session(&model_path.join(REC_FILE))?,
                dict: load_dict(&dict_text),
            })
        }

        fn profile(&self) -> EngineProfile {
            profile()
        }

        fn detect_and_recognize(
            &mut self,
            image: &DynamicImage,
        ) -> KaleidoResult<Vec<EngineBlock>> {
            let rgb = image.to_rgb8();
            let (img_w, img_h) = rgb.dimensions();
            if img_w == 0 || img_h == 0 {
                return Ok(Vec::new());
            }
            let (data, det_w, det_h) = det_preprocess(&rgb);
            let (shape, prob) =
                Self::run_f32(&mut self.det, [1, 3, det_h as usize, det_w as usize], &data)?;
            let (map_h, map_w) = match shape[..] {
                [_, _, h, w] => (h.max(0) as usize, w.max(0) as usize),
                _ => return Err(engine_err(format!("unexpected det output shape {shape:?}"))),
            };
            let (sx, sy) = (
                img_w as f32 / map_w.max(1) as f32,
                img_h as f32 / map_h.max(1) as f32,
            );

            let mut blocks = Vec::new();
            for b in db_boxes(&prob, map_w, map_h) {
                let x0 = (b.x0 * sx).floor().clamp(0.0, img_w as f32 - 1.0) as u32;
                let y0 = (b.y0 * sy).floor().clamp(0.0, img_h as f32 - 1.0) as u32;
                let x1 = (b.x1 * sx).ceil().clamp(x0 as f32 + 1.0, img_w as f32) as u32;
                let y1 = (b.y1 * sy).ceil().clamp(y0 as f32 + 1.0, img_h as f32) as u32;
                let crop = image::imageops::crop_imm(&rgb, x0, y0, x1 - x0, y1 - y0).to_image();
                let (text, conf) = self.recognize(&crop)?;
                if text.trim().is_empty() {
                    continue;
                }
                let rect = Rect {
                    x0: x0 as f32 / img_w as f32,
                    y0: y0 as f32 / img_h as f32,
                    x1: x1 as f32 / img_w as f32,
                    y1: y1 as f32 / img_h as f32,
                };
                blocks.push(EngineBlock::new(ENGINE_ID, &text, Some(rect), Some(conf)));
            }
            Ok(blocks)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn det_input_size_caps_long_side_and_rounds_to_32() {
        assert_eq!(det_input_size(640, 480), (640, 480));
        assert_eq!(det_input_size(1920, 1080), (960, 544));
        assert_eq!(det_input_size(10, 10), (32, 32));
        assert_eq!(det_input_size(100, 50), (128, 64));
    }

    #[test]
    fn chw_bgr_puts_blue_in_first_plane() {
        let img = RgbImage::from_pixel(2, 1, image::Rgb([255, 0, 0]));
        let data = to_chw_bgr(&img, [0.0; 3], [1.0; 3]);
        assert_eq!(data, vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn db_finds_one_box_per_blob_and_unclips() {
        let (w, h) = (40, 20);
        let mut prob = vec![0.0f32; w * h];
        // Blob A: x 2..=11, y 4..=9 (10×6). Blob B: x 25..=34, y 4..=9.
        for y in 4..=9 {
            for x in (2..=11).chain(25..=34) {
                prob[y * w + x] = 0.9;
            }
        }
        // Weak blob below the box threshold is dropped.
        for y in 14..=17 {
            for x in 2..=8 {
                prob[y * w + x] = 0.4;
            }
        }
        let boxes = db_boxes(&prob, w, h);
        assert_eq!(boxes.len(), 2, "{boxes:?}");
        let a = boxes[0];
        assert!((a.score - 0.9).abs() < 1e-5);
        // offset = 10·6·1.5 / (2·16) = 2.8125
        assert!((a.x0 - (2.0 - 2.8125)).abs() < 1e-4 || a.x0 == 0.0);
        assert!((a.y0 - (4.0 - 2.8125)).abs() < 1e-4);
        assert!((a.x1 - (12.0 + 2.8125)).abs() < 1e-4);
        assert!((a.y1 - (10.0 + 2.8125)).abs() < 1e-4);
    }

    #[test]
    fn db_ignores_tiny_blobs_and_short_maps() {
        let mut prob = vec![0.0f32; 100];
        prob[55] = 1.0;
        assert!(db_boxes(&prob, 10, 10).is_empty());
        assert!(db_boxes(&prob[..10], 10, 10).is_empty());
    }

    #[test]
    fn ctc_collapses_repeats_and_blanks() {
        let dict = load_dict("a\nb\n");
        // classes: 0 blank, 1 'a', 2 'b', 3 ' '
        let classes = 4;
        let rows = [1usize, 1, 0, 1, 2, 3, 2];
        let mut probs = vec![0.0f32; rows.len() * classes];
        for (t, &c) in rows.iter().enumerate() {
            probs[t * classes + c] = 0.8;
        }
        let (text, conf) = ctc_decode(&probs, rows.len(), classes, &dict);
        assert_eq!(text, "aab b");
        assert!((conf - 0.8).abs() < 1e-6);
        assert_eq!(ctc_decode(&[1.0, 0.0, 0.0, 0.0], 1, classes, &dict).0, "");
    }

    #[test]
    fn rec_preprocess_pads_to_min_width_and_rotates_tall_crops() {
        let wide = RgbImage::from_pixel(96, 48, image::Rgb([255, 255, 255]));
        let (data, w) = rec_preprocess(&wide);
        assert_eq!(w, REC_MIN_WIDTH);
        assert_eq!(data.len(), (3 * REC_HEIGHT * w) as usize);
        assert!((data[0] - 1.0).abs() < 1e-6, "white maps to 1.0");
        assert_eq!(data[(REC_MIN_WIDTH - 1) as usize], 0.0, "padding is zero");

        // 20×400 tall crop rotates to 400×20, i.e. width 960 at height 48.
        let tall = RgbImage::from_pixel(20, 400, image::Rgb([0, 0, 0]));
        assert_eq!(rec_preprocess(&tall).1, 960);
    }
}
