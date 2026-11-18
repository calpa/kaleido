//! OCR engine abstraction: every engine (fake, PP-OCRv5, later Tesseract and
//! VLMs) turns an image into per-engine [`EngineBlock`]s, which
//! [`run_ensemble`] merges into evidence-graded [`OcrBlock`]s through the M3
//! ensemble (`docs/research/ocr-vlm-models.md` §4).

use crate::error::KaleidoResult;
use crate::ocr::{ensemble, EngineBlock, EngineProfile, OcrBlock, Rect, ScriptSet};
use image::DynamicImage;
use std::path::Path;

/// One OCR engine.
///
/// Engines emit raw [`EngineBlock`]s (text, normalized box, self-confidence)
/// rather than [`OcrBlock`]s: evidence levels only exist after the ensemble
/// has compared engines, so grading is done once, in [`run_ensemble`].
pub trait OcrEngine {
    /// Load the engine from its model directory (as resolved by
    /// [`crate::model_store::ModelStore`]).
    fn init(model_path: &Path) -> KaleidoResult<Self>
    where
        Self: Sized;

    /// Static capability profile used by the ensemble (id, weight, scripts).
    fn profile(&self) -> EngineProfile;

    /// Detect text regions and recognize their contents.
    fn detect_and_recognize(&mut self, image: &DynamicImage) -> KaleidoResult<Vec<EngineBlock>>;
}

/// Run every engine on `image` and merge their outputs with the ensemble.
pub fn run_ensemble(
    engines: &mut [Box<dyn OcrEngine>],
    image: &DynamicImage,
) -> KaleidoResult<Vec<OcrBlock>> {
    let mut profiles = Vec::with_capacity(engines.len());
    let mut blocks = Vec::new();
    for engine in engines.iter_mut() {
        profiles.push(engine.profile());
        blocks.extend(engine.detect_and_recognize(image)?);
    }
    Ok(ensemble(&blocks, &profiles))
}

/// Deterministic stand-in engine: needs no model files and returns canned
/// blocks. Keeps the default build (and CI) free of model downloads.
#[derive(Debug, Clone)]
pub struct FakeEngine {
    profile: EngineProfile,
    blocks: Vec<EngineBlock>,
}

impl FakeEngine {
    /// Engine id used by [`FakeEngine::init`] and the CLI.
    pub const ID: &'static str = "fake";
    /// Text emitted by the default fake engine.
    pub const TEXT: &'static str = "kaleido fake ocr";

    /// Fake engine with a custom profile that always returns `blocks`
    /// (their `engine` field is rewritten to the profile id).
    pub fn with_blocks(profile: EngineProfile, blocks: Vec<EngineBlock>) -> Self {
        let blocks = blocks
            .into_iter()
            .map(|mut b| {
                b.engine = profile.id.clone();
                b
            })
            .collect();
        Self { profile, blocks }
    }
}

impl Default for FakeEngine {
    /// One full-frame Latin block with confidence 1.0.
    fn default() -> Self {
        let full = Rect {
            x0: 0.0,
            y0: 0.0,
            x1: 1.0,
            y1: 1.0,
        };
        Self::with_blocks(
            EngineProfile::new(Self::ID, 1.0, ScriptSet::LATIN, true),
            vec![EngineBlock::new(
                Self::ID,
                Self::TEXT,
                Some(full),
                Some(1.0),
            )],
        )
    }
}

impl OcrEngine for FakeEngine {
    /// The fake engine has no model files; `model_path` is ignored.
    fn init(_model_path: &Path) -> KaleidoResult<Self> {
        Ok(Self::default())
    }

    fn profile(&self) -> EngineProfile {
        self.profile.clone()
    }

    fn detect_and_recognize(&mut self, _image: &DynamicImage) -> KaleidoResult<Vec<EngineBlock>> {
        Ok(self.blocks.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ocr::Evidence;

    fn blank() -> DynamicImage {
        DynamicImage::new_rgb8(8, 8)
    }

    fn rect(x0: f32, y0: f32, x1: f32, y1: f32) -> Rect {
        Rect { x0, y0, x1, y1 }
    }

    #[test]
    fn default_fake_engine_yields_single_low_block() {
        let mut engines: Vec<Box<dyn OcrEngine>> =
            vec![Box::new(FakeEngine::init(Path::new("unused")).unwrap())];
        let out = run_ensemble(&mut engines, &blank()).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, FakeEngine::TEXT);
        assert_eq!(out[0].model, FakeEngine::ID);
        // A lone detector without corroboration stays low (research §4 step 6).
        assert_eq!(out[0].evidence, Evidence::Low);
    }

    #[test]
    fn two_agreeing_fake_engines_reach_medium_or_better() {
        let both = ScriptSet {
            han: true,
            kana: true,
            latin: true,
        };
        let block = |id: &str| {
            EngineBlock::new(
                id,
                "請求書 2026",
                Some(rect(0.1, 0.1, 0.5, 0.2)),
                Some(0.95),
            )
        };
        let a = FakeEngine::with_blocks(EngineProfile::new("a", 1.0, both, true), vec![block("x")]);
        let b = FakeEngine::with_blocks(EngineProfile::new("b", 0.6, both, true), vec![block("y")]);
        let mut engines: Vec<Box<dyn OcrEngine>> = vec![Box::new(a), Box::new(b)];
        let out = run_ensemble(&mut engines, &blank()).unwrap();
        assert_eq!(out.len(), 1);
        assert_ne!(out[0].evidence, Evidence::Low);
    }
}
