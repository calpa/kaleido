//! Real PP-OCRv5 mobile run on `tests/fixtures/ocr-sample.png`.
//!
//! Needs `--features engine-ort` and the model files
//! (`scripts/fetch-models.sh paddle-ocrv5-mobile`); `KALEIDO_MODELS`
//! overrides the models root.
#![cfg(feature = "engine-ort")]

use kaleido_core::{run_ensemble, ModelStore, OcrEngine, PaddleOcrV5};
use std::path::{Path, PathBuf};

fn models_root() -> PathBuf {
    std::env::var_os("KALEIDO_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models"))
}

#[test]
fn ppocrv5_reads_sample_image() {
    let store = ModelStore::open(models_root()).expect("manifest");
    let dir = store
        .verify(kaleido_core::ppocr::ENGINE_ID)
        .expect("run scripts/fetch-models.sh paddle-ocrv5-mobile");
    let engine = PaddleOcrV5::init(&dir).expect("load PP-OCRv5");
    let image =
        image::open(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ocr-sample.png"))
            .expect("sample image");

    let mut engines: Vec<Box<dyn OcrEngine>> = vec![Box::new(engine)];
    let blocks = run_ensemble(&mut engines, &image).expect("ocr");
    println!("{}", serde_json::to_string_pretty(&blocks).unwrap());

    let texts: Vec<&str> = blocks.iter().map(|b| b.text.as_str()).collect();
    for want in ["Kaleido OCR 2026", "東京都渋谷区", "こんにちは世界"] {
        assert!(texts.contains(&want), "missing {want:?} in {texts:?}");
    }
    for b in &blocks {
        assert_eq!(b.model, "paddle-ocrv5-mobile");
        let r = b.bbox.expect("detector boxes");
        assert!(0.0 <= r.x0 && r.x0 < r.x1 && r.x1 <= 1.0);
        assert!(0.0 <= r.y0 && r.y0 < r.y1 && r.y1 <= 1.0);
    }
}
