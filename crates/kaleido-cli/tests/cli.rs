//! End-to-end CLI behavior: exit-code semantics and output formats.

use std::path::PathBuf;
use std::process::Command;

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("kaleido-cli-{}-{}", std::process::id(), tag));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_png(dir: &std::path::Path, name: &str, rgb: [u8; 3]) -> PathBuf {
    let img =
        image::DynamicImage::from(image::ImageBuffer::from_fn(32, 32, |_, _| image::Rgb(rgb)));
    let path = dir.join(name);
    img.save(&path).unwrap();
    path
}

fn write_gradient_png(dir: &std::path::Path, name: &str) -> PathBuf {
    // Solid images all share the all-zero dHash, so the "different" image
    // must carry structure. Brightness decreasing with x ⇒ every
    // left>right bit set ⇒ maximally far from any solid color's hash.
    let img = image::DynamicImage::from(image::ImageBuffer::from_fn(64, 64, |x, _| {
        image::Rgb([
            ((63 - x) * 4) as u8,
            ((63 - x) * 4) as u8,
            ((63 - x) * 4) as u8,
        ])
    }));
    let path = dir.join(name);
    img.save(&path).unwrap();
    path
}

fn kaleido(args: &[&str]) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_kaleido"));
    cmd.args(args);
    cmd
}

#[test]
fn dupes_reports_groups_and_exits_zero() {
    let dir = temp_dir("dupes-hit");
    write_png(&dir, "a.png", [10, 20, 30]);
    write_png(&dir, "b.png", [10, 20, 30]);
    write_gradient_png(&dir, "c.png");

    let out = kaleido(&["dupes", dir.to_str().unwrap()]).output().unwrap();
    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        text.contains("a.png") && text.contains("b.png"),
        "got: {text}"
    );
    assert!(!text.contains("c.png"));

    // JSON variant carries the same groups plus metadata.
    let out = kaleido(&["dupes", dir.to_str().unwrap(), "--json"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(0));
    let parsed: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let groups = parsed["groups"].as_array().unwrap();
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0]["paths"].as_array().unwrap().len(), 2);
    assert_eq!(parsed["algorithm"], "dhash");
    assert_eq!(parsed["scanned"], 3);

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn dupes_without_duplicates_exits_one() {
    let dir = temp_dir("dupes-miss");
    write_png(&dir, "only.png", [10, 20, 30]);

    let out = kaleido(&["dupes", dir.to_str().unwrap(), "--json"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    let parsed: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(parsed["groups"].as_array().unwrap().len(), 0);

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn missing_dir_is_an_error_exiting_two() {
    let missing = std::env::temp_dir().join(format!("kaleido-cli-missing-{}", std::process::id()));
    let out = kaleido(&["scan", missing.to_str().unwrap()])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn scan_json_includes_exif_object() {
    let dir = temp_dir("scan-json");
    let img = write_png(&dir, "a.png", [1, 2, 3]);

    let out = kaleido(&["scan", dir.to_str().unwrap()]).output().unwrap();
    assert_eq!(out.status.code(), Some(0));
    let parsed: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let entries = parsed["entries"].as_array().unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0]["path"].as_str().unwrap(), img.to_str().unwrap());
    assert!(entries[0]["exif"].is_object(), "exif object expected");

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn palette_css_flag_prints_css_variables() {
    let dir = temp_dir("css");
    let img = write_png(&dir, "p.png", [26, 43, 60]);

    let out = kaleido(&["palette", img.to_str().unwrap(), "--css"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(0));
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains(":root"), "got: {text}");
    assert!(text.contains("--c1: #1a2b3c"), "got: {text}");

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn ocr_fake_engine_prints_json_blocks_with_evidence() {
    let dir = temp_dir("ocr-fake");
    let img = write_png(&dir, "a.png", [255, 255, 255]);
    let out = kaleido(&["ocr", img.to_str().unwrap(), "--engine", "fake", "--json"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let blocks: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let blocks = blocks.as_array().unwrap();
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0]["text"], "kaleido fake ocr");
    assert_eq!(blocks[0]["model"], "fake");
    assert_eq!(blocks[0]["evidence"], "low");

    let text = kaleido(&["ocr", img.to_str().unwrap()]).output().unwrap();
    assert!(text.status.success());
    let line = String::from_utf8(text.stdout).unwrap();
    assert!(line.starts_with("low"), "{line}");
    assert!(line.contains("kaleido fake ocr"), "{line}");
}

#[test]
fn ocr_unknown_engine_is_runtime_error() {
    let dir = temp_dir("ocr-unknown");
    let img = write_png(&dir, "a.png", [0, 0, 0]);
    let out = kaleido(&["ocr", img.to_str().unwrap(), "--engine", "nope"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("unknown engine"));
}

#[cfg(not(feature = "engine-ort"))]
#[test]
fn ocr_paddle_without_feature_explains_how_to_enable() {
    let dir = temp_dir("ocr-nofeature");
    let img = write_png(&dir, "a.png", [0, 0, 0]);
    let out = kaleido(&["ocr", img.to_str().unwrap(), "--engine", "ppocrv5"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("engine-ort"));
}

// --- web search / fetch (against a wiremock Openverse) ---

mod web {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn solid_png(rgb: [u8; 3]) -> Vec<u8> {
        let mut buf = image::RgbImage::new(8, 8);
        for px in buf.pixels_mut() {
            *px = image::Rgb(rgb);
        }
        let mut out = Vec::new();
        image::DynamicImage::ImageRgb8(buf)
            .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
            .unwrap();
        out
    }

    fn openverse_body(image_url: &str) -> serde_json::Value {
        serde_json::json!({
            "results": [{
                "id": "u1",
                "title": "A test cat",
                "creator": "tester",
                "license": "cc0",
                "license_version": "1.0",
                "foreign_landing_url": "https://example.com/page",
                "url": image_url,
                "provider": "flickr",
                "width": 8,
                "height": 8
            }]
        })
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn search_json_lists_mock_results_exiting_zero() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/images/"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(openverse_body("https://cdn.example.com/a.jpg")),
            )
            .mount(&server)
            .await;
        let out = kaleido(&["search", "cats", "--limit", "5", "--json"])
            .env("KALEIDO_OPENVERSE_API", server.uri())
            .output()
            .unwrap();
        assert_eq!(out.status.code(), Some(0));
        let stdout = String::from_utf8(out.stdout).unwrap();
        let results: serde_json::Value = serde_json::from_str(&stdout).unwrap();
        assert_eq!(results[0]["id"], "u1");
        assert_eq!(results[0]["image_url"], "https://cdn.example.com/a.jpg");
        assert_eq!(results[0]["license"], "cc0 1.0");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn search_without_results_exits_one() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/images/"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "results": []
            })))
            .mount(&server)
            .await;
        let out = kaleido(&["search", "nothing-here"])
            .env("KALEIDO_OPENVERSE_API", server.uri())
            .output()
            .unwrap();
        assert_eq!(out.status.code(), Some(1));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn fetch_downloads_into_dir_and_writes_manifest() {
        let server = MockServer::start().await;
        let png = solid_png([77, 33, 5]);
        Mock::given(method("GET"))
            .and(path("/images/"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(openverse_body(&format!("{}/img/a.jpg", server.uri()))),
            )
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/img/a.jpg"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_bytes(png.clone())
                    .insert_header("Content-Type", "image/jpeg"),
            )
            .mount(&server)
            .await;
        let dir = temp_dir("fetch");
        let out = kaleido(&[
            "fetch",
            "cats",
            "--dir",
            dir.to_str().unwrap(),
            "--limit",
            "3",
        ])
        .env("KALEIDO_OPENVERSE_API", server.uri())
        .output()
        .unwrap();
        assert_eq!(
            out.status.code(),
            Some(0),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let saved: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert!(saved.iter().any(|f| f == "manifest.json"));
        assert!(saved
            .iter()
            .any(|f| f.starts_with("kaleido-") && f.ends_with(".jpg")));
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap())
                .unwrap();
        assert_eq!(manifest["images"][0]["provider"], "openverse");
        assert_eq!(manifest["images"][0]["query"], "cats");
        assert_eq!(
            manifest["images"][0]["source_page"],
            "https://example.com/page"
        );
        assert_eq!(
            std::fs::read(dir.join(manifest["images"][0]["file"].as_str().unwrap())).unwrap(),
            png
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn fetch_with_dead_image_source_exits_one() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/images/"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(openverse_body(&format!("{}/img/missing.png", server.uri()))),
            )
            .mount(&server)
            .await;
        let dir = temp_dir("fetch-dead");
        let out = kaleido(&["fetch", "cats", "--dir", dir.to_str().unwrap()])
            .env("KALEIDO_OPENVERSE_API", server.uri())
            .output()
            .unwrap();
        assert_eq!(out.status.code(), Some(1), "nothing saved must exit 1");
        let stdout = String::from_utf8(out.stdout).unwrap();
        assert!(stdout.contains("skipped"), "{stdout}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
