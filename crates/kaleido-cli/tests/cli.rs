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
