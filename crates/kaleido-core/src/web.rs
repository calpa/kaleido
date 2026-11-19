//! Web image search and download: pluggable provider trait with an Openverse
//! default (keyless CC search), plus a download manager that validates content
//! type and size, deduplicates via dHash, and maintains a `manifest.json`.

use crate::error::{KaleidoError, KaleidoResult};
use crate::phash::{dhash, Hash64};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::io;
use std::path::Path;

/// Default Openverse API root.
pub const OPENVERSE_API: &str = "https://api.openverse.org/v1";
/// Maximum results per search request (Openverse anonymous API caps at 20).
pub const MAX_PAGE_SIZE: u32 = 20;

/// One image found by an [`ImageProvider`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Provider-local identifier.
    pub id: String,
    /// Work title, when the upstream API provides one.
    pub title: Option<String>,
    /// Author or uploader, when known.
    pub creator: Option<String>,
    /// Human-readable license, e.g. `cc0 1.0`.
    pub license: String,
    /// Landing page of the work.
    pub source_page: String,
    /// Direct URL of the image file.
    pub image_url: String,
    /// Upstream source the provider indexed (e.g. `flickr`).
    pub source: String,
    pub width: Option<u32>,
    pub height: Option<u32>,
}

/// A web image search backend.
///
/// Implementations turn a free-text query into [`SearchResult`]s. Network
/// failures surface as [`KaleidoError::Network`]; providers never panic.
#[async_trait]
pub trait ImageProvider: Send + Sync {
    /// Provider identifier recorded in download manifests.
    fn name(&self) -> &'static str;

    /// Search for images matching `query`, returning at most `limit` results.
    async fn search(&self, query: &str, limit: u32) -> KaleidoResult<Vec<SearchResult>>;
}

/// Keyless default provider backed by the Openverse CC-image search API.
pub struct OpenverseProvider {
    client: reqwest::Client,
    base_url: String,
}

impl OpenverseProvider {
    /// Provider pointed at the public Openverse API.
    pub fn new() -> KaleidoResult<Self> {
        Self::with_base_url(OPENVERSE_API)
    }

    /// Provider pointed at a custom API root (tests, self-hosted instances).
    pub fn with_base_url(base_url: &str) -> KaleidoResult<Self> {
        let client = reqwest::Client::builder()
            .user_agent(concat!(
                "kaleido/",
                env!("CARGO_PKG_VERSION"),
                " (https://github.com/calpa/kaleido)"
            ))
            .build()?;
        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
        })
    }
}

#[async_trait]
impl ImageProvider for OpenverseProvider {
    fn name(&self) -> &'static str {
        "openverse"
    }

    async fn search(&self, query: &str, limit: u32) -> KaleidoResult<Vec<SearchResult>> {
        let query = query.trim();
        if query.is_empty() {
            return Err(KaleidoError::InvalidInput("query is empty".to_string()));
        }
        let limit = limit.clamp(1, MAX_PAGE_SIZE);
        let page_size = limit.to_string();
        let endpoint = format!("{}/images/", self.base_url);
        let resp = self
            .client
            .get(&endpoint)
            .query(&[("q", query), ("page_size", page_size.as_str())])
            .send()
            .await?
            .error_for_status()?;
        let text = resp.text().await?;
        let body: OpenverseResponse = serde_json::from_str(&text)?;
        Ok(body
            .results
            .into_iter()
            .filter_map(|img| {
                // Results without a direct image URL cannot be downloaded.
                let image_url = img.url?;
                Some(SearchResult {
                    id: img.id,
                    title: img.title,
                    creator: img.creator,
                    license: license_label(img.license, img.license_version),
                    source_page: img.foreign_landing_url.unwrap_or_else(|| image_url.clone()),
                    image_url,
                    source: img.provider.unwrap_or_else(|| "unknown".to_string()),
                    width: img.width.filter(|w| *w > 0),
                    height: img.height.filter(|h| *h > 0),
                })
            })
            .take(limit as usize)
            .collect())
    }
}

#[derive(Deserialize)]
struct OpenverseResponse {
    #[serde(default)]
    results: Vec<OpenverseImage>,
}

#[derive(Deserialize)]
struct OpenverseImage {
    id: String,
    title: Option<String>,
    creator: Option<String>,
    license: Option<String>,
    license_version: Option<String>,
    foreign_landing_url: Option<String>,
    url: Option<String>,
    provider: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
}

fn license_label(license: Option<String>, version: Option<String>) -> String {
    match (license, version) {
        (Some(l), Some(v)) => format!("{l} {v}"),
        (Some(l), None) => l,
        (None, _) => "unknown".to_string(),
    }
}

/// Limits and dedup policy for [`Downloader`].
#[derive(Debug, Clone)]
pub struct DownloadConfig {
    /// Reject downloads larger than this many bytes.
    pub max_bytes: u64,
    /// dHash hamming distance at or below which an image counts as duplicate.
    pub dup_threshold: u32,
}

impl Default for DownloadConfig {
    fn default() -> Self {
        Self {
            max_bytes: 20 * 1024 * 1024,
            dup_threshold: 5,
        }
    }
}

/// Downloads search results into a directory: content-type and size checks,
/// dHash dedup (within the run and against an existing manifest), and a
/// `manifest.json` recording the source of every saved file.
pub struct Downloader {
    client: reqwest::Client,
    config: DownloadConfig,
}

impl Downloader {
    /// Downloader with [`DownloadConfig::default`].
    pub fn new() -> KaleidoResult<Self> {
        Self::with_config(DownloadConfig::default())
    }

    /// Downloader with explicit limits.
    pub fn with_config(config: DownloadConfig) -> KaleidoResult<Self> {
        let client = reqwest::Client::builder().build()?;
        Ok(Self { client, config })
    }

    /// Download every result into `dir`, merging entries into
    /// `dir/manifest.json`. A failing item is recorded as a skip with its
    /// reason; it never aborts the batch. Fatal errors (creating the
    /// directory, rewriting the manifest) propagate as [`KaleidoError`].
    pub async fn fetch_all(
        &self,
        results: &[SearchResult],
        dir: &Path,
        query: &str,
        provider: &str,
    ) -> KaleidoResult<FetchReport> {
        std::fs::create_dir_all(dir)?;
        let manifest_path = dir.join("manifest.json");
        let mut manifest = load_manifest(&manifest_path)?;
        let mut known: Vec<Hash64> = manifest
            .images
            .iter()
            .filter_map(|e| parse_hash(&e.dhash))
            .collect();
        let mut outcomes = Vec::with_capacity(results.len());
        for r in results {
            match self.fetch_one(r, dir, provider, query, &known).await {
                Ok(entry) => {
                    if let Some(h) = parse_hash(&entry.dhash) {
                        known.push(h);
                    }
                    manifest.images.push(entry.clone());
                    outcomes.push(ItemOutcome::Saved(entry));
                }
                Err(err) => outcomes.push(ItemOutcome::Skipped {
                    title: r.title.clone(),
                    url: r.image_url.clone(),
                    reason: err.to_string(),
                }),
            }
        }
        let json = serde_json::to_string_pretty(&manifest)?;
        std::fs::write(&manifest_path, json)?;
        Ok(FetchReport { outcomes })
    }

    async fn fetch_one(
        &self,
        r: &SearchResult,
        dir: &Path,
        provider: &str,
        query: &str,
        known: &[Hash64],
    ) -> KaleidoResult<ManifestEntry> {
        let resp = self
            .client
            .get(&r.image_url)
            .send()
            .await?
            .error_for_status()?;
        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .split(';')
            .next()
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();
        if !content_type.starts_with("image/") {
            return Err(KaleidoError::ContentType(content_type));
        }
        let bytes = read_capped(resp, self.config.max_bytes).await?;
        let img = image::load_from_memory(&bytes)?;
        let hash = dhash(&img);
        let hex = hash_hex(hash);
        if known
            .iter()
            .any(|h| h.similar(hash, self.config.dup_threshold))
        {
            return Err(KaleidoError::Duplicate(hex));
        }
        let file = format!(
            "kaleido-{}.{}",
            &hex[..8],
            ext_for(&content_type, &r.image_url)
        );
        std::fs::write(dir.join(&file), &bytes)?;
        Ok(ManifestEntry {
            file,
            provider: provider.to_string(),
            query: query.to_string(),
            title: r.title.clone(),
            creator: r.creator.clone(),
            license: r.license.clone(),
            source_page: r.source_page.clone(),
            image_url: r.image_url.clone(),
            dhash: hex,
            width: r.width,
            height: r.height,
        })
    }
}

/// Read a response body, bailing out once it exceeds `max_bytes`.
async fn read_capped(mut resp: reqwest::Response, max_bytes: u64) -> KaleidoResult<Vec<u8>> {
    if let Some(len) = resp.content_length() {
        if len > max_bytes {
            return Err(KaleidoError::TooLarge {
                size: len,
                limit: max_bytes,
            });
        }
    }
    let mut bytes = Vec::new();
    let mut size: u64 = 0;
    while let Some(chunk) = resp.chunk().await? {
        size += chunk.len() as u64;
        if size > max_bytes {
            return Err(KaleidoError::TooLarge {
                size,
                limit: max_bytes,
            });
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

/// One attempted download.
#[derive(Debug)]
pub enum ItemOutcome {
    /// Image saved and recorded in the manifest.
    Saved(ManifestEntry),
    /// Image skipped; `reason` carries the [`KaleidoError`] display.
    Skipped {
        title: Option<String>,
        url: String,
        reason: String,
    },
}

/// Batch summary returned by [`Downloader::fetch_all`].
#[derive(Debug, Default)]
pub struct FetchReport {
    pub outcomes: Vec<ItemOutcome>,
}

impl FetchReport {
    /// Number of images actually saved.
    pub fn saved_count(&self) -> usize {
        self.outcomes
            .iter()
            .filter(|o| matches!(o, ItemOutcome::Saved(_)))
            .count()
    }
}

/// One downloaded image as recorded in `manifest.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestEntry {
    /// File name inside the download directory.
    pub file: String,
    /// Provider that served the result.
    pub provider: String,
    /// Search phrase that produced the result.
    pub query: String,
    pub title: Option<String>,
    pub creator: Option<String>,
    pub license: String,
    /// Landing page of the work.
    pub source_page: String,
    /// Direct image URL that was downloaded.
    pub image_url: String,
    /// dHash of the saved image (16 hex digits).
    pub dhash: String,
    pub width: Option<u32>,
    pub height: Option<u32>,
}

/// Contents of a directory's `manifest.json`.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct FetchManifest {
    pub images: Vec<ManifestEntry>,
}

fn load_manifest(path: &Path) -> KaleidoResult<FetchManifest> {
    match std::fs::read_to_string(path) {
        Ok(text) => Ok(serde_json::from_str(&text)?),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(FetchManifest::default()),
        Err(e) => Err(e.into()),
    }
}

fn hash_hex(h: Hash64) -> String {
    format!("{:016x}", h.as_u64())
}

fn parse_hash(s: &str) -> Option<Hash64> {
    u64::from_str_radix(s, 16).ok().map(Hash64)
}

fn ext_for(content_type: &str, url: &str) -> String {
    let known = match content_type {
        "image/jpeg" | "image/jpg" => Some("jpg"),
        "image/png" => Some("png"),
        "image/gif" => Some("gif"),
        "image/webp" => Some("webp"),
        "image/tiff" => Some("tif"),
        "image/bmp" => Some("bmp"),
        _ => None,
    };
    known.map(str::to_string).unwrap_or_else(|| {
        Path::new(url)
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .filter(|e| (1..=5).contains(&e.len()) && e.chars().all(|c| c.is_ascii_alphanumeric()))
            .unwrap_or_else(|| "img".to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn solid_png(rgb: [u8; 3]) -> Vec<u8> {
        let mut buf = image::RgbImage::new(8, 8);
        for px in buf.pixels_mut() {
            *px = image::Rgb(rgb);
        }
        encode_png(image::DynamicImage::ImageRgb8(buf))
    }

    /// Horizontal light-to-dark ramp: with dHash's `left > right` bit rule,
    /// every comparison is set, so the hash differs maximally from any solid
    /// image (whose hash is 0).
    fn gradient_png() -> Vec<u8> {
        let mut buf = image::RgbImage::new(8, 8);
        for (x, _, px) in buf.enumerate_pixels_mut() {
            *px = image::Rgb([(7 - x) as u8 * 30, (7 - x) as u8 * 30, (7 - x) as u8 * 30]);
        }
        encode_png(image::DynamicImage::ImageRgb8(buf))
    }

    fn encode_png(img: image::DynamicImage) -> Vec<u8> {
        let mut out = Vec::new();
        img.write_to(&mut io::Cursor::new(&mut out), image::ImageFormat::Png)
            .unwrap();
        out
    }

    fn result(id: &str, url: &str) -> SearchResult {
        SearchResult {
            id: id.to_string(),
            title: Some(id.to_string()),
            creator: Some("someone".to_string()),
            license: "cc0 1.0".to_string(),
            source_page: url.to_string(),
            image_url: url.to_string(),
            source: "test".to_string(),
            width: Some(8),
            height: Some(8),
        }
    }

    fn openverse_body(images: serde_json::Value) -> serde_json::Value {
        serde_json::json!({ "result_count": 42, "results": images })
    }

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("kaleido-web-{}-{}", std::process::id(), tag));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[tokio::test]
    async fn search_parses_results_and_filters_urlless() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/images/"))
            .and(query_param("q", "cats"))
            .and(query_param("page_size", "2"))
            .respond_with(ResponseTemplate::new(200).set_body_json(openverse_body(
                serde_json::json!([
                    {
                        "id": "u1", "title": null, "creator": null,
                        "license": "cc0", "license_version": "1.0",
                        "foreign_landing_url": "https://example.com/a",
                        "url": "https://cdn.example.com/a.jpg",
                        "provider": "flickr", "width": 0, "height": 0
                    },
                    { "id": "u2", "url": null }
                ]),
            )))
            .mount(&server)
            .await;
        let provider = OpenverseProvider::with_base_url(&server.uri()).unwrap();
        let results = provider.search("cats", 2).await.unwrap();
        assert_eq!(results.len(), 1, "url-less result must be dropped");
        let r = &results[0];
        assert_eq!(r.id, "u1");
        assert_eq!(r.license, "cc0 1.0");
        assert_eq!(r.source, "flickr");
        assert_eq!(r.image_url, "https://cdn.example.com/a.jpg");
        assert_eq!(r.source_page, "https://example.com/a");
        assert_eq!(r.width, None, "zero dimensions are not reported");
    }

    #[tokio::test]
    async fn search_rejects_empty_query_without_network() {
        let server = MockServer::start().await;
        let provider = OpenverseProvider::with_base_url(&server.uri()).unwrap();
        let err = provider.search("   ", 5).await.unwrap_err();
        assert!(matches!(err, KaleidoError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn search_http_error_is_network_error() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/images/"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;
        let provider = OpenverseProvider::with_base_url(&server.uri()).unwrap();
        let err = provider.search("cats", 5).await.unwrap_err();
        assert!(matches!(err, KaleidoError::Network(_)), "{err:?}");
    }

    #[tokio::test]
    async fn search_malformed_json_is_json_error() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/images/"))
            .respond_with(ResponseTemplate::new(200).set_body_string("<html>"))
            .mount(&server)
            .await;
        let provider = OpenverseProvider::with_base_url(&server.uri()).unwrap();
        let err = provider.search("cats", 5).await.unwrap_err();
        assert!(matches!(err, KaleidoError::Json(_)), "{err:?}");
    }

    #[tokio::test]
    async fn download_saves_image_and_manifest() {
        let server = MockServer::start().await;
        let png = solid_png([10, 20, 30]);
        Mock::given(method("GET"))
            .and(path("/img/a.png"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_bytes(png.clone())
                    .insert_header("Content-Type", "image/png"),
            )
            .mount(&server)
            .await;
        let dir = temp_dir("save-manifest");
        let results = vec![result("r1", &format!("{}/img/a.png", server.uri()))];
        let report = Downloader::new()
            .unwrap()
            .fetch_all(&results, &dir, "cats", "openverse")
            .await
            .unwrap();
        assert_eq!(report.saved_count(), 1);
        let manifest: FetchManifest =
            serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap())
                .unwrap();
        assert_eq!(manifest.images.len(), 1);
        let entry = &manifest.images[0];
        assert_eq!(entry.provider, "openverse");
        assert_eq!(entry.query, "cats");
        assert_eq!(entry.license, "cc0 1.0");
        let saved = std::fs::read(dir.join(&entry.file)).unwrap();
        assert_eq!(saved, png);
        let hash = dhash(&image::load_from_memory(&png).unwrap());
        assert_eq!(entry.dhash, hash_hex(hash));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn download_skips_wrong_content_type() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/img/a"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string("<html>not an image</html>")
                    .insert_header("Content-Type", "text/html"),
            )
            .mount(&server)
            .await;
        let dir = temp_dir("content-type");
        let results = vec![result("r1", &format!("{}/img/a", server.uri()))];
        let report = Downloader::new()
            .unwrap()
            .fetch_all(&results, &dir, "cats", "openverse")
            .await
            .unwrap();
        assert_eq!(report.saved_count(), 0);
        match &report.outcomes[0] {
            ItemOutcome::Skipped { reason, .. } => {
                assert!(reason.contains("content type"), "{reason}");
            }
            ItemOutcome::Saved(_) => panic!("expected skip"),
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn download_enforces_size_cap() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/img/a.png"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_bytes(solid_png([1, 2, 3]))
                    .insert_header("Content-Type", "image/png"),
            )
            .mount(&server)
            .await;
        let dir = temp_dir("size-cap");
        let results = vec![result("r1", &format!("{}/img/a.png", server.uri()))];
        let config = DownloadConfig {
            max_bytes: 8,
            ..DownloadConfig::default()
        };
        let report = Downloader::with_config(config)
            .unwrap()
            .fetch_all(&results, &dir, "cats", "openverse")
            .await
            .unwrap();
        assert_eq!(report.saved_count(), 0);
        match &report.outcomes[0] {
            ItemOutcome::Skipped { reason, .. } => {
                assert!(reason.contains("too large"), "{reason}");
            }
            ItemOutcome::Saved(_) => panic!("expected skip"),
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn download_skips_corrupt_payload() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/img/a.png"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_bytes(b"definitely not png bytes".to_vec())
                    .insert_header("Content-Type", "image/png"),
            )
            .mount(&server)
            .await;
        let dir = temp_dir("corrupt");
        let results = vec![result("r1", &format!("{}/img/a.png", server.uri()))];
        let report = Downloader::new()
            .unwrap()
            .fetch_all(&results, &dir, "cats", "openverse")
            .await
            .unwrap();
        assert_eq!(report.saved_count(), 0);
        match &report.outcomes[0] {
            ItemOutcome::Skipped { reason, .. } => {
                assert!(reason.contains("image decode error"), "{reason}");
            }
            ItemOutcome::Saved(_) => panic!("expected skip"),
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn download_dedupes_within_run_and_against_manifest() {
        let server = MockServer::start().await;
        let png = solid_png([9, 9, 9]);
        let grad = gradient_png();
        for (p, body) in [("/img/solid.png", &png), ("/img/grad.png", &grad)] {
            Mock::given(method("GET"))
                .and(path(p))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_bytes(body.clone())
                        .insert_header("Content-Type", "image/png"),
                )
                .mount(&server)
                .await;
        }
        // The first two results serve identical bytes: perceptual duplicates.
        let results = vec![
            result("r1", &format!("{}/img/solid.png", server.uri())),
            result("r2", &format!("{}/img/solid.png", server.uri())),
            result("r3", &format!("{}/img/grad.png", server.uri())),
        ];
        let dir = temp_dir("dedupe");
        let downloader = Downloader::new().unwrap();
        let report = downloader
            .fetch_all(&results, &dir, "cats", "openverse")
            .await
            .unwrap();
        assert_eq!(report.saved_count(), 2, "solid + gradient saved");
        match &report.outcomes[1] {
            ItemOutcome::Skipped { reason, .. } => {
                assert!(reason.contains("duplicate"), "{reason}");
            }
            ItemOutcome::Saved(_) => panic!("expected duplicate skip"),
        }
        // Re-running against the same directory: manifest hashes dedupe everything.
        let again = downloader
            .fetch_all(&results, &dir, "cats", "openverse")
            .await
            .unwrap();
        assert_eq!(again.saved_count(), 0, "manifest dedup skips all");
        let manifest: FetchManifest =
            serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap())
                .unwrap();
        assert_eq!(manifest.images.len(), 2, "manifest stays at two entries");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
