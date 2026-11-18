//! Model manager: a `models/` directory described by `manifest.json`.
//!
//! Model binaries are never committed. The manifest pins, per model, every
//! file's download URL, size and SHA-256; [`ModelStore::verify`] checks the
//! files on disk against it before an engine is allowed to load them.
//!
//! Layout: `<root>/manifest.json` and `<root>/<model id>/<file name>`.

use crate::error::{KaleidoError, KaleidoResult};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};

/// Manifest file name inside the models root.
pub const MANIFEST_FILE: &str = "manifest.json";

/// Parsed `manifest.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// Manifest schema version (currently 1).
    pub version: u32,
    pub models: Vec<ManifestModel>,
}

/// One model: an id matching [`crate::models::MODEL_SPECS`] plus its files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestModel {
    pub id: String,
    pub files: Vec<ManifestFile>,
}

/// One downloadable file of a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestFile {
    /// File name inside `<root>/<model id>/`.
    pub name: String,
    pub url: String,
    /// Size in bytes.
    pub size: u64,
    /// Lowercase hex SHA-256.
    pub sha256: String,
}

impl Manifest {
    /// Parse a manifest from JSON text.
    pub fn from_json(text: &str) -> KaleidoResult<Self> {
        let manifest: Manifest = serde_json::from_str(text)
            .map_err(|e| KaleidoError::Model(format!("invalid {MANIFEST_FILE}: {e}")))?;
        for model in &manifest.models {
            for file in &model.files {
                let safe = !file.name.is_empty()
                    && !file.name.contains(['/', '\\'])
                    && file.name != "."
                    && file.name != "..";
                if !safe {
                    return Err(KaleidoError::Model(format!(
                        "{}: file name {:?} must be a plain file name",
                        model.id, file.name
                    )));
                }
            }
        }
        Ok(manifest)
    }

    /// Look up a model entry by id.
    pub fn model(&self, id: &str) -> Option<&ManifestModel> {
        self.models.iter().find(|m| m.id == id)
    }
}

/// On-disk state of one manifest file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FileState {
    /// Present and checksum matches.
    Ok,
    /// Not on disk.
    Missing,
    /// On disk but size or SHA-256 differs from the manifest.
    Corrupt,
}

/// Verification result for one file.
#[derive(Debug, Clone, Serialize)]
pub struct FileStatus {
    pub name: String,
    pub path: PathBuf,
    pub url: String,
    pub sha256: String,
    pub state: FileState,
}

/// Verification result for one model.
#[derive(Debug, Clone, Serialize)]
pub struct ModelStatus {
    pub id: String,
    pub dir: PathBuf,
    pub files: Vec<FileStatus>,
}

impl ModelStatus {
    /// True when every file is present with a matching checksum.
    pub fn is_ready(&self) -> bool {
        self.files.iter().all(|f| f.state == FileState::Ok)
    }
}

/// A models root directory plus its parsed manifest.
#[derive(Debug, Clone)]
pub struct ModelStore {
    root: PathBuf,
    manifest: Manifest,
}

impl ModelStore {
    /// Open `<root>/manifest.json`.
    pub fn open(root: impl Into<PathBuf>) -> KaleidoResult<Self> {
        let root = root.into();
        let path = root.join(MANIFEST_FILE);
        let text = std::fs::read_to_string(&path)
            .map_err(|e| KaleidoError::Model(format!("cannot read {}: {e}", path.display())))?;
        Ok(Self {
            manifest: Manifest::from_json(&text)?,
            root,
        })
    }

    /// Build a store from an already-parsed manifest.
    pub fn with_manifest(root: impl Into<PathBuf>, manifest: Manifest) -> Self {
        Self {
            root: root.into(),
            manifest,
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Directory holding a model's files (`<root>/<id>`).
    pub fn model_dir(&self, id: &str) -> PathBuf {
        self.root.join(id)
    }

    /// Check every file of model `id` against the manifest.
    pub fn status(&self, id: &str) -> KaleidoResult<ModelStatus> {
        let model = self
            .manifest
            .model(id)
            .ok_or_else(|| KaleidoError::Model(format!("model {id:?} not in {MANIFEST_FILE}")))?;
        let dir = self.model_dir(id);
        let mut files = Vec::with_capacity(model.files.len());
        for file in &model.files {
            let path = dir.join(&file.name);
            let state = file_state(&path, file)?;
            files.push(FileStatus {
                name: file.name.clone(),
                path,
                url: file.url.clone(),
                sha256: file.sha256.clone(),
                state,
            });
        }
        Ok(ModelStatus {
            id: id.to_string(),
            dir,
            files,
        })
    }

    /// Status of every model in the manifest, in manifest order.
    pub fn status_all(&self) -> KaleidoResult<Vec<ModelStatus>> {
        self.manifest
            .models
            .iter()
            .map(|m| self.status(&m.id))
            .collect()
    }

    /// Verify model `id` and return its directory, or an error naming the
    /// missing/corrupt files and how to fetch them.
    pub fn verify(&self, id: &str) -> KaleidoResult<PathBuf> {
        let status = self.status(id)?;
        if status.is_ready() {
            return Ok(status.dir);
        }
        let bad: Vec<String> = status
            .files
            .iter()
            .filter(|f| f.state != FileState::Ok)
            .map(|f| format!("{} ({:?})", f.path.display(), f.state))
            .collect();
        Err(KaleidoError::Model(format!(
            "model {id:?} not ready: {}; run scripts/fetch-models.sh",
            bad.join(", ")
        )))
    }
}

fn file_state(path: &Path, file: &ManifestFile) -> KaleidoResult<FileState> {
    let meta = match std::fs::metadata(path) {
        Ok(meta) => meta,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(FileState::Missing),
        Err(e) => return Err(e.into()),
    };
    if !meta.is_file() || meta.len() != file.size {
        return Ok(FileState::Corrupt);
    }
    let digest = sha256_file(path)?;
    Ok(if digest.eq_ignore_ascii_case(&file.sha256) {
        FileState::Ok
    } else {
        FileState::Corrupt
    })
}

/// Lowercase hex SHA-256 of a file, streamed.
pub fn sha256_file(path: &Path) -> KaleidoResult<String> {
    let mut reader = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("kaleido-store-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    // sha256("hello\n")
    const HELLO_SHA: &str = "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03";

    fn manifest(name: &str) -> Manifest {
        Manifest::from_json(&format!(
            r#"{{"version":1,"models":[{{"id":"m","files":[{{"name":"{name}","url":"https://example.invalid/{name}","size":6,"sha256":"{HELLO_SHA}"}}]}}]}}"#
        ))
        .unwrap()
    }

    #[test]
    fn sha256_matches_known_vector() {
        let root = temp_root("sha");
        let path = root.join("hello.txt");
        std::fs::write(&path, b"hello\n").unwrap();
        assert_eq!(sha256_file(&path).unwrap(), HELLO_SHA);
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn status_reports_missing_corrupt_and_ok() {
        let root = temp_root("status");
        let store = ModelStore::with_manifest(&root, manifest("a.bin"));
        assert_eq!(
            store.status("m").unwrap().files[0].state,
            FileState::Missing
        );
        assert!(store.verify("m").is_err());

        std::fs::create_dir_all(root.join("m")).unwrap();
        std::fs::write(root.join("m/a.bin"), b"HELLO\n").unwrap();
        assert_eq!(
            store.status("m").unwrap().files[0].state,
            FileState::Corrupt
        );
        assert!(store.verify("m").is_err());

        std::fs::write(root.join("m/a.bin"), b"hello\n").unwrap();
        assert!(store.status("m").unwrap().is_ready());
        assert_eq!(store.verify("m").unwrap(), root.join("m"));

        assert!(store.status("unknown").is_err());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn manifest_rejects_path_traversal_names() {
        for bad in ["../x", "a/b", "..", ""] {
            let json = format!(
                r#"{{"version":1,"models":[{{"id":"m","files":[{{"name":"{bad}","url":"u","size":1,"sha256":"00"}}]}}]}}"#
            );
            assert!(Manifest::from_json(&json).is_err(), "{bad:?} accepted");
        }
    }

    #[test]
    fn repo_manifest_parses_and_covers_paddle() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
        let store = ModelStore::open(&root).unwrap();
        let paddle = store.manifest().model("paddle-ocrv5-mobile").unwrap();
        assert_eq!(paddle.files.len(), 3);
        for file in &paddle.files {
            assert_eq!(file.sha256.len(), 64);
            assert!(file.url.starts_with("https://"));
        }
    }
}
