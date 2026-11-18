//! kaleido CLI: scan / similar / palette / dupes / ocr commands over kaleido-core.
//!
//! Exit codes: 0 = success (and for `dupes`: groups found), 1 = `dupes` with
//! no duplicate groups found, 2 = runtime error.

use anyhow::Context;
use clap::{Parser, Subcommand, ValueEnum};
use kaleido_core::{
    cluster_similar, dhash_file, palette_file, phash_file, run_ensemble, scan, FakeEngine, Hash64,
    OcrEngine,
};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[derive(Parser)]
#[command(name = "kaleido", version, about = "Fast local image gallery toolkit", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum HashAlgo {
    /// 64-bit difference hash (fast, gradient-based).
    Dhash,
    /// 64-bit DCT-based perceptual hash (robust to rescaling).
    Phash,
}

impl HashAlgo {
    fn name(self) -> &'static str {
        match self {
            HashAlgo::Dhash => "dhash",
            HashAlgo::Phash => "phash",
        }
    }
}

#[derive(Subcommand)]
enum Command {
    /// Scan a folder and list supported images as JSON.
    Scan {
        /// Folder to scan recursively.
        #[arg(value_name = "DIR")]
        root: PathBuf,
    },
    /// Find images similar to the reference within a folder (dHash + hamming).
    Similar {
        /// Reference image path.
        #[arg(value_name = "IMAGE")]
        reference: PathBuf,
        /// Folder to search.
        #[arg(value_name = "DIR")]
        root: PathBuf,
        /// Hamming distance threshold (0-64).
        #[arg(long, default_value_t = 10)]
        threshold: u32,
    },
    /// Extract a k-color palette from an image.
    Palette {
        /// Image path.
        #[arg(value_name = "IMAGE")]
        image: PathBuf,
        /// Number of palette colors.
        #[arg(long, default_value_t = 5)]
        colors: u8,
        /// Print CSS custom properties (`:root --c1..--ck`) instead of JSON.
        #[arg(long)]
        css: bool,
    },
    /// Detect duplicate and near-duplicate image groups in a folder.
    ///
    /// Exit codes: 0 = duplicate groups found, 1 = none found, 2 = error.
    Dupes {
        /// Folder to scan recursively.
        #[arg(value_name = "DIR")]
        root: PathBuf,
        /// Hamming distance threshold (0-64) for grouping.
        #[arg(long, default_value_t = 5)]
        threshold: u32,
        /// Perceptual hash algorithm to compare with.
        #[arg(long, value_enum, default_value_t = HashAlgo::Dhash)]
        hash: HashAlgo,
        /// Emit machine-readable JSON instead of text.
        #[arg(long)]
        json: bool,
    },
    /// OCR an image through the evidence-graded engine ensemble.
    ///
    /// Engines: `fake` (always built) and `paddle-ocrv5-mobile` (alias
    /// `ppocrv5`, needs the `engine-ort` feature and downloaded models).
    Ocr {
        /// Image path.
        #[arg(value_name = "IMAGE")]
        image: PathBuf,
        /// Engine to run; repeat to ensemble several engines.
        #[arg(long = "engine", value_name = "NAME", default_value = "fake")]
        engines: Vec<String>,
        /// Models root holding manifest.json (defaults to $KALEIDO_MODELS, else ./models).
        #[arg(
            long,
            value_name = "DIR",
            env = "KALEIDO_MODELS",
            default_value = "models"
        )]
        models: PathBuf,
        /// Emit machine-readable JSON instead of text.
        #[arg(long)]
        json: bool,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(code) => ExitCode::from(code),
        Err(err) => {
            eprintln!("kaleido: {err:#}");
            ExitCode::from(2)
        }
    }
}

fn run(cli: Cli) -> anyhow::Result<u8> {
    match cli.command {
        Command::Scan { root } => {
            let stats = scan(&root).context("scan failed")?;
            println!("{}", serde_json::to_string_pretty(&stats)?);
            Ok(0)
        }
        Command::Similar {
            reference,
            root,
            threshold,
        } => {
            use rayon::prelude::*;
            let reference_hash: Hash64 =
                dhash_file(&reference).context("reference image unreadable")?;
            let stats = scan(&root).context("scan failed")?;
            let mut matches: Vec<(String, u32)> = stats
                .entries
                .par_iter()
                .filter_map(|e| {
                    let h = dhash_file(&e.path).ok()?;
                    let d = reference_hash.hamming(h);
                    (d <= threshold).then(|| (e.path.display().to_string(), d))
                })
                .collect();
            matches.sort_by_key(|(_, d)| *d);
            println!("{}", serde_json::to_string_pretty(&matches)?);
            Ok(0)
        }
        Command::Palette { image, colors, css } => {
            let p = palette_file(&image, colors).context("palette failed")?;
            if css {
                println!("{}", p.to_css_vars());
            } else {
                println!("{}", serde_json::to_string_pretty(&p)?);
            }
            Ok(0)
        }
        Command::Dupes {
            root,
            threshold,
            hash,
            json,
        } => dupes(&root, threshold, hash, json),
        Command::Ocr {
            image,
            engines,
            models,
            json,
        } => ocr(&image, &engines, &models, json),
    }
}

#[derive(Serialize)]
struct DupesReport<'a> {
    algorithm: &'static str,
    threshold: u32,
    scanned: u64,
    groups: Vec<DupeGroup<'a>>,
}

#[derive(Serialize)]
struct DupeGroup<'a> {
    paths: Vec<&'a str>,
    /// Largest pairwise hamming distance inside the group.
    max_hamming: u32,
}

fn ocr(image: &Path, engines: &[String], models: &Path, json: bool) -> anyhow::Result<u8> {
    let mut built: Vec<Box<dyn OcrEngine>> = Vec::with_capacity(engines.len());
    for name in engines {
        built.push(build_engine(name, models)?);
    }
    let img = image::open(image).with_context(|| format!("cannot open {}", image.display()))?;
    let blocks = run_ensemble(&mut built, &img).context("ocr failed")?;
    if json {
        println!("{}", serde_json::to_string_pretty(&blocks)?);
        return Ok(0);
    }
    for b in &blocks {
        let bbox = b.bbox.map_or_else(
            || "-".to_string(),
            |r| format!("[{:.3}, {:.3}, {:.3}, {:.3}]", r.x0, r.y0, r.x1, r.y1),
        );
        let flag = if b.possible_hallucination {
            " (possible hallucination)"
        } else {
            ""
        };
        let evidence = serde_json::to_value(b.evidence)?;
        let evidence = evidence.as_str().unwrap_or("?");
        println!("{evidence:<6} {:<20} {bbox} {}{flag}", b.model, b.text);
    }
    Ok(0)
}

fn build_engine(name: &str, models: &Path) -> anyhow::Result<Box<dyn OcrEngine>> {
    match name {
        FakeEngine::ID => Ok(Box::new(FakeEngine::default())),
        kaleido_core::ppocr::ENGINE_ID | "ppocrv5" => paddle_engine(models),
        other => anyhow::bail!(
            "unknown engine {other:?} (available: {}, {})",
            FakeEngine::ID,
            kaleido_core::ppocr::ENGINE_ID
        ),
    }
}

#[cfg(feature = "engine-ort")]
fn paddle_engine(models: &Path) -> anyhow::Result<Box<dyn OcrEngine>> {
    let store = kaleido_core::ModelStore::open(models)?;
    let dir = store.verify(kaleido_core::ppocr::ENGINE_ID)?;
    Ok(Box::new(kaleido_core::PaddleOcrV5::init(&dir)?))
}

#[cfg(not(feature = "engine-ort"))]
fn paddle_engine(_models: &Path) -> anyhow::Result<Box<dyn OcrEngine>> {
    anyhow::bail!(
        "engine {:?} requires a build with `--features engine-ort`",
        kaleido_core::ppocr::ENGINE_ID
    )
}

fn dupes(root: &std::path::Path, threshold: u32, hash: HashAlgo, json: bool) -> anyhow::Result<u8> {
    use rayon::prelude::*;

    let stats = scan(root).context("scan failed")?;
    let hashed: Vec<(String, Hash64)> = stats
        .entries
        .par_iter()
        .filter_map(|e| {
            let h = match hash {
                HashAlgo::Dhash => dhash_file(&e.path),
                HashAlgo::Phash => phash_file(&e.path),
            }
            .ok()?;
            Some((e.path.display().to_string(), h))
        })
        .collect();

    let hashes: Vec<Hash64> = hashed.iter().map(|(_, h)| *h).collect();
    let index_groups = cluster_similar(&hashes, threshold, 2);
    let groups: Vec<DupeGroup<'_>> = index_groups
        .iter()
        .map(|members| {
            let paths: Vec<&str> = members.iter().map(|&i| hashed[i].0.as_str()).collect();
            let max_hamming = members
                .iter()
                .flat_map(|&i| members.iter().map(move |&j| (i, j)))
                .filter(|(i, j)| i < j)
                .map(|(i, j)| hashed[i].1.hamming(hashed[j].1))
                .max()
                .unwrap_or(0);
            DupeGroup { paths, max_hamming }
        })
        .collect();

    let found = !groups.is_empty();
    if json {
        let report = DupesReport {
            algorithm: hash.name(),
            threshold,
            scanned: stats.scanned,
            groups,
        };
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else if found {
        println!(
            "{} duplicate group(s) in {} image(s):",
            groups.len(),
            stats.scanned
        );
        for (n, g) in groups.iter().enumerate() {
            println!(
                "group {} ({} image(s), max hamming {})",
                n + 1,
                g.paths.len(),
                g.max_hamming
            );
            for path in &g.paths {
                println!("  {path}");
            }
        }
    } else {
        println!(
            "no duplicate groups ({} image(s) scanned, threshold {threshold})",
            stats.scanned
        );
    }
    // Exit-code semantics: 0 = groups found, 1 = none found.
    Ok(if found { 0 } else { 1 })
}
