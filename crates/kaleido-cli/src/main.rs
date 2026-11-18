//! kaleido CLI: scan / similar / palette / dupes commands over kaleido-core.
//!
//! Exit codes: 0 = success (and for `dupes`: groups found), 1 = `dupes` with
//! no duplicate groups found, 2 = runtime error.

use anyhow::Context;
use clap::{Parser, Subcommand, ValueEnum};
use kaleido_core::{cluster_similar, dhash_file, palette_file, phash_file, scan, Hash64};
use serde::Serialize;
use std::path::PathBuf;
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
