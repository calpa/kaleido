//! kaleido CLI: scan / similar / palette commands over kaleido-core.

use anyhow::Context;
use clap::{Parser, Subcommand};
use kaleido_core::{dhash_file, palette_file, scan, Hash64};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "kaleido", version, about = "Fast local image gallery toolkit", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
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
    /// Extract a k-color palette from an image as JSON.
    Palette {
        /// Image path.
        #[arg(value_name = "IMAGE")]
        image: PathBuf,
        /// Number of palette colors.
        #[arg(long, default_value_t = 5)]
        colors: u8,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Scan { root } => {
            let stats = scan(&root).context("scan failed")?;
            println!("{}", serde_json::to_string_pretty(&stats)?);
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
        }
        Command::Palette { image, colors } => {
            let p = palette_file(&image, colors).context("palette failed")?;
            println!("{}", serde_json::to_string_pretty(&p)?);
        }
    }
    Ok(())
}
