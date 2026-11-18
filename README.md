<div align="center">

# 🎞️ kaleido

**Fast, local-first image gallery for macOS — written in Rust.**

Similar-image search · color palettes · OCR with multi-model evidence · duplicate detection · machine-readiness benchmarks

[![CI](https://github.com/calpa/kaleido/actions/workflows/ci.yml/badge.svg)](https://github.com/calpa/kaleido/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust)](https://www.rust-lang.org)
[![Platform](https://img.shields.io/badge/platform-macOS-black?logo=apple)](https://github.com/calpa/kaleido)
[![crates.io](https://img.shields.io/crates/v/kaleido-core.svg)](https://crates.io/crates/kaleido-core)

<!-- TODO(milestone M2): replace with real screenshot
![kaleido screenshot](docs/assets/screenshot-grid.png)
-->

</div>

## Why kaleido

Your photos stay on your machine. kaleido indexes, analyzes, and searches them
**100% locally** — no cloud uploads, no telemetry. Built Rust-first for speed
and memory safety, with optional small local vision models (2B-class) for OCR.

## Features

### Gallery
- 🖼️ Fast thumbnail grid with smooth **zoom in/out**
- 🔍 **Search**: filenames, EXIF metadata, and — thanks to on-device OCR — **text inside images**
- 🌐 **Web image search & download** (explicit user action, opt-in network use)
- ⭐ Favorites, tags, folders; dark & light themes
- 🗺️ EXIF inspector (camera, lens, settings)

### Analysis (all local)
- 🔁 **Similar-image search** — 64-bit dHash perceptual hashing, hamming-distance ranking
- 🧹 **Duplicate detection** — exact + near-duplicate clustering
- 🎨 **Color palettes** — k-means swatches with share %, dominant color, export as JSON/CSS

### OCR ensemble (multi-model, evidence-graded)
- Run **several OCR/vision models** over the same image
- Merge results into one answer with **agreement-based evidence levels** (high / medium / low)
- Models ship as small local downloads; **pick your model in the GUI**

### Machine readiness
- `kaleido bench` measures your machine (RAM, Metal GPU, disk) against each
  model's requirements and prints a **readiness verdict** before you download anything

## Install

> Prebuilt binaries land with v0.1. Until then, build from source:

```bash
git clone https://github.com/calpa/kaleido
cd kaleido
cargo install --path crates/kaleido-cli
kaleido scan ~/Pictures
```

## CLI

```bash
kaleido scan <DIR>                     # list supported images as JSON
kaleido similar <IMAGE> <DIR>          # find perceptually similar images
kaleido palette <IMAGE> --colors 5     # extract a k-color palette
```

## Architecture

```text
crates/
  kaleido-core/   # pure library: scan · dHash · palette · EXIF · OCR ensemble
  kaleido-cli/    # clap CLI over the core
  kaleido-app/    # native macOS GUI (milestone M2, framework under research)
docs/
  research/       # framework & model research (Devin-generated, human-reviewed)
```

**Rust-first**: analysis is a pure library crate usable from CLI, GUI, or your
own code. Local inference rides on Metal via `llama-cpp-2` / `ort` (see research).

## Roadmap

| Milestone | Scope | Status |
|---|---|---|
| M0 | Repo bootstrap, CI green, core skeleton | ✅ |
| M1 | Core solid: scan · dHash · palette · duplicates · EXIF + CLI | 🚧 |
| M2 | Wireframes → native GUI shell: grid, zoom, themes (screenshot QA per milestone) | ⏳ |
| M3 | OCR ensemble: model manager (GUI selection), evidence levels, search-by-text | ⏳ |
| M4 | Machine-readiness benchmark suite + verdict panel | ⏳ |
| M5 | Web search & download, polish, v0.1 release | ⏳ |

## Contributing

PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Rustfmt + clippy clean,
tests included. Design/UX proposals: open an issue with the `design` label.

## License

[MIT](LICENSE) © 2026 Calpa
